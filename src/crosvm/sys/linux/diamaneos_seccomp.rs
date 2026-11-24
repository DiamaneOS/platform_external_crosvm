// Copyright (C) 2026 DiamaneOS
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! DiamaneOS: crosvm's seccomp filter. Android starts crosvm with --disable-sandbox, so no device
//! runs in a jail and crosvm's own policies (jail/seccomp) never apply. A guest that escaped its
//! kernel reaches crosvm's vCPU threads and device workers first. crosvm installs this filter
//! itself in run_control, once the vCPUs passed their start barrier: the VM exists, its memory is
//! lent, every device and vCPU is set up. TSYNC puts it on every thread. crosvm starts no program
//! afterwards, so no_new_privs costs nothing here. From then on crosvm may only:
//!
//! - make the system calls in `ALLOWED`: the control loop, the vCPU loop, the device workers of the
//!   devices Android's VMs use (block, console, serial, vhost-vsock), tubes, logging and the
//!   allocator.
//! - use ioctl with the commands in `IOCTLS` only: the Gunyah calls after setup, vhost-vsock's
//!   activation and reset, descriptor flags. A filter cannot see the descriptor, so these are
//!   commands, never whole ioctl types.
//! - start threads, never processes: clone only with CLONE_THREAD, clone3 answered ENOSYS (its
//!   flags are in memory a filter cannot read, bionic then uses clone), no execve.
//! - signal itself only (vCPU kicks are tgkill to its own threads): kill, tgkill, rt_sigqueueinfo
//!   and rt_tgsigqueueinfo with its own pid. tkill, pidfd_open and pidfd_getfd are refused.
//! - open AF_UNIX sockets only (logd's). The control socket and the tubes exist already.
//! - map or protect memory without PROT_EXEC: crosvm maps no code after setup.
//! - at its exit: restore stdin's terminal mode (TCGETS and TCSETS on descriptor 0 only) and reap
//!   children without waiting (wait4 for any child with WNOHANG). With --disable-sandbox crosvm
//!   has no children.
//! - read its descriptor limit (getrlimit with RLIMIT_NOFILE only). bionic's fdsan reads it once,
//!   on whichever thread first uses a descriptor above 127.
//!
//! The vCPU watchdog (vmwdt) reads each vCPU thread's /proc stat file, which run_control opens
//! before the filter goes on (devices/src/diamaneos_vcpu_stat.rs), so openat stays refused.
//!
//! Anything else kills crosvm on user builds (KILL_ON_USER_BUILDS): the VM ends and its VM manager
//! reports it dead. Debuggable builds log it (audit type 1326) and allow it, so that a phone run
//! of every VM use shows what is missing, unless the test hook (TEST_ENV) asks for the user-build
//! action. The user-build action is also text in the program (USER_BUILDS), which the start line
//! names and the image check of diamaneos-tools reads from the built crosvm.
//!
//! Where the list comes from: the code that runs after the install point (run_control's loop,
//! vcpu_loop, the virtio workers of block, console and vhost-vsock, the control socket requests the
//! VM manager sends), crosvm's own aarch64 device policies for those devices, and an strace of
//! every VM use on the phone.

#![allow(dead_code)]

use anyhow::ensure;
use anyhow::Result;

/// One BPF instruction (struct sock_filter).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Insn {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}

const BPF_LD_W_ABS: u16 = 0x20;
const BPF_JEQ_K: u16 = 0x15;
const BPF_JSET_K: u16 = 0x45;
const BPF_RET_K: u16 = 0x06;

pub const RET_KILL_PROCESS: u32 = 0x8000_0000;
pub const RET_ERRNO: u32 = 0x0005_0000;
pub const RET_LOG: u32 = 0x7ffc_0000;
pub const RET_ALLOW: u32 = 0x7fff_0000;

/// Offsets in struct seccomp_data.
const NR: u32 = 0;
const ARCH: u32 = 4;
const fn arg_lo(i: u32) -> u32 {
    16 + 8 * i
}

#[cfg(target_arch = "aarch64")]
const AUDIT_ARCH: u32 = 0xC000_00B7; // AUDIT_ARCH_AARCH64
#[cfg(target_arch = "x86_64")]
const AUDIT_ARCH: u32 = 0xC000_003E; // AUDIT_ARCH_X86_64

/// Newer calls, numbered alike on every architecture since Linux 5.1.
const SYS_PIDFD_OPEN: libc::c_long = 434;
const SYS_CLONE3: libc::c_long = 435;
const SYS_PIDFD_GETFD: libc::c_long = 438;

/// The system calls crosvm makes after the filter.
pub const ALLOWED: &[(libc::c_long, &str)] = &[
    // Memory: bionic, scudo, Rust's std, guest memory teardown. mmap and mprotect are in `build`
    // (no PROT_EXEC).
    (libc::SYS_brk, "brk"),
    (libc::SYS_munmap, "munmap"),
    (libc::SYS_mremap, "mremap"),
    (libc::SYS_madvise, "madvise"),
    (libc::SYS_prctl, "prctl"),
    (libc::SYS_getrandom, "getrandom"),
    (libc::SYS_getcpu, "getcpu"),
    // Threads: device workers started when the guest activates a device, their sync, Rust's std
    // threads.
    (libc::SYS_futex, "futex"),
    (libc::SYS_set_robust_list, "set_robust_list"),
    (libc::SYS_sched_yield, "sched_yield"),
    (libc::SYS_sched_getaffinity, "sched_getaffinity"),
    (libc::SYS_gettid, "gettid"),
    (libc::SYS_getpid, "getpid"),
    (libc::SYS_getuid, "getuid"),
    (libc::SYS_geteuid, "geteuid"),
    (libc::SYS_exit, "exit"),
    (libc::SYS_exit_group, "exit_group"),
    (libc::SYS_restart_syscall, "restart_syscall"),
    // Signals: handlers, masks, the vCPU kick signal cleared again (sigtimedwait, sigpending).
    // Sending is in `build`.
    (libc::SYS_rt_sigaction, "rt_sigaction"),
    (libc::SYS_rt_sigprocmask, "rt_sigprocmask"),
    (libc::SYS_rt_sigreturn, "rt_sigreturn"),
    (libc::SYS_rt_sigtimedwait, "rt_sigtimedwait"),
    (libc::SYS_rt_sigpending, "rt_sigpending"),
    (libc::SYS_sigaltstack, "sigaltstack"),
    // Time and timers.
    (libc::SYS_clock_gettime, "clock_gettime"),
    (libc::SYS_clock_getres, "clock_getres"),
    (libc::SYS_clock_nanosleep, "clock_nanosleep"),
    (libc::SYS_nanosleep, "nanosleep"),
    (libc::SYS_gettimeofday, "gettimeofday"),
    (libc::SYS_timerfd_create, "timerfd_create"),
    (libc::SYS_timerfd_settime, "timerfd_settime"),
    (libc::SYS_timerfd_gettime, "timerfd_gettime"),
    // Descriptors: disks, console and log pipes, tubes, events, the wait context.
    (libc::SYS_read, "read"),
    (libc::SYS_write, "write"),
    (libc::SYS_readv, "readv"),
    (libc::SYS_writev, "writev"),
    (libc::SYS_pread64, "pread64"),
    (libc::SYS_pwrite64, "pwrite64"),
    (libc::SYS_preadv, "preadv"),
    (libc::SYS_pwritev, "pwritev"),
    (libc::SYS_lseek, "lseek"),
    (libc::SYS_close, "close"),
    (libc::SYS_dup, "dup"),
    (libc::SYS_dup3, "dup3"),
    (libc::SYS_fcntl, "fcntl"),
    #[cfg(target_arch = "aarch64")]
    (80, "fstat"),
    #[cfg(target_arch = "x86_64")]
    (libc::SYS_fstat, "fstat"),
    (libc::SYS_statx, "statx"),
    (libc::SYS_fsync, "fsync"),
    (libc::SYS_fdatasync, "fdatasync"),
    (libc::SYS_fallocate, "fallocate"),
    (libc::SYS_eventfd2, "eventfd2"),
    (libc::SYS_ppoll, "ppoll"),
    (libc::SYS_epoll_create1, "epoll_create1"),
    (libc::SYS_epoll_ctl, "epoll_ctl"),
    (libc::SYS_epoll_pwait, "epoll_pwait"),
    // Sockets: the control socket (a connection per VM manager request), tubes, liblog. socket
    // itself is in `build`.
    (libc::SYS_accept4, "accept4"),
    (libc::SYS_connect, "connect"),
    (libc::SYS_socketpair, "socketpair"),
    (libc::SYS_sendmsg, "sendmsg"),
    (libc::SYS_recvmsg, "recvmsg"),
    (libc::SYS_sendto, "sendto"),
    (libc::SYS_recvfrom, "recvfrom"),
    (libc::SYS_getsockopt, "getsockopt"),
    (libc::SYS_setsockopt, "setsockopt"),
    (libc::SYS_shutdown, "shutdown"),
    #[cfg(target_arch = "aarch64")]
    (293, "rseq"),
    #[cfg(target_arch = "x86_64")]
    (334, "rseq"),
    #[cfg(target_arch = "x86_64")]
    (libc::SYS_poll, "poll"),
];

/// ioctl request encoding (asm-generic/ioctl.h).
const fn io(ty: u32, nr: u32) -> u32 {
    (ty << 8) | nr
}
const fn iow(ty: u32, nr: u32, size: u32) -> u32 {
    (1 << 30) | (size << 16) | (ty << 8) | nr
}
const fn ior(ty: u32, nr: u32, size: u32) -> u32 {
    (2 << 30) | (size << 16) | (ty << 8) | nr
}
const fn iowr(ty: u32, nr: u32, size: u32) -> u32 {
    (3 << 30) | (size << 16) | (ty << 8) | nr
}

/// GH_IOCTL_TYPE and GH_ANDROID_IOCTL_TYPE (hypervisor/src/gunyah/gunyah_sys.rs).
const GH: u32 = 0x47;
const GH_ANDROID: u32 = 0x41;
/// VHOST_VIRTIO (linux/vhost.h).
const VHOST: u32 = 0xAF;

/// ioctl commands crosvm uses after setup.
pub const IOCTLS: &[(u32, &str)] = &[
    // Gunyah: the vCPU loop, ioeventfds and irqfds the guest's PCI setup adds and removes
    // (gh_fn_desc, 16 bytes).
    (io(GH, 0x5), "GH_VCPU_RUN"),
    (iow(GH, 0x4, 16), "GH_VM_ADD_FUNCTION"),
    (iow(GH, 0x7, 16), "GH_VM_REMOVE_FUNCTION"),
    // vhost-vsock: activation when the guest's driver starts the device, and its reset.
    (ior(VHOST, 0x00, 8), "VHOST_GET_FEATURES"),
    (iow(VHOST, 0x00, 8), "VHOST_SET_FEATURES"),
    (io(VHOST, 0x01), "VHOST_SET_OWNER"),
    (io(VHOST, 0x02), "VHOST_RESET_OWNER"),
    (iow(VHOST, 0x03, 8), "VHOST_SET_MEM_TABLE"),
    (iow(VHOST, 0x10, 8), "VHOST_SET_VRING_NUM"),
    (iow(VHOST, 0x11, 40), "VHOST_SET_VRING_ADDR"),
    (iow(VHOST, 0x12, 8), "VHOST_SET_VRING_BASE"),
    (iowr(VHOST, 0x12, 8), "VHOST_GET_VRING_BASE"),
    (iow(VHOST, 0x20, 8), "VHOST_SET_VRING_KICK"),
    (iow(VHOST, 0x21, 8), "VHOST_SET_VRING_CALL"),
    (iow(VHOST, 0x22, 8), "VHOST_SET_VRING_ERR"),
    (iow(VHOST, 0x60, 8), "VHOST_VSOCK_SET_GUEST_CID"),
    (iow(VHOST, 0x61, 4), "VHOST_VSOCK_SET_RUNNING"),
    // A raw card's discard (base::linux::discard_block, _IO(0x12, 119) as crosvm defines it).
    (0x1277, "BLKDISCARD"),
    // Descriptor flags and the bytes waiting on a socket.
    (0x5421, "FIONBIO"),
    (0x541B, "FIONREAD"),
    (0x5451, "FIOCLEX"),
    (0x5450, "FIONCLEX"),
];

/// stdin's terminal state: isatty, and the raw and canonical mode around the run
/// (base::Terminal in run_control).
const TCGETS: u32 = 0x5401;
const TCSETS: u32 = 0x5402;

/// The filter for crosvm with pid `pid`. `default` is what happens to any other call:
/// RET_KILL_PROCESS on user builds (KILL_ON_USER_BUILDS), RET_LOG otherwise.
pub fn build(pid: u32, default: u32) -> Vec<Insn> {
    let ld = |k| Insn {
        code: BPF_LD_W_ABS,
        jt: 0,
        jf: 0,
        k,
    };
    let ret = |k| Insn {
        code: BPF_RET_K,
        jt: 0,
        jf: 0,
        k,
    };
    let jeq = |k, jt, jf| Insn {
        code: BPF_JEQ_K,
        jt,
        jf,
        k,
    };
    let jset = |k, jt, jf| Insn {
        code: BPF_JSET_K,
        jt,
        jf,
        k,
    };
    let deny = RET_ERRNO | libc::EPERM as u32;
    let mut p = vec![
        ld(ARCH),
        jeq(AUDIT_ARCH, 1, 0),
        ret(RET_KILL_PROCESS),
        ld(NR),
    ];
    // Signals with a pid in the first argument: its own only.
    for nr in [
        libc::SYS_kill,
        libc::SYS_tgkill,
        libc::SYS_rt_sigqueueinfo,
        libc::SYS_rt_tgsigqueueinfo,
    ] {
        p.extend([
            jeq(nr as u32, 0, 4),
            ld(arg_lo(0)),
            jeq(pid, 1, 0),
            ret(deny),
            ret(RET_ALLOW),
        ]);
    }
    // Never: a thread id may be any process's, and pidfds of other processes.
    for nr in [libc::SYS_tkill, SYS_PIDFD_OPEN, SYS_PIDFD_GETFD] {
        p.extend([jeq(nr as u32, 0, 1), ret(deny)]);
    }
    // Threads only, no new process.
    p.extend([
        jeq(libc::SYS_clone as u32, 0, 4),
        ld(arg_lo(0)),
        jset(libc::CLONE_THREAD as u32, 0, 1),
        ret(RET_ALLOW),
        ret(default),
    ]);
    p.extend([
        jeq(SYS_CLONE3 as u32, 0, 1),
        ret(RET_ERRNO | libc::ENOSYS as u32),
    ]);
    // No new code: mmap and mprotect without PROT_EXEC (the protection is the third argument of
    // both).
    for nr in [libc::SYS_mmap, libc::SYS_mprotect] {
        p.extend([
            jeq(nr as u32, 0, 4),
            ld(arg_lo(2)),
            jset(libc::PROT_EXEC as u32, 0, 1),
            ret(default),
            ret(RET_ALLOW),
        ]);
    }
    // bionic's fdsan sizes its descriptor table from RLIMIT_NOFILE, once per process.
    p.extend([
        jeq(libc::SYS_getrlimit as u32, 0, 4),
        ld(arg_lo(0)),
        jeq(libc::RLIMIT_NOFILE as u32, 0, 1),
        ret(RET_ALLOW),
        ret(default),
    ]);
    // At exit: reap any child without waiting (src/sys/linux/main.rs, wait_all_children).
    p.extend([
        jeq(libc::SYS_wait4 as u32, 0, 6),
        ld(arg_lo(0)),
        jeq(u32::MAX, 0, 3),
        ld(arg_lo(2)),
        jeq(libc::WNOHANG as u32, 0, 1),
        ret(RET_ALLOW),
        ret(default),
    ]);
    // Unix sockets only.
    p.extend([
        jeq(libc::SYS_socket as u32, 0, 4),
        ld(arg_lo(0)),
        jeq(libc::AF_UNIX as u32, 0, 1),
        ret(RET_ALLOW),
        ret(default),
    ]);
    // ioctl on stdin: its terminal state only. Any other ioctl goes on to the list below.
    p.extend([
        jeq(libc::SYS_ioctl as u32, 0, 7),
        ld(arg_lo(0)),
        jeq(0, 0, 4),
        ld(arg_lo(1)),
        jeq(TCGETS, 1, 0),
        jeq(TCSETS, 0, 1),
        ret(RET_ALLOW),
        ld(NR),
    ]);
    // ioctl: the commands in IOCTLS.
    p.extend([
        jeq(libc::SYS_ioctl as u32, 0, (IOCTLS.len() + 3) as u8),
        ld(arg_lo(1)),
    ]);
    for (i, (cmd, _)) in IOCTLS.iter().enumerate() {
        p.push(jeq(*cmd, (IOCTLS.len() - i) as u8, 0));
    }
    p.extend([ret(default), ret(RET_ALLOW)]);
    for (nr, _) in ALLOWED {
        p.extend([jeq(*nr as u32, 0, 1), ret(RET_ALLOW)]);
    }
    p.push(ret(default));
    p
}

#[repr(C)]
struct Prog {
    len: u16,
    filter: *const Insn,
}

const SECCOMP_SET_MODE_FILTER: libc::c_uint = 1;
const SECCOMP_FILTER_FLAG_TSYNC: libc::c_uint = 1;

/// Installs `filter` on every thread of this process.
fn install_tsync(filter: &[Insn]) -> Result<()> {
    ensure!(filter.len() < u16::MAX as usize, "filter too long");
    // SAFETY: prctl with these arguments only sets no_new_privs for this process.
    let r = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    ensure!(r == 0, "no_new_privs: {}", std::io::Error::last_os_error());
    let prog = Prog {
        len: filter.len() as u16,
        filter: filter.as_ptr(),
    };
    // SAFETY: seccomp reads the program, which stays valid for the call. The kernel copies it.
    let r = unsafe {
        libc::syscall(
            libc::SYS_seccomp,
            SECCOMP_SET_MODE_FILTER,
            SECCOMP_FILTER_FLAG_TSYNC,
            &prog as *const Prog,
        )
    };
    ensure!(r >= 0, "seccomp: {}", std::io::Error::last_os_error());
    // With TSYNC a positive result is the id of a thread that could not take the filter, and then
    // no thread has it.
    ensure!(r == 0, "seccomp: thread {r} could not take the filter");
    Ok(())
}

/// Whether user builds kill on a call outside the list: true since the phone gate ran every VM use
/// with zero audit type 1326 lines from crosvm.
pub const KILL_ON_USER_BUILDS: bool = true;

/// The user-build action as text in the program: the start line names it, and the image check of
/// diamaneos-tools (vm-filters-kill-on-user-builds) reads it from the built crosvm. `action` reads
/// it too, so it stays in the program.
pub const USER_BUILDS: &str = if KILL_ON_USER_BUILDS {
    "DiamaneOS seccomp filter: user builds kill"
} else {
    "DiamaneOS seccomp filter: user builds log"
};

/// The test hook of debuggable builds, read from crosvm's environment, which it has from its VM
/// manager and that from its client (`vm run-microdroid` in an adb shell): "user" gives the filter
/// the user-build action. "probe-crosvm" does too, and right after the filter is on crosvm makes
/// one call that no VM filter lists (PROBE_CALL), so it ends by SIGSYS where user builds kill.
/// Any other value, or none, changes nothing. User builds never read it.
pub const TEST_ENV: &str = "DIAMANEOS_VM_FILTER_TEST";

/// The probe's call: getpgid, on no list of the VM filters, harmless where it is only logged.
pub const PROBE_CALL: libc::c_long = libc::SYS_getpgid;

/// What the filter does with a call outside the list (`default` of `build`), and whether crosvm
/// makes one on purpose. `test` is TEST_ENV's value on a debuggable build.
pub fn action(debuggable: bool, test: Option<&str>) -> (u32, bool) {
    let user = if std::hint::black_box(USER_BUILDS).ends_with(" kill") {
        RET_KILL_PROCESS
    } else {
        RET_LOG
    };
    if !debuggable {
        return (user, false);
    }
    match test {
        Some("user") => (user, false),
        Some("probe-crosvm") => (user, true),
        _ => (RET_LOG, false),
    }
}

/// The probe: one call outside the list, on purpose.
fn probe() {
    base::warn!("seccomp filter test: a call outside the list (getpgid) on purpose");
    // SAFETY: getpgid(0) only reads this process's group.
    unsafe { libc::syscall(PROBE_CALL, 0) };
    base::warn!("seccomp filter test: still running after the call outside the list");
}

/// Whether this is a debuggable build (ro.debuggable=1): the filter then logs instead of killing,
/// unless the test hook asks otherwise.
#[cfg(target_os = "android")]
fn debuggable() -> bool {
    const PROP_VALUE_MAX: usize = 92;
    let mut v = [0u8; PROP_VALUE_MAX];
    // SAFETY: __system_property_get writes at most PROP_VALUE_MAX bytes into `v`, NUL-ended.
    let n =
        unsafe { libc::__system_property_get(c"ro.debuggable".as_ptr(), v.as_mut_ptr().cast()) };
    n == 1 && v[0] == b'1'
}

#[cfg(not(target_os = "android"))]
fn debuggable() -> bool {
    true
}

/// crosvm's filter on every thread: kill on user builds, log on debuggable builds. Any failure
/// ends crosvm (run_control returns it), on every build.
pub fn install() -> Result<()> {
    let debuggable = debuggable();
    let test = if debuggable {
        std::env::var(TEST_ENV).ok()
    } else {
        None
    };
    let (default, probe_now) = action(debuggable, test.as_deref());
    install_tsync(&build(std::process::id(), default))?;
    base::info!(
        "seccomp filter on ({} anything else), {}",
        if default == RET_KILL_PROCESS {
            "killing"
        } else {
            "logging"
        },
        USER_BUILDS
    );
    if probe_now {
        probe();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the program on one call, as the kernel would: the action it returns.
    fn eval(p: &[Insn], nr: libc::c_long, args: [u64; 6]) -> u32 {
        let field = |k: u32| -> u32 {
            match k {
                NR => nr as u32,
                ARCH => AUDIT_ARCH,
                k if (16..64).contains(&k) && k % 8 == 0 => args[((k - 16) / 8) as usize] as u32,
                k => panic!("load at {k}"),
            }
        };
        let (mut pc, mut a) = (0usize, 0u32);
        loop {
            let i = p[pc];
            pc += 1;
            match i.code {
                BPF_LD_W_ABS => a = field(i.k),
                BPF_JEQ_K => pc += if a == i.k { i.jt } else { i.jf } as usize,
                BPF_JSET_K => pc += if a & i.k != 0 { i.jt } else { i.jf } as usize,
                BPF_RET_K => return i.k,
                c => panic!("code {c:#x}"),
            }
        }
    }

    #[test]
    fn ioctl_numbers() {
        // From the kernel headers and crosvm's bindings, as strace prints them.
        let n = |s: &str| IOCTLS.iter().find(|(_, x)| *x == s).unwrap().0;
        assert_eq!(n("GH_VCPU_RUN"), 0x4705);
        assert_eq!(n("GH_VM_ADD_FUNCTION"), 0x4010_4704);
        assert_eq!(n("GH_VM_REMOVE_FUNCTION"), 0x4010_4707);
        assert_eq!(n("VHOST_GET_FEATURES"), 0x8008_af00);
        assert_eq!(n("VHOST_SET_MEM_TABLE"), 0x4008_af03);
        assert_eq!(n("VHOST_SET_VRING_ADDR"), 0x4028_af11);
        assert_eq!(n("VHOST_GET_VRING_BASE"), 0xc008_af12);
        assert_eq!(n("VHOST_VSOCK_SET_RUNNING"), 0x4004_af61);
    }

    #[test]
    fn what_the_filter_decides() {
        let me = 4242u64;
        let p = build(me as u32, RET_KILL_PROCESS);
        let eperm = RET_ERRNO | libc::EPERM as u32;
        let call = |nr, args| eval(&p, nr, args);
        // Signals: itself only.
        assert_eq!(call(libc::SYS_tgkill, [me, 4243, 34, 0, 0, 0]), RET_ALLOW);
        assert_eq!(call(libc::SYS_kill, [me, 6, 0, 0, 0, 0]), RET_ALLOW);
        assert_eq!(call(libc::SYS_kill, [1, 9, 0, 0, 0, 0]), eperm);
        assert_eq!(call(libc::SYS_kill, [0, 9, 0, 0, 0, 0]), eperm);
        assert_eq!(
            call(libc::SYS_kill, [u32::MAX as u64, 9, 0, 0, 0, 0]),
            eperm
        );
        assert_eq!(call(libc::SYS_tgkill, [7, 7, 9, 0, 0, 0]), eperm);
        assert_eq!(call(libc::SYS_tkill, [me, 9, 0, 0, 0, 0]), eperm);
        assert_eq!(call(libc::SYS_rt_sigqueueinfo, [8, 9, 0, 0, 0, 0]), eperm);
        assert_eq!(call(SYS_PIDFD_OPEN, [1, 0, 0, 0, 0, 0]), eperm);
        // Threads, never processes or programs.
        let thread = (libc::CLONE_VM | libc::CLONE_THREAD | libc::CLONE_SIGHAND) as u64;
        assert_eq!(call(libc::SYS_clone, [thread, 0, 0, 0, 0, 0]), RET_ALLOW);
        assert_eq!(
            call(libc::SYS_clone, [libc::SIGCHLD as u64, 0, 0, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        assert_eq!(call(SYS_CLONE3, [0; 6]), RET_ERRNO | libc::ENOSYS as u32);
        assert_eq!(call(libc::SYS_execve, [0; 6]), RET_KILL_PROCESS);
        // Memory, never executable.
        let rw = (libc::PROT_READ | libc::PROT_WRITE) as u64;
        let rx = (libc::PROT_READ | libc::PROT_EXEC) as u64;
        assert_eq!(
            call(libc::SYS_mmap, [0, 4096, rw, 0x22, u32::MAX as u64, 0]),
            RET_ALLOW
        );
        assert_eq!(
            call(libc::SYS_mmap, [0, 4096, rx, 0x22, u32::MAX as u64, 0]),
            RET_KILL_PROCESS
        );
        assert_eq!(
            call(libc::SYS_mprotect, [0x1000, 4096, 0, 0, 0, 0]),
            RET_ALLOW
        );
        assert_eq!(
            call(libc::SYS_mprotect, [0x1000, 4096, rx, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        // Unix sockets only, and the listed ioctls.
        assert_eq!(
            call(libc::SYS_socket, [libc::AF_UNIX as u64, 2, 0, 0, 0, 0]),
            RET_ALLOW
        );
        assert_eq!(
            call(libc::SYS_socket, [libc::AF_INET as u64, 2, 0, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        assert_eq!(
            call(libc::SYS_socket, [libc::AF_VSOCK as u64, 1, 0, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        assert_eq!(call(libc::SYS_ioctl, [3, 0x4705, 0, 0, 0, 0]), RET_ALLOW);
        // GH_CREATE_VM: setup only.
        assert_eq!(
            call(libc::SYS_ioctl, [3, 0x4700, 0, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        assert_eq!(
            call(libc::SYS_ioctl, [3, 0x40086602, 0, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        for (cmd, name) in IOCTLS {
            assert_eq!(
                call(libc::SYS_ioctl, [3, *cmd as u64, 0, 0, 0, 0]),
                RET_ALLOW,
                "{name}"
            );
        }
        // stdin's terminal mode at exit: on descriptor 0 only, and no other terminal ioctl.
        assert_eq!(call(libc::SYS_ioctl, [0, TCGETS as u64, 0, 0, 0, 0]), RET_ALLOW);
        assert_eq!(call(libc::SYS_ioctl, [0, TCSETS as u64, 0, 0, 0, 0]), RET_ALLOW);
        assert_eq!(
            call(libc::SYS_ioctl, [3, TCGETS as u64, 0, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        assert_eq!(
            call(libc::SYS_ioctl, [3, TCSETS as u64, 0, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        // TIOCSTI
        assert_eq!(
            call(libc::SYS_ioctl, [0, 0x5412, 0, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        assert_eq!(call(libc::SYS_ioctl, [0, 0x4705, 0, 0, 0, 0]), RET_ALLOW);
        // Reaping at exit: any child, never waiting.
        let any = u32::MAX as u64;
        let nohang = libc::WNOHANG as u64;
        assert_eq!(call(libc::SYS_wait4, [any, 0, nohang, 0, 0, 0]), RET_ALLOW);
        assert_eq!(
            call(libc::SYS_wait4, [any, 0, 0, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        assert_eq!(
            call(libc::SYS_wait4, [4243, 0, nohang, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        // fdsan's one read of the descriptor limit, no other limit.
        let nofile = libc::RLIMIT_NOFILE as u64;
        let stack = libc::RLIMIT_STACK as u64;
        assert_eq!(call(libc::SYS_getrlimit, [nofile, 0, 0, 0, 0, 0]), RET_ALLOW);
        assert_eq!(
            call(libc::SYS_getrlimit, [stack, 0, 0, 0, 0, 0]),
            RET_KILL_PROCESS
        );
        assert_eq!(call(libc::SYS_prlimit64, [0; 6]), RET_KILL_PROCESS);
        // The list, and nothing beyond it.
        for (nr, name) in ALLOWED {
            assert_eq!(call(*nr, [0; 6]), RET_ALLOW, "{name}");
        }
        for nr in [
            libc::SYS_openat,
            libc::SYS_ptrace,
            libc::SYS_mount,
            libc::SYS_unshare,
            libc::SYS_bpf,
            libc::SYS_execveat,
            libc::SYS_memfd_create,
            libc::SYS_process_vm_writev,
        ] {
            assert_eq!(call(nr, [0; 6]), RET_KILL_PROCESS, "{nr}");
        }
        // Debuggable builds log instead of killing, and still refuse signals to others.
        let d = build(me as u32, RET_LOG);
        assert_eq!(eval(&d, libc::SYS_execve, [0; 6]), RET_LOG);
        assert_eq!(eval(&d, libc::SYS_kill, [1, 9, 0, 0, 0, 0]), eperm);
    }

    #[test]
    fn the_test_hook_and_the_user_build_action() {
        let user = if KILL_ON_USER_BUILDS {
            RET_KILL_PROCESS
        } else {
            RET_LOG
        };
        assert_eq!(USER_BUILDS.ends_with(" kill"), KILL_ON_USER_BUILDS);
        // User builds: the switch alone, whatever the environment says.
        for test in [None, Some("user"), Some("probe-crosvm"), Some("log")] {
            assert_eq!(action(false, test), (user, false), "{test:?}");
        }
        // Debuggable builds: log, unless the hook asks for the user-build action or the probe.
        assert_eq!(action(true, None), (RET_LOG, false));
        assert_eq!(action(true, Some("")), (RET_LOG, false));
        assert_eq!(action(true, Some("kill")), (RET_LOG, false));
        assert_eq!(action(true, Some("probe-virtmgr")), (RET_LOG, false));
        assert_eq!(action(true, Some("probe-fsvmd")), (RET_LOG, false));
        assert_eq!(action(true, Some("user")), (user, false));
        assert_eq!(action(true, Some("probe-crosvm")), (user, true));
        // The probe's call is on no list: the filter gives it the default action.
        let p = build(4242, RET_KILL_PROCESS);
        assert_eq!(eval(&p, PROBE_CALL, [0; 6]), RET_KILL_PROCESS);
        assert!(!ALLOWED.iter().any(|(nr, _)| *nr == PROBE_CALL));
    }

    #[test]
    fn jumps_stay_inside_the_program() {
        let p = build(1234, RET_KILL_PROCESS);
        for (i, insn) in p.iter().enumerate() {
            if insn.code == BPF_JEQ_K || insn.code == BPF_JSET_K {
                assert!(
                    i + 1 + (insn.jt as usize) < p.len() && i + 1 + (insn.jf as usize) < p.len(),
                    "jump at {i}"
                );
            }
        }
        assert_eq!(p.last().unwrap().k, RET_KILL_PROCESS);
        let mut nrs: Vec<_> = ALLOWED.iter().map(|(n, _)| *n).collect();
        nrs.sort();
        nrs.dedup();
        assert_eq!(nrs.len(), ALLOWED.len(), "a call listed twice");
        for nr in [
            libc::SYS_execve,
            libc::SYS_execveat,
            libc::SYS_ptrace,
            libc::SYS_process_vm_readv,
            libc::SYS_process_vm_writev,
            libc::SYS_kill,
            libc::SYS_tkill,
            libc::SYS_clone,
            libc::SYS_mmap,
            libc::SYS_mprotect,
            libc::SYS_socket,
            libc::SYS_ioctl,
            SYS_PIDFD_OPEN,
            libc::SYS_mount,
            libc::SYS_unshare,
            libc::SYS_setns,
            libc::SYS_bpf,
            libc::SYS_perf_event_open,
            libc::SYS_openat,
            libc::SYS_wait4,
            libc::SYS_getrlimit,
        ] {
            assert!(!nrs.contains(&nr), "{nr} must not be allowed outright");
        }
    }
}
