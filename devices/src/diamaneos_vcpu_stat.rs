// Copyright 2026 DiamaneOS
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! DiamaneOS: the vCPU threads' /proc stat files for the watchdog (vmwdt). Upstream opens
//! /proc/PID/task/TID/stat on every watchdog write and timer event to read the thread's guest
//! time. crosvm's seccomp filter (src/crosvm/sys/linux/diamaneos_seccomp.rs) refuses openat once
//! the vCPUs run. So run_control opens each vCPU thread's file here, right before the filter goes
//! on, and the watchdog reads it again from the start with pread. procfs builds the contents
//! afresh on each read from offset 0, so the guest time is always current.

use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;

/// The open stat files by (process id, thread id).
static FILES: Mutex<BTreeMap<(u32, u32), Arc<File>>> = Mutex::new(BTreeMap::new());

fn files() -> MutexGuard<'static, BTreeMap<(u32, u32), Arc<File>>> {
    FILES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Opens the stat file of each thread. A thread opened again gets its new file.
pub fn open(pid_tids: impl IntoIterator<Item = (u32, u32)>) -> io::Result<()> {
    for (pid, tid) in pid_tids {
        let file = File::open(format!("/proc/{pid}/task/{tid}/stat"))?;
        files().insert((pid, tid), Arc::new(file));
    }
    Ok(())
}

/// The thread's stat file as it reads now, or None when `open` did not open it.
pub fn read(pid: u32, tid: u32) -> Option<io::Result<String>> {
    let file = files().get(&(pid, tid)).cloned()?;
    Some(read_from_start(&file))
}

fn read_from_start(file: &File) -> io::Result<String> {
    let mut buf = vec![0u8; 1024];
    let mut len = 0;
    loop {
        if len == buf.len() {
            buf.resize(len * 2, 0);
        }
        let n = file.read_at(&mut buf[len..], len as u64)?;
        if n == 0 {
            break;
        }
        len += n;
    }
    buf.truncate(len);
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::process;
    use std::thread;
    use std::thread::sleep;
    use std::time::Duration;

    use base::gettid;
    use base::poll_assert;
    use base::Tube;
    use base::VmEventType;
    use vm_control::VmResponse;

    use super::super::*;
    use super::*;

    /// Forgets a test thread's file when the test ends: the system reuses thread ids.
    struct Opened(u32, u32);

    impl Opened {
        fn this_thread() -> Opened {
            let (pid, tid) = (process::id(), gettid() as u32);
            open([(pid, tid)]).unwrap();
            Opened(pid, tid)
        }
    }

    impl Drop for Opened {
        fn drop(&mut self) {
            files().remove(&(self.0, self.1));
        }
    }

    /// From here on this thread and the threads it starts get EPERM for every open, as crosvm's
    /// filter refuses openat after setup.
    fn refuse_opens() {
        #[repr(C)]
        struct Insn {
            code: u16,
            jt: u8,
            jf: u8,
            k: u32,
        }
        #[repr(C)]
        struct Prog {
            len: u16,
            filter: *const Insn,
        }
        let ld_nr = Insn {
            code: 0x20,
            jt: 0,
            jf: 0,
            k: 0,
        };
        let jeq = |k: libc::c_long| Insn {
            code: 0x15,
            jt: 0,
            jf: 1,
            k: k as u32,
        };
        let ret = |k| Insn {
            code: 0x06,
            jt: 0,
            jf: 0,
            k,
        };
        let eperm = 0x0005_0000 | libc::EPERM as u32;
        let mut p = vec![ld_nr, jeq(libc::SYS_openat), ret(eperm)];
        #[cfg(target_arch = "x86_64")]
        p.extend([jeq(libc::SYS_open), ret(eperm)]);
        p.push(ret(0x7fff_0000));
        let prog = Prog {
            len: p.len() as u16,
            filter: p.as_ptr(),
        };
        // SAFETY: both calls only change this thread's own system call rules. The kernel copies
        // the program, which stays valid for the call.
        unsafe {
            assert_eq!(libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
            assert_eq!(
                libc::prctl(
                    22, // PR_SET_SECCOMP
                    2,  // SECCOMP_MODE_FILTER
                    &prog as *const Prog,
                ),
                0,
                "seccomp: {}",
                io::Error::last_os_error()
            );
        }
        let e = File::open("/proc/self/stat").unwrap_err();
        assert_eq!(e.raw_os_error(), Some(libc::EPERM), "the filter is not on");
    }

    #[test]
    fn guest_time_without_open() {
        thread::spawn(|| {
            let me = Opened::this_thread();
            refuse_opens();
            let first = read(me.0, me.1).unwrap().unwrap();
            assert!(first.starts_with(&format!("{} (", me.1)), "{first}");
            assert!(first.split_whitespace().count() > PROCSTAT_GUEST_TIME_INDX);
            // Read again from the start, never from where the last read ended.
            let again = read(me.0, me.1).unwrap().unwrap();
            assert!(again.starts_with(&format!("{} (", me.1)), "{again}");
            assert!(Vmwdt::get_guest_time_ms(me.0, me.1).is_ok());
            // A thread nobody opened still takes upstream's way.
            assert!(read(me.0, me.1 ^ 0x4000_0000).is_none());
        })
        .join()
        .unwrap();
    }

    /// Upstream's expiration test with every open refused: the guest's writes are honoured and
    /// a stalled vCPU is still caught, first with the stall interrupt, then with a reset.
    #[test]
    fn watchdog_without_open() {
        thread::spawn(|| {
            let me = Opened::this_thread();
            let (vm_evt_wrtube, vm_evt_rdtube) = Tube::directional_pair().unwrap();
            let (vm_ctrl_wrtube, vm_ctrl_rdtube) = Tube::pair().unwrap();
            vm_ctrl_wrtube
                .send(&VmResponse::VcpuPidTidResponse {
                    pid_tid_map: BTreeMap::from([(0, (me.0, me.1))]),
                })
                .unwrap();
            let irq = IrqEdgeEvent::new().unwrap();
            let mut device = Vmwdt::new(1, vm_evt_wrtube, irq, vm_ctrl_rdtube).unwrap();
            refuse_opens();
            let at = |offset: u32| BusAccessInfo {
                offset: offset as u64,
                address: 0x3000,
                id: 0,
            };
            // 10 Hz, one tick, on. The load reads the guest time on this thread, the worker
            // (started by the status write, under the same filter) on every timer event.
            device.write(at(VMWDT_REG_CLOCK_FREQ_HZ), &[10, 0, 0, 0]);
            device.write(at(VMWDT_REG_LOAD_CNT), &[1, 0, 0, 0]);
            device.write(at(VMWDT_REG_STATUS), &[1, 0, 0, 0]);
            {
                let wdts = device.vm_wdts.lock();
                assert_eq!(wdts[0].timer_freq_hz, 10);
                assert_eq!(wdts[0].next_expiration_interval_ms, 100);
                assert!(wdts[0].is_enabled);
                assert_eq!(wdts[0].thread_id, me.1);
            }
            // The guest ran past its interval without a write: a stall.
            device.vm_wdts.lock()[0].last_guest_time_ms = -100;
            poll_assert!(10, || {
                sleep(Duration::from_millis(50));
                device.vm_wdts.lock()[0].stall_evt_ppi_triggered
            });
            device.vm_wdts.lock()[0].last_guest_time_ms = -100;
            poll_assert!(10, || {
                sleep(Duration::from_millis(50));
                matches!(
                    vm_evt_rdtube.recv::<VmEventType>(),
                    Ok(VmEventType::WatchdogReset)
                )
            });
        })
        .join()
        .unwrap();
    }
}
