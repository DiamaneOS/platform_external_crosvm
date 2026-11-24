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

//! DiamaneOS: the vCPUs run guest code only once crosvm's seccomp filter is on.
//!
//! Upstream lets a running VM's vCPUs into their loop at the start barrier, while run_control
//! still finishes setup and only then installs the filter (diamaneos_seccomp.rs). A guest could
//! so run, and its device accesses be handled, before the filter. Here the vCPU threads start in
//! the suspended run mode instead (parked in vcpu_loop, waiting for a message, no guest code).
//! run_control then opens the watchdog's files, installs the filter (TSYNC, every thread) and only
//! after that sends the vCPUs the run mode the VM asked for. If any of it fails the vCPUs get
//! Exiting and leave their loop, and crosvm does not start the VM.
//!
//! A VM restored from a snapshot starts suspended upstream too. Its last step, the switch to the
//! run mode it asked for, waits for the filter the same way.

use anyhow::Result;
use vm_control::VcpuControl;
use vm_control::VmRunMode;

/// The run mode the vCPU threads start in, and the run mode to release them into after the
/// filter (None: they stay in the mode they started in, which runs no guest code). `restored` is
/// the run mode a VM restored from a snapshot asks for, None for a VM not restored.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub fn park(run_mode: VmRunMode, restored: Option<VmRunMode>) -> (VmRunMode, Option<VmRunMode>) {
    match (run_mode, restored) {
        (_, Some(mode)) => (run_mode, Some(mode)),
        (VmRunMode::Running, None) => (VmRunMode::Suspending, Some(VmRunMode::Running)),
        (other, None) => (other, None),
    }
}

/// Runs `seal` (the watchdog's files, then the filter), then sends the vCPUs `release` through
/// `kick`. When `seal` fails the vCPUs get Exiting instead and the error is returned.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub fn seal_then_release(
    release: Option<VmRunMode>,
    seal: impl FnOnce() -> Result<()>,
    mut kick: impl FnMut(VcpuControl),
) -> Result<()> {
    if let Err(e) = seal() {
        kick(VcpuControl::RunState(VmRunMode::Exiting));
        return Err(e);
    }
    if let Some(mode) = release {
        kick(VcpuControl::RunState(mode));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use anyhow::anyhow;

    use super::*;

    fn modes(sent: &[VcpuControl]) -> Vec<VmRunMode> {
        sent.iter()
            .map(|m| match m {
                VcpuControl::RunState(mode) => *mode,
                other => panic!("unexpected message {other:?}"),
            })
            .collect()
    }

    #[test]
    fn running_vm_starts_parked() {
        assert_eq!(
            park(VmRunMode::Running, None),
            (VmRunMode::Suspending, Some(VmRunMode::Running))
        );
    }

    #[test]
    fn suspended_and_breakpoint_stay() {
        assert_eq!(
            park(VmRunMode::Suspending, None),
            (VmRunMode::Suspending, None)
        );
        assert_eq!(
            park(VmRunMode::Breakpoint, None),
            (VmRunMode::Breakpoint, None)
        );
    }

    #[test]
    fn restored_vm_released_after_seal() {
        assert_eq!(
            park(VmRunMode::Suspending, Some(VmRunMode::Running)),
            (VmRunMode::Suspending, Some(VmRunMode::Running))
        );
    }

    #[test]
    fn released_only_after_seal() {
        let events = RefCell::new(Vec::new());
        seal_then_release(
            Some(VmRunMode::Running),
            || {
                events.borrow_mut().push("seal");
                Ok(())
            },
            |m| {
                assert_eq!(modes(&[m]), [VmRunMode::Running]);
                events.borrow_mut().push("release");
            },
        )
        .unwrap();
        assert_eq!(*events.borrow(), ["seal", "release"]);
    }

    #[test]
    fn failed_seal_sends_exiting() {
        let mut sent = Vec::new();
        let r = seal_then_release(
            Some(VmRunMode::Running),
            || Err(anyhow!("no filter")),
            |m| sent.push(m),
        );
        assert!(r.is_err());
        assert_eq!(modes(&sent), [VmRunMode::Exiting]);
    }

    #[test]
    fn nothing_to_release() {
        let mut sent = Vec::new();
        seal_then_release(None, || Ok(()), |m| sent.push(m)).unwrap();
        assert!(sent.is_empty());
    }
}
