// DiamaneOS: rustc --edition 2021 --test tools/diamaneos/worker_resources.rs.
// The Event boundary injects host allocation and duplication failures into the real worker code.
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
pub type Error = io::Error;
static FAIL: AtomicUsize = AtomicUsize::new(0);
pub struct Event;
impl Event {
    pub fn new() -> Result<Self, Error> {
        if FAIL.load(Ordering::SeqCst) == 1 {
            Err(io::Error::other("event allocation"))
        } else {
            Ok(Self)
        }
    }
    pub fn try_clone(&self) -> Result<Self, Error> {
        if FAIL.load(Ordering::SeqCst) == 2 {
            Err(io::Error::other("event duplication"))
        } else {
            Ok(Self)
        }
    }
    pub fn signal(&self) -> Result<(), Error> {
        Ok(())
    }
}
#[path = "../../base/src/worker_thread.rs"]
mod worker;
#[test]
fn allocation_errors_are_returned_without_a_panic() {
    for fail in [1, 2] {
        FAIL.store(fail, Ordering::SeqCst);
        let result = std::panic::catch_unwind(|| worker::WorkerThread::try_start("test", |_| ()));
        assert!(result.is_ok(), "resource failure panicked");
        assert!(result.unwrap().is_err());
    }
}
