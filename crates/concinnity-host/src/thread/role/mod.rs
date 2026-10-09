//! How the operating system schedules a thread the engine starts.
//!
//! Whoever spawns a thread names what it is for with [`set_current_thread_role`]
//! as the thread's first act. The platform calls behind it are the only place
//! the engine touches scheduler state, so a thread's role is decided where it
//! is created and nowhere else.
//!
//! | Platform      | [`ThreadRole::Frame`]                        | [`ThreadRole::Background`]                |
//! | ------------- | -------------------------------------------- | ----------------------------------------- |
//! | macOS, iOS    | `QOS_CLASS_USER_INTERACTIVE`                 | `QOS_CLASS_UTILITY`                       |
//! | Windows       | above normal priority, power throttling off  | normal priority, power throttling (EcoQoS) |
//! | anything else | unchanged                                    | unchanged                                 |
//!
//! On Linux and Android a thread keeps the scheduler's defaults: neither has an
//! equivalent that an unprivileged process can set without side effects on the
//! rest of the system.

use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(not(any(target_vendor = "apple", windows)))]
mod unchanged;
#[cfg(windows)]
mod win32;

#[cfg(target_vendor = "apple")]
use apple as platform;
#[cfg(not(any(target_vendor = "apple", windows)))]
use unchanged as platform;
#[cfg(windows)]
use win32 as platform;

/// What a thread does for the frame, which decides how it is scheduled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThreadRole {
    /// On the path of every frame: the render and sim threads and the job
    /// workers they fan out to. Woken promptly and kept on performance cores.
    Frame,
    /// Work a frame never waits on: streaming, shader builds, audio decode,
    /// file writes, dev tooling. Yields performance cores to frame work.
    Background,
}

/// Apply `role` to the calling thread.
///
/// A platform that refuses the change leaves the thread as it was; the first
/// refusal in a process is logged and the rest are ignored, since a thread
/// that keeps its default scheduling still runs correctly.
pub fn set_current_thread_role(role: ThreadRole) {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if let Err(error) = platform::apply(role) {
        report_once(&REPORTED, role, &error);
    }
}

// Log `error` unless `reported` says one already was. Returns whether it logged.
fn report_once(reported: &AtomicBool, role: ThreadRole, error: &std::io::Error) -> bool {
    if reported.swap(true, Ordering::Relaxed) {
        return false;
    }
    tracing::warn!("thread role {role:?} not applied, keeping default scheduling: {error}");
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_first_failure_is_reported() {
        let reported = AtomicBool::new(false);
        let error = std::io::Error::other("refused");
        assert!(report_once(&reported, ThreadRole::Frame, &error));
        assert!(!report_once(&reported, ThreadRole::Background, &error));
        assert!(!report_once(&reported, ThreadRole::Frame, &error));
    }

    #[test]
    fn every_role_applies_on_a_fresh_thread() {
        for role in [ThreadRole::Frame, ThreadRole::Background] {
            let applied =
                std::thread::spawn(move || platform::apply(role).map_err(|e| e.to_string()))
                    .join()
                    .expect("the thread runs");
            assert_eq!(applied, Ok(()), "{role:?}");
        }
    }

    #[test]
    fn a_thread_can_change_role() {
        let applied = std::thread::spawn(|| {
            platform::apply(ThreadRole::Background)?;
            platform::apply(ThreadRole::Frame)
        })
        .join()
        .expect("the thread runs");
        assert!(applied.is_ok(), "{applied:?}");
    }
}
