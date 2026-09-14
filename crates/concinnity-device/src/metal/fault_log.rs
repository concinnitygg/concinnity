//! Logging for command buffers the GPU faulted. One fault discards every other
//! buffer in flight as its victim, so the buffer that caused it is easy to lose
//! in the cascade. Causes and victims are throttled apart, so a burst of victims
//! can never spend the budget that names the cause.

use core::fmt::Display;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU32, Ordering};

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};

// Faulted buffers of each role logged per process, so a fault that repeats
// every frame cannot flood the log.
const CAUSE_LOG_LIMIT: u32 = 16;
const VICTIM_LOG_LIMIT: u32 = 16;

// The IOGPU callback errors Metal reports for a buffer it discarded because of
// a fault elsewhere: a victim of the recovery, or a submission ignored after
// this process had already faulted.
const VICTIM_MARKERS: [&str; 2] = [
    "kIOGPUCommandBufferCallbackErrorInnocentVictim",
    "kIOGPUCommandBufferCallbackErrorSubmissionsIgnored",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultRole {
    Cause,
    Victim,
}

// Classify a faulted buffer from its error description.
fn fault_role(description: &str) -> FaultRole {
    if VICTIM_MARKERS
        .iter()
        .any(|marker| description.contains(marker))
    {
        FaultRole::Victim
    } else {
        FaultRole::Cause
    }
}

struct FaultThrottle {
    causes: AtomicU32,
    victims: AtomicU32,
}

impl FaultThrottle {
    const fn new() -> Self {
        Self {
            causes: AtomicU32::new(0),
            victims: AtomicU32::new(0),
        }
    }

    // Whether another fault of `role` may be logged, counting it if so.
    fn admit(&self, role: FaultRole) -> bool {
        let (logged, limit) = match role {
            FaultRole::Cause => (&self.causes, CAUSE_LOG_LIMIT),
            FaultRole::Victim => (&self.victims, VICTIM_LOG_LIMIT),
        };
        logged
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n < limit).then_some(n + 1)
            })
            .is_ok()
    }
}

static THROTTLE: FaultThrottle = FaultThrottle::new();

// Log `cb` if the GPU faulted it. Called from a completion handler; `what`
// names the work the buffer carried.
pub(super) fn report_fault(cb: &ProtocolObject<dyn MTLCommandBuffer>, what: impl Display) {
    if cb.status() != MTLCommandBufferStatus::Error {
        return;
    }
    let description = cb.error().map_or_else(
        || "no error object".to_string(),
        |e| e.localizedDescription().to_string(),
    );
    let role = fault_role(&description);
    if !THROTTLE.admit(role) {
        return;
    }
    match role {
        FaultRole::Cause => tracing::error!("{what} command buffer faulted: {description}"),
        FaultRole::Victim => {
            tracing::error!("{what} command buffer discarded by GPU recovery: {description}")
        }
    }
}

// Report `cmd` from its own completion handler if the GPU faults it. Must be
// attached before `cmd` is committed.
pub(super) fn attach_fault_logger(cmd: &ProtocolObject<dyn MTLCommandBuffer>, what: &'static str) {
    let handler = block2::RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
        // SAFETY: Metal hands the completion handler a live command buffer, and the borrow does
        // not escape the block.
        report_fault(unsafe { cb.as_ref() }, what);
    });
    // SAFETY: addCompletedHandler copies the block, so the RcBlock may drop here.
    unsafe {
        cmd.addCompletedHandler(block2::RcBlock::as_ptr(&handler));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn innocent_victim_is_a_victim() {
        let description = "Discarded (victim of GPU error/recovery) \
            (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)";
        assert_eq!(fault_role(description), FaultRole::Victim);
    }

    #[test]
    fn ignored_submission_is_a_victim() {
        let description = "Ignored (for causing prior/excessive GPU errors) \
            (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)";
        assert_eq!(fault_role(description), FaultRole::Victim);
    }

    #[test]
    fn page_fault_is_a_cause() {
        let description = "Caused GPU Address Fault Error \
            (0000000b:kIOGPUCommandBufferCallbackErrorPageFault)";
        assert_eq!(fault_role(description), FaultRole::Cause);
    }

    #[test]
    fn each_role_stops_at_its_limit() {
        let throttle = FaultThrottle::new();
        for _ in 0..VICTIM_LOG_LIMIT {
            assert!(throttle.admit(FaultRole::Victim));
        }
        assert!(!throttle.admit(FaultRole::Victim));
        for _ in 0..CAUSE_LOG_LIMIT {
            assert!(throttle.admit(FaultRole::Cause));
        }
        assert!(!throttle.admit(FaultRole::Cause));
    }

    #[test]
    fn victims_never_spend_the_cause_budget() {
        let throttle = FaultThrottle::new();
        for _ in 0..VICTIM_LOG_LIMIT * 4 {
            throttle.admit(FaultRole::Victim);
        }
        assert!(throttle.admit(FaultRole::Cause));
    }
}
