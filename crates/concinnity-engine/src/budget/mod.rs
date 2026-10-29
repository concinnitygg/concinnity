//! Process-level resource budgets, published as world resources so systems can
//! read them, and the resident-memory reading they are compared against. The
//! host measures the machine (core count, physical RAM, resident set) and
//! hands the numbers in; the policy that turns them into budgets lives here.
//!
//!   `ThreadBudget`  how many worker threads the shared job pool runs, and
//!                 which frame threads run above normal priority.
//!   `MemoryBudget`  a soft ceiling on host memory the runtime aims to stay under.
//!   cpu           process CPU utilization from successive clock readings.
//!   drift         whether long-session growth came from the Rust heap or from
//!                 outside it.
//!
//! The budgets are advisory: they are computed, logged, and reported, and
//! nothing here aborts or caps an allocation.

use crate::components::FramePriority;

pub mod cpu;
pub mod drift;

// Absolute default cap on the memory budget regardless of how much RAM the
// machine has, so a workstation with hundreds of GiB does not implicitly invite
// the runtime to grow without bound. An `AppConfig` override or a smaller
// machine lowers it; nothing but an override raises it.
const HARD_CEILING_BYTES: u64 = 16 * 1024 * 1024 * 1024;
// Default budget as a percentage of total RAM (whichever is smaller than the
// hard ceiling wins).
const DEFAULT_FRACTION_PCT: u64 = 70;
// An `AppConfig` override may not exceed this percentage of total RAM: a game
// cannot ask for more memory than the machine can safely give.
const MAX_FRACTION_PCT: u64 = 85;

/// How many threads the runtime plans to run, computed from the machine's core
/// count and the optional `AppConfig` override, and how they are scheduled.
/// Advisory as a count: it sizes the shared job pool and is reported, but does
/// not cap the streaming workers or the audio thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadBudget {
    /// Logical cores the machine reports.
    pub total_cores: usize,
    /// Worker threads for the shared job pool.
    pub job_threads: usize,
    /// Which frame threads run above normal priority (`AppConfig`'s setting).
    pub frame_priority: FramePriority,
}

impl ThreadBudget {
    /// The budget for a machine with `total_cores`. A `job_threads_override` of
    /// 0 means "auto" and takes `default_threads`, the job pool's own default; a
    /// non-zero override is honored but never exceeds the core count.
    pub fn compute(
        total_cores: usize,
        default_threads: usize,
        job_threads_override: u32,
        frame_priority: FramePriority,
    ) -> Self {
        let job_threads = if job_threads_override > 0 {
            (job_threads_override as usize).min(total_cores)
        } else {
            default_threads
        };
        Self {
            total_cores,
            job_threads,
            frame_priority,
        }
    }
}

/// A soft ceiling on host memory the runtime aims to stay under, computed from
/// total RAM and the optional `AppConfig` override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Total physical RAM, or `None` when the platform query failed (the budget
    /// then falls back to the hard ceiling, or a bare override).
    pub total_ram_bytes: Option<u64>,
    /// The effective budget in bytes.
    pub budget_bytes: u64,
    /// Whether an `AppConfig` override set the budget (vs. the computed default).
    pub overridden: bool,
}

impl MemoryBudget {
    /// The budget for `total_ram_bytes`. A `max_memory_mb_override` of 0 means
    /// "auto": `min(hard ceiling, 70% of RAM)`. A non-zero override is honored
    /// but clamped to 85% of RAM. When total RAM is unknown, the default is the
    /// hard ceiling and an override passes through.
    pub fn compute(total_ram_bytes: Option<u64>, max_memory_mb_override: u32) -> Self {
        let override_bytes =
            (max_memory_mb_override > 0).then(|| u64::from(max_memory_mb_override) * 1024 * 1024);
        let budget_bytes = match (total_ram_bytes, override_bytes) {
            (Some(ram), Some(want)) => want.min(ram * MAX_FRACTION_PCT / 100),
            (Some(ram), None) => HARD_CEILING_BYTES.min(ram * DEFAULT_FRACTION_PCT / 100),
            (None, Some(want)) => want,
            (None, None) => HARD_CEILING_BYTES,
        };
        Self {
            total_ram_bytes,
            budget_bytes,
            overridden: override_bytes.is_some(),
        }
    }

    /// The budget in whole mebibytes, for logging and reporting.
    pub fn budget_mib(&self) -> u64 {
        self.budget_bytes / (1024 * 1024)
    }
}

/// The process's resident set in bytes, read on demand, installed as a world
/// resource by a host that can measure it. `None` from the call when the
/// platform cannot report it.
#[derive(Debug, Clone, Copy)]
pub struct ResidentMemory(pub fn() -> Option<u64>);

impl ResidentMemory {
    /// The resident set now, or `None` when the platform cannot report it.
    pub fn read(&self) -> Option<u64> {
        (self.0)()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn auto_thread_budget_takes_the_pool_default() {
        let tb = ThreadBudget::compute(8, 7, 0, FramePriority::default());
        assert_eq!(tb.job_threads, 7);
        assert_eq!(tb.total_cores, 8);
    }

    #[test]
    fn thread_override_is_honored_but_capped_at_core_count() {
        let tb = ThreadBudget::compute(8, 7, 2, FramePriority::default());
        assert_eq!(tb.job_threads, 2);
        let huge = ThreadBudget::compute(8, 7, 9999, FramePriority::default());
        assert_eq!(huge.job_threads, 8);
    }

    #[test]
    fn thread_budget_keeps_the_configured_priority() {
        for priority in [FramePriority::Normal, FramePriority::AllThreads] {
            let budget = ThreadBudget::compute(4, 3, 0, priority);
            assert_eq!(budget.frame_priority, priority);
        }
    }

    #[test]
    fn default_memory_budget_is_the_smaller_of_ceiling_and_fraction() {
        let small = MemoryBudget::compute(Some(8 * GIB), 0);
        assert_eq!(small.budget_bytes, 8 * GIB * 70 / 100);
        assert!(!small.overridden);

        let large = MemoryBudget::compute(Some(256 * GIB), 0);
        assert_eq!(large.budget_bytes, HARD_CEILING_BYTES);
    }

    #[test]
    fn memory_override_is_honored_but_clamped_to_a_safe_fraction() {
        let ok = MemoryBudget::compute(Some(32 * GIB), 4096);
        assert_eq!(ok.budget_bytes, 4096 * 1024 * 1024);
        assert!(ok.overridden);

        let greedy = MemoryBudget::compute(Some(8 * GIB), 16384);
        assert_eq!(greedy.budget_bytes, 8 * GIB * 85 / 100);
        assert!(greedy.overridden);
    }

    #[test]
    fn unknown_ram_falls_back_to_ceiling_or_bare_override() {
        let fallback = MemoryBudget::compute(None, 0);
        assert_eq!(fallback.budget_bytes, HARD_CEILING_BYTES);
        assert!(!fallback.overridden);

        let bare_override = MemoryBudget::compute(None, 2048);
        assert_eq!(bare_override.budget_bytes, 2048 * 1024 * 1024);
        assert!(bare_override.overridden);
    }

    #[test]
    fn resident_memory_reads_through_the_installed_probe() {
        #[expect(
            clippy::unnecessary_wraps,
            reason = "a `ResidentMemory` probe, which returns `None` where unsupported"
        )]
        fn fixed() -> Option<u64> {
            Some(42)
        }
        assert_eq!(ResidentMemory(fixed).read(), Some(42));
    }
}
