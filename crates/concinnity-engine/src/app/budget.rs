// src/app/budget.rs
//
// Process-level resource budgets computed once at App start from the host
// machine and the world's `AppConfig` overrides, then published as world
// resources so systems (and the debug server) can read them. Two budgets:
//
//   ThreadBudget  how many worker threads the shared job pool runs.
//   MemoryBudget  a soft ceiling on host memory the runtime aims to stay under.
//
// The budgets are advisory today: they are computed, logged, and reported.
// Cooperative enforcement (streaming byte budgets, back-off near the ceiling)
// is a separate follow-up; nothing here aborts or caps an allocation.

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
/// count and the optional `AppConfig` override. Advisory: it sizes the shared
/// job pool (`jobs::configure`) and is reported, but does not cap the streaming
/// workers or the audio thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadBudget {
    /// Logical cores the machine reports.
    pub total_cores: usize,
    /// Worker threads for the shared rayon job pool.
    pub job_threads: usize,
}

impl ThreadBudget {
    // `job_threads_override` of 0 means "auto": one worker per core, less one
    // for the main thread (the historical `available_parallelism() - 1`). A
    // non-zero override is honored but never exceeds the core count.
    pub(crate) fn compute(job_threads_override: u32) -> Self {
        let total_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let job_threads = if job_threads_override > 0 {
            (job_threads_override as usize).min(total_cores)
        } else {
            total_cores.saturating_sub(1).max(1)
        };
        Self {
            total_cores,
            job_threads,
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
    // `max_memory_mb_override` of 0 means "auto": `min(hard ceiling, 70% of
    // RAM)`. A non-zero override is honored but clamped to 85% of RAM so a game
    // cannot budget past what the machine can safely give. When total RAM is
    // unknown, the default is the hard ceiling and an override passes through.
    pub(crate) fn compute(total_ram_bytes: Option<u64>, max_memory_mb_override: u32) -> Self {
        let override_bytes =
            (max_memory_mb_override > 0).then(|| (max_memory_mb_override as u64) * 1024 * 1024);
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

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn auto_thread_budget_leaves_a_core_for_the_main_thread() {
        let tb = ThreadBudget::compute(0);
        assert_eq!(tb.job_threads, tb.total_cores.saturating_sub(1).max(1));
        assert!(tb.job_threads >= 1);
    }

    #[test]
    fn thread_override_is_honored_but_capped_at_core_count() {
        let tb = ThreadBudget::compute(2);
        assert_eq!(tb.job_threads, 2.min(tb.total_cores));
        // An absurd override never exceeds the machine's cores.
        let huge = ThreadBudget::compute(9999);
        assert_eq!(huge.job_threads, huge.total_cores);
    }

    #[test]
    fn default_memory_budget_is_the_smaller_of_ceiling_and_fraction() {
        // A small machine: 70% of RAM is under the ceiling, so the fraction wins.
        let small = MemoryBudget::compute(Some(8 * GIB), 0);
        assert_eq!(small.budget_bytes, 8 * GIB * 70 / 100);
        assert!(!small.overridden);

        // A large machine: 70% of RAM exceeds the ceiling, so the ceiling caps it.
        let large = MemoryBudget::compute(Some(256 * GIB), 0);
        assert_eq!(large.budget_bytes, HARD_CEILING_BYTES);
    }

    #[test]
    fn memory_override_is_honored_but_clamped_to_a_safe_fraction() {
        // A reasonable override under 85% of RAM passes through.
        let ok = MemoryBudget::compute(Some(32 * GIB), 4096);
        assert_eq!(ok.budget_bytes, 4096 * 1024 * 1024);
        assert!(ok.overridden);

        // An override past 85% of RAM is clamped to that safety fraction.
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
}
