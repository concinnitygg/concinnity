// src/gfx/streaming_system/pressure.rs
//
// Process-RAM back-off valve: the pure decision half of the streaming memory
// safety loop. StreamingSystem samples live process RSS against the world's
// `MemoryBudget` a few times a second and feeds the pair here; this module
// decides whether streaming keeps running normally, stops dispatching new
// loads, or actively shrinks residency. It performs no syscalls and holds no
// state of its own, so the whole policy is unit-testable from synthetic
// rss/budget pairs.
//
// The VRAM-bounded byte budgets (StreamPlanner's `byte_budget`) remain the
// primary residency control; this valve only backs streaming off when process
// RSS nears the RAM ceiling, with hysteresis so it does not flap at the
// threshold.

// Back-off engages when RSS exceeds this percentage of the memory budget and
// releases below `RELEASE_PCT`; the gap between them is the hysteresis band that
// stops the valve toggling on every sample near the threshold.
const ENGAGE_PCT: u64 = 90;
const RELEASE_PCT: u64 = 80;
// Above this percentage (or while RSS keeps climbing under back-off) the valve
// escalates from merely gating new loads to actively evicting residents.
const EVICT_PCT: u64 = 95;

// Byte-budget scale applied when eviction first engages, the step it tightens by
// on each subsequent sample that stays under deep pressure, and the floor it
// stops at. Reducing a pool's byte budget makes the planner evict its farthest
// residents until resident bytes fit the smaller budget.
const EVICT_FACTOR_START: f64 = 0.75;
const EVICT_FACTOR_STEP: f64 = 0.85;
const EVICT_FACTOR_FLOOR: f64 = 0.50;

// How hard the RAM valve is backing streaming off.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum StreamPressureStage {
    // RSS is comfortably under the budget: streaming runs on its normal
    // byte-budget policy.
    #[default]
    None,
    // Stage 1: RSS is high, so stop dispatching new loads. Residency is left
    // untouched -- streaming just stops growing.
    Gate,
    // Stage 2: RSS is high and not falling, so shrink residency by reducing the
    // pool byte budgets, evicting the farthest residents.
    Evict,
}

impl StreamPressureStage {
    // Whether new load dispatch should be suppressed this frame. Only stage 1
    // freezes; stage 2 keeps running the planner so it can evict under the
    // reduced budget.
    pub(crate) fn freezes_loads(self) -> bool {
        matches!(self, Self::Gate)
    }
}

// The valve's decision for one throttled sample: the new stage plus the
// byte-budget scale to apply to each pool (1.0 = the derived baseline).
pub(crate) struct PressureDecision {
    pub(crate) stage: StreamPressureStage,
    pub(crate) budget_factor: f64,
}

// Re-evaluate the back-off valve from a fresh RSS sample.
//
// `rss` / `budget` are live process RSS and the `MemoryBudget` ceiling; `rising`
// is whether RSS grew since the previous sample; `prev_stage` / `prev_factor`
// are the last decision. Returns the new stage plus the byte-budget scale the
// caller applies to the pools. A zero budget (no meaningful ceiling) always
// releases.
pub(crate) fn step_pressure(
    rss: u64,
    budget: u64,
    rising: bool,
    prev_stage: StreamPressureStage,
    prev_factor: f64,
) -> PressureDecision {
    let stage = next_stage(rss, budget, rising, prev_stage);
    let budget_factor = match stage {
        StreamPressureStage::Evict => {
            if prev_stage == StreamPressureStage::Evict {
                (prev_factor * EVICT_FACTOR_STEP).max(EVICT_FACTOR_FLOOR)
            } else {
                EVICT_FACTOR_START
            }
        }
        // None and Gate leave residency at the baseline budget.
        StreamPressureStage::None | StreamPressureStage::Gate => 1.0,
    };
    PressureDecision {
        stage,
        budget_factor,
    }
}

// The hysteresis state machine. Engages at `ENGAGE_PCT`, releases at
// `RELEASE_PCT`, and latches between; escalates to eviction above `EVICT_PCT` or
// when RSS keeps rising while still engaged.
fn next_stage(
    rss: u64,
    budget: u64,
    rising: bool,
    prev: StreamPressureStage,
) -> StreamPressureStage {
    use StreamPressureStage::*;
    if budget == 0 {
        return None;
    }
    // Integer percentage comparisons avoid float rounding at the thresholds.
    let over = |pct: u64| rss.saturating_mul(100) > budget.saturating_mul(pct);
    let under = |pct: u64| rss.saturating_mul(100) < budget.saturating_mul(pct);
    match prev {
        None => {
            if over(EVICT_PCT) {
                Evict
            } else if over(ENGAGE_PCT) {
                Gate
            } else {
                None
            }
        }
        Gate => {
            if under(RELEASE_PCT) {
                None
            } else if over(EVICT_PCT) || (over(ENGAGE_PCT) && rising) {
                Evict
            } else {
                Gate
            }
        }
        // Once evicting, hold the reduced budget until RSS fully releases, so a
        // dip back under `EVICT_PCT` does not restore-then-re-shrink and churn.
        Evict => {
            if under(RELEASE_PCT) {
                None
            } else {
                Evict
            }
        }
    }
}

// Scale a baseline byte budget by the valve's current factor. A factor of 1.0
// (or above) restores the baseline exactly; a smaller factor shrinks it.
pub(crate) fn scale_budget(baseline: u64, factor: f64) -> u64 {
    if factor >= 1.0 {
        baseline
    } else {
        (baseline as f64 * factor) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Budget pairs are expressed as (rss, budget) so a ratio like 0.92 reads
    // directly. 1000 as the budget keeps the percentages exact.
    const BUDGET: u64 = 1000;

    fn step(rss: u64, rising: bool, prev: StreamPressureStage, factor: f64) -> PressureDecision {
        step_pressure(rss, BUDGET, rising, prev, factor)
    }

    #[test]
    fn holds_normal_below_engage() {
        let d = step(500, true, StreamPressureStage::None, 1.0);
        assert_eq!(d.stage, StreamPressureStage::None);
        assert_eq!(d.budget_factor, 1.0);
    }

    #[test]
    fn engages_gate_between_90_and_95() {
        // Above the engage mark but under the evict mark: stage 1 only, and the
        // byte budget stays at baseline (residency untouched).
        let d = step(920, false, StreamPressureStage::None, 1.0);
        assert_eq!(d.stage, StreamPressureStage::Gate);
        assert_eq!(d.budget_factor, 1.0);
    }

    #[test]
    fn jumps_to_evict_above_95() {
        let d = step(970, false, StreamPressureStage::None, 1.0);
        assert_eq!(d.stage, StreamPressureStage::Evict);
        assert_eq!(d.budget_factor, EVICT_FACTOR_START);
    }

    #[test]
    fn latches_in_the_hysteresis_band() {
        // Between release (80%) and engage (90%): whatever we were, we stay.
        let engaged = step(850, false, StreamPressureStage::Gate, 1.0);
        assert_eq!(engaged.stage, StreamPressureStage::Gate);
        let released = step(850, true, StreamPressureStage::None, 1.0);
        assert_eq!(released.stage, StreamPressureStage::None);
    }

    #[test]
    fn releases_below_80() {
        let from_gate = step(790, false, StreamPressureStage::Gate, 1.0);
        assert_eq!(from_gate.stage, StreamPressureStage::None);
        let from_evict = step(500, false, StreamPressureStage::Evict, EVICT_FACTOR_START);
        assert_eq!(from_evict.stage, StreamPressureStage::None);
        // Releasing restores the baseline budget exactly.
        assert_eq!(from_evict.budget_factor, 1.0);
    }

    #[test]
    fn gate_escalates_to_evict_when_rss_keeps_rising() {
        // Still above engage (but under evict) and climbing: cut load rate was
        // not enough, so escalate to eviction.
        let d = step(920, true, StreamPressureStage::Gate, 1.0);
        assert_eq!(d.stage, StreamPressureStage::Evict);
        assert_eq!(d.budget_factor, EVICT_FACTOR_START);
    }

    #[test]
    fn gate_holds_when_not_rising_below_evict() {
        let d = step(920, false, StreamPressureStage::Gate, 1.0);
        assert_eq!(d.stage, StreamPressureStage::Gate);
    }

    #[test]
    fn evict_holds_until_full_release() {
        // Dipping under the evict mark but still above release keeps evicting,
        // rather than restoring the baseline and re-shrinking next sample.
        let d = step(920, false, StreamPressureStage::Evict, EVICT_FACTOR_START);
        assert_eq!(d.stage, StreamPressureStage::Evict);
    }

    #[test]
    fn evict_factor_tightens_across_sustained_samples_down_to_the_floor() {
        // Each sustained evict sample multiplies the factor down by the step.
        let a = step(990, true, StreamPressureStage::Evict, EVICT_FACTOR_START);
        assert_eq!(a.budget_factor, EVICT_FACTOR_START * EVICT_FACTOR_STEP);
        // From just above the floor it clamps at the floor rather than going under.
        let b = step(
            990,
            true,
            StreamPressureStage::Evict,
            EVICT_FACTOR_FLOOR + 0.01,
        );
        assert_eq!(b.budget_factor, EVICT_FACTOR_FLOOR);
    }

    #[test]
    fn zero_budget_disables_the_valve() {
        let d = step_pressure(
            9_999,
            0,
            true,
            StreamPressureStage::Evict,
            EVICT_FACTOR_START,
        );
        assert_eq!(d.stage, StreamPressureStage::None);
        assert_eq!(d.budget_factor, 1.0);
    }

    #[test]
    fn only_stage_one_freezes_loads() {
        assert!(!StreamPressureStage::None.freezes_loads());
        assert!(StreamPressureStage::Gate.freezes_loads());
        assert!(!StreamPressureStage::Evict.freezes_loads());
    }

    #[test]
    fn scale_budget_restores_at_one_and_reduces_below() {
        // A factor of 1.0 (or above) is the exact baseline; below it shrinks.
        assert_eq!(scale_budget(1000, 1.0), 1000);
        assert_eq!(scale_budget(1000, 1.5), 1000);
        assert_eq!(scale_budget(1000, 0.75), 750);
        assert_eq!(scale_budget(1000, 0.5), 500);
    }
}
