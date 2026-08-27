// src/app/fixed_timestep.rs
//
// The virtual clock a headless run keeps. Every tick advances the same fixed
// dt whatever it cost to compute, which is what makes an unpaced run
// reproducible: the simulation sees the same timing sequence on a slow host as
// on a fast one, and a host with no clock at all can still drive it.
//
// The windowed driver's accumulator (engine `app::clock`) is the other half of
// this seam: it turns real elapsed time into the same budget, emitting zero
// ticks on a short frame and several on a long one.

use crate::ecs::SimTiming;

// One tick per step at the fixed rate, with no accumulator remainder to blend
// against, which is exactly what `SimTiming::default` describes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FixedTimestep {
    dt: f32,
    ticks: u64,
}

impl Default for FixedTimestep {
    fn default() -> Self {
        Self {
            dt: SimTiming::TICK_DT,
            ticks: 0,
        }
    }
}

impl FixedTimestep {
    // The next tick's simulation budget.
    pub(crate) fn advance(&mut self) -> SimTiming {
        self.ticks = self.ticks.saturating_add(1);
        SimTiming {
            ticks: 1,
            tick_dt: self.dt,
            alpha: 1.0,
        }
    }

    // Ticks advanced so far.
    pub(crate) fn ticks(&self) -> u64 {
        self.ticks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every tick is the same budget: one fixed step, fully caught up, so
    // nothing downstream blends between two simulated states.
    #[test]
    fn every_tick_publishes_one_unblended_fixed_step() {
        let mut sim = FixedTimestep::default();
        for _ in 0..3 {
            let timing = sim.advance();
            assert_eq!(timing.ticks, 1);
            assert_eq!(timing.tick_dt, SimTiming::TICK_DT);
            assert_eq!(timing.alpha, 1.0);
        }
    }

    // The budget matches what a world with no driver at all assumes, so a
    // headless run and a bare `World::step` loop simulate identically.
    #[test]
    fn the_budget_matches_an_undriven_worlds_default() {
        let published = FixedTimestep::default().advance();
        let default = SimTiming::default();
        assert_eq!(published.ticks, default.ticks);
        assert_eq!(published.tick_dt, default.tick_dt);
        assert_eq!(published.alpha, default.alpha);
    }

    #[test]
    fn ticks_count_the_advances() {
        let mut sim = FixedTimestep::default();
        assert_eq!(sim.ticks(), 0);
        sim.advance();
        sim.advance();
        assert_eq!(sim.ticks(), 2);
    }
}
