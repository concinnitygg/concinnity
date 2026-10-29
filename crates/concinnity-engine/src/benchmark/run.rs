// The armed run: the config it was armed from and where its phases fall on the
// run's clock.

use alloc::string::String;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

use crate::components::BenchmarkConfig;

/// The benchmark a world was armed with, published as a resource before the
/// world starts. Its presence is what marks a run as benchmarked.
///
/// Times are seconds on the run's clock, which counts from the start of the
/// warm-up: the first phase opens at `warmup_seconds`.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkRun {
    /// The config the run was armed from, declared or default.
    pub config: BenchmarkConfig,
    names: Vec<String>,
    // Where each phase ends on the run's clock, in phase order.
    ends: Vec<f32>,
    // Where the launch stops the run, when before its last phase ends.
    stop_at: Option<f32>,
}

impl BenchmarkRun {
    /// The run `config` describes.
    pub fn new(config: BenchmarkConfig) -> Self {
        let warmup = config.warmup_seconds.max(0.0);
        let mut end = warmup;
        let (names, ends) = config
            .resolved_phases()
            .into_iter()
            .map(|phase| {
                end += phase.seconds.max(0.0);
                (phase.name, end)
            })
            .unzip();
        Self {
            config,
            names,
            ends,
            stop_at: None,
        }
    }

    /// The same run, stopped `seconds` into it when that is before its last
    /// phase ends.
    #[must_use]
    pub fn with_stop_at(mut self, seconds: Option<f32>) -> Self {
        self.stop_at = seconds;
        self
    }

    /// The phase names, in order. A phase's index into this list is how a
    /// frame records which phase it was drawn in.
    pub fn phase_names(&self) -> &[String] {
        &self.names
    }

    /// The phase `run_seconds` falls in, or `None` during the warm-up. Past the
    /// end the last phase still answers.
    pub fn phase_at(&self, run_seconds: f32) -> Option<u32> {
        if run_seconds < self.config.warmup_seconds {
            return None;
        }
        let last = self.ends.len().checked_sub(1)?;
        let at = self
            .ends
            .iter()
            .position(|&end| run_seconds < end)
            .unwrap_or(last);
        Some(u32::saturating_from(at))
    }

    /// When the last phase ends and the run is complete.
    pub fn end_seconds(&self) -> f32 {
        self.ends
            .last()
            .copied()
            .unwrap_or(self.config.warmup_seconds)
    }

    /// When the run stops: its end, or the launch's stop when that is sooner.
    pub fn stop_seconds(&self) -> f32 {
        let end = self.end_seconds();
        self.stop_at.map_or(end, |stop| stop.min(end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::BenchmarkPhase;

    fn run(warmup_seconds: f32, phases: &[(&str, f32)]) -> BenchmarkRun {
        BenchmarkRun::new(BenchmarkConfig {
            warmup_seconds,
            phases: phases
                .iter()
                .map(|&(name, seconds)| BenchmarkPhase {
                    name: name.into(),
                    seconds,
                })
                .collect(),
            ..Default::default()
        })
    }

    #[test]
    fn the_default_run_is_ten_seconds_after_its_warm_up() {
        let run = BenchmarkRun::new(BenchmarkConfig::default());
        assert_eq!(run.phase_names(), ["run"]);
        assert_eq!(run.end_seconds(), 12.0);
    }

    #[test]
    fn phases_follow_the_warm_up_back_to_back() {
        let run = run(2.0, &[("ground", 3.0), ("orbit", 5.0)]);
        assert_eq!(run.phase_at(0.0), None);
        assert_eq!(run.phase_at(1.99), None);
        assert_eq!(run.phase_at(2.0), Some(0));
        assert_eq!(run.phase_at(4.99), Some(0));
        assert_eq!(run.phase_at(5.0), Some(1));
        assert_eq!(run.end_seconds(), 10.0);
    }

    #[test]
    fn the_last_phase_answers_past_the_end() {
        let run = run(0.0, &[("a", 1.0), ("b", 1.0)]);
        assert_eq!(run.phase_at(600.0), Some(1));
    }

    #[test]
    fn a_negative_duration_takes_no_time() {
        let run = run(1.0, &[("a", -4.0), ("b", 2.0)]);
        assert_eq!(run.phase_at(1.0), Some(1));
        assert_eq!(run.end_seconds(), 3.0);
    }

    #[test]
    fn a_launch_stop_cuts_the_run_short_but_never_extends_it() {
        let full = run(1.0, &[("a", 4.0), ("b", 5.0)]);
        assert_eq!(full.stop_seconds(), 10.0);
        let cut = full.clone().with_stop_at(Some(3.5));
        assert_eq!(cut.stop_seconds(), 3.5);
        assert_eq!(cut.end_seconds(), 10.0);
        assert_eq!(full.with_stop_at(Some(60.0)).stop_seconds(), 10.0);
    }
}
