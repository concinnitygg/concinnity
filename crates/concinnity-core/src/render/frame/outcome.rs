//! What one frame's submission leaves in the world: the memory-pressure
//! signal, the op failures the streaming pools roll back next tick, the
//! frame's render stats for the profiler, and the frame count a launch's cap
//! stops the run at.

use crate::ecs::{GpuMemoryPressure, PipelineContext, StepResult};
use crate::math::SaturatingFrom;
use crate::profile::RenderStats;
use crate::render::frame::extract::FrameExtractor;
use crate::render::frame::ops::ReplayOutcome;
use crate::render::frame::queues::ActiveRenderQueues;
use crate::render::frame::snapshot::FrameScalars;
use crate::render::streaming::pools::system::RenderOpFailures;

/// Frames submitted so far, against a launch's cap.
#[derive(Debug, Default)]
pub struct FrameCounter {
    frames: u64,
    max: Option<u64>,
}

impl FrameCounter {
    /// A count from zero that stops at `max` frames, or never.
    pub fn new(max: Option<u64>) -> Self {
        Self { frames: 0, max }
    }

    /// Frames counted so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Count one more frame, returning whether the cap is reached. A reached
    /// cap ends the run, and is logged on the frame that first reaches it.
    pub fn count(&mut self) -> bool {
        self.frames += 1;
        let Some(max) = self.max else {
            return false;
        };
        if self.frames == max.max(1) {
            tracing::info!("graphics: max_frames ({max}) reached");
        }
        self.frames >= max
    }

    /// The result of a step whose frame was submitted with `result`. A
    /// continuing frame is counted, and the one that reaches the cap ends the
    /// run with `Stop` rather than `Done` (which only retires the system, so the
    /// process would spin on with nothing to present). The flag says the cap
    /// ended it, so the device should drain before the run stops.
    pub fn end_step(&mut self, result: StepResult) -> (StepResult, bool) {
        if result == StepResult::Continue && self.count() {
            (StepResult::Stop, true)
        } else {
            (result, false)
        }
    }
}

/// One submitted frame, as its outcome comes back.
pub struct Submitted<'a> {
    /// The frame policy dropped its draw.
    pub skipped: bool,
    /// The frame's scalars, whose history reset a dropped draw hands on.
    pub frame: &'a FrameScalars,
    /// What its op replay produced.
    pub replay: ReplayOutcome,
    /// The backend's stats for it.
    pub stats: Option<RenderStats>,
}

/// Settle one submitted frame: a dropped draw's history reset is reissued on
/// the next frame `extractor` extracts, and the replay outcome and stats are
/// applied to the world at frame `frame`.
pub fn settle(
    ctx: &mut PipelineContext<'_>,
    extractor: &mut FrameExtractor,
    frame: u64,
    submitted: Submitted<'_>,
) {
    if submitted.skipped {
        extractor.reissue_history_reset(submitted.frame);
    }
    apply(ctx, frame, submitted.replay, submitted.stats);
}

// Apply frame `frame`'s replay outcome and render stats to the world. A backend
// without GPU-timed stats reports the default (all zeros), which the profiler
// shows as "--".
fn apply(
    ctx: &mut PipelineContext<'_>,
    frame: u64,
    replay: ReplayOutcome,
    stats: Option<RenderStats>,
) {
    if replay.memory_pressure {
        note_memory_pressure(ctx, frame);
    }
    if !replay.failures.is_empty() {
        match ctx.resource_mut::<RenderOpFailures>() {
            Some(pending) => pending.0.extend_from_slice(&replay.failures),
            None => {
                ctx.insert_resource(RenderOpFailures(replay.failures));
            }
        }
    }
    if let Some(mut stats) = stats {
        stats.skinned_pool_free = skinned_pool_free(ctx);
        ctx.profile.render = stats;
    }
}

// The skinned instance pool's free count, for the profiler's pool chip.
fn skinned_pool_free(ctx: &PipelineContext<'_>) -> u32 {
    ctx.resource::<ActiveRenderQueues>()
        .and_then(|slot| slot.0.as_ref())
        .map_or(0, |queues| {
            u32::saturating_from(queues.slots.skinned_free())
        })
}

// Count a device-memory failure where the streaming valve observes it.
fn note_memory_pressure(ctx: &mut PipelineContext<'_>, frame: u64) {
    match ctx.resource_mut::<GpuMemoryPressure>() {
        Some(pressure) => {
            pressure.events += 1;
            pressure.last_frame = frame;
        }
        None => {
            ctx.insert_resource(GpuMemoryPressure {
                events: 1,
                last_frame: frame,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    // Only a continuing frame counts toward the cap, and the frame reaching it
    // stops the run and asks for a drain.
    #[test]
    fn the_capped_frame_stops_the_run_and_drains() {
        use super::{FrameCounter, StepResult};
        let mut frames = FrameCounter::new(Some(2));
        assert_eq!(
            frames.end_step(StepResult::Continue),
            (StepResult::Continue, false)
        );
        assert_eq!(frames.end_step(StepResult::Done), (StepResult::Done, false));
        assert_eq!(frames.frames(), 1);
        assert_eq!(
            frames.end_step(StepResult::Continue),
            (StepResult::Stop, true)
        );
        let mut uncapped = FrameCounter::new(None);
        assert_eq!(
            uncapped.end_step(StepResult::Continue),
            (StepResult::Continue, false)
        );
    }

    use super::*;
    use crate::ecs::World;
    use crate::render::frame::ops::OpFailure;
    use alloc::vec;
    use alloc::vec::Vec;

    fn pressured() -> ReplayOutcome {
        ReplayOutcome {
            failures: Vec::new(),
            memory_pressure: true,
        }
    }

    #[test]
    fn the_counter_stops_at_its_cap() {
        let mut uncapped = FrameCounter::default();
        assert!(!(0..10).any(|_| uncapped.count()));
        assert_eq!(uncapped.frames(), 10);

        let mut capped = FrameCounter::new(Some(3));
        assert_eq!(
            [capped.count(), capped.count(), capped.count()],
            [false, false, true]
        );
        assert!(capped.count(), "past the cap stays reached");
        // A cap of zero stops after the first frame.
        assert!(FrameCounter::new(Some(0)).count());
    }

    #[test]
    fn memory_pressure_counts_each_failing_frame() {
        let mut world = World::new();
        apply(&mut world.context(), 3, pressured(), None);
        apply(&mut world.context(), 7, pressured(), None);
        let pressure = world.resource::<GpuMemoryPressure>().unwrap();
        assert_eq!((pressure.events, pressure.last_frame), (2, 7));
    }

    #[test]
    fn op_failures_accumulate_until_streaming_drains_them() {
        let mut world = World::new();
        let failed = |failures| ReplayOutcome {
            failures,
            memory_pressure: false,
        };
        apply(&mut world.context(), 1, failed(Vec::new()), None);
        assert!(world.resource::<RenderOpFailures>().is_none());
        apply(
            &mut world.context(),
            1,
            failed(vec![OpFailure::MeshUpload { stream_id: 0 }]),
            None,
        );
        apply(
            &mut world.context(),
            2,
            failed(vec![OpFailure::MeshUpload { stream_id: 0 }]),
            None,
        );
        assert_eq!(world.resource::<RenderOpFailures>().unwrap().0.len(), 2);
        assert!(world.resource::<GpuMemoryPressure>().is_none());
    }

    #[test]
    fn stats_land_in_the_profile() {
        let mut world = World::new();
        let stats = RenderStats {
            draw_calls: 9,
            ..Default::default()
        };
        let mut ctx = world.context();
        apply(&mut ctx, 1, ReplayOutcome::default(), Some(stats));
        assert_eq!(ctx.profile.render.draw_calls, 9);
        assert_eq!(ctx.profile.render.skinned_pool_free, 0);
    }
}
