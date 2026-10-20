//! The wall-clock side of the simulation clock. Runs at the runtime level, after
//! the frame pacer and before the world steps: each frame's wall delta feeds
//! core's [`TickAccumulator`], which publishes the frame's clock.

use concinnity_engine::ecs::{TickAccumulator, World};
use std::num::NonZeroU32;
use std::time::Instant;

// One per `Runtime`, advanced once per world step.
#[derive(Debug, Default)]
pub(crate) struct SimClock {
    last: Option<Instant>,
    ticks: TickAccumulator,
}

impl SimClock {
    // Publish the clock of a frame starting at `now` into `world`.
    pub(crate) fn publish(&mut self, world: &mut World, now: Instant, fixed: Option<NonZeroU32>) {
        let elapsed = self.wall_elapsed(now);
        self.ticks.publish_frame(world, fixed, elapsed);
    }

    // Seconds since the previous frame started; the first frame has none.
    fn wall_elapsed(&mut self, now: Instant) -> f32 {
        let elapsed = self
            .last
            .map_or(0.0, |t| now.duration_since(t).as_secs_f32());
        self.last = Some(now);
        elapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::ecs::{FrameTime, SimTiming};

    #[test]
    fn first_frame_has_no_elapsed_time() {
        let mut clock = SimClock::default();
        let mut world = World::new();
        clock.publish(&mut world, Instant::now(), None);
        assert_eq!(world.resource::<SimTiming>().unwrap().ticks, 0);
        assert_eq!(world.resource::<FrameTime>().unwrap().elapsed, 0.0);
    }
}
