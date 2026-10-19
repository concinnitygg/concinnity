//! FPS-counter overlay behavior. An internal system (not a declarable asset):
//! `World::start` constructs one from the world's `FpsCounter` component and it
//! updates that component's `label` with the current rate once per second.

use crate::ecs::{Access, FrameTime, PipelineContext, Ref, StepResult, System, World};
use alloc::format;

use crate::components::{FpsCounter, TextLabel};
use crate::ecs::asset_id::AssetId;

use super::rate_window::RateWindow;

// Seconds each rate readout averages over.
const WINDOW_SECS: f32 = 1.0;

/// The FPS counter, built from a world's `FpsCounter` request component.
#[derive(Debug)]
pub struct FpsCounterSystem {
    window: RateWindow,
    label: Option<AssetId>,
}

impl FpsCounterSystem {
    /// The system for a world that declares an `FpsCounter`.
    pub fn for_world(world: &World) -> Option<Self> {
        world.query::<FpsCounter>().next().cloned().map(Self::new)
    }

    /// Build the counter from a world's `FpsCounter` request component.
    pub fn new(config: FpsCounter) -> Self {
        Self {
            window: RateWindow::default(),
            label: config.label.map(Ref::id),
        }
    }
}

impl System for FpsCounterSystem {
    fn access(&self) -> Access {
        Access::new()
            .writes_components(crate::component_mask![TextLabel])
            .reads_resources(crate::resource_mask![crate::ecs::EntityById, FrameTime])
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        let dt = ctx.resource::<FrameTime>().copied().unwrap_or_default().dt;
        if let Some((frames, secs)) = self.window.tick(dt, WINDOW_SECS) {
            let fps = frames as f32 / secs;
            crate::ecs::by_asset_id::update::<TextLabel>(ctx, self.label, |l| {
                l.content = format!("FPS: {fps:.0}");
            });
        }
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{Phase, World};
    use crate::test_support::start_with;
    use alloc::string::String;

    // Once a second of frame time has accumulated the step writes the rate
    // into the counter's label.
    #[test]
    fn rate_written_into_label_after_a_second() {
        let config = FpsCounter {
            label: Some(Ref::new(AssetId(1))),
        };
        let mut world = World::new();
        world.push_identified(AssetId(1), TextLabel::default());
        start_with(
            &mut world,
            Phase::Late,
            "FpsCounterSystem",
            FpsCounterSystem::new(config),
        );

        // Two half-second frames: the second closes the 1s window.
        world.insert_resource(FrameTime {
            dt: 0.5,
            elapsed: 0.0,
        });
        world.step();
        assert_eq!(content(&world), "", "the window has not closed yet");
        world.step();
        assert_eq!(content(&world), "FPS: 2");
    }

    fn content(world: &World) -> String {
        world
            .get_by_id::<TextLabel>(AssetId(1))
            .map(|l| l.content.clone())
            .unwrap_or_default()
    }
}
