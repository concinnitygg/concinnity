//! FPS-counter overlay behavior. An internal system (not a declarable asset):
//! `World::start` constructs one from the world's `FpsCounter` component and it
//! updates that component's `label` with the current rate once per second.

use concinnity_core::components::FpsCounter;
use concinnity_core::components::TextLabel;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{Access, FrameTime, PipelineContext, StepResult, System};

use super::rate_window::RateWindow;

// Seconds each rate readout averages over.
const WINDOW_SECS: f32 = 1.0;

#[derive(Debug)]
pub(crate) struct FpsCounterSystem {
    window: RateWindow,
    label: Option<AssetId>,
}

impl FpsCounterSystem {
    // Build the counter from a world's `FpsCounter` request component.
    pub(crate) fn new(config: FpsCounter) -> Self {
        Self {
            window: RateWindow::default(),
            label: config.label,
        }
    }
}

impl System for FpsCounterSystem {
    fn access(&self) -> Access {
        Access::new()
            .writes_components(crate::component_mask![TextLabel])
            .reads_resources(crate::resource_mask![FrameTime])
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let dt = ctx.resource::<FrameTime>().copied().unwrap_or_default().dt;
        if let Some((frames, secs)) = self.window.tick(dt, WINDOW_SECS) {
            let fps = frames as f32 / secs;
            if let Some(label_id) = self.label {
                for lbl in ctx.query_mut::<TextLabel>() {
                    if lbl.asset_id == label_id {
                        lbl.content = format!("FPS: {:.0}", fps);
                        break;
                    }
                }
            }
        }
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use crate::ecs::SYSTEMS;
    use concinnity_core::components::FpsCounter;
    use concinnity_core::ecs::World;

    // An FpsCounter component spawns the internal counter system.
    #[test]
    fn fps_counter_component_spawns_internal_system() {
        let mut world = World::new();
        world.add_component(FpsCounter::default());
        world.start(SYSTEMS).unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["FpsCounter"]);
    }

    #[test]
    fn no_fps_counter_no_system() {
        let mut world = World::new();
        world.start(SYSTEMS).unwrap();
        assert!(world.systems().is_empty());
    }

    // Once a second of frame time has accumulated the step writes the rate
    // into the counter's label.
    #[test]
    fn rate_written_into_label_after_a_second() {
        use concinnity_core::components::TextLabel;
        use concinnity_core::ecs::FrameTime;
        use concinnity_core::ecs::asset_id::AssetId;

        let mut world = World::new();
        world.add_component(FpsCounter {
            label: Some(AssetId(1)),
        });
        world.add_component(TextLabel {
            asset_id: AssetId(1),
            ..Default::default()
        });
        world.start(SYSTEMS).unwrap();

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
        use concinnity_core::components::TextLabel;
        use concinnity_core::ecs::asset_id::AssetId;

        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(1))
            .map(|l| l.content.clone())
            .unwrap_or_default()
    }
}
