// src/hud/fps_counter.rs
//
// FPS-counter overlay behavior. An internal system (not a declarable asset):
// `World::start` constructs one from the world's `FpsCounter` component and it
// updates that component's `label` with the current rate once per second.

use crate::components::{FpsCounter, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use std::time::Instant;

#[derive(Debug)]
pub struct FpsCounterSystem {
    last_time: Instant,
    frame_count: u32,
    label: Option<AssetId>,
}

impl FpsCounterSystem {
    // Build the counter from a world's `FpsCounter` request component.
    pub fn new(config: FpsCounter) -> Self {
        Self {
            last_time: Instant::now(),
            frame_count: 0,
            label: config.label,
        }
    }
}

impl System for FpsCounterSystem {
    fn access(&self) -> crate::ecs::Access {
        crate::ecs::Access::new()
            .writes_components(crate::component_mask![crate::components::TextLabel])
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_time).as_secs_f64();
        if elapsed >= 1.0 {
            let fps = self.frame_count as f64 / elapsed;
            if let Some(label_id) = self.label {
                for lbl in ctx.query_mut::<TextLabel>() {
                    if lbl.asset_id == label_id {
                        lbl.content = format!("FPS: {:.0}", fps);
                        break;
                    }
                }
            }
            self.frame_count = 0;
            self.last_time = now;
        }
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use crate::components::FpsCounter;
    use crate::ecs::World;

    // An FpsCounter component spawns the internal counter system.
    #[test]
    fn fps_counter_component_spawns_internal_system() {
        let mut world = World::new();
        world.add_component(FpsCounter::default());
        world.start().unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["FpsCounter"]);
    }

    #[test]
    fn no_fps_counter_no_system() {
        let mut world = World::new();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }

    // Once a second of wall time has elapsed the step writes the rate into the
    // counter's label. The elapsed is injected by backdating `last_time` (an
    // in-file-accessible field), so no real sleep is needed.
    #[test]
    fn rate_written_into_label_after_a_second() {
        use crate::components::TextLabel;
        use crate::ecs::asset_id::AssetId;
        use std::time::{Duration, Instant};

        let mut world = World::new();
        world.add_component(FpsCounter {
            label: Some(AssetId(1)),
        });
        world.add_component(TextLabel {
            asset_id: AssetId(1),
            ..Default::default()
        });
        world.start().unwrap();

        // Backdate the window start so the next step crosses the 1s threshold.
        for system in world.systems_mut() {
            if let crate::ecs::SystemAsset::FpsCounter(s) = system {
                s.last_time = Instant::now() - Duration::from_millis(1100);
            }
        }
        world.step();

        let content = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(1))
            .map(|l| l.content.clone())
            .unwrap_or_default();
        assert!(content.starts_with("FPS: "), "{content}");
    }
}
