use crate::ecs::{Entity, MenuActive, PipelineContext, SimTiming, StepResult, System, World};
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

use crate::components::{SkyRotation, Transform};
use crate::sky::{SkyFrame, SkyOrientation};

const FULL_TURN_DEG: f32 = 360.0;

/// Advances the world's celestial-sphere rotation on the fixed timestep.
///
/// Publishes the tick's [`SkyOrientation`] and writes it onto the
/// [`SkyRotation`] entity's [`Transform`], so the renderer, the lights and the
/// transform hierarchy all read one rotation. Frozen while a world-pausing
/// screen is open, like the rest of the simulation.
#[derive(Debug)]
pub struct SkyRotationSystem {
    axis: [f32; 3],
    degrees_per_second: f32,
    angle_deg: f32,
    turns: i32,
    pivots: Vec<Entity>,
}

impl SkyRotationSystem {
    /// The system for a world that declares a `SkyRotation`.
    pub fn for_world(world: &World) -> Option<Self> {
        world.query::<SkyRotation>().next().map(Self::new)
    }

    /// The system for a world's authored rotation.
    pub fn new(rotation: &SkyRotation) -> Self {
        Self {
            axis: rotation.axis,
            degrees_per_second: rotation.degrees_per_second,
            angle_deg: rotation.angle_deg,
            turns: 0,
            pivots: Vec::new(),
        }
    }

    // Publish the current orientation and carry it onto every pivot entity.
    fn publish(&self, ctx: &mut PipelineContext<'_>) {
        let mut sky = SkyOrientation::new(self.axis, self.angle_deg).with_turns(self.turns);
        if let Some(frame) = ctx.resource::<SkyFrame>() {
            sky = sky.in_frame(&frame.0);
        }
        let rotation_deg = sky.euler_deg();
        for &pivot in &self.pivots {
            match ctx.get_mut::<Transform>(pivot) {
                Some(transform) => transform.rotation_deg = rotation_deg,
                None => {
                    ctx.insert(
                        pivot,
                        Transform {
                            rotation_deg,
                            ..Default::default()
                        },
                    );
                }
            }
        }
        ctx.insert_resource(sky);
    }
}

impl System for SkyRotationSystem {
    fn init(&mut self, ctx: &mut PipelineContext<'_>) {
        self.pivots = ctx
            .query_with_entity::<SkyRotation>()
            .map(|(entity, _)| entity)
            .collect();
        self.publish(ctx);
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        if ctx.resource::<MenuActive>().is_some_and(|m| m.0) {
            return StepResult::Continue;
        }
        let timing = ctx.resource::<SimTiming>().copied().unwrap_or_default();
        let turned = self.degrees_per_second * timing.ticks as f32 * timing.tick_dt;
        // Kept within one turn, so a long session never loses precision to a
        // large angle.
        let total = self.angle_deg + turned;
        let wrapped = total % FULL_TURN_DEG;
        self.angle_deg = if wrapped < 0.0 {
            wrapped + FULL_TURN_DEG
        } else {
            wrapped
        };
        let whole = i32::saturating_from(concinnity_core::math::round(
            (total - self.angle_deg) / FULL_TURN_DEG,
        ));
        self.turns = self.turns.saturating_add(whole);
        self.publish(ctx);
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::World;

    fn world_with(rotation: SkyRotation) -> (World, SkyRotationSystem) {
        let system = SkyRotationSystem::new(&rotation);
        let mut world = World::new();
        world.add_component(rotation);
        (world, system)
    }

    // One second of a 90-degree-per-second sky is a quarter turn, arrived at
    // through 60 fixed ticks rather than a wall clock.
    #[test]
    fn the_angle_integrates_on_the_fixed_timestep() {
        let (mut world, mut system) = world_with(SkyRotation {
            degrees_per_second: 90.0,
            ..Default::default()
        });
        system.init(&mut world.context());
        for _ in 0..60 {
            system.step(&mut world.context());
        }
        let sky = world
            .resource::<SkyOrientation>()
            .copied()
            .expect("published");
        assert!((sky.angle_deg - 90.0).abs() < 1e-3, "{}", sky.angle_deg);
        let up = sky.rotate([0.0, 0.0, 1.0]);
        assert!(up[1] > 0.99, "a quarter turn puts +Z overhead: {up:?}");
    }

    // The authored start angle is where the world opens, before any tick.
    #[test]
    fn the_authored_angle_is_live_from_init() {
        let (mut world, mut system) = world_with(SkyRotation {
            degrees_per_second: 0.0,
            angle_deg: 180.0,
            ..Default::default()
        });
        system.init(&mut world.context());
        let sky = world
            .resource::<SkyOrientation>()
            .copied()
            .expect("published");
        assert_eq!(sky.angle_deg, 180.0);
    }

    // The pivot entity carries the rotation as its own transform, which is what
    // a parented prop orbits on.
    #[test]
    fn the_pivot_entity_carries_the_rotation() {
        let (mut world, mut system) = world_with(SkyRotation {
            degrees_per_second: 60.0,
            ..Default::default()
        });
        system.init(&mut world.context());
        {
            let ctx = world.context();
            let mut transforms = ctx.query::<Transform>();
            let t = transforms.next().expect("the pivot gained a transform");
            assert_eq!(t.rotation_deg, [0.0, 0.0, 0.0]);
            assert!(transforms.next().is_none(), "one pivot, one transform");
        }
        for _ in 0..30 {
            system.step(&mut world.context());
        }
        let ctx = world.context();
        let t = ctx
            .query::<Transform>()
            .next()
            .copied()
            .expect("still there");
        // Half a second at 60 deg/s is 30 degrees of pitch, in the sense that
        // lifts +Z toward +Y (a negative right-handed turn about +X).
        assert!(
            (t.rotation_deg[0] + 30.0).abs() < 1e-2,
            "{:?}",
            t.rotation_deg
        );
    }

    // A full turn lands back where it started, and a negative rate counts
    // down from the top of the turn rather than below zero.
    #[test]
    fn the_angle_stays_within_one_turn() {
        let (mut world, mut system) = world_with(SkyRotation {
            degrees_per_second: -90.0,
            angle_deg: 30.0,
            ..Default::default()
        });
        system.init(&mut world.context());
        for _ in 0..60 {
            system.step(&mut world.context());
        }
        let angle = world
            .resource::<SkyOrientation>()
            .expect("published")
            .angle_deg;
        assert!((angle - 300.0).abs() < 1e-2, "{angle}");
        // Wrapping below zero counted one turn back.
        let turns = world.resource::<SkyOrientation>().expect("published").turns;
        assert_eq!(turns, -1);
        let orientation = SkyOrientation::new([1.0, 0.0, 0.0], -60.0).rotate([0.0, 0.0, 1.0]);
        let published = world
            .resource::<SkyOrientation>()
            .expect("published")
            .rotate([0.0, 0.0, 1.0]);
        assert!(
            (0..3).all(|i| (orientation[i] - published[i]).abs() < 1e-3),
            "{orientation:?} {published:?}"
        );
    }

    // The turns run on past the angle's wrap, one per full turn.
    #[test]
    fn full_turns_are_counted() {
        let (mut world, mut system) = world_with(SkyRotation {
            degrees_per_second: 400.0,
            ..Default::default()
        });
        system.init(&mut world.context());
        for _ in 0..120 {
            system.step(&mut world.context());
        }
        let sky = world
            .resource::<SkyOrientation>()
            .copied()
            .expect("published");
        assert_eq!(sky.turns, 2);
        assert!((sky.angle_deg - 80.0).abs() < 0.1, "{}", sky.angle_deg);
    }

    // A paused world holds still: the sky is simulation, not presentation.
    #[test]
    fn an_open_menu_freezes_the_sky() {
        let (mut world, mut system) = world_with(SkyRotation {
            degrees_per_second: 90.0,
            ..Default::default()
        });
        system.init(&mut world.context());
        world.insert_resource(MenuActive(true));
        for _ in 0..60 {
            system.step(&mut world.context());
        }
        assert_eq!(
            world
                .resource::<SkyOrientation>()
                .expect("published")
                .angle_deg,
            0.0
        );
    }
}
