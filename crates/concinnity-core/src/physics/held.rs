// The planet ground's stand-in rule: the ground only exists in the patch
// around the camera, so a freely simulated body the patch has moved away from
// would fall into the planet. Such a body is held where it is, driven by
// nothing and pulled by nothing, until the patch covers it again, and is then
// handed back to the solver at rest.

use alloc::vec::Vec;

use crate::physics::{BodyHandle, Simulation};
use crate::planet::GroundPatch;

// How far inside the patch's edge a body must stay to keep simulating, and
// how far inside it a held body must be before it is let go. The gap between
// the two keeps a body near the line from flickering between the two states.
const HOLD_INSET: f32 = 2.0;
const RELEASE_INSET: f32 = 4.0;

/// Where on the local `X`/`Z` plane a ground patch reaches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Footprint {
    center: [f32; 2],
    half: [f32; 2],
}

impl Footprint {
    pub(super) fn of(patch: &GroundPatch) -> Self {
        Self {
            center: [patch.center[0], patch.center[2]],
            half: patch.grid.extent(),
        }
    }

    // Whether `p` lies at least `inset` inside the footprint's edge.
    fn covers(&self, p: [f32; 3], inset: f32) -> bool {
        (p[0] - self.center[0]).abs() <= self.half[0] - inset
            && (p[2] - self.center[1]).abs() <= self.half[1] - inset
    }
}

/// The freely simulated bodies currently held off a planet's missing ground.
#[derive(Debug, Default)]
pub(super) struct HeldBodies {
    held: Vec<BodyHandle>,
}

impl HeldBodies {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            held: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn is_held(&self, handle: BodyHandle) -> bool {
        self.held.contains(&handle)
    }

    /// Hold every body in `bodies` that has strayed off `footprint`, and let
    /// go of every held one it covers again. `bodies` are the freely
    /// simulated bodies the caller tracks; a held body no longer among them
    /// is forgotten.
    pub(super) fn update(
        &mut self,
        world: &mut Simulation,
        bodies: impl Iterator<Item = BodyHandle> + Clone,
        footprint: &Footprint,
    ) {
        self.held
            .retain(|h| bodies.clone().any(|b| b == *h) && world.body_pose_quat(*h).is_some());
        for handle in bodies {
            let Some((position, _)) = world.body_pose_quat(handle) else {
                continue;
            };
            let held = self.is_held(handle);
            if !held && !footprint.covers(position, HOLD_INSET) {
                if world.make_kinematic(handle) {
                    self.held.push(handle);
                }
            } else if held && footprint.covers(position, RELEASE_INSET) {
                world.make_dynamic(handle, [0.0; 3]);
                self.held.retain(|h| *h != handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{ColliderShape, SimConfig};
    use crate::physics::{DynamicParams, LayerMask};
    use crate::terrain::TerrainGrid;
    use alloc::vec;

    const TICK: f32 = 1.0 / 60.0;

    fn patch(center: [f32; 3]) -> GroundPatch {
        GroundPatch {
            center,
            grid: TerrainGrid::new(4, [16.0, 16.0], vec![0.0; 25]).unwrap(),
        }
    }

    fn ball(world: &mut Simulation, at: [f32; 3]) -> BodyHandle {
        world
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                at,
                [0.0; 3],
                DynamicParams {
                    mass: 1.0,
                    friction: 0.6,
                    restitution: 0.0,
                    gravity_scale: 1.0,
                    linear_damping: 0.0,
                },
                LayerMask::ALL,
            )
            .unwrap()
    }

    // A world with one patch-sized ground grid at `center`, as the physics
    // system keeps it, plus `balls` dropped at their positions.
    fn world_on(center: [f32; 3], balls: &[[f32; 3]]) -> (Simulation, BodyHandle, Vec<BodyHandle>) {
        let config = SimConfig {
            allow_sleep: false,
            ..SimConfig::default()
        };
        let mut world = Simulation::new(config, 8);
        let p = patch(center);
        let ground = world
            .add_heightfield(
                5,
                5,
                vec![0.0; 25],
                [32.0, 1.0, 32.0],
                p.center,
                LayerMask::ALL,
            )
            .unwrap();
        let handles = balls.iter().map(|&b| ball(&mut world, b)).collect();
        (world, ground, handles)
    }

    fn step(
        world: &mut Simulation,
        held: &mut HeldBodies,
        bodies: &[BodyHandle],
        at: &GroundPatch,
    ) {
        held.update(world, bodies.iter().copied(), &Footprint::of(at));
        world.step(TICK);
    }

    fn y(world: &Simulation, h: BodyHandle) -> f32 {
        world.body_pose_quat(h).unwrap().0[1]
    }

    // A body the patch leaves behind stays exactly where it rested, however
    // long the camera is away.
    #[test]
    fn a_body_left_behind_stays_put() {
        let (mut world, ground, balls) = world_on([0.0; 3], &[[0.0, 0.6, 0.0]]);
        let mut held = HeldBodies::default();
        let here = patch([0.0; 3]);
        for _ in 0..90 {
            step(&mut world, &mut held, &balls, &here);
        }
        let rest = world.body_pose_quat(balls[0]).unwrap().0;

        let away = patch([100.0, 0.0, 0.0]);
        assert!(world.replace_heightfield(
            ground,
            5,
            5,
            vec![0.0; 25],
            [32.0, 1.0, 32.0],
            away.center
        ));
        for _ in 0..300 {
            step(&mut world, &mut held, &balls, &away);
        }
        assert!(held.is_held(balls[0]));
        assert_eq!(world.body_pose_quat(balls[0]).unwrap().0, rest);
    }

    // When the patch comes back over a held body, it is let go at rest and
    // settles on the ground like any other.
    #[test]
    fn a_body_the_patch_returns_over_settles() {
        let (mut world, ground, balls) = world_on([100.0, 0.0, 0.0], &[[0.0, 3.0, 0.0]]);
        let mut held = HeldBodies::default();
        let away = patch([100.0, 0.0, 0.0]);
        for _ in 0..120 {
            step(&mut world, &mut held, &balls, &away);
        }
        assert_eq!(y(&world, balls[0]), 3.0, "held in the air, never pulled");

        let back = patch([0.0; 3]);
        assert!(world.replace_heightfield(
            ground,
            5,
            5,
            vec![0.0; 25],
            [32.0, 1.0, 32.0],
            back.center
        ));
        step(&mut world, &mut held, &balls, &back);
        assert!(!held.is_held(balls[0]));
        let fall = (2.0 * 2.5 / crate::physics::GRAVITY).sqrt();
        let ticks = (fall / TICK) as usize + 60;
        for _ in 0..ticks {
            step(&mut world, &mut held, &balls, &back);
        }
        assert!(
            (y(&world, balls[0]) - 0.5).abs() < 0.05,
            "{}",
            y(&world, balls[0])
        );
    }

    // A body the patch covers simulates exactly as it would with no rule.
    #[test]
    fn a_body_inside_the_patch_is_unaffected() {
        let (mut with_rule, _, a) = world_on([0.0; 3], &[[3.0, 4.0, -2.0]]);
        let (mut without, _, b) = world_on([0.0; 3], &[[3.0, 4.0, -2.0]]);
        let mut held = HeldBodies::default();
        let here = patch([0.0; 3]);
        for _ in 0..120 {
            step(&mut with_rule, &mut held, &a, &here);
            without.step(TICK);
        }
        assert!(!held.is_held(a[0]));
        assert_eq!(
            with_rule.body_pose_quat(a[0]).unwrap(),
            without.body_pose_quat(b[0]).unwrap()
        );
    }

    #[test]
    fn the_footprint_keeps_a_margin_inside_the_edge() {
        let f = Footprint::of(&patch([10.0, 0.0, -5.0]));
        assert!(f.covers([10.0, 99.0, -5.0], RELEASE_INSET));
        assert!(f.covers([10.0 + 13.0, 0.0, -5.0], HOLD_INSET));
        assert!(!f.covers([10.0 + 13.0, 0.0, -5.0], RELEASE_INSET));
        assert!(!f.covers([10.0, 0.0, -5.0 - 15.0], HOLD_INSET));
    }
}
