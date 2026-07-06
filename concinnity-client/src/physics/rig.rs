// src/physics/rig.rs
//
// The character-rig drive: one kinematic capsule per `CharacterRig`
// component (a `SkinnedMesh` that declared a `capsule`). Each frame the
// capsule moves by the target's `RootMotion` displacement -- mapped through
// the rig's authored rotation/scale -- plus gravity, sliding against the
// scene like the player capsule; the resolved position is written back to
// the rig component for GraphicsSystem's render follow.

use crate::assets::{CharacterRig, RootMotion};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{EventCursor, PipelineContext};
use crate::gfx::root_motion::add3;

use super::{BodyHandle, PhysicsWorld};

// Physics-side state for one rig.
#[derive(Debug)]
pub(crate) struct RigPhysics {
    pub target: AssetId,
    pub handle: BodyHandle,
    // Current vertical velocity (world units/second).
    pub vy: f32,
}

// Capsule centre for a rig's mesh-origin position: the capsule stands on
// the origin (the mesh's feet).
fn center_of(rig: &CharacterRig) -> [f32; 3] {
    [
        rig.position[0],
        rig.position[1] + rig.half_height + rig.radius,
        rig.position[2],
    ]
}

// Create one kinematic capsule per published `CharacterRig`.
pub(crate) fn init_rigs(world: &mut PhysicsWorld, ctx: &mut PipelineContext) -> Vec<RigPhysics> {
    let rigs: Vec<RigPhysics> = ctx
        .query::<CharacterRig>()
        .map(|rig| RigPhysics {
            target: rig.target,
            handle: world.add_character(rig.half_height, rig.radius, center_of(rig)),
            vy: 0.0,
        })
        .collect();
    if !rigs.is_empty() {
        tracing::debug!("PhysicsSystem: {} character rig(s)", rigs.len());
    }
    rigs
}

// Step every rig capsule: root-motion displacement plus gravity, resolved
// against the scene. Runs every frame -- a rig with no motion events still
// settles under gravity.
pub(crate) fn step_rigs(
    world: &mut PhysicsWorld,
    ctx: &mut PipelineContext,
    rigs: &mut [RigPhysics],
    cursor: &mut EventCursor,
    dt: f32,
    gravity: f32,
) {
    if rigs.is_empty() {
        return;
    }
    // This frame's displacements (published by AnimationSystem after last
    // frame's physics step; the events queue holds them for one cycle).
    let motions: Vec<RootMotion> = ctx
        .events::<RootMotion>()
        .map(|ev| ev.read(cursor).into_iter().copied().collect())
        .unwrap_or_default();
    for rig_body in rigs.iter_mut() {
        let Some(rig) = ctx
            .query_mut::<CharacterRig>()
            .find(|r| r.target == rig_body.target)
        else {
            continue;
        };
        let mut local = [0.0f32; 3];
        for motion in motions.iter().filter(|m| m.target == rig_body.target) {
            local = add3(local, motion.delta);
        }
        let displacement = rig.world_delta(local);
        rig_body.vy -= gravity * dt;
        let center = center_of(rig);
        let desired = [
            displacement[0],
            displacement[1] + rig_body.vy * dt,
            displacement[2],
        ];
        let moved = world.move_character(rig.half_height, rig.radius, center, desired, dt);
        let new_center = add3(center, moved.translation);
        world.set_kinematic_translation(rig_body.handle, new_center);
        if moved.grounded && rig_body.vy < 0.0 {
            rig_body.vy = 0.0;
        }
        rig.grounded = moved.grounded;
        let new_pos = [
            new_center[0],
            new_center[1] - rig.half_height - rig.radius,
            new_center[2],
        ];
        if new_pos != rig.position {
            rig.position = new_pos;
            rig.moved = true;
        }
    }
}
