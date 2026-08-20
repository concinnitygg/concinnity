// concinnity-physics/src/rig.rs
//
// The character-rig drive: one kinematic capsule per `CharacterRig`
// component (a `SkinnedMesh` that declared a `capsule`). Each fixed tick the
// capsule moves by the target's `RootMotion` displacement -- mapped through
// the rig's authored rotation/scale -- plus gravity, sliding against the
// scene like the player capsule; each frame the blended capsule position is
// written back to the rig component for GraphicsSystem's render follow.

use concinnity_core::assets::{CharacterRig, RootMotion};
use concinnity_core::ecs::{EventCursor, PipelineContext, SkinnedMeshHandle};
use concinnity_core::gfx::root_motion::add3;

use super::interp::PointInterp;
use super::{BodyHandle, CharacterMoveInput, CharacterShape, LayerMask, PhysicsWorld};

// Physics-side state for one rig.
#[derive(Debug)]
pub(crate) struct RigPhysics {
    pub target: SkinnedMeshHandle,
    pub handle: BodyHandle,
    // The capsule the tick's move is resolved against, resized when the rig
    // component's dimensions change.
    shape: CharacterShape,
    // Current vertical velocity (world units/second).
    pub vy: f32,
    // Authoritative simulated capsule centre with its render blend snapshots.
    center: PointInterp,
    // The rig position written back last frame. A component position that
    // differs was moved externally and is adopted with no blend.
    written_pos: Option<[f32; 3]>,
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

// Small height a rig capsule spawns above its authored position. A capsule
// whose bottom starts exactly in contact with the ground never gets its
// downward moves clipped (the character shape-cast ignores hits it starts
// inside), so it would sink a little every frame; spawning just above lets
// the first gravity steps settle it onto the controller's contact offset.
const SPAWN_LIFT: f32 = 0.05;

// Create one kinematic capsule per published `CharacterRig`.
pub(crate) fn init_rigs(
    world: &mut PhysicsWorld,
    ctx: &mut PipelineContext,
    mask: LayerMask,
) -> Vec<RigPhysics> {
    let rigs: Vec<RigPhysics> = ctx
        .query_mut::<CharacterRig>()
        .map(|rig| {
            rig.position[1] += SPAWN_LIFT;
            rig.moved = true;
            let center = center_of(rig);
            RigPhysics {
                target: rig.target,
                handle: world.add_character(rig.half_height, rig.radius, center, mask),
                shape: CharacterShape::capsule(rig.half_height, rig.radius),
                vy: 0.0,
                center: PointInterp::new(center),
                written_pos: Some(rig.position),
            }
        })
        .collect();
    if !rigs.is_empty() {
        tracing::debug!("PhysicsSystem: {} character rig(s)", rigs.len());
    }
    rigs
}

// Read the root-motion displacements published since last frame (by
// AnimationSystem, which runs after physics; the events queue holds them for
// one cycle) into `out`, cleared first so the caller's buffer is reused.
pub(crate) fn drain_motions_into(
    ctx: &PipelineContext,
    cursor: &mut EventCursor,
    out: &mut Vec<RootMotion>,
) {
    out.clear();
    if let Some(ev) = ctx.events::<RootMotion>() {
        out.extend(ev.read(cursor).copied());
    }
}

// Adopt externally moved rig components before the frame's ticks run: a
// position that differs from the one written back last frame was not ours,
// so the capsule snaps to it with no blend across the jump.
pub(crate) fn sync_rigs(ctx: &mut PipelineContext, rigs: &mut [RigPhysics]) {
    for rig_body in rigs.iter_mut() {
        let Some(rig) = ctx
            .query::<CharacterRig>()
            .find(|r| r.target == rig_body.target)
        else {
            continue;
        };
        if rig_body.written_pos != Some(rig.position) {
            rig_body.center.snap(center_of(rig));
        }
    }
}

// Advance every rig capsule one fixed tick: root-motion displacement plus
// gravity, resolved against the scene. A rig with no motion still settles
// under gravity.
pub(crate) fn tick_rigs(
    world: &mut PhysicsWorld,
    ctx: &mut PipelineContext,
    rigs: &mut [RigPhysics],
    motions: &[RootMotion],
    dt: f32,
    gravity: f32,
    mask: LayerMask,
) {
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
        // Root-motion displacement plus the controller's direct drive.
        let displacement = add3(
            rig.world_delta(local),
            [
                rig.desired_move[0] * dt,
                rig.desired_move[1] * dt,
                rig.desired_move[2] * dt,
            ],
        );
        // A one-shot jump only takes off from the ground; discard it either
        // way so a press cannot latch until the next landing.
        if rig.jump_velocity > 0.0 {
            if rig.grounded {
                rig_body.vy = rig.jump_velocity;
            }
            rig.jump_velocity = 0.0;
        }
        rig_body.vy -= gravity * dt;
        let center = rig_body.center.current();
        let desired = [
            displacement[0],
            displacement[1] + rig_body.vy * dt,
            displacement[2],
        ];
        rig_body.shape.resize(rig.half_height, rig.radius);
        let moved = world.move_character(
            &rig_body.shape,
            &CharacterMoveInput {
                center,
                desired,
                dt,
                exclude: rig_body.handle,
                mask,
            },
        );
        let new_center = add3(center, moved.translation);
        world.set_kinematic_translation(rig_body.handle, new_center);
        if moved.grounded && rig_body.vy < 0.0 {
            rig_body.vy = 0.0;
        }
        rig.grounded = moved.grounded;
        rig_body.center.push(new_center);
    }
}

// Write each rig's blended capsule position back to its component for the
// render follow.
pub(crate) fn publish_rigs(ctx: &mut PipelineContext, rigs: &mut [RigPhysics], alpha: f32) {
    for rig_body in rigs.iter_mut() {
        let Some(rig) = ctx
            .query_mut::<CharacterRig>()
            .find(|r| r.target == rig_body.target)
        else {
            continue;
        };
        let center = rig_body.center.sample(alpha);
        let new_pos = [
            center[0],
            center[1] - rig.half_height - rig.radius,
            center[2],
        ];
        rig_body.written_pos = Some(new_pos);
        if new_pos != rig.position {
            rig.position = new_pos;
            rig.moved = true;
        }
    }
}
