// The camera's capsule: who gets one, and building it.

use crate::components::{Camera3D, CameraTrack, RigidBody};
use crate::ecs::PipelineContext;
use crate::physics::{CharacterCapsule, Simulation};

use super::PlayerPhysics;
use crate::physics::interp::PointInterp;
use crate::physics::layers::LayerMask;

/// Whether a camera track drives the camera, which then has no capsule.
pub(super) fn camera_is_tracked(ctx: &PipelineContext<'_>) -> bool {
    ctx.query::<CameraTrack>().next().is_some()
}

/// The capsule for the world's camera, added to `world` on `mask`, or `None`
/// for a world without a first-person camera.
///
/// Every first-person camera is collided as a capsule. A `RigidBody` upgrades it
/// from a free-flying spectator to a grounded, gravity-bound character. A
/// third-person camera (a controller with a `follow` block) gets no capsule: it
/// is a virtual orbit around the followed character, whose own rig capsule is
/// the collided body. `floor_y` is set from the first camera to get one.
pub(super) fn spawn(
    world: &mut Simulation,
    ctx: &PipelineContext<'_>,
    floor_y: &mut f32,
    mask: LayerMask,
) -> Option<PlayerPhysics> {
    let cam_pos = ctx
        .query::<Camera3D>()
        .next()
        .filter(|c| {
            c.controller
                .as_ref()
                .is_none_or(|ctrl| ctrl.follow.is_none())
        })
        .map(|c| c.position)?;
    let rb_opt = ctx.query::<RigidBody>().next().cloned();
    let has_gravity = rb_opt.is_some();
    let rb = rb_opt.unwrap_or_default();
    if *floor_y == 0.0 {
        *floor_y = cam_pos[1];
    }
    let radius = rb.capsule_radius.max(0.05);
    let half_height = ((rb.capsule_height * 0.5) - radius).max(0.05);
    // a grounded character's eye sits at the capsule top; a flying
    // camera's capsule is centered on the eye.
    let eye_offset = if has_gravity {
        (rb.capsule_height * 0.5).max(radius + 0.05)
    } else {
        0.0
    };
    let center = [cam_pos[0], cam_pos[1] - eye_offset, cam_pos[2]];
    world.configure_character(rb.max_slope_deg, rb.step_height, has_gravity);
    let handle = world.add_character(half_height, radius, center, mask)?;
    Some(PlayerPhysics {
        handle,
        shape: CharacterCapsule::new(half_height, radius),
        eye_offset,
        has_gravity,
        gravity_scale: rb.gravity_scale.max(0.0),
        jump_height: rb.jump_height.max(0.0),
        vy: 0.0,
        grounded: true,
        center: PointInterp::new(center),
        written_eye: None,
    })
}
