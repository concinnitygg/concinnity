// concinnity-physics/src/probes.rs
//
// Raycast probe answering: each frame the ground probes the animation IK
// published and the third-person camera's occlusion probe get their rays
// resolved against the scene. Both exchanges are one frame behind their
// writer (the probing systems run after physics), which is invisible at
// frame rates.

use concinnity_core::assets::{CameraProbe, GroundProbes};
use concinnity_core::ecs::PipelineContext;

use super::PhysicsWorld;
use super::rig::RigPhysics;

// Kept between the camera and the nearest obstruction so the near plane
// never clips into it.
const CAMERA_MARGIN: f32 = 0.15;
// The camera never pulls closer to the pivot than this.
const CAMERA_MIN_DISTANCE: f32 = 0.3;

// Answer every probe component. Character capsules are transparent to their
// own probes (a foot ray starts inside the rig's capsule; the camera ray
// starts at a pivot inside it).
pub(crate) fn step_probes(world: &PhysicsWorld, ctx: &mut PipelineContext, rigs: &[RigPhysics]) {
    let handle_of = |target| rigs.iter().find(|r| r.target == target).map(|r| r.handle);

    for probes in ctx.query_mut::<GroundProbes>() {
        let exclude = handle_of(probes.target);
        for probe in &mut probes.probes {
            probe.hit = world
                .raycast(probe.origin, [0.0, -1.0, 0.0], probe.max_dist, exclude)
                .map(|hit| (hit.point, hit.normal));
        }
    }

    for probe in ctx.query_mut::<CameraProbe>() {
        let dir = [
            probe.desired[0] - probe.pivot[0],
            probe.desired[1] - probe.pivot[1],
            probe.desired[2] - probe.pivot[2],
        ];
        let dist = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        probe.clearance = world
            .raycast(probe.pivot, dir, dist, handle_of(probe.target))
            .map(|hit| (hit.distance - CAMERA_MARGIN).max(CAMERA_MIN_DISTANCE));
    }
}
