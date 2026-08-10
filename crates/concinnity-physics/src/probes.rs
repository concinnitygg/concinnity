// concinnity-physics/src/probes.rs
//
// Raycast probe answering: each frame the ground probes the animation IK
// published and the third-person camera's occlusion probe get their rays
// resolved against the scene. Both exchanges are one frame behind their
// writer (the probing systems run after physics), which is invisible at
// frame rates.

use concinnity_core::assets::{AudioOcclusionProbe, CameraProbe, GroundProbes};
use concinnity_core::ecs::PipelineContext;

use super::rig::RigPhysics;
use super::{LayerMask, PhysicsWorld};

// Kept between the camera and the nearest obstruction so the near plane
// never clips into it.
const CAMERA_MARGIN: f32 = 0.15;
// The camera never pulls closer to the pivot than this.
const CAMERA_MIN_DISTANCE: f32 = 0.3;

// Answer every probe component. Character capsules are transparent to their
// own probes (a foot ray starts inside the rig's capsule; the camera ray
// starts at a pivot inside it); `mask` keeps the rays on scene geometry
// (world and prop layers), so trigger regions and other characters never
// register as ground or as a camera obstruction.
pub(crate) fn step_probes(
    world: &PhysicsWorld,
    ctx: &mut PipelineContext,
    rigs: &[RigPhysics],
    mask: LayerMask,
) {
    let handle_of = |target| rigs.iter().find(|r| r.target == target).map(|r| r.handle);

    for probes in ctx.query_mut::<GroundProbes>() {
        let exclude = handle_of(probes.target);
        for probe in &mut probes.probes {
            probe.hit = world
                .raycast(
                    probe.origin,
                    [0.0, -1.0, 0.0],
                    probe.max_dist,
                    exclude,
                    mask,
                )
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
            .raycast(probe.pivot, dir, dist, handle_of(probe.target), mask)
            .map(|hit| (hit.distance - CAMERA_MARGIN).max(CAMERA_MIN_DISTANCE));
    }

    // Audio occlusion: whether scene geometry interrupts the listener ->
    // emitter segment. A degenerate segment reads as clear (raycast rejects
    // a zero-length direction).
    for probe in ctx.query_mut::<AudioOcclusionProbe>() {
        let dir = [
            probe.to[0] - probe.from[0],
            probe.to[1] - probe.from[1],
            probe.to[2] - probe.from[2],
        ];
        let dist = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        probe.blocked = Some(world.raycast(probe.from, dir, dist, None, mask).is_some());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColliderShape;
    use concinnity_core::ecs::{Arena, ComponentStorage, FrameContext, Resources};
    use concinnity_core::gfx::profile::FrameProfile;
    use concinnity_store::blob::BlobData;

    struct ProbeWorld {
        components: ComponentStorage,
        blob: BlobData,
        profile: FrameProfile,
        resources: Resources,
        scratch: Arena,
    }

    impl ProbeWorld {
        fn new() -> Self {
            Self {
                components: ComponentStorage::default(),
                blob: BlobData::empty(),
                profile: FrameProfile::default(),
                resources: Resources::new(),
                scratch: Arena::with_capacity(64 * 1024),
            }
        }

        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
                frame: FrameContext::new(&self.scratch),
            }
        }
    }

    // An audio occlusion probe reads blocked when a wall interrupts its
    // segment, clear when nothing does, and clear on a degenerate
    // zero-length segment.
    #[test]
    fn audio_probes_answer_blocked_and_clear() {
        let mut world = PhysicsWorld::new(20.0);
        // A wall at x = 5 crossing the first probe's segment.
        world.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [0.5, 5.0, 5.0],
            },
            [5.0, 0.0, 0.0],
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        );
        world.step(1.0 / 60.0);

        let mut w = ProbeWorld::new();
        let probe = |to: [f32; 3]| AudioOcclusionProbe {
            from: [0.0; 3],
            to,
            blocked: None,
        };
        w.components.push_typed(probe([10.0, 0.0, 0.0]));
        w.components.push_typed(probe([0.0, 0.0, 10.0]));
        w.components.push_typed(probe([0.0; 3]));

        let mut ctx = w.ctx();
        step_probes(&world, &mut ctx, &[], LayerMask::ALL);

        let answers: Vec<Option<bool>> = ctx
            .query::<AudioOcclusionProbe>()
            .map(|p| p.blocked)
            .collect();
        assert_eq!(
            answers,
            vec![Some(true), Some(false), Some(false)],
            "through the wall / past it / zero-length"
        );
    }
}
