// src/assets/root_motion_event.rs

use crate::ecs::SkinnedMeshHandle;

/// Per-frame character displacement extracted from root-motion clips.
///
/// `AnimationSystem` publishes one event per animated target whose active
/// clips carry a baked root track, holding the frame's blended displacement
/// in the mesh's local space (unrotated, unscaled). Consumers -- the
/// `CharacterRig` capsule drive in `PhysicsSystem` -- map it through the
/// mesh's model transform and move the character. Events, not components:
/// a missed frame must not accumulate.
#[derive(Debug, Clone, Copy)]
pub struct RootMotionEvent {
    /// The `SkinnedMesh` resource whose animation produced the displacement.
    pub target: SkinnedMeshHandle,
    /// Displacement in the mesh's local space for this frame.
    pub delta: [f32; 3],
}
