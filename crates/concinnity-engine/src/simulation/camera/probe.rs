use crate::ecs::{Entity, SkinnedMeshHandle};

/// What a follow camera follows: a character rig or a prop's body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Followed {
    /// The rig of a `SkinnedMesh`.
    Rig(SkinnedMeshHandle),
    /// A prop's entity.
    Prop(Entity),
}

/// Runtime-only occlusion probe for the third-person follow camera.
///
/// The third-person controller publishes one and refreshes `pivot` /
/// `desired` (the orbit pivot and the unclamped camera position) every
/// frame; `PhysicsSystem` (which steps earlier) raycasts from the pivot
/// toward the previous frame's desired position and writes back the largest
/// unobstructed `clearance`, excluding the followed rig's own capsule or the
/// followed prop's own collider. The controller then pulls the camera inside
/// that distance so walls do not cut between the camera and what it follows.
///
/// Not authored in world files: it has no `args`.
#[derive(Debug, Clone)]
pub struct CameraProbe {
    /// What the camera follows, whose own body the ray ignores.
    pub followed: Followed,
    /// Orbit pivot in world space (the ray origin).
    pub pivot: [f32; 3],
    /// Unclamped camera position the controller wants this frame.
    pub desired: [f32; 3],
    /// Largest unobstructed distance from `pivot` toward `desired`, or
    /// `None` when the path is clear (or unanswered).
    pub clearance: Option<f32>,
}
