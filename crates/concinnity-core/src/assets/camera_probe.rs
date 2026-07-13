// src/assets/camera_probe.rs

use crate::ecs::SkinnedMeshHandle;

/// Runtime-only occlusion probe for the third-person follow camera.
///
/// The third-person controller publishes one and refreshes `pivot` /
/// `desired` (the orbit pivot and the unclamped camera position) every
/// frame; `PhysicsSystem` (which steps earlier) raycasts from the pivot
/// toward the previous frame's desired position and writes back the largest
/// unobstructed `clearance`, excluding the followed character's own capsule.
/// The controller then pulls the camera inside that distance so walls do not
/// cut between camera and character.
///
/// Not authored in world files: it has no `args`.
#[derive(Debug, Clone, Default)]
pub struct CameraProbe {
    /// The followed `SkinnedMesh`, whose rig capsule the ray must ignore.
    pub target: SkinnedMeshHandle,
    /// Orbit pivot in world space (the ray origin).
    pub pivot: [f32; 3],
    /// Unclamped camera position the controller wants this frame.
    pub desired: [f32; 3],
    /// Largest unobstructed distance from `pivot` toward `desired`, or
    /// `None` when the path is clear (or unanswered).
    pub clearance: Option<f32>,
}
