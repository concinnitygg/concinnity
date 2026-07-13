// src/assets/character_rig.rs

use crate::ecs::SkinnedMeshHandle;
use crate::gfx::skinning::Mat4;

/// Runtime-only link between a skinned mesh and its character capsule.
///
/// `GraphicsSystem` publishes one `CharacterRig` per `SkinnedMesh` that
/// declares a `capsule`, carrying the authored model transform and capsule
/// dimensions. `PhysicsSystem` creates the kinematic capsule from it, then
/// each frame consumes the target's `RootMotion` events, resolves the
/// displacement against the scene, and writes the new `position` back here;
/// `GraphicsSystem` moves the rendered mesh to follow. The one-frame
/// producer/consumer hand-offs are invisible at animation rates.
///
/// Not authored in world files: it has no `args`.
#[derive(Debug, Clone)]
pub struct CharacterRig {
    /// The `SkinnedMesh` resource this rig moves.
    pub target: SkinnedMeshHandle,
    /// Index of the mesh's skinned draw object in the render backend.
    pub skinned_index: usize,
    /// The mesh's authored model matrix. Root-motion deltas are mapped
    /// through its rotation/scale, and the moved mesh keeps its orientation.
    pub base_model: Mat4,
    /// Current mesh-origin position in world space. Seeded to the authored
    /// position; overwritten by `PhysicsSystem` as the capsule moves.
    pub position: [f32; 3],
    /// Capsule half-height (cylindrical section) in world units.
    pub half_height: f32,
    /// Capsule radius in world units.
    pub radius: f32,
    /// Runtime facing yaw in radians, applied on top of the authored
    /// rotation (about world Y). Written by a character controller; `0`
    /// keeps the authored facing.
    pub yaw: f32,
    /// World-space drive velocity in units per second, added to the
    /// root-motion displacement each physics step. Written every frame by a
    /// direct-drive character controller; stays zero otherwise.
    pub desired_move: [f32; 3],
    /// One-shot takeoff velocity in units per second. Consumed by
    /// `PhysicsSystem` on the next step: applied if the capsule is grounded,
    /// discarded either way.
    pub jump_velocity: f32,
    /// Whether the capsule rested on ground after the last move.
    pub grounded: bool,
    /// Set by `PhysicsSystem` when `position` changed and by a controller
    /// when `yaw` changed; cleared by `GraphicsSystem` once the render
    /// transform caught up.
    pub moved: bool,
}

impl CharacterRig {
    /// A rig at its authored placement. `base_model` is the mesh's model
    /// matrix; its translation column doubles as the starting position.
    pub fn new(
        target: SkinnedMeshHandle,
        skinned_index: usize,
        base_model: Mat4,
        half_height: f32,
        radius: f32,
    ) -> Self {
        Self {
            target,
            skinned_index,
            base_model,
            position: [base_model[3][0], base_model[3][1], base_model[3][2]],
            half_height,
            radius,
            yaw: 0.0,
            desired_move: [0.0; 3],
            jump_velocity: 0.0,
            grounded: false,
            moved: false,
        }
    }

    /// The current model matrix: the authored rotation/scale, turned by the
    /// runtime facing `yaw`, at the live position.
    pub fn model(&self) -> Mat4 {
        let mut m = self.base_model;
        for col in m.iter_mut().take(3) {
            let turned = self.turn([col[0], col[1], col[2]]);
            col[0] = turned[0];
            col[1] = turned[1];
            col[2] = turned[2];
        }
        m[3][0] = self.position[0];
        m[3][1] = self.position[1];
        m[3][2] = self.position[2];
        m
    }

    /// Map a mesh-local displacement into world space through the authored
    /// rotation/scale (the linear part of `base_model`) and the runtime
    /// facing `yaw`.
    pub fn world_delta(&self, local: [f32; 3]) -> [f32; 3] {
        let m = &self.base_model;
        self.turn([
            m[0][0] * local[0] + m[1][0] * local[1] + m[2][0] * local[2],
            m[0][1] * local[0] + m[1][1] * local[1] + m[2][1] * local[2],
            m[0][2] * local[0] + m[1][2] * local[1] + m[2][2] * local[2],
        ])
    }

    // Rotate a world-space vector by the runtime facing yaw (about world Y).
    fn turn(&self, v: [f32; 3]) -> [f32; 3] {
        let (s, c) = self.yaw.sin_cos();
        [v[0] * c + v[2] * s, v[1], v[2] * c - v[0] * s]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_delta_maps_through_rotation_and_scale() {
        // 90-degree yaw + uniform scale 2: local +Z becomes world +X, doubled.
        let pose = crate::gfx::skinning::JointPose {
            translation: [5.0, 0.0, 1.0],
            rotation_deg: [0.0, 90.0, 0.0],
            scale: [2.0, 2.0, 2.0],
        };
        let rig = CharacterRig::new(SkinnedMeshHandle(1), 0, pose.to_matrix(), 0.5, 0.3);
        assert_eq!(rig.position, [5.0, 0.0, 1.0]);
        let d = rig.world_delta([0.0, 0.0, 1.0]);
        assert!((d[0] - 2.0).abs() < 1e-4, "{d:?}");
        assert!(d[1].abs() < 1e-4 && d[2].abs() < 1e-4, "{d:?}");
    }

    #[test]
    fn yaw_turns_local_forward_to_the_heading() {
        // Facing yaw pi/2: local -Z travel becomes world -X (the camera
        // convention, where yaw 0 looks down -Z).
        let mut rig = CharacterRig::new(
            SkinnedMeshHandle(1),
            0,
            crate::gfx::skinning::IDENTITY,
            0.5,
            0.3,
        );
        rig.yaw = std::f32::consts::FRAC_PI_2;
        let d = rig.world_delta([0.0, 0.0, -1.0]);
        assert!((d[0] + 1.0).abs() < 1e-4, "{d:?}");
        assert!(d[1].abs() < 1e-4 && d[2].abs() < 1e-4, "{d:?}");
        // The model matrix turns the same way: its third column (local +Z)
        // lands on world +X.
        let m = rig.model();
        assert!(
            (m[2][0] - 1.0).abs() < 1e-4 && m[2][2].abs() < 1e-4,
            "{m:?}"
        );
        // Yaw composes on top of the authored rotation: with a 90-degree
        // authored yaw as well, local -Z ends up at world +Z (180 total).
        let pose = crate::gfx::skinning::JointPose {
            translation: [0.0; 3],
            rotation_deg: [0.0, 90.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let mut rig = CharacterRig::new(SkinnedMeshHandle(1), 0, pose.to_matrix(), 0.5, 0.3);
        rig.yaw = std::f32::consts::FRAC_PI_2;
        let d = rig.world_delta([0.0, 0.0, -1.0]);
        assert!((d[2] - 1.0).abs() < 1e-4, "{d:?}");
        assert!(d[0].abs() < 1e-4 && d[1].abs() < 1e-4, "{d:?}");
    }

    #[test]
    fn model_replaces_translation_only() {
        let pose = crate::gfx::skinning::JointPose {
            translation: [1.0, 2.0, 3.0],
            rotation_deg: [0.0, 45.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let mut rig = CharacterRig::new(SkinnedMeshHandle(1), 0, pose.to_matrix(), 0.5, 0.3);
        rig.position = [9.0, 2.0, -4.0];
        let m = rig.model();
        assert_eq!([m[3][0], m[3][1], m[3][2]], [9.0, 2.0, -4.0]);
        // Rotation/scale columns untouched.
        assert_eq!(m[0], rig.base_model[0]);
        assert_eq!(m[1], rig.base_model[1]);
        assert_eq!(m[2], rig.base_model[2]);
    }
}
