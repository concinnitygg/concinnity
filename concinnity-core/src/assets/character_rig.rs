// src/assets/character_rig.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};
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
    /// The `SkinnedMesh` asset this rig moves.
    pub target: AssetId,
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
    /// Whether the capsule rested on ground after the last move.
    pub grounded: bool,
    /// Set by `PhysicsSystem` when `position` changed; cleared by
    /// `GraphicsSystem` once the render transform caught up.
    pub moved: bool,
}

impl CharacterRig {
    /// A rig at its authored placement. `base_model` is the mesh's model
    /// matrix; its translation column doubles as the starting position.
    pub fn new(
        target: AssetId,
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
            grounded: false,
            moved: false,
        }
    }

    /// The current model matrix: the authored rotation/scale at the live
    /// position.
    pub fn model(&self) -> Mat4 {
        let mut m = self.base_model;
        m[3][0] = self.position[0];
        m[3][1] = self.position[1];
        m[3][2] = self.position[2];
        m
    }

    /// Map a mesh-local displacement into world space through the authored
    /// rotation/scale (the linear part of `base_model`).
    pub fn world_delta(&self, local: [f32; 3]) -> [f32; 3] {
        let m = &self.base_model;
        [
            m[0][0] * local[0] + m[1][0] * local[1] + m[2][0] * local[2],
            m[0][1] * local[0] + m[1][1] * local[1] + m[2][1] * local[2],
            m[0][2] * local[0] + m[1][2] * local[1] + m[2][2] * local[2],
        ]
    }
}

/// `CharacterRig` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct CharacterRigArgs {}

impl Component for CharacterRig {
    const NAME: &'static str = "CharacterRig";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = CharacterRigArgs;

    fn to_args(&self) -> CharacterRigArgs {
        CharacterRigArgs {}
    }
    fn from_args(_: CharacterRigArgs) -> Self {
        // A RuntimeOnly component never round-trips through args; real
        // instances are built by GraphicsSystem via `new`.
        Self::new(
            AssetId::default(),
            0,
            crate::gfx::skinning::IDENTITY,
            0.5,
            0.3,
        )
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
        let rig = CharacterRig::new(AssetId(1), 0, pose.to_matrix(), 0.5, 0.3);
        assert_eq!(rig.position, [5.0, 0.0, 1.0]);
        let d = rig.world_delta([0.0, 0.0, 1.0]);
        assert!((d[0] - 2.0).abs() < 1e-4, "{d:?}");
        assert!(d[1].abs() < 1e-4 && d[2].abs() < 1e-4, "{d:?}");
    }

    #[test]
    fn model_replaces_translation_only() {
        let pose = crate::gfx::skinning::JointPose {
            translation: [1.0, 2.0, 3.0],
            rotation_deg: [0.0, 45.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let mut rig = CharacterRig::new(AssetId(1), 0, pose.to_matrix(), 0.5, 0.3);
        rig.position = [9.0, 2.0, -4.0];
        let m = rig.model();
        assert_eq!([m[3][0], m[3][1], m[3][2]], [9.0, 2.0, -4.0]);
        // Rotation/scale columns untouched.
        assert_eq!(m[0], rig.base_model[0]);
        assert_eq!(m[1], rig.base_model[1]);
        assert_eq!(m[2], rig.base_model[2]);
    }
}
