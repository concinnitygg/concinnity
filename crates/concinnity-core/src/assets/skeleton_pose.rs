// src/assets/skeleton_pose.rs

use alloc::vec::Vec;

use crate::ecs::SkinnedMeshHandle;
use crate::gfx::pose_scratch::PoseScratch;
use crate::gfx::proportions::ProportionLayer;
use crate::gfx::skeleton::Skeleton;
use crate::gfx::transform::Mat4;

/// Runtime-only link between a skinned mesh and its animation state.
///
/// `GraphicsSystem` publishes one `SkeletonPose` per `SkinnedMesh` during
/// init: it carries the resolved bind-pose `Skeleton` and the index of the
/// mesh's skinned draw object in the backend. `AnimationSystem` then ticks the
/// matching `Animation` clip each frame and writes the resulting skinning
/// matrices into `joint_matrices`; `GraphicsSystem` reads them back and
/// uploads them to the GPU. The one-frame producer/consumer hand-off is
/// invisible at animation rates.
///
/// A `CharacterShape` targeting the mesh seeds the static layers: `morph_base`
/// sits under every clip's morph track and `proportions` re-shapes every
/// sampled pose before skinning.
///
/// Not authored in world files: it has no `args`.
#[derive(Debug)]
pub struct SkeletonPose {
    /// The `SkinnedMesh` resource this pose belongs to. Used by
    /// `AnimationSystem` to match an `Animation` clip to its target.
    pub mesh_id: SkinnedMeshHandle,
    /// Index of this mesh's skinned draw object in the render backend.
    pub skinned_index: usize,
    /// Bind-pose joint hierarchy, used to compose skinning matrices.
    pub skeleton: Skeleton,
    /// Current skinning matrices, one per joint. Seeded to the bind pose
    /// (identity skinning) and overwritten by `AnimationSystem` each frame.
    pub joint_matrices: Vec<Mat4>,
    /// Current morph-target weights, one per target of the mesh: the base
    /// layer plus whatever a clip's morph track adds. Empty for a mesh with
    /// neither a base layer nor a morph clip.
    pub morph_weights: Vec<f32>,
    /// Static morph weights from the mesh's `CharacterShape`, one per target;
    /// empty without one. Clip morph tracks are added onto these.
    pub morph_base: Vec<f32>,
    /// Per-joint proportion changes from the mesh's `CharacterShape`, applied
    /// to every sampled pose before the skinning matrices are built.
    pub proportions: ProportionLayer,
    /// True while `joint_matrices` / `morph_weights` hold data the render
    /// backend has not consumed yet. Set by whoever writes the pose, cleared
    /// after upload, so an unanimated pose is uploaded exactly once.
    pub updated: bool,
    /// Reusable sampling buffers owned by this pose, so the per-frame
    /// sample/blend/skinning chain allocates nothing in steady state.
    pub scratch: PoseScratch,
}

impl SkeletonPose {
    /// Build a pose for `mesh_id`'s skinned draw object, seeded to the bind
    /// pose so the mesh renders undeformed until an animation drives it.
    pub fn new(mesh_id: SkinnedMeshHandle, skinned_index: usize, skeleton: Skeleton) -> Self {
        let joint_matrices = skeleton.bind_skinning_matrices();
        Self {
            mesh_id,
            skinned_index,
            skeleton,
            joint_matrices,
            morph_weights: Vec::new(),
            morph_base: Vec::new(),
            proportions: ProportionLayer::default(),
            updated: true,
            scratch: PoseScratch::default(),
        }
    }

    /// Install the static shape layers and re-seed the rest pose through them,
    /// so a mesh with no clip renders shaped.
    pub fn with_shape(mut self, morph_base: Vec<f32>, proportions: ProportionLayer) -> Self {
        self.set_shape(morph_base, proportions);
        self
    }

    /// Replace the static shape layers in place and re-seed the rest pose
    /// through them; an animated pose picks the new layers up on its next
    /// sample. For editing a shape on a live pose without rebuilding it.
    pub fn set_shape(&mut self, morph_base: Vec<f32>, proportions: ProportionLayer) {
        self.morph_weights = morph_base.clone();
        self.morph_base = morph_base;
        self.proportions = proportions;
        self.scratch.locals.clear();
        self.scratch
            .locals
            .extend_from_slice(self.skeleton.bind_locals());
        self.proportions.apply(&mut self.scratch.locals);
        self.skeleton
            .skinning_matrices_into(&self.scratch.locals, &mut self.joint_matrices);
        self.updated = true;
    }

    /// A fresh pose sharing this one's skeleton and shape layers, for a
    /// runtime-spawned copy of the mesh at draw slot `skinned_index`.
    pub fn clone_for_slot(&self, skinned_index: usize) -> Self {
        Self::new(self.mesh_id, skinned_index, self.skeleton.clone())
            .with_shape(self.morph_base.clone(), self.proportions.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::skeleton::{Joint, JointPose};
    use alloc::string::String;
    use alloc::vec;

    fn two_joint_chain() -> Skeleton {
        Skeleton::new(vec![
            Joint {
                name: String::from("root"),
                parent: None,
                bind: JointPose::default(),
            },
            Joint {
                name: String::from("tip"),
                parent: Some(0),
                bind: JointPose {
                    translation: [0.0, 1.0, 0.0],
                    ..Default::default()
                },
            },
        ])
    }

    #[test]
    fn a_shaped_rest_pose_is_seeded_through_the_layers() {
        let skeleton = two_joint_chain();
        let layer = ProportionLayer::resolve(
            &skeleton,
            &[concinnity_asset::JointProportion {
                joint: String::from("root"),
                scale: 2.0,
                length: 0.0,
            }],
        );
        let pose =
            SkeletonPose::new(SkinnedMeshHandle(1), 3, skeleton).with_shape(vec![0.25, 0.5], layer);
        assert_eq!(pose.morph_weights, [0.25, 0.5]);
        // Root doubled: its skinning matrix is a pure scale of 2 (bind is
        // identity there), and the tip's bind position (0, 1, 0) skins to
        // (0, 2, 0).
        assert_eq!(pose.joint_matrices[0][0][0], 2.0);
        let tip = pose.joint_matrices[1];
        assert!((tip[1][1] + tip[3][1] - 2.0).abs() < 1e-5, "{tip:?}");
        assert!(pose.updated);
        let copy = pose.clone_for_slot(7);
        assert_eq!(copy.skinned_index, 7);
        assert_eq!(copy.joint_matrices, pose.joint_matrices);
        assert_eq!(copy.morph_base, pose.morph_base);
    }
}
