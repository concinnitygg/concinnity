//! The skinned draw path: what a backend is handed for an animated mesh.
//!
//! One upload at load time, then per-frame joint palettes and morph weights,
//! and the reveal / retire pair that hands a pre-reserved instance to a
//! runtime spawn. The static equivalents live on [`RenderBackend`] itself,
//! since a world with no skinning still needs them.
//!
//! [`RenderBackend`]: crate::render::backend::RenderBackend

use super::SceneHost;
use crate::gfx::mesh_payload::SkinnedVertex;
use crate::gfx::render_types::{SkinnedDrawObject, SkinnedIndex};
use crate::render::backend::error::RenderResult;
use crate::render::frame::snapshot::SpanBuffer;
use crate::transform::Mat4;
use alloc::vec::Vec;

/// One frame's skinned updates, borrowed from the snapshot that carried them.
#[derive(Clone, Copy)]
pub struct SkinnedFrame<'a> {
    /// Changed joint matrices, one span per instance.
    pub poses: &'a SpanBuffer<Mat4>,
    /// Changed morph-target weights, one span per instance.
    pub morphs: &'a SpanBuffer<f32>,
    /// Moved instances' model-to-world matrices, in push order.
    pub models: &'a [(SkinnedIndex, Mat4)],
}

/// The animated half of the draw set: skinned upload, pose, morph and
/// instance reveal.
///
/// The upload is required. The per-frame pose, weight and model pushes and the
/// instance reveal and retire only edit the scene's
/// `SkinnedSlots`, so they are
/// defaulted over [`SceneHost`]; without a scene they do nothing, and a skinned
/// spawn finds nothing to claim.
pub trait SkinnedDraws: SceneHost {
    /// Upload the world's skinned geometry and draw objects, once at load. The
    /// skinned pipelines pair the world default Shader's programs where the
    /// world declares one, which the backend kept from init.
    fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        draw_objects: Vec<SkinnedDrawObject>,
    ) -> RenderResult<()>;
    /// Push this frame's skinned updates: each changed instance's joint
    /// matrices and morph weights, then each moved instance's model, applied in
    /// order. Batched so the trait is crossed once per frame. Out-of-range
    /// indices are ignored.
    fn update_skinned(&mut self, frame: SkinnedFrame<'_>) {
        if let Some(scene) = self.scene_mut() {
            for (index, matrices) in frame.poses.iter() {
                scene.skinned.update_pose(index, matrices);
            }
            for (index, weights) in frame.morphs.iter() {
                scene.skinned.update_morph_weights(index, weights);
            }
            scene.skinned.update_models(frame.models);
        }
    }

    /// Attach morph-target data to the skinned draw objects, called once after
    /// `upload_skinned`: `morphs[i]` belongs to draw object `i` (instance
    /// copies share their template's data via the `Arc`). Default: nothing to
    /// attach, for a backend without a morph deformation path.
    fn upload_skinned_morphs(
        &mut self,
        _morphs: &[Option<alloc::sync::Arc<crate::gfx::mesh_payload::PayloadMorphs>>],
    ) -> RenderResult<()> {
        Ok(())
    }

    // Runtime skinned spawn (pre-reserved instance pool): a backend pre-reserves
    // hidden bind-pose copies at load (`SkinnedMesh.max_instances`) and reveals
    // one per skinned SpawnRequest.

    /// Reveal the pre-reserved skinned instance at `instance_index` (a hidden
    /// bind-pose copy expanded at load): show it at `model` and reset its
    /// palette to bind so it does not flash a previous occupant's pose. Which
    /// instance to use is decided by the engine's instance pool; the backend
    /// only applies it. A no-op if the index is out of range.
    fn reveal_skinned_instance(&mut self, instance_index: SkinnedIndex, model: [[f32; 4]; 4]) {
        if let Some(scene) = self.scene_mut() {
            scene.reveal_skinned_instance(instance_index, model);
        }
    }

    /// Hide a live skinned instance. The engine's instance pool returns the
    /// slot for reuse; the backend only hides it. A no-op if the index is out
    /// of range.
    fn retire_skinned_draw_object(&mut self, skinned_index: SkinnedIndex) {
        if let Some(scene) = self.scene_mut() {
            scene.skinned.retire(skinned_index);
        }
    }
}
