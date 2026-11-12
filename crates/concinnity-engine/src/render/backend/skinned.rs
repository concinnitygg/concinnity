//! The skinned draw path: what a backend is handed for an animated mesh.
//!
//! One upload at load time, then per-frame joint palettes and morph weights,
//! the reveal / retire pair that hands a pre-reserved instance to a runtime
//! spawn, and the in-place geometry and skeleton writes an asset hot-reload
//! makes. The static draws' per-frame pushes live on [`RenderBackend`], and
//! their hot-reload rewrites, with the skinned buffer repack, on
//! [`GeometryRebuild`].
//!
//! [`RenderBackend`]: crate::render::backend::RenderBackend
//! [`GeometryRebuild`]: crate::render::backend::GeometryRebuild

use super::SceneHost;
use crate::gfx::mesh_payload::SkinnedVertex;
use crate::gfx::render_types::{SkinnedDrawObject, SkinnedIndex};
use crate::render::backend::error::{RenderError, RenderResult};
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
    /// The move this frame carries the world through, if any, which every
    /// moved instance's change includes.
    pub rebase: Option<&'a crate::environment::planet::Rebase>,
}

/// The animated half of the draw set: skinned upload, pose, morph and
/// instance reveal.
///
/// The upload is required. The per-frame pose, weight and model pushes, the
/// instance reveal and retire, and a skeleton resize only edit the scene's
/// `SkinnedSlots`, so they are defaulted over [`SceneHost`]; without a scene
/// they do nothing, and a skinned spawn finds nothing to claim. The in-place
/// geometry write a hot reload makes defaults to
/// [`RenderError::Unsupported`].
pub trait SkinnedDraws: SceneHost {
    /// Upload the world's skinned geometry and draw objects, once at load. The
    /// skinned pipelines pair the world default Shader's programs where the
    /// world declares one, which the backend kept from init.
    ///
    /// # Errors
    ///
    /// Returns an error if the skinned buffers or pipelines cannot be created.
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
            scene.update_skinned(&frame);
        }
    }

    /// Attach morph-target data to the skinned draw objects, called once after
    /// `upload_skinned`: `morphs[i]` belongs to draw object `i` (instance
    /// copies share their template's data via the `Arc`). Default: nothing to
    /// attach, for a backend without a morph deformation path.
    ///
    /// # Errors
    ///
    /// Returns an error if the morph buffers cannot be created.
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
            scene.retire_skinned_instance(skinned_index);
        }
    }

    /// Replace a `SkinnedMesh` draw slot's vertex + index data in place.
    /// Driven by asset hot-reload (`cn run -f` only). Reuses the slot's
    /// existing vertex region + index region in the shared skinned vertex /
    /// index buffers (created once by [`Self::upload_skinned`]), so the new
    /// geometry must match the slot's init-time vertex count + index count
    /// and the new skeleton must keep the same joint count; pipelines stay
    /// untouched, only the bytes change. `vertex_base` is the init-time
    /// vertex offset (in vertex units) into the shared buffer; indices are
    /// rebased onto it before writing.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without in-place
    /// skinned writes, or an error if the slot is unknown or the new geometry
    /// does not match its layout.
    fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: SkinnedIndex,
        vertex_base: u32,
        verts: &[SkinnedVertex],
        idxs: &[u16],
    ) -> RenderResult<()> {
        let _ = (skinned_index, vertex_base, verts, idxs);
        Err(RenderError::Unsupported {
            op: "update_skinned_mesh_geometry",
        })
    }

    /// Update a skinned slot's joint count and resize the backend's per-slot
    /// joint-matrix buffers to match. Driven by asset hot-reload (`cn run -f`
    /// only) when a re-imported `.glb`'s skeleton has a different joint
    /// count than the slot was initialized with. Shrinking truncates the
    /// per-slot Vec; growing seeds the new entries to identity so the slot
    /// renders undeformed on the next `update_skinned_pose`. The skinned
    /// shaders consume the joints buffer through a pointer (not a fixed-
    /// size array) and use vertex-attribute-encoded joint indices, so no
    /// pipeline or shader rebuild is required for a joint-count change;
    /// only the CPU-side per-slot buffer and `SkinnedDrawObject.joint_count`
    /// change.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene, or an error if
    /// `skinned_index` names no slot.
    fn update_skinned_skeleton(
        &mut self,
        skinned_index: SkinnedIndex,
        new_joint_count: usize,
    ) -> RenderResult<()> {
        match self.scene_mut() {
            Some(scene) => scene.update_skinned_skeleton(skinned_index, new_joint_count),
            None => Err(RenderError::Unsupported {
                op: "update_skinned_skeleton",
            }),
        }
    }
}
