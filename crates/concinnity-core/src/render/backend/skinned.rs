//! The skinned draw path: what a backend is handed for an animated mesh.
//!
//! One upload at load time, then per-frame joint palettes and morph weights,
//! and the reveal / retire pair that hands a pre-reserved instance to a
//! runtime spawn. The static equivalents live on [`RenderBackend`] itself,
//! since a world with no skinning still needs them.
//!
//! [`RenderBackend`]: crate::render::backend::RenderBackend

use crate::gfx::mesh_payload::SkinnedVertex;
use crate::gfx::render_types::SkinnedDrawObject;
use crate::render::error::RenderResult;
use alloc::vec::Vec;

/// The animated half of the draw set: skinned upload, pose, morph and
/// instance reveal.
///
/// The upload and the per-frame palette push are required; a backend without a
/// morph deformation path or a pre-reserved instance pool keeps the defaults,
/// and a skinned spawn there finds nothing to claim.
pub trait SkinnedDraws {
    /// Upload the world's skinned geometry and draw objects, once at load. The
    /// skinned pipelines pair the world default Shader's programs where the
    /// world declares one, which the backend kept from init.
    fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        draw_objects: Vec<SkinnedDrawObject>,
    ) -> RenderResult<()>;
    /// Push one skinned slot's joint matrices for this frame.
    fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]);

    /// Attach morph-target data to the skinned draw objects, called once after
    /// `upload_skinned`: `morphs[i]` belongs to draw object `i` (instance
    /// copies share their template's data via the `Arc`). Default no-op for a
    /// backend without a morph deformation path.
    fn upload_skinned_morphs(
        &mut self,
        _morphs: Vec<Option<alloc::sync::Arc<crate::gfx::mesh_payload::PayloadMorphs>>>,
    ) {
    }

    /// Push a skinned object's current morph-target weights, sampled by the
    /// animation system each frame. A no-op when the index is out of range or
    /// the object carries no morph targets.
    fn update_morph_weights(&mut self, _skinned_index: usize, _weights: &[f32]) {}

    // Runtime skinned spawn (pre-reserved instance pool): a backend pre-reserves
    // hidden bind-pose copies at load (`SkinnedMesh.max_instances`) and reveals
    // one per skinned SpawnRequest. The default no-op implementations are a
    // fallback for a backend that has not wired runtime skinned spawn, where a
    // skinned SpawnRequest finds nothing to claim and is dropped.

    /// Reveal the pre-reserved skinned instance at `instance_index` (a hidden
    /// bind-pose copy expanded at load): show it at `model` and reset its
    /// palette to bind so it does not flash a previous occupant's pose. Which
    /// instance to use is decided by the engine's instance pool; the backend
    /// only applies it. A no-op if the index is out of range.
    fn reveal_skinned_instance(&mut self, _instance_index: usize, _model: [[f32; 4]; 4]) {}

    /// Hide a live skinned instance. The engine's instance pool returns the
    /// slot for reuse; the backend only hides it. A no-op if the index is out
    /// of range.
    fn retire_skinned_draw_object(&mut self, _skinned_index: usize) {}

    /// Push this frame's changed skinned model-to-world matrices, one
    /// `(skinned index, matrix)` entry per moved instance, applied in order
    /// (a skinned object animates in place unless something moves it). Cheap:
    /// the per-frame cull rebuild reads the object's model directly, so this
    /// just writes the fields. Out-of-range indices are ignored; default
    /// no-op for a backend without movable skinned instances.
    fn update_skinned_models(&mut self, _updates: &[(u32, [[f32; 4]; 4])]) {}
}
