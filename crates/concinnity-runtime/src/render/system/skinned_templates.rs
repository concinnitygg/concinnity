// Skinned mesh templates: the shared geometry upload, then one posed entity
// per mesh for AnimationSystem to drive and a runtime spawn to clone.

use std::sync::Arc;

use concinnity_core::ecs::PipelineContext;
use concinnity_core::gfx::mesh_payload::{PayloadMorphs, SkinnedVertex};
use concinnity_core::gfx::render_types::SkinnedDrawObject;
use concinnity_core::render::backend::RenderBackend;
use concinnity_core::render::backend::error::RenderResult;
use concinnity_core::render::world_load::skinned::{self, SkinnedTemplate};

use super::GraphicsSystem;

// The shared skinned buffers and per-slot records the backend uploads once.
pub(super) struct SkinnedUpload {
    pub(super) vertices: Vec<SkinnedVertex>,
    pub(super) indices: Vec<u32>,
    pub(super) draw_objects: Vec<SkinnedDrawObject>,
    pub(super) morphs: Vec<Option<Arc<PayloadMorphs>>>,
}

// Upload the skinned geometry, then the morph targets when any slot has some.
// The hidden instance copies among the draw objects seed the skinned pool.
fn upload_skinned_geometry(
    backend: &mut dyn RenderBackend,
    upload: SkinnedUpload,
) -> RenderResult<()> {
    backend.upload_skinned(&upload.vertices, &upload.indices, upload.draw_objects)?;
    if upload.morphs.iter().any(Option::is_some) {
        backend.upload_skinned_morphs(&upload.morphs)?;
    }
    Ok(())
}

impl GraphicsSystem {
    // Upload the skinned geometry and spawn one template per skinned mesh. The
    // poses are published whatever the backend, so the system graph is identical.
    pub(super) fn install_skinned_templates(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        upload: SkinnedUpload,
        templates: Vec<SkinnedTemplate>,
    ) -> RenderResult<()> {
        if templates.is_empty() {
            return Ok(());
        }
        if let Some(backend) = self.backend.as_deref_mut() {
            upload_skinned_geometry(backend, upload)?;
        }
        let count = templates.len();
        skinned::spawn_skinned_templates(ctx, templates, self.extractor.pick_candidates_mut());
        tracing::info!("GraphicsSystem: {count} skinned mesh(es) ready");
        Ok(())
    }
}
