// src/directx/wireframe.rs
//
// Wireframe view-mode pipeline variants. D3D12 fill mode lives in the PSO, so
// unlike Metal (where it is one encoder flag every inherited indirect draw
// picks up) the mode needs a second pipeline per main-pass path. They are built
// on the first wireframe frame and dropped whenever the shaders they were built
// from are rebuilt, so a shipped runtime never pays for them.
//
// The variants render the engine's built-in main shaders: material shader
// buckets own their own pipelines and keep drawing solid, which is the one
// place this diverges from Metal's encoder-state fill mode.

use windows::Win32::Graphics::Direct3D12::*;

use super::builtins;
use super::context::{DxContext, dump_on_err};
use super::init::pipelines::{compile_main_bindless_shaders, create_main_pso_wireframe};
use super::resources::create_skinned_pso_wireframe;
use super::texture::HDR_FORMAT;

// Wireframe twins of the engine main-pass pipelines, keyed to the solid
// pipeline each mirrors. A `None` entry means the solid pipeline it mirrors is
// not live either (or its build failed), in which case the pass falls back to
// the solid one for that frame.
#[derive(Default)]
pub(super) struct DxWireframe {
    pub(super) bindless: Option<ID3D12PipelineState>,
    pub(super) main: Option<ID3D12PipelineState>,
    pub(super) instanced: Option<ID3D12PipelineState>,
    pub(super) skinned: Option<ID3D12PipelineState>,
    // Set once a build has run so a failure is not retried every frame.
    built: bool,
}

impl DxContext {
    // Build the wireframe pipeline variants if the view mode needs them and
    // they are not built yet. Called from `draw_frame` before the frame is
    // recorded, so the `&self` pass encoders can just read them.
    pub(super) fn ensure_wireframe_pipelines(&mut self) {
        if self.view_mode != concinnity_core::gfx::view_modes::ViewMode::Wireframe
            || self.wireframe.built
        {
            return;
        }
        self.wireframe.built = true;
        if let Err(e) = self.build_wireframe_pipelines() {
            tracing::warn!("wireframe view: {e}; falling back to solid fill");
        }
    }

    // Drop the built variants so the next wireframe frame rebuilds them against
    // the current shaders. Called wherever a main-pass pipeline is rebuilt
    // (shader hot-reload, world shader swap).
    pub(super) fn invalidate_wireframe_pipelines(&mut self) {
        self.wireframe = DxWireframe::default();
    }

    fn build_wireframe_pipelines(&mut self) -> Result<(), String> {
        let device = self.device.clone();
        let iq = self.info_queue.clone();
        let msaa = self.hdr.msaa_samples;
        let mut built = DxWireframe {
            built: true,
            ..Default::default()
        };

        // Bindless static main pass. Its bytecode is already retained on the
        // context (the world-shader bucket rebuild reads it), so this is a
        // straight PSO create.
        if let Some(root_sig) = self.cull.main_bindless_root_sig.as_ref() {
            let (vs, ps) = if self.bindless_main_shaders.vs.is_empty() {
                compile_main_bindless_shaders(self.hot_reload.enabled)?
            } else {
                (
                    self.bindless_main_shaders.vs.clone(),
                    self.bindless_main_shaders.ps.clone(),
                )
            };
            built.bindless = Some(dump_on_err(
                iq.as_ref(),
                create_main_pso_wireframe(&device, root_sig, &vs, &ps, HDR_FORMAT, msaa),
            )?);
        }

        // Legacy per-draw main pass + the instanced and skinned paths. All three
        // are rebuilt from the built-in shaders rather than a world's custom
        // ones: a custom fragment stage is free to ignore the fill mode's intent
        // and the edges only need to be visible.
        let main_ps = builtins::MAIN_FRAG.compile(self.hot_reload.enabled)?;
        let main_vs = builtins::MAIN_VERT.compile(self.hot_reload.enabled)?;
        built.main = Some(dump_on_err(
            iq.as_ref(),
            create_main_pso_wireframe(
                &device,
                &self.main_root_sig,
                &main_vs,
                &main_ps,
                HDR_FORMAT,
                msaa,
            ),
        )?);

        if let Some(root_sig) = self.instanced.root_sig.as_ref() {
            let vs = builtins::MAIN_VERT_INSTANCED.compile(self.hot_reload.enabled)?;
            built.instanced = Some(dump_on_err(
                iq.as_ref(),
                create_main_pso_wireframe(&device, root_sig, &vs, &main_ps, HDR_FORMAT, msaa),
            )?);
        }

        // The skinned main pass reuses the instanced root signature. Both halves
        // go live together in `upload_skinned`, and the draw site binds the root
        // signature from `skinned.root_sig`, so a twin is only reachable when
        // that is `Some` too.
        if self.skinned.pso.is_some()
            && let Some(root_sig) = self.skinned.root_sig.as_ref()
        {
            let vs = builtins::SKINNED_VERT.compile(self.hot_reload.enabled)?;
            built.skinned = Some(dump_on_err(
                iq.as_ref(),
                create_skinned_pso_wireframe(&device, root_sig, &vs, &main_ps, HDR_FORMAT, msaa),
            )?);
        }

        self.wireframe = built;
        Ok(())
    }

    // The pipeline the main pass should bind for `solid`'s path this frame: the
    // wireframe twin while that view mode is active and the twin built, else the
    // solid pipeline itself.
    pub(in crate::directx) fn wireframe_or<'a>(
        &'a self,
        solid: &'a ID3D12PipelineState,
        twin: Option<&'a ID3D12PipelineState>,
    ) -> &'a ID3D12PipelineState {
        match twin {
            Some(w) if self.view_mode == concinnity_core::gfx::view_modes::ViewMode::Wireframe => w,
            _ => solid,
        }
    }
}
