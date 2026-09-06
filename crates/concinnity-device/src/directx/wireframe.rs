// src/directx/wireframe.rs
//
// Wireframe view-mode pipeline variants. D3D12 fill mode lives in the PSO, so
// unlike Metal (where it is one encoder flag every inherited indirect draw
// picks up) the mode needs a second pipeline per main-pass path. They are built
// on the first wireframe frame and dropped whenever the shaders they were built
// from are rebuilt, so a shipped runtime never pays for them.
//
// The twin renders the engine's own bindless pair rather than a world Shader's:
// a world fragment is free to ignore the fill mode's intent and the edges only
// need to be visible. Every shader bucket shares the bindless root signature,
// so the one twin stands in for every bucket's PSO while the mode is on, as
// Metal's encoder-state fill mode does.

use windows::Win32::Graphics::Direct3D12::*;

use super::context::{DxContext, dump_on_err};
use super::init::pipelines::create_main_pso_wireframe;
use super::texture::HDR_FORMAT;

// The Wireframe twin of the GPU-driven main PSO. `None` means the pass it
// mirrors is not live either (or the build failed), in which case the pass
// keeps the solid PSO.
#[derive(Default)]
pub(super) struct DxWireframe {
    pub(super) bindless: Option<ID3D12PipelineState>,
    // Set once a build has run so a failure is not retried every frame.
    built: bool,
}

impl DxContext {
    // Build the wireframe twin if the view mode needs it and it is not built
    // yet. Called from `draw_frame` before the frame is
    // recorded, so the `&self` pass encoders can just read them.
    pub(super) fn ensure_wireframe_pipelines(&mut self) {
        if self.view.mode != concinnity_core::gfx::view_modes::ViewMode::Wireframe
            || self.wireframe.built
        {
            return;
        }
        self.wireframe.built = true;
        if let Err(e) = self.build_wireframe_pipelines() {
            tracing::warn!("wireframe view: {e}; falling back to solid fill");
        }
    }

    // Drop the built twin so the next wireframe frame rebuilds it against the
    // current shaders. Called wherever a main-pass pipeline is rebuilt
    // (shader hot-reload, world shader swap).
    pub(super) fn invalidate_wireframe_pipelines(&mut self) {
        self.wireframe = DxWireframe::default();
    }

    fn build_wireframe_pipelines(&mut self) -> Result<(), String> {
        let device = self.device.clone();
        let iq = self.diagnostics.info_queue.clone();
        let msaa = self.hdr.msaa_samples;
        let mut built = DxWireframe {
            built: true,
            ..Default::default()
        };

        // The engine's pair is retained on the context past init (the
        // world-shader bucket rebuild reads it), so this is a straight PSO create.
        if let Some(root_sig) = self.cull.main_bindless_root_sig.as_ref() {
            let vs = &self.bindless_main_shaders.vs;
            let ps = &self.bindless_main_shaders.ps;
            built.bindless = Some(dump_on_err(
                iq.as_ref(),
                create_main_pso_wireframe(&device, root_sig, vs, ps, HDR_FORMAT, msaa),
            )?);
        }

        self.wireframe = built;
        Ok(())
    }

    // The pipeline the main pass should bind in place of `solid` this frame: the
    // wireframe twin while that view mode is active and the twin built, else
    // the solid pipeline itself.
    pub(in crate::directx) fn wireframe_or<'a>(
        &'a self,
        solid: &'a ID3D12PipelineState,
        twin: Option<&'a ID3D12PipelineState>,
    ) -> &'a ID3D12PipelineState {
        match twin {
            Some(w) if self.view.mode == concinnity_core::gfx::view_modes::ViewMode::Wireframe => w,
            _ => solid,
        }
    }
}
