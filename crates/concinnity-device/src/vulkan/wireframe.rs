// src/vulkan/wireframe.rs
//
// Wireframe view-mode pipeline variants. Without
// `VK_EXT_extended_dynamic_state3` the polygon mode lives in the pipeline, so
// unlike Metal (where it is one encoder flag) the mode needs a second pipeline
// per main-pass path. They are built on the first wireframe frame and destroyed
// whenever the shaders they were built from are rebuilt, so a shipped runtime
// never pays for them.
//
// The twin renders the engine's own bindless pair rather than a world Shader's:
// a world fragment is free to ignore the fill mode's intent and the edges only
// need to be visible. Every shader bucket shares the bindless pipeline layout,
// so the one twin stands in for every bucket's pipeline while the mode is on,
// as Metal's encoder-state fill mode does. A device without `fillModeNonSolid`
// gets no twin and keeps solid fill.

use crate::vulkan::owned::OwnedPipeline;

use super::context::VkContext;
use super::pipeline::{MeshPipelineTargets, create_main_pipeline_wireframe};

// The Wireframe twin of the GPU-driven main pipeline. `None` means the pass
// it mirrors is not live either (or the build failed), in which case the pass
// keeps the solid pipeline.
#[derive(Default)]
pub(super) struct VkWireframe {
    pub(super) bindless: Option<OwnedPipeline>,
    // Set once a build has run so a failure is not retried every frame.
    built: bool,
}

impl VkWireframe {
    // Retire every built pipeline, leaving the set unbuilt.
    pub(super) fn destroy(&mut self) {
        *self = VkWireframe::default();
    }
}

impl VkContext {
    // Build the wireframe twin if the view mode needs it and it is not built
    // yet. Called from `draw_frame` before the frame is
    // recorded.
    pub(super) fn ensure_wireframe_pipelines(&mut self) {
        if self.view.mode != concinnity_core::gfx::view_modes::ViewMode::Wireframe
            || self.wireframe.built
        {
            return;
        }
        self.wireframe.built = true;
        // `VK_POLYGON_MODE_LINE` is only legal with `fillModeNonSolid`; without
        // it the mode falls back to solid fill rather than failing the frame.
        // SAFETY: a property query on a live handle; it only reads.
        let supported = unsafe {
            self.instance
                .get_physical_device_features(self.physical_device)
        };
        if supported.fill_mode_non_solid == 0 {
            tracing::warn!("wireframe view: device lacks fillModeNonSolid; using solid fill");
            return;
        }
        if let Err(e) = self.build_wireframe_pipelines() {
            tracing::warn!("wireframe view: {e}; falling back to solid fill");
        }
    }

    // Destroy the built twin so the next wireframe frame rebuilds it against
    // the current shaders. Called wherever the main-pass pipeline is rebuilt
    // (shader hot-reload, world shader swap) and at teardown.
    pub(super) fn invalidate_wireframe_pipelines(&mut self) {
        self.wireframe.destroy();
    }

    fn build_wireframe_pipelines(&mut self) -> Result<(), String> {
        let device = self.device.clone();
        let msaa = self.msaa_samples;
        let format = self.swapchain.format;
        let render_pass = self.main_render_pass.handle();
        let mut built = VkWireframe {
            built: true,
            ..Default::default()
        };
        let mut build = || -> Result<(), String> {
            if let (Some(_), Some(layout)) = (
                self.cull.bindless_pipeline.as_ref(),
                self.cull.bindless_pipeline_layout.as_ref(),
            ) {
                // The engine's pair, retained on the context past init.
                let (vs, fs) = &self.cull.bindless_main_spv;
                built.bindless = Some(create_main_pipeline_wireframe(
                    &device,
                    MeshPipelineTargets {
                        render_pass,
                        layout: layout.handle(),
                        vert_spv: vs,
                        frag_spv: fs,
                    },
                    msaa,
                    format,
                )?);
            }

            Ok(())
        };
        match build() {
            Ok(()) => {
                self.wireframe = built;
                Ok(())
            }
            Err(e) => {
                // Keep `built` set so the failure is not retried every frame.
                built.destroy();
                built.built = true;
                self.wireframe = built;
                Err(e)
            }
        }
    }

    // The pipeline the main pass should bind in place of `solid` this frame:
    // the wireframe twin while that view mode is active and the twin built,
    // else the solid pipeline itself.
    pub(in crate::vulkan) fn wireframe_or<'a>(
        &self,
        solid: &'a OwnedPipeline,
        twin: Option<&'a OwnedPipeline>,
    ) -> &'a OwnedPipeline {
        match twin {
            Some(w) if self.view.mode == concinnity_core::gfx::view_modes::ViewMode::Wireframe => w,
            _ => solid,
        }
    }
}
