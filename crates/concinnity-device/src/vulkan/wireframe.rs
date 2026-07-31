// src/vulkan/wireframe.rs
//
// Wireframe view-mode pipeline variants. Without
// `VK_EXT_extended_dynamic_state3` the polygon mode lives in the pipeline, so
// unlike Metal (where it is one encoder flag) the mode needs a second pipeline
// per main-pass path. They are built on the first wireframe frame and destroyed
// whenever the shaders they were built from are rebuilt, so a shipped runtime
// never pays for them.
//
// The variants render the engine's built-in main shaders: material shader
// buckets own their own pipelines and keep drawing solid, which is the one
// place this diverges from Metal's encoder-state fill mode. A device without
// `fillModeNonSolid` gets no variants and keeps solid fill.

use ash::vk;

use super::context::VkContext;
use super::pipeline::{
    MeshPipelineTargets, compile_bindless_shaders, compile_skinned_shaders,
    create_main_pipeline_wireframe, create_skinned_pipeline_wireframe, resolve_instanced_shader,
    resolve_main_shaders,
};

// Wireframe twins of the engine main-pass pipelines, keyed to the solid
// pipeline each mirrors. A `None` entry means the pipeline it mirrors is not
// live either (or its build failed), in which case the pass keeps the solid one.
#[derive(Default)]
pub(super) struct VkWireframe {
    pub(super) bindless: Option<vk::Pipeline>,
    pub(super) main: Option<vk::Pipeline>,
    pub(super) instanced: Option<vk::Pipeline>,
    pub(super) skinned: Option<vk::Pipeline>,
    // Set once a build has run so a failure is not retried every frame.
    built: bool,
}

impl VkWireframe {
    // Destroy every built pipeline. The caller must have drained the GPU.
    pub(super) fn destroy(&mut self, device: &ash::Device) {
        for p in [self.bindless, self.main, self.instanced, self.skinned]
            .into_iter()
            .flatten()
        {
            unsafe { device.destroy_pipeline(p, None) };
        }
        *self = VkWireframe::default();
    }
}

impl VkContext {
    // Build the wireframe pipeline variants if the view mode needs them and
    // they are not built yet. Called from `draw_frame` before the frame is
    // recorded.
    pub(super) fn ensure_wireframe_pipelines(&mut self) {
        if self.view_mode != concinnity_core::gfx::view_modes::ViewMode::Wireframe
            || self.wireframe.built
        {
            return;
        }
        self.wireframe.built = true;
        // `VK_POLYGON_MODE_LINE` is only legal with `fillModeNonSolid`; without
        // it the mode falls back to solid fill rather than failing the frame.
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

    // Destroy the built variants so the next wireframe frame rebuilds them
    // against the current shaders. Called wherever a main-pass pipeline is
    // rebuilt (shader hot-reload, world shader swap) and at teardown.
    pub(super) fn invalidate_wireframe_pipelines(&mut self) {
        let device = self.device.clone();
        self.wireframe.destroy(&device);
    }

    fn build_wireframe_pipelines(&mut self) -> Result<(), String> {
        let device = self.device.clone();
        let hr = self.hot_reload;
        let msaa = self.msaa_samples;
        let format = self.swapchain_format;
        let render_pass = self.main_render_pass;
        let mut built = VkWireframe {
            built: true,
            ..Default::default()
        };
        // Any failure past this point leaves `built`'s successful pipelines
        // unowned, so destroy what was made before propagating.
        let mut build = || -> Result<(), String> {
            if let (Some(_), Some(layout)) = (
                self.cull.bindless_pipeline,
                self.cull.bindless_pipeline_layout,
            ) {
                let (vs, fs) = compile_bindless_shaders(hr, self.textures.len())?;
                built.bindless = Some(create_main_pipeline_wireframe(
                    &device,
                    MeshPipelineTargets {
                        render_pass,
                        layout,
                        vert_spv: &vs,
                        frag_spv: &fs,
                    },
                    msaa,
                    format,
                )?);
            }

            // The legacy, instanced, and skinned paths are rebuilt from the
            // built-in shaders rather than a world's custom ones: a custom
            // fragment stage is free to ignore the fill mode's intent and the
            // edges only need to be visible.
            let (main_vs, main_fs) = resolve_main_shaders(hr, &[], &[])?;
            built.main = Some(create_main_pipeline_wireframe(
                &device,
                MeshPipelineTargets {
                    render_pass,
                    layout: self.main_pipeline_layout,
                    vert_spv: &main_vs,
                    frag_spv: &main_fs,
                },
                msaa,
                format,
            )?);

            if let (Some(_), Some(layout)) =
                (self.instanced.pipeline, self.instanced.pipeline_layout)
                && let Some(vs) = resolve_instanced_shader(hr, &[], true)?
            {
                built.instanced = Some(create_main_pipeline_wireframe(
                    &device,
                    MeshPipelineTargets {
                        render_pass,
                        layout,
                        vert_spv: &vs,
                        frag_spv: &main_fs,
                    },
                    msaa,
                    format,
                )?);
            }

            if let (Some(_), Some(layout)) = (self.skinned.pipeline, self.skinned.pipeline_layout) {
                let (skinned_vs, _, frag) = compile_skinned_shaders(hr, &[])?;
                built.skinned = Some(create_skinned_pipeline_wireframe(
                    &device,
                    MeshPipelineTargets {
                        render_pass,
                        layout,
                        vert_spv: &skinned_vs,
                        frag_spv: &frag,
                    },
                    msaa,
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
                built.destroy(&device);
                built.built = true;
                self.wireframe = built;
                Err(e)
            }
        }
    }

    // The pipeline the main pass should bind for `solid`'s path this frame: the
    // wireframe twin while that view mode is active and the twin built, else
    // the solid pipeline itself.
    pub(in crate::vulkan) fn wireframe_or(
        &self,
        solid: vk::Pipeline,
        twin: Option<vk::Pipeline>,
    ) -> vk::Pipeline {
        match twin {
            Some(w) if self.view_mode == concinnity_core::gfx::view_modes::ViewMode::Wireframe => w,
            _ => solid,
        }
    }
}
