// src/vulkan/post/ssgi.rs
//
// Vulkan's share of screen-space global illumination, which is its settings and
// where the pass reads and writes this frame. The gather and composite -- their
// pipelines, the reduced gather target and both draws -- are written once in
// `concinnity_core::render::post::ssgi` and reach Vulkan through `VkPostDevice`.

use ash::vk;
use concinnity_core::gfx::ssgi::SsgiSettings;
use concinnity_core::render::post::ssgi::{SsgiInputs, SsgiPass, SsgiPipelines};

use crate::vulkan::context::VkContext;
use crate::vulkan::post::post_device::{PostPipeline, PostTarget, VkPostDevice, post_extent};

// SSGI resources held by `VkContext` when `PostProcessConfig.indirect_lighting`
// is `ssgi`.
pub(in crate::vulkan) struct SsgiResources {
    // Resolved authored tunables; turned into a per-frame `SsgiParams` push.
    pub(in crate::vulkan) settings: SsgiSettings,
    pass: SsgiPass<PostPipeline, PostTarget>,
}

impl SsgiResources {
    // Build both pipelines and the gather target for a render resolution of
    // `extent`.
    pub(in crate::vulkan) fn new(
        device: &VkPostDevice,
        settings: SsgiSettings,
        extent: vk::Extent2D,
    ) -> Result<Self, String> {
        Ok(Self {
            settings,
            pass: SsgiPass::new(device, settings.gi_scale, post_extent(extent))?,
        })
    }

    // Recreate the gather target at a new render extent. The caller has already
    // idled the device and dropped the framebuffers naming the old view.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &VkPostDevice,
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        self.pass.resize(device, post_extent(extent))
    }

    // Swap in freshly built pipelines after a hot reload. The caller has already
    // idled the device.
    pub(in crate::vulkan) fn swap_pipelines(&mut self, pipelines: SsgiPipelines<PostPipeline>) {
        self.pass.swap_pipelines(pipelines);
    }
}

impl VkContext {
    // Encode the SSGI gather + composite: hemisphere rays marched over the
    // G-buffer into the reduced gather target, then blurred and added into this
    // frame's HDR resolve. Runs on the hdr_resolve read-modify-write chain after
    // the main pass.
    pub(in crate::vulkan) fn encode_ssgi(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
    ) {
        let Some(ssgi) = &self.ssgi else { return };
        // With no G-buffer there is nothing to gather against, so skip rather
        // than read a stale view.
        let Some(gbuffer) = &self.gbuffer else { return };
        let params = ssgi.settings.params(fov_y_radians, aspect);
        let device = self.post_device(frame_idx);
        let scene = self.hdr_scene_attachment(frame_idx);
        if let Err(e) = ssgi.pass.encode(
            &device,
            &cmd,
            SsgiInputs {
                scene: scene.view,
                scene_target: scene,
                normal_depth: gbuffer.normal_depth_view(frame_idx),
            },
            &params,
        ) {
            tracing::error!("SSGI: {e}");
        }
    }
}
