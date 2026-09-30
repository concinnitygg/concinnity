//! Vulkan's share of screen-space global illumination, which is its settings and
//! where the pass reads and writes this frame. Every stage -- the pipelines, the
//! depth pyramids, the accumulation and all the draws -- is written once in
//! `concinnity_core::render::post::ssgi` and reaches Vulkan through
//! `VkPostDevice`.

use ash::vk;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::ssgi::settings::SsgiSettings;
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
    // Build every pipeline and target for a render resolution of `extent`.
    pub(in crate::vulkan) fn new(
        device: &VkPostDevice,
        settings: SsgiSettings,
        extent: vk::Extent2D,
    ) -> RenderResult<Self> {
        Ok(Self {
            settings,
            pass: SsgiPass::new(device, settings.gi_scale, post_extent(extent))?,
        })
    }

    // Recreate the targets at a new render extent. The caller has already idled
    // the device and dropped the framebuffers naming the old views.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &VkPostDevice,
        extent: vk::Extent2D,
    ) -> RenderResult<()> {
        self.pass.resize(device, post_extent(extent))
    }

    // Swap in freshly built pipelines after a hot reload. The caller has already
    // idled the device.
    pub(in crate::vulkan) fn swap_pipelines(&mut self, pipelines: SsgiPipelines<PostPipeline>) {
        self.pass.swap_pipelines(pipelines);
    }

    // Step the accumulation ring once the frame is recorded.
    pub(in crate::vulkan) fn advance(&mut self) {
        self.pass.advance();
    }
}

impl VkContext {
    // Encode SSGI: the depth pyramid, the trace over the G-buffer, the
    // accumulation, and the composite into this frame's HDR resolve. Runs on
    // the hdr_resolve read-modify-write chain after the main pass.
    pub(in crate::vulkan) fn encode_ssgi(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
    ) {
        let Some(ssgi) = &self.ssgi else { return };
        // With no G-buffer there is nothing to trace against, so skip rather
        // than read a stale view.
        let Some(gbuffer) = &self.gbuffer else { return };
        let params = ssgi
            .settings
            .params(fov_y_radians, aspect, ssgi.pass.frame());
        let device = self.post_device(frame_idx);
        let scene = self.hdr_scene_attachment(frame_idx);
        if let Err(e) = ssgi.pass.encode(
            &device,
            &cmd,
            SsgiInputs {
                scene: scene.view,
                scene_target: scene,
                normal_depth: gbuffer.normal_depth_view(frame_idx),
                velocity: gbuffer.velocity_view(frame_idx),
            },
            &params,
        ) {
            tracing::error!("SSGI: {e}");
        }
    }
}
