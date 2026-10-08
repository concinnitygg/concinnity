//! Vulkan's share of SSAO (GTAO), which is the settings and where the kernel's
//! inputs and the blur's output come from this frame. The depth copy, kernel
//! and blur -- their pipelines, the depth copy and raw occlusion they own and
//! every draw -- are written once in `concinnity_core::render::post::ssao` and
//! reach Vulkan through `VkPostDevice`. The depth + normal they read come from the unified G-buffer
//! pre-pass.
//!
//! The main pass samples the blurred occlusion, the pool's `ao_output`, at set 0
//! binding 6 to modulate its ambient term; when SSAO is disabled the renderer
//! binds the 1×1 `ssao_white` fallback at that slot so the multiplier is a
//! pass-through 1.0.

use ash::vk;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::post::ssao::settings::SsaoSettings;
use concinnity_core::render::post::ssao::{OCCLUSION_FORMAT, SsaoInputs, SsaoPass};

use super::super::context::VkContext;
use crate::vulkan::post::pass_cache::AttachmentRest;
use crate::vulkan::post::post_device::{
    PostPipeline, PostTarget, VkAttachment, VkPostDevice, post_extent,
};

// SSAO resources held by `VkContext` when `PostProcessConfig.ssao` is on.
pub(in crate::vulkan) struct SsaoResources {
    // Resolved authored tunables; turned into a per-frame `SsaoParams` push.
    pub(in crate::vulkan) settings: SsaoSettings,
    pass: SsaoPass<PostPipeline, PostTarget>,
}

impl SsaoResources {
    pub(in crate::vulkan) fn new(
        device: &VkPostDevice,
        settings: SsaoSettings,
        extent: vk::Extent2D,
    ) -> RenderResult<Self> {
        Ok(Self {
            settings,
            pass: SsaoPass::new(device, post_extent(extent))?,
        })
    }

    // Recreate the raw occlusion at a new render resolution. The caller has
    // already idled the device.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &VkPostDevice,
        extent: vk::Extent2D,
    ) -> RenderResult<()> {
        self.pass.resize(device, post_extent(extent))
    }

    // Swap in freshly built pipelines after a hot reload.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        pipelines: concinnity_core::render::post::ssao::SsaoPipelines<PostPipeline>,
    ) {
        self.pass.swap_pipelines(pipelines);
    }
}

impl VkContext {
    // Encode the GTAO kernel and the depth-aware blur over the unified
    // pre-pass's normal+depth into this slot's pooled `ao_output`, ahead of the
    // main pass that samples it via set 0 binding 6. No-op when SSAO is
    // disabled.
    pub(in crate::vulkan) fn encode_ssao(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
    ) -> RenderResult<()> {
        let Some(ssao) = &self.ssao else {
            return Ok(());
        };
        let Some(gbuffer) = self.gbuffer_targets().and_then(|gb| gb.frame(frame_idx)) else {
            return Ok(());
        };
        let view = self
            .targets
            .transient_pool
            .view_for("ao_output", frame_idx)
            .ok_or_else(|| RenderError::Other("ao_output missing from transient pool".into()))?;
        ssao.pass.encode(
            &self.post_device(frame_idx),
            &cmd,
            SsaoInputs {
                normal_depth: gbuffer.normal_depth,
                output: VkAttachment {
                    view,
                    extent: self.targets.render_extent,
                    format: OCCLUSION_FORMAT,
                    // The executor moves `ao_output` into the attachment layout
                    // ahead of this pass and out before the main pass samples it.
                    rest: AttachmentRest::Graph,
                },
            },
            &ssao.settings.params(fov_y_radians, aspect),
        )
    }
}
