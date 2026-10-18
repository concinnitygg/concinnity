//! Vulkan's share of screen-space global illumination, which is its settings and
//! where the pass reads and writes this frame. Every stage -- the pipelines, the
//! depth pyramids, the accumulation and all the draws -- is written once in
//! `concinnity_core::render::post::ssgi` and reaches Vulkan through
//! `VkPostDevice`.

use ash::vk;
use concinnity_core::gfx::render_types::SsgiParams;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::ssgi::settings::{SsgiFrame, SsgiSettings};
use concinnity_core::render::post::ssgi::{
    SsgiPass, SsgiPipelines, SsgiResolveInputs, SsgiTraceInputs,
};

use crate::vulkan::context::VkContext;
use crate::vulkan::post::post_device::{PostPipeline, PostTarget, VkPostDevice, post_extent};

// SSGI resources held by `VkContext` when `IndirectLighting.technique`
// is `ssgi`.
pub(in crate::vulkan) struct SsgiResources {
    // Resolved authored tunables; turned into a per-frame `SsgiParams` push.
    pub(in crate::vulkan) settings: SsgiSettings,
    pass: SsgiPass<PostPipeline, PostTarget>,
}

impl SsgiResources {
    // Where the accumulation stands this frame.
    pub(in crate::vulkan) fn frame(&self) -> SsgiFrame {
        self.pass.frame()
    }

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

    // Forget the accumulation for a frame that does not continue the last.
    pub(in crate::vulkan) fn reset_history(&mut self) {
        self.pass.reset_history();
    }
}

impl SsgiResources {
    // The resolve output's sampled view, which the main pass's global set binds.
    pub(in crate::vulkan) fn indirect_view(&self) -> vk::ImageView {
        self.pass.indirect().view()
    }
}

impl VkContext {
    // Whether the main pass's screen-indirect binding carries SSGI's bounce:
    // whenever SSGI is built, which is when the graph runs its resolve.
    pub(in crate::vulkan) fn screen_bounce(&self) -> bool {
        self.ssgi.is_some()
    }

    // Encode the resolve ahead of the main pass: last frame's accumulation
    // reprojected onto this frame's G-buffer, folded with the occlusion.
    pub(in crate::vulkan) fn encode_ssgi_resolve(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        params: Option<&SsgiParams>,
    ) {
        let (Some(ssgi), Some(params)) = (&self.ssgi, params) else {
            return;
        };
        let Some(gbuffer) = self.gbuffer_targets().and_then(|gb| gb.frame(frame_idx)) else {
            return;
        };
        let occlusion = self
            .targets
            .transient_pool
            .view_for("ao_output", frame_idx)
            .unwrap_or(self.scene.ssao_white.view);
        if let Err(e) = ssgi.pass.encode_resolve(
            &self.post_device(frame_idx),
            &cmd,
            SsgiResolveInputs {
                normal_depth: gbuffer.normal_depth,
                velocity: gbuffer.velocity,
                occlusion,
            },
            params,
        ) {
            tracing::error!("SSGI resolve: {e}");
            return;
        }
        // The output rests shader-readable like every pass-owned post target,
        // but the main render pass's dependency orders no color write before
        // its fragment reads, so make the resolve's writes visible to them here.
        let visible = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            self.hw.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&visible),
                &[],
                &[],
            );
        }
    }

    // Encode SSGI after the main pass: the depth pyramid, the age, the trace
    // over the G-buffer and the accumulation the next frame's resolve reads.
    pub(in crate::vulkan) fn encode_ssgi(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        params: Option<&SsgiParams>,
    ) {
        let (Some(ssgi), Some(params)) = (&self.ssgi, params) else {
            return;
        };
        // With no G-buffer there is nothing to trace against, so skip rather
        // than read a stale view.
        let Some(gbuffer) = self.gbuffer_targets().and_then(|gb| gb.frame(frame_idx)) else {
            return;
        };
        if let Err(e) = ssgi.pass.encode(
            &self.post_device(frame_idx),
            &cmd,
            SsgiTraceInputs {
                scene: self.hdr_scene_attachment(frame_idx).view,
                normal_depth: gbuffer.normal_depth,
                velocity: gbuffer.velocity,
            },
            params,
        ) {
            tracing::error!("SSGI: {e}");
        }
    }
}
