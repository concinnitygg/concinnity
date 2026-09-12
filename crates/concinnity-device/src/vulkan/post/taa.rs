// src/vulkan/post/taa.rs
//
// Vulkan's share of temporal anti-aliasing, which is the jitter counter and
// where the resolve's two inputs come from this frame. The resolve itself --
// its pipeline, its accumulation ring, the history-validity gate and the draw --
// is written once in `concinnity_core::render::post::taa` and reaches Vulkan
// through `VkPostDevice`.
//
// The accumulation ring is one image per frame in flight rather than a bare
// ping-pong, because the bloom prefilter and the composite bind their scene
// input per frame slot: a frame's target has to be the one its consumers were
// wired to. Frame slot `f` writes image `f` and samples `f - 1`, which is the
// previous frame's output either way.

use ash::vk;
use concinnity_core::render::post::taa::{TaaInputs, TaaPass, TaaRing};

use crate::vulkan::context::VkContext;
use crate::vulkan::post::post_device::{PostPipeline, PostTarget, VkPostDevice, post_extent};

// The shared temporal resolve, holding Vulkan's own pipeline and target types.
pub(in crate::vulkan) type VkTaaPass = TaaPass<PostPipeline, PostTarget>;

// Temporal-anti-aliasing state. Built only when the world's `PostProcessConfig`
// asked for it, or when temporal upscaling forced the velocity pre-pass on.
pub(in crate::vulkan) struct TaaResources {
    // The shared resolve.
    pub(in crate::vulkan) pass: VkTaaPass,
    // Drives the Halton jitter sequence. Here rather than in the shared pass
    // because it also drives the temporal upscalers, which run in the resolve's
    // place.
    pub(in crate::vulkan) taa_frame: u32,
}

impl TaaResources {
    // Build the resolve at `extent`, with one accumulation image per frame in
    // flight.
    pub(in crate::vulkan) fn new(
        device: &VkPostDevice,
        frames: usize,
        extent: vk::Extent2D,
    ) -> Result<Self, String> {
        Ok(Self {
            pass: TaaPass::new(device, TaaRing::per_frame(frames), post_extent(extent))?,
            taa_frame: 0,
        })
    }

    // The accumulation image frame slot `f` writes: what the bloom prefilter and
    // the composite sample instead of the raw HDR resolve.
    pub(in crate::vulkan) fn output_view(&self, frame: usize) -> vk::ImageView {
        self.pass.target(frame).view()
    }

    // Rebuild the accumulation images at a new extent. The caller
    // (`rebuild_swapchain`) has already idled the device and cleared the
    // framebuffer cache, which keys on the views this drops.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &VkPostDevice,
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        self.pass.resize(device, post_extent(extent))?;
        // Stale history cannot be reprojected onto the new resolution.
        self.taa_frame = 0;
        Ok(())
    }

    // Swap in a freshly built pipeline. Driven by shader hot reload; the caller
    // has already idled the device.
    pub(in crate::vulkan) fn swap_pipelines(&mut self, pipeline: PostPipeline) {
        self.pass.swap_pipeline(pipeline);
    }
}

impl VkContext {
    // Encode the TAA resolve pass: one fullscreen-triangle draw blending this
    // frame's scene with the reprojected, neighborhood-clipped history into
    // frame slot `frame_idx`'s accumulation image. Runs before bloom, and only
    // when TAA is on.
    pub(in crate::vulkan) fn encode_taa(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        let Some(taa) = &self.taa else { return };
        let Some(velocity) = self.velocity_view_for_post(frame_idx) else {
            tracing::error!("TAA enabled but the G-buffer velocity view is missing");
            return;
        };
        let device = self.post_device(frame_idx);
        if let Err(e) = taa.pass.encode(
            &device,
            &cmd,
            frame_idx,
            TaaInputs {
                scene: self.scene_view_for_post(frame_idx),
                velocity,
            },
        ) {
            tracing::error!("TAA resolve: {e}");
        }
    }

    // The scene image the post stack treats as pre-TAA: the reflection
    // composite's output when a reflection path owns the scene, else this frame
    // slot's raw HDR resolve.
    pub(in crate::vulkan) fn scene_view_for_post(&self, frame: usize) -> vk::ImageView {
        match self.reflection_composite.as_ref() {
            Some(rc) => rc.output.view,
            None => self.hdr_resolve_images[frame % self.hdr_resolve_images.len()].view,
        }
    }

    // This frame slot's per-pixel motion vectors, from the unified G-buffer
    // pre-pass.
    fn velocity_view_for_post(&self, frame: usize) -> Option<vk::ImageView> {
        let views = self.gbuffer.as_ref()?.velocity_views();
        if views.is_empty() {
            return None;
        }
        Some(views[frame % views.len()])
    }
}
