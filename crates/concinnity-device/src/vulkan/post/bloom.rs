//! Vulkan's share of bloom, which is where the chain's scene and top octave
//! come from this frame. The chain itself -- its pipelines, the octaves below
//! the top, and every draw -- is written once in
//! `concinnity_core::render::post::bloom` and reaches Vulkan through
//! `VkPostDevice`.

use ash::vk;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::post::bloom::{BloomInputs, BloomPass, top_extent};
use concinnity_core::render::render_graph::PixelFormat;

use crate::vulkan::context::VkContext;
use crate::vulkan::post::pass_cache::AttachmentRest;
use crate::vulkan::post::post_device::{PostPipeline, PostTarget, VkAttachment, post_extent};
use crate::vulkan::transient_pool::TransientImagePool;

// The shared chain, holding Vulkan's own pipeline and target types.
pub(in crate::vulkan) type VkBloomPass = BloomPass<PostPipeline, PostTarget>;

// What the composite samples as bloom for frame slot `frame`: the pool's
// `bloom_top`. The pool always manages it; `fallback` only keeps the binding
// valid should it not, and the composite skips the sample while bloom is off.
pub(in crate::vulkan) fn composite_bloom_view(
    pool: &TransientImagePool,
    fallback: vk::ImageView,
    frame: usize,
) -> vk::ImageView {
    pool.view_for("bloom_top", frame).unwrap_or(fallback)
}

impl VkContext {
    // Encode the chain for frame slot `frame_idx` over this frame's scene color.
    // On return the pool's `bloom_top` holds the glow the composite samples.
    // Called only when `post_process.bloom_intensity > 0`.
    pub(in crate::vulkan) fn encode_bloom(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
    ) -> RenderResult<()> {
        let Some(bloom) = &self.bloom else {
            return Ok(());
        };
        let view = self
            .targets
            .transient_pool
            .view_for("bloom_top", frame_idx)
            .ok_or_else(|| RenderError::Other("bloom_top missing from transient pool".into()))?;
        let extent = top_extent(post_extent(self.swapchain.extent));
        let top = VkAttachment {
            view,
            extent: vk::Extent2D {
                width: extent.width,
                height: extent.height,
            },
            format: PixelFormat::Rgba16Float,
            rest: AttachmentRest::Sampled,
        };
        bloom.encode(
            &self.post_device(frame_idx),
            &cmd,
            BloomInputs {
                scene: self.scene_color_view(frame_idx),
                top,
                top_ref: view,
            },
            &self.post_process,
        )
    }

    // The scene the bloom prefilter and the composite read for frame slot
    // `frame`: the upscaler's output, else the TAA output, else the pre-TAA
    // scene.
    pub(in crate::vulkan) fn scene_color_view(&self, frame: usize) -> vk::ImageView {
        if let Some(up) = &self.upscale {
            up.output_image().view
        } else if let Some(taa) = &self.taa {
            taa.output_view(frame)
        } else {
            self.post_scene_image(frame).view
        }
    }
}
