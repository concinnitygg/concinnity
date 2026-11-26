//! Vulkan's share of the reactive mask's disocclusion term: the shared pass is
//! kept while an upscaler that reads the mask resolves the frame, and
//! each frame it reads this slot's G-buffer and writes this slot's mask. The
//! mark and the snapshot are written once in
//! `concinnity_engine::render::post::disocclusion`.

use ash::vk;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::post::disocclusion::{
    self, DisocclusionInputs, DisocclusionPass, MASK_FORMAT,
};

use crate::vulkan::context::VkContext;
use crate::vulkan::post::pass_cache::AttachmentRest;
use crate::vulkan::post::post_device::{PostPipeline, PostTarget, VkAttachment, post_extent};
use crate::vulkan::record::Recorder;

// The shared pass, holding Vulkan's own pipeline and target handles.
pub(in crate::vulkan) type VkDisocclusionPass = DisocclusionPass<PostPipeline, PostTarget>;

impl VkContext {
    // Whether the pass is wanted beside the upscaler (when there is one) and
    // the G-buffer.
    pub(in crate::vulkan) fn disocclusion_wanted(&self) -> bool {
        disocclusion::wanted(
            self.gbuffer_targets().is_some(),
            self.upscale.as_ref().map(|u| u.reactive_reader().reads()),
        )
    }

    // Build, resize or drop the pass to match the upscaler and the G-buffer at
    // render extent `extent`. The caller has already idled the device.
    pub(in crate::vulkan) fn sync_disocclusion(
        &mut self,
        extent: vk::Extent2D,
    ) -> RenderResult<()> {
        let wanted = self.disocclusion_wanted();
        self.disocclusion = match (wanted, self.disocclusion.take()) {
            (false, _) => None,
            (true, Some(mut pass)) => {
                pass.resize(&self.post_device(0), post_extent(extent))?;
                Some(pass)
            }
            (true, None) => Some(DisocclusionPass::new(
                &self.post_device(0),
                post_extent(extent),
            )?),
        };
        Ok(())
    }

    // Encode the mark into frame slot `frame_idx`'s reactive mask, then this
    // frame's snapshot.
    pub(in crate::vulkan) fn encode_disocclusion(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
    ) -> RenderResult<()> {
        let missing = |what: &str| RenderError::Other(format!("disocclusion: no {what}"));
        let pass = self.disocclusion.as_ref().ok_or_else(|| missing("pass"))?;
        let gbuffer = self
            .gbuffer_targets()
            .and_then(|gb| gb.frame(frame_idx))
            .ok_or_else(|| missing("G-buffer"))?;
        let mask = self
            .reactive_mask(frame_idx)
            .ok_or_else(|| missing("reactive mask"))?;
        pass.encode(
            &self.post_device(frame_idx),
            rec,
            &DisocclusionInputs {
                normal_depth: gbuffer.normal_depth,
                velocity: gbuffer.velocity,
                mask: VkAttachment {
                    view: mask.view,
                    extent: self.targets.render_extent,
                    format: MASK_FORMAT,
                    // The mask rests sampled and its writers move it in and
                    // out; the mark covers every pixel, so it opens discarded.
                    rest: AttachmentRest::Sampled,
                },
            },
        )
    }
}
