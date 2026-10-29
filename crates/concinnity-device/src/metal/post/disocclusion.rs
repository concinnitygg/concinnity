//! Metal's share of the reactive mask's disocclusion term: the shared pass is
//! built beside a `MetalFX` scaler that reads the mask, and each frame it reads
//! the pooled G-buffer and writes the HDR targets' mask. The mark and the
//! snapshot are written once in `concinnity_engine::render::post::disocclusion`.

use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::post::device::PostExtent;
use concinnity_engine::render::post::disocclusion::{DisocclusionInputs, DisocclusionPass};

use crate::metal::context::MtlContext;
use crate::metal::post::UpscaleState;
use crate::metal::post::post_device::{MtlPostDevice, MtlPostPipeline, MtlPostTarget};

// The shared pass, holding Metal's own pipeline and target handles.
pub(crate) type MtlDisocclusionPass = DisocclusionPass<MtlPostPipeline, MtlPostTarget>;

// The pass, when the scaler reads the reactive mask and the G-buffer pre-pass
// that the mark reads is built.
pub(in crate::metal) fn build_disocclusion(
    post_device: &MtlPostDevice<'_>,
    upscale: &UpscaleState,
    gbuffer: bool,
    render: (u32, u32),
) -> RenderResult<Option<MtlDisocclusionPass>> {
    if !gbuffer || !upscale.scaler.as_ref().is_some_and(|s| s.reactive) {
        return Ok(None);
    }
    DisocclusionPass::new(
        post_device,
        PostExtent {
            width: render.0,
            height: render.1,
        },
    )
    .map(Some)
}

impl MtlContext {
    // Encode the mark into the reactive mask, then this frame's snapshot.
    pub(in crate::metal) fn encode_disocclusion(
        &self,
        cmd_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    ) -> RenderResult<u32> {
        let missing = |what: &str| RenderError::Other(format!("disocclusion: no {what}"));
        let pass = self.disocclusion.as_ref().ok_or_else(|| missing("pass"))?;
        pass.encode(
            &self.post_device(),
            cmd_buf,
            &DisocclusionInputs {
                normal_depth: self
                    .gbuffer_normal_depth()
                    .ok_or_else(|| missing("G-buffer normal+depth"))?,
                velocity: self
                    .gbuffer_velocity()
                    .ok_or_else(|| missing("G-buffer velocity"))?,
                mask: self.targets.hdr.reactive_mask.as_ref(),
            },
        )?;
        Ok(0)
    }
}
