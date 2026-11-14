//! Metal's share of the reactive mask's disocclusion term: the shared pass is
//! built while TAA or a `MetalFX` scaler that reads the mask resolves the frame,
//! and each frame it reads the pooled G-buffer and writes the HDR targets'
//! mask. The mark and the snapshot are written once in
//! `concinnity_engine::render::post::disocclusion`.

use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::post::disocclusion::{self, DisocclusionInputs, DisocclusionPass};

use crate::metal::context::MtlContext;
use crate::metal::post::UpscaleState;
use crate::metal::post::post_device::{PostPipeline, PostTarget};

// The shared pass, holding Metal's own pipeline and target handles.
pub(crate) type MtlDisocclusionPass = DisocclusionPass<PostPipeline, PostTarget>;

// Whether the pass is wanted beside the scaler (when there is one), the
// G-buffer pre-pass and the TAA resolve.
pub(in crate::metal) fn disocclusion_wanted(
    upscale: &UpscaleState,
    gbuffer: bool,
    taa: bool,
) -> bool {
    disocclusion::wanted(gbuffer, upscale.scaler.as_ref().map(|s| s.reactive), taa)
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
