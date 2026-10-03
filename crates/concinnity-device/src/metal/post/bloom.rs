//! Metal's share of bloom, which is where the chain's scene and top octave come
//! from this frame. The chain itself -- its pipelines, the octaves below the
//! top, and every draw -- is written once in
//! `concinnity_core::render::post::bloom` and reaches Metal through
//! `MtlPostDevice`.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::bloom::{BloomInputs, BloomPass};
use concinnity_core::render::post::device::PostExtent;
use objc2::runtime::ProtocolObject;

use crate::metal::context::MtlContext;
use crate::metal::post::post_device::{MtlPostDevice, MtlPostPipeline, MtlPostTarget};

// The shared chain, holding Metal's own pipeline and target handles.
pub(crate) type MtlBloomPass = BloomPass<MtlPostPipeline, MtlPostTarget>;

// Build the chain for an output of `width` x `height`.
pub(crate) fn build_bloom_pass(
    device: &MtlPostDevice,
    width: u32,
    height: u32,
) -> RenderResult<MtlBloomPass> {
    BloomPass::new(device, PostExtent { width, height })
}

impl MtlContext {
    // Encode the chain over `scene_color`, the post-TAA scene (or the HDR
    // resolve when TAA is off). On return the pool's `bloom_top` holds the glow
    // the composite samples. Scene-less worlds build no chain and the graph
    // never inserts the Bloom pass, so a missing one is a no-op.
    pub(in crate::metal) fn encode_bloom(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        scene_color: &ProtocolObject<dyn objc2_metal::MTLTexture>,
    ) -> RenderResult<u32> {
        let Some(pass) = &self.bloom else {
            return Ok(0);
        };
        // Pool-owned, so it is fetched at encode time: a pool rebuild repacks
        // every slot.
        let top = self.targets.transient_pool.bloom_top()?;
        pass.encode(
            &self.post_device(),
            cmd_buf,
            BloomInputs {
                scene: scene_color,
                top,
                top_ref: top,
            },
            &self.post_process,
        )?;
        Ok(0)
    }
}
