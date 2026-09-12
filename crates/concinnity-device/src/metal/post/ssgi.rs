// src/metal/post/ssgi.rs
//
// Metal's share of screen-space global illumination, which is its settings and
// where the pass reads and writes this frame. The gather and composite -- their
// pipelines, the reduced gather target and both draws -- are written once in
// `concinnity_core::render::post::ssgi` and reach Metal through `MtlPostDevice`.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::render_types;
use concinnity_core::gfx::ssgi::SsgiSettings;
use concinnity_core::render::post::ssgi::{SsgiInputs, SsgiPass};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLTexture;

use crate::metal::context::MtlContext;
use crate::metal::post::post_device::MtlPostPipeline;

// The shared gather + composite, holding Metal's own pipeline and target handles.
pub(crate) type MtlSsgiPass = SsgiPass<MtlPostPipeline, Retained<ProtocolObject<dyn MTLTexture>>>;

// Screen-space GI state: the resolved tunables, and the pass when SSGI is on
// (and the unified G-buffer it gathers against therefore exists).
pub(crate) struct SsgiState {
    pub settings: Option<SsgiSettings>,
    pub pass: Option<MtlSsgiPass>,
}

impl MtlContext {
    // Encode the SSGI gather + composite: hemisphere rays marched over the
    // G-buffer into the reduced gather target, then blurred and added into
    // `hdr_resolve`. Runs on the hdr_resolve read-modify-write chain after the
    // main pass.
    pub(in crate::metal) fn encode_ssgi(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        ssgi_params: &render_types::SsgiParams,
    ) -> Result<u32, String> {
        // With no G-buffer there is nothing to gather against, so skip the pass.
        let (Some(pass), Some(normal_depth)) = (&self.ssgi.pass, self.gbuffer_normal_depth())
        else {
            return Ok(0);
        };
        let scene = self.hdr_targets.hdr_resolve.as_ref();
        pass.encode(
            &self.post_device(),
            cmd_buf,
            SsgiInputs {
                scene,
                scene_target: scene,
                normal_depth,
            },
            ssgi_params,
        )?;
        Ok(0)
    }
}
