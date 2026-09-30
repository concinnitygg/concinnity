//! Metal's share of screen-space global illumination, which is its settings and
//! where the pass reads and writes this frame. Every stage -- the pipelines, the
//! depth pyramids, the accumulation and all the draws -- is written once in
//! `concinnity_core::render::post::ssgi` and reaches Metal through
//! `MtlPostDevice`.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::render_types;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::post::ssgi::settings::SsgiSettings;
use concinnity_core::render::post::ssgi::{SsgiInputs, SsgiPass};
use objc2::runtime::ProtocolObject;

use crate::metal::context::MtlContext;
use crate::metal::post::post_device::{MtlPostPipeline, MtlPostTarget};

// The shared SSGI pass, holding Metal's own pipeline and target handles.
pub(crate) type MtlSsgiPass = SsgiPass<MtlPostPipeline, MtlPostTarget>;

// Screen-space GI state: the resolved tunables, and the pass when SSGI is on
// (and the unified G-buffer it traces against therefore exists).
pub(crate) struct SsgiState {
    pub settings: Option<SsgiSettings>,
    pub pass: Option<MtlSsgiPass>,
}

impl MtlContext {
    // Encode SSGI: the depth pyramid, the trace over the G-buffer, the
    // accumulation, and the composite into `hdr_resolve`. Runs on the
    // hdr_resolve read-modify-write chain after the main pass.
    pub(in crate::metal) fn encode_ssgi(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        ssgi_params: &render_types::SsgiParams,
    ) -> RenderResult<u32> {
        // With no G-buffer there is nothing to trace against, so skip the pass.
        let (Some(pass), Some(normal_depth)) = (&self.ssgi.pass, self.gbuffer_normal_depth())
        else {
            return Ok(0);
        };
        // Pool-owned, so it is fetched at encode time, like the normals.
        let velocity = self.gbuffer_velocity().ok_or_else(|| {
            RenderError::Other("SSGI enabled but the pooled G-buffer velocity is missing".into())
        })?;
        let scene = self.targets.hdr.hdr_resolve.as_ref();
        pass.encode(
            &self.post_device(),
            cmd_buf,
            SsgiInputs {
                scene,
                scene_target: scene,
                normal_depth,
                velocity,
            },
            ssgi_params,
        )?;
        Ok(0)
    }
}
