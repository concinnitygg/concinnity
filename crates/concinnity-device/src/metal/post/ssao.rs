//! Metal's share of SSAO (GTAO): the settings, the white fallback the forward
//! pass binds while it is off, and where the kernel's inputs come from this
//! frame. The depth copy, kernel and blur -- their pipelines, the depth copy and
//! raw occlusion they own and every draw -- are written once in
//! `concinnity_core::render::post::ssao` and reach Metal through
//! `MtlPostDevice`. The depth + normal they read come from the unified G-buffer
//! pre-pass.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::render_types;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::post::ssao::settings::SsaoSettings;
use concinnity_core::render::post::ssao::{SsaoInputs, SsaoPass};
use objc2::runtime::ProtocolObject;

use crate::metal::context::MtlContext;
use crate::metal::post::post_device::{MtlPostPipeline, MtlPostTarget};

// The shared depth copy, kernel and blur, holding Metal's own pipeline and
// target handles.
pub(crate) type MtlSsaoPass = SsaoPass<MtlPostPipeline, MtlPostTarget>;

// All SSAO (GTAO) state grouped into one feature unit: the resolved settings,
// the shared pass, and the 1×1 white fallback. `settings` and `pass` are `Some`
// only when SSAO is enabled; `white` is always present so
// `MtlContext::ao_output_texture` can return a constant-1.0 sample when SSAO is
// off. The blurred occlusion the main pass samples lives in the transient pool.
pub(crate) struct SsaoState {
    pub settings: Option<SsaoSettings>,
    pub pass: Option<MtlSsaoPass>,
    pub white: crate::metal::allocator::PooledTexture,
}

impl MtlContext {
    // Encode the SSAO kernel and blur over the unified G-buffer pre-pass's
    // normal + depth into the pool's `ao_output`. Runs before the main pass so
    // `shade_surface` can sample the blurred occlusion and modulate its ambient
    // term. Only called when SSAO is enabled.
    pub(in crate::metal) fn encode_ssao(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        ssao_params: &render_types::SsaoParams,
    ) -> RenderResult<u32> {
        // Pool-owned, so it is fetched here rather than cached.
        let (Some(pass), Some(normal_depth)) = (&self.ssao.pass, self.gbuffer_normal_depth())
        else {
            return Ok(0);
        };
        // The pool always holds `ao_output` when SSAO is on (both gate on the
        // same setting), sharing a heap slot with `bloom_top`.
        let output = self
            .targets
            .transient_pool
            .texture_for("ao_output")
            .ok_or_else(|| RenderError::Other("ao_output missing from transient pool".into()))?;
        pass.encode(
            &self.post_device(),
            cmd_buf,
            SsaoInputs {
                normal_depth,
                output,
            },
            ssao_params,
        )?;
        Ok(0)
    }
}
