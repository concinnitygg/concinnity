//! Metal's share of screen-space global illumination, which is its settings and
//! where the pass reads and writes this frame. Every stage -- the pipelines, the
//! depth pyramids, the accumulation and all the draws -- is written once in
//! `concinnity_engine::render::post::ssgi` and reaches Metal through
//! `MtlPostDevice`.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::gfx::render_types;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::post::ssgi::settings::SsgiSettings;
use concinnity_engine::render::post::ssgi::{SsgiPass, SsgiResolveInputs, SsgiTraceInputs};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLTexture;

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
    // Whether the main pass's screen-indirect texture carries SSGI's bounce:
    // whenever SSGI is built, which is when the graph runs its resolve.
    pub(in crate::metal) fn screen_bounce(&self) -> bool {
        self.ssgi.pass.is_some()
    }

    // The texture the main pass and the bindless argument buffer sample for
    // the screen-space indirect light: SSGI's resolved visibility and bounce
    // while it is built, else the ambient occlusion (or its white fallback).
    pub(in crate::metal) fn screen_indirect_texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        match &self.ssgi.pass {
            Some(pass) => pass.indirect().texture(),
            None => self.ao_output_texture(),
        }
    }

    // Encode the resolve ahead of the main pass: last frame's accumulation
    // reprojected onto this frame's G-buffer, folded with the occlusion.
    pub(in crate::metal) fn encode_ssgi_resolve(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        ssgi_params: &render_types::SsgiParams,
    ) -> RenderResult<u32> {
        let (Some(pass), Some(normal_depth)) = (&self.ssgi.pass, self.gbuffer_normal_depth())
        else {
            return Ok(0);
        };
        pass.encode_resolve(
            &self.post_device(),
            cmd_buf,
            &SsgiResolveInputs {
                normal_depth,
                velocity: self.ssgi_velocity()?,
                occlusion: self.ao_output_texture(),
            },
            ssgi_params,
        )?;
        Ok(0)
    }

    // Encode SSGI after the main pass: the depth pyramid, the age, the trace
    // over the G-buffer and the accumulation the next frame's resolve reads.
    pub(in crate::metal) fn encode_ssgi(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        ssgi_params: &render_types::SsgiParams,
    ) -> RenderResult<u32> {
        // With no G-buffer there is nothing to trace against, so skip the pass.
        let (Some(pass), Some(normal_depth), Some(reflectance)) = (
            &self.ssgi.pass,
            self.gbuffer_normal_depth(),
            self.gbuffer_roughness(),
        ) else {
            return Ok(0);
        };
        pass.encode(
            &self.post_device(),
            cmd_buf,
            &SsgiTraceInputs {
                scene: self.targets.hdr.hdr_resolve.as_ref(),
                normal_depth,
                velocity: self.ssgi_velocity()?,
                reflectance,
            },
            ssgi_params,
        )?;
        Ok(0)
    }

    // Pool-owned, so it is fetched at encode time, like the normals.
    fn ssgi_velocity(&self) -> RenderResult<&ProtocolObject<dyn MTLTexture>> {
        self.gbuffer_velocity().ok_or_else(|| {
            RenderError::Other("SSGI enabled but the pooled G-buffer velocity is missing".into())
        })
    }
}
