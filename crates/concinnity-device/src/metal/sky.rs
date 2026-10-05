//! The environment drawn as the background (see `concinnity_core::render::sky`):
//! a fullscreen draw at the tail of each opaque scene pass (the main camera, a
//! reflection-probe face, a planar mirror), and one at the tail of the G-buffer
//! pre-pass for the sky's motion. Both ride the encoder the pass already has
//! open, so the sky costs no extra load or store of its targets.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::view_modes::ViewMode;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::sky;
use concinnity_core::render::uniforms::{GBufferView, ViewUniforms};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandEncoder as _, MTLDevice, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState,
};

use super::builtin_shaders::{
    self, GBUFFER_PREPASS_FRAG_BINDLESS, GBUFFER_SKY_VERT, SKY_FRAG, SKY_VERT,
};
use super::context::MtlContext;
use super::encode::RenderEncode;

pub(super) struct SkyState {
    // Draws the environment behind the opaque scene, at the main pass's sample
    // count, which the probe faces and mirrors share.
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    // Writes the sky's motion behind the pre-pass geometry, into the pre-pass's
    // three targets.
    pub(super) velocity_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    // The world draws its environment map as the background.
    pub(super) background: bool,
}

impl SkyState {
    pub(super) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        sample_count: u32,
        background: bool,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        Ok(Self {
            pipeline: build_sky_pipeline(device, sample_count, hot_reload)?,
            velocity_pipeline: build_sky_velocity_pipeline(device, hot_reload)?,
            background,
        })
    }
}

// The sky over the HDR color and the main depth: no blending, and no vertex
// stream, since the vertex stage generates its triangle.
pub(super) fn build_sky_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    sample_count: u32,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert = builtin_shaders::entry_function(device, &SKY_VERT, hot_reload)?;
    let frag = builtin_shaders::entry_function(device, &SKY_FRAG, hot_reload)?;
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vert));
    desc.setFragmentFunction(Some(&frag));
    desc.setRasterSampleCount(sample_count as usize);
    // SAFETY: plain descriptor property setters; the subscripted slot is one this descriptor
    // declares.
    unsafe {
        desc.colorAttachments()
            .objectAtIndexedSubscript(0)
            .setPixelFormat(MTLPixelFormat::RGBA16Float);
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| RenderError::ShaderCompile(format!("sky pipeline state: {e:?}")))
}

// The sky's motion into the pre-pass's normal + depth, roughness and velocity
// targets, through the pre-pass's own fragment.
pub(super) fn build_sky_velocity_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert = builtin_shaders::entry_function(device, &GBUFFER_SKY_VERT, hot_reload)?;
    let frag = builtin_shaders::entry_function(device, &GBUFFER_PREPASS_FRAG_BINDLESS, hot_reload)?;
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vert));
    desc.setFragmentFunction(Some(&frag));
    desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let targets = desc.colorAttachments();
        for (i, format) in super::post::gbuffer::GBUFFER_FORMATS.iter().enumerate() {
            targets.objectAtIndexedSubscript(i).setPixelFormat(*format);
        }
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| RenderError::ShaderCompile(format!("sky velocity pipeline state: {e:?}")))
}

impl MtlContext {
    // Whether a view rendered in `mode` draws the sky.
    pub(super) fn draws_sky(&self, mode: ViewMode) -> bool {
        sky::draws_sky(
            self.scene.env_map.prefilter_mip_count > 0,
            self.sky.background,
            mode,
        )
    }

    // Draw the environment behind everything `enc` drew so far, from the
    // viewpoint `view` describes. The encoder's targets must be the HDR color
    // and the depth its geometry tested against.
    pub(super) fn encode_sky(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        view: &ViewUniforms,
    ) {
        enc.pushDebugGroup(objc2_foundation::ns_string!("sky"));
        enc.set_pipeline(&self.sky.pipeline);
        enc.set_depth_stencil(&self.targets.depth_state_read_only);
        enc.set_vertex_value(view, 0);
        enc.set_fragment_value(view, 0);
        enc.set_fragment_texture(&self.scene.env_map.prefilter, 0);
        enc.set_fragment_sampler(&self.scene.cube_sampler, 0);
        // SAFETY: the vertex stage generates all three vertices, so the draw reads no bound vertex
        // buffer.
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        enc.popDebugGroup();
    }

    // Write the sky's motion behind everything the pre-pass drew, from the
    // pre-pass's own view block.
    pub(super) fn encode_sky_velocity(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        view: &GBufferView,
    ) {
        enc.set_pipeline(&self.sky.velocity_pipeline);
        enc.set_depth_stencil(&self.targets.depth_state_read_only);
        enc.set_vertex_value(view, 0);
        // SAFETY: the vertex stage generates all three vertices, so the draw reads no bound vertex
        // buffer.
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
    }
}
