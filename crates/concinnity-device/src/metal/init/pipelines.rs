//! Main-pass and shadow pipeline builders shared by the init stages and the
//! shader hot reload:
//!   * The shared vertex descriptor (interleaved [pos, normal, tangent, color, uv]).
//!   * The main static pipeline, the world shader bucket pipelines, and the
//!     bindless argument encoders and sampler block.
//!   * The cascade shadow pipelines.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::gfx::mesh_payload::Vertex;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLArgumentBuffersTier, MTLArgumentEncoder, MTLDevice, MTLFunction, MTLPixelFormat,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};

pub(crate) use crate::metal::bucket_pipelines::{
    BucketBuild, BucketPipelines, build_bucket_pipelines, replacement,
};
use crate::metal::context::{BINDLESS_SAMPLER_ARG_BUFFER_INDEX, BINDLESS_TEXTURE_ARG_BUFFER_INDEX};
use crate::metal::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};
use crate::metal::error::allocation_failed;

// Describes the per-vertex buffer layout so Metal can map [[stage_in]]:
//   buffer(1): interleaved [float3 pos, float3 normal, float3 tangent, float3 color, float2 uv]
//   stride = sizeof(Vertex) = 56 bytes
pub(in crate::metal) fn make_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    vertex_descriptor(
        &[
            VertexAttr {
                index: 0,
                format: MTLVertexFormat::Float3,
                offset: 0,
                buffer_index: 1,
            },
            VertexAttr {
                index: 1,
                format: MTLVertexFormat::Float3,
                offset: 12,
                buffer_index: 1,
            },
            VertexAttr {
                index: 2,
                format: MTLVertexFormat::Float3,
                offset: 24,
                buffer_index: 1,
            },
            VertexAttr {
                index: 3,
                format: MTLVertexFormat::Float3,
                offset: 36,
                buffer_index: 1,
            },
            VertexAttr {
                index: 4,
                format: MTLVertexFormat::Float2,
                offset: 48,
                buffer_index: 1,
            },
        ],
        &[VertexLayout {
            buffer_index: 1,
            stride: size_of::<Vertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    )
}

// The main-pass pipeline over `vert_fn` / `frag_fn`: the off-screen HDR pass,
// RGBA16Float color at the world's resolved sample count. Output is linear
// light; ACES tonemap + gamma + FXAA run in the composite pass.
pub(crate) fn build_main_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    vert_fn: &ProtocolObject<dyn MTLFunction>,
    frag_fn: &ProtocolObject<dyn MTLFunction>,
    sample_count: u32,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let pipeline_desc = MTLRenderPipelineDescriptor::new();
    pipeline_desc.setVertexDescriptor(Some(vert_desc));
    pipeline_desc.setVertexFunction(Some(vert_fn));
    pipeline_desc.setFragmentFunction(Some(frag_fn));
    pipeline_desc.setRasterSampleCount(sample_count as usize);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        pipeline_desc
            .colorAttachments()
            .objectAtIndexedSubscript(0)
            .setPixelFormat(MTLPixelFormat::RGBA16Float);
    }
    pipeline_desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    pipeline_desc.setSupportIndirectCommandBuffers(true);

    device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
        .map_err(|e| RenderError::ShaderCompile(format!("main pipeline state: {e:?}")))
}

// Fail unless the device and fragment support the bindless main pass:
// argument-buffer tier 2 and a `BindlessTextures` block at buffer(7) laid out as
// the resource-id writes assume, checked against the encoder's length.
fn check_bindless_support(
    device: &ProtocolObject<dyn MTLDevice>,
    fragment: &ProtocolObject<dyn MTLFunction>,
) -> RenderResult<()> {
    use crate::metal::bindless_args::bindless_block_len;
    if device.argumentBuffersSupport() != MTLArgumentBuffersTier::Tier2 {
        return Err(RenderError::Other(
            "the bindless main pass needs argument-buffer tier 2, which this device lacks".into(),
        ));
    }
    // SAFETY: the index is the static buffer index the engine fragment declares
    // its texture argument buffer at (locked by the ABI tests).
    let texture =
        unsafe { fragment.newArgumentEncoderWithBufferIndex(BINDLESS_TEXTURE_ARG_BUFFER_INDEX) };
    let expected = bindless_block_len(1);
    if texture.encodedLength() != expected {
        return Err(RenderError::Other(format!(
            "the bindless texture block encodes to {} bytes, not the {expected} its resource-id \
             layout assumes",
            texture.encodedLength()
        )));
    }
    Ok(())
}

// Write the sampler block at buffer(10) once: pool, shadow compare and cube
// samplers at ids s0..s2. Its encoder comes from the engine's fragment, since a
// world `shade` that samples nothing lets the compiler drop the block.
pub(crate) fn build_bindless_sampler_args(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
    tex_sampler: &ProtocolObject<dyn objc2_metal::MTLSamplerState>,
    shadow_sampler: &ProtocolObject<dyn objc2_metal::MTLSamplerState>,
    cube_sampler: &ProtocolObject<dyn objc2_metal::MTLSamplerState>,
) -> RenderResult<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>> {
    use objc2_metal::MTLResourceOptions;
    let fragment = super::super::builtin_shaders::entry_function(
        device,
        &super::super::builtin_shaders::MAIN_BINDLESS_FRAG,
        hot_reload,
    )?;
    check_bindless_support(device, &fragment)?;
    // SAFETY: the index is the static buffer index the engine fragment declares
    // its sampler argument buffer at (locked by the ABI tests).
    let encoder =
        unsafe { fragment.newArgumentEncoderWithBufferIndex(BINDLESS_SAMPLER_ARG_BUFFER_INDEX) };
    let len = encoder.encodedLength().max(16);
    let buf = device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| allocation_failed("the sampler argument buffer"))?;
    // SAFETY: `buf` was sized to the encoder's `encodedLength()`.
    unsafe { encoder.setArgumentBuffer_offset(Some(&buf), 0) };
    // SAFETY: id 0 is the first EngineSamplers member, the pool sampler, in `buf`.
    unsafe { encoder.setSamplerState_atIndex(Some(tex_sampler), 0) };
    // SAFETY: id 1 is the shadow compare sampler member, next in declaration order.
    unsafe { encoder.setSamplerState_atIndex(Some(shadow_sampler), 1) };
    // SAFETY: id 2 is the cube sampler, the last EngineSamplers member.
    unsafe { encoder.setSamplerState_atIndex(Some(cube_sampler), 2) };
    Ok(buf)
}

// The GPU-driven shadow pipeline: depth-only with ICB support, reading each
// record's model from buffer(9) by `[[base_instance]]`.
pub(crate) fn build_shadow_bindless_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    depth_format: MTLPixelFormat,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let shadow_fn = super::super::builtin_shaders::entry_function(
        device,
        &super::super::builtin_shaders::SHADOW_VERT_BINDLESS,
        hot_reload,
    )?;
    let shadow_pipeline_desc = MTLRenderPipelineDescriptor::new();
    shadow_pipeline_desc.setVertexDescriptor(Some(vert_desc));
    shadow_pipeline_desc.setVertexFunction(Some(&shadow_fn));
    shadow_pipeline_desc.setRasterSampleCount(1);
    shadow_pipeline_desc.setDepthAttachmentPixelFormat(depth_format);
    shadow_pipeline_desc.setSupportIndirectCommandBuffers(true);
    device
        .newRenderPipelineStateWithDescriptor_error(&shadow_pipeline_desc)
        .map_err(|e| RenderError::ShaderCompile(format!("shadow bindless pipeline state: {e:?}")))
}
