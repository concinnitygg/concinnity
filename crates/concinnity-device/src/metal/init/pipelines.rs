// src/metal/init/pipelines.rs
//
// Core render-pipeline construction extracted from MtlContext::new:
//   * The shared vertex descriptor (interleaved [pos, normal, tangent, color, uv]).
//   * The main static pipeline (with its GPU-driven cull pipeline and the
//     bindless texture argument encoder).
//   * The shared depth-stencil state used by main + shadow passes.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLArgumentEncoder, MTLCompareFunction, MTLDepthStencilDescriptor, MTLDepthStencilState,
    MTLDevice, MTLFunction as _, MTLLibrary as _, MTLPixelFormat, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction,
};

use crate::gfx::mesh_payload::Vertex;
use crate::metal::context::{
    BINDLESS_SAMPLER_ARG_BUFFER_INDEX, BINDLESS_TEXTURE_ARG_BUFFER_INDEX, HDR_SAMPLE_COUNT,
};
use crate::metal::cull::{CullPipeline, build_cull_pipeline};
use crate::metal::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};
use crate::metal::pipeline::{ns_str, world_library};
use concinnity_core::components::ShaderPrograms;

pub(crate) struct MainPipelineBundle {
    pub pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    // The GPU-driven cull's pipelines, phase 2 included: it is built whenever
    // the bindless path is active (cheap: one extra compute pipeline) and only
    // used when `occlusion_two_pass` is on at runtime.
    pub cull: CullPipeline,
    pub bindless_tex_arg_encoder: Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    // Encoder for the engine sampler block at buffer(10).
    pub bindless_sampler_arg_encoder: Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
}

// Describes the per-vertex buffer layout so Metal can map [[stage_in]]:
//   buffer(1): interleaved [float3 pos, float3 normal, float3 tangent, float3 color, float2 uv]
//   stride = sizeof(Vertex) = 56 bytes
pub(crate) fn make_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
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
            stride: std::mem::size_of::<Vertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    )
}

// Build the main static pipeline together with everything the GPU-driven pass
// implies: the pipeline opts into indirect command buffers, and a compute cull
// pipeline, an ICB argument encoder and the BindlessTextures argument encoder
// come with it. The pair is the engine's own, or the world Shader's compile of
// the same source.
pub(crate) fn build_main_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    world: Option<&ShaderPrograms>,
    hot_reload: bool,
) -> Result<MainPipelineBundle, String> {
    // Both pairs come from the single-source bindless program: the engine's
    // own, or the world's compile of the same file with its hooks spliced in.
    // The static pass is always GPU-driven now.
    let (vert_fn, main_frag_fn) = match world {
        None => {
            let vert_library = super::super::slang_shaders::MAIN_BINDLESS_VERT
                .library(device, hot_reload)
                .map_err(|e| format!("failed to load engine vertex library: {e}"))?;
            let frag_library = super::super::slang_shaders::MAIN_BINDLESS_FRAG
                .library(device, hot_reload)
                .map_err(|e| format!("failed to load engine fragment library: {e}"))?;
            let vert_fn = vert_library
                .newFunctionWithName(&ns_str("vertex_main_bindless"))
                .ok_or("vertex_main_bindless not found in engine library")?;
            let frag_fn = frag_library
                .newFunctionWithName(&ns_str("fragment_main_bindless"))
                .ok_or("fragment_main_bindless not found in engine library")?;
            (vert_fn, frag_fn)
        }
        Some(programs) => {
            // One library holds the pair: the cook groups the bindless
            // entries into one MSL translation unit on this host.
            let library = world_library(device, hot_reload, programs, "fragment_main_bindless")
                .map_err(|e| format!("failed to load the world's main library: {e}"))?;
            let vert_fn = library
                .newFunctionWithName(&ns_str("vertex_main_bindless"))
                .ok_or("vertex_main_bindless not found in the world's main library")?;
            let frag_fn = library
                .newFunctionWithName(&ns_str("fragment_main_bindless"))
                .ok_or("fragment_main_bindless not found in the world's main library")?;
            (vert_fn, frag_fn)
        }
    };
    let pipeline_desc = MTLRenderPipelineDescriptor::new();
    pipeline_desc.setVertexDescriptor(Some(vert_desc));
    pipeline_desc.setVertexFunction(Some(&vert_fn));
    pipeline_desc.setFragmentFunction(Some(&main_frag_fn));
    // Off-screen HDR pass: RGBA16Float colour + 4x MSAA. Output is linear
    // light; ACES tonemap + gamma + FXAA run in the composite pass.
    pipeline_desc.setRasterSampleCount(HDR_SAMPLE_COUNT as usize);
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

    let pipeline_state = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
        .map_err(|e| format!("failed to create pipeline state: {:?}", e))?;

    let cull = build_cull_pipeline(device, hot_reload)?;

    // The argument encoders for the `BindlessTextures` buffer at buffer(7) and
    // the sampler block at buffer(10) describe the engine's layouts, so they
    // come from the engine's own fragment. A world's compile of the same file
    // declares the same blocks, but a `shade` that samples nothing lets the
    // compiler drop them, and an encoder cannot be derived from a parameter
    // that is not there.
    let encoder_frag_fn = match world {
        None => main_frag_fn.clone(),
        Some(_) => super::super::slang_shaders::entry_function(
            device,
            &super::super::slang_shaders::MAIN_BINDLESS_FRAG,
            hot_reload,
        )?,
    };
    // SAFETY: both indices are the static buffer indices the engine fragment
    // declares its argument buffers at (locked by the build script's ABI
    // assertion).
    let (bindless_tex_arg_encoder, bindless_sampler_arg_encoder) = unsafe {
        (
            Some(
                encoder_frag_fn
                    .newArgumentEncoderWithBufferIndex(BINDLESS_TEXTURE_ARG_BUFFER_INDEX),
            ),
            Some(
                encoder_frag_fn
                    .newArgumentEncoderWithBufferIndex(BINDLESS_SAMPLER_ARG_BUFFER_INDEX),
            ),
        )
    };

    Ok(MainPipelineBundle {
        pipeline_state,
        cull,
        bindless_tex_arg_encoder,
        bindless_sampler_arg_encoder,
    })
}

// Write the engine sampler block once: the pool sampler (trilinear +
// anisotropic + repeat), the shadow compare sampler, and the cube sampler. Member order mirrors EngineSamplers
// in `src/shaders/main_bindless.slang`. Samplers never stream, so unlike the
// texture argument buffer this is written a single time at init.
pub(crate) fn build_bindless_sampler_args(
    device: &ProtocolObject<dyn MTLDevice>,
    encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
    tex_sampler: &ProtocolObject<dyn objc2_metal::MTLSamplerState>,
    shadow_sampler: &ProtocolObject<dyn objc2_metal::MTLSamplerState>,
    cube_sampler: &ProtocolObject<dyn objc2_metal::MTLSamplerState>,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>, String> {
    use objc2_metal::MTLResourceOptions;
    let len = encoder.encodedLength().max(16);
    let buf = device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        .ok_or("failed to allocate sampler argument buffer")?;
    // SAFETY: `buf` was sized to the encoder's `encodedLength()`, and the
    // indices 0..2 are the EngineSamplers member ids in declaration order.
    unsafe {
        encoder.setArgumentBuffer_offset(Some(&buf), 0);
        encoder.setSamplerState_atIndex(Some(tex_sampler), 0);
        encoder.setSamplerState_atIndex(Some(shadow_sampler), 1);
        encoder.setSamplerState_atIndex(Some(cube_sampler), 2);
    }
    Ok(buf)
}

// The main-pass pipelines of the material-referenced world shaders, indexed by
// `shader_bucket - 1`. `None` marks a bucket whose Shader is not resident.
pub(crate) type WorldPipelineTable =
    Vec<Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>>;

// Pipelines for the material-referenced world shaders past the default
// (ShaderHandle 1..), in bucket order. Extra world shaders render only through
// the GPU-driven bindless path (the cull kernel routes their draws into
// per-bucket ICBs), from the bindless pair the cook compiled for each.
//
// A bucket flagged `deferred` (its Shader is owned by a scene that has not
// pinned) stays `None` until
// [`super::super::MtlContext::install_world_shader`] builds it.
pub(crate) fn build_world_pipeline_table(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    extra_shaders: &[crate::gfx::backend_init::WorldShader<'_>],
    hot_reload: bool,
) -> Result<WorldPipelineTable, String> {
    let mut table = Vec::with_capacity(extra_shaders.len());
    for (i, shader) in extra_shaders.iter().enumerate() {
        // A bucket whose Shader a non-start scene owns has no payload yet; the
        // streaming pump installs it when that scene pins.
        let Some(programs) = shader.programs.filter(|_| !shader.deferred) else {
            table.push(None);
            continue;
        };
        table.push(Some(build_bucket_pipeline(
            device,
            vert_desc,
            i + 1,
            programs,
            hot_reload,
        )?));
    }
    Ok(table)
}

// One material-referenced shader bucket's bindless main-pass pipeline.
pub(crate) fn build_bucket_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    bucket: usize,
    programs: &ShaderPrograms,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let library = world_library(device, hot_reload, programs, "fragment_main_bindless")
        .map_err(|e| format!("shader bucket {bucket}: {e}"))?;
    let vert_fn = library
        .newFunctionWithName(&ns_str("vertex_main_bindless"))
        .ok_or_else(|| format!("shader bucket {bucket}: vertex_main_bindless not found"))?;
    let frag_fn = library
        .newFunctionWithName(&ns_str("fragment_main_bindless"))
        .ok_or_else(|| format!("shader bucket {bucket}: fragment_main_bindless not found"))?;

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(HDR_SAMPLE_COUNT as usize);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        desc.colorAttachments()
            .objectAtIndexedSubscript(0)
            .setPixelFormat(MTLPixelFormat::RGBA16Float);
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    desc.setSupportIndirectCommandBuffers(true);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("shader bucket {bucket}: failed to create pipeline: {e:?}"))
}

// Shadow pipeline: depth-only, no fragment function, no MSAA. Compiled from the
// engine-internal single source (`shadow.slang`, entry `shadow_vertex_main`).
// Shared by init (one-shot at startup) and the internal-shader hot-reload path
// (`reload_shaders`) so the two stay consistent.
pub(crate) fn build_shadow_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let shadow_fn = super::super::slang_shaders::entry_function(
        device,
        &super::super::slang_shaders::SHADOW_VERT,
        hot_reload,
    )?;
    let shadow_pipeline_desc = MTLRenderPipelineDescriptor::new();
    shadow_pipeline_desc.setVertexDescriptor(Some(vert_desc));
    shadow_pipeline_desc.setVertexFunction(Some(&shadow_fn));
    shadow_pipeline_desc.setRasterSampleCount(1);
    shadow_pipeline_desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    device
        .newRenderPipelineStateWithDescriptor_error(&shadow_pipeline_desc)
        .map_err(|e| format!("failed to create shadow pipeline state: {:?}", e))
}

// GPU-driven cascaded-shadow render pipeline: depth-only, no
// fragment, no MSAA, but `supportIndirectCommandBuffers` so each cascade's
// casters can draw through the shadow ICB the shadow cull's encode dispatch
// fills. Entry `shadow_vertex_bindless` reads the per-object model matrix from
// the GpuObjectData buffer at buffer(9) by `[[base_instance]]` (the record id
// the cull baked), exactly like the main bindless `vertex_main`. Reuses the
// full static vertex descriptor (the VS consumes only attribute(0) = position;
// the deformed skinned tail shares the same 56-byte layout).
pub(crate) fn build_shadow_bindless_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let shadow_fn = super::super::slang_shaders::entry_function(
        device,
        &super::super::slang_shaders::SHADOW_VERT_BINDLESS,
        hot_reload,
    )?;
    let shadow_pipeline_desc = MTLRenderPipelineDescriptor::new();
    shadow_pipeline_desc.setVertexDescriptor(Some(vert_desc));
    shadow_pipeline_desc.setVertexFunction(Some(&shadow_fn));
    shadow_pipeline_desc.setRasterSampleCount(1);
    shadow_pipeline_desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    shadow_pipeline_desc.setSupportIndirectCommandBuffers(true);
    device
        .newRenderPipelineStateWithDescriptor_error(&shadow_pipeline_desc)
        .map_err(|e| format!("failed to create shadow bindless pipeline state: {:?}", e))
}

// Depth-stencil state: less-than test, writes enabled (shared for main and
// shadow pass).
pub(crate) fn make_depth_state(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>, String> {
    let depth_desc = MTLDepthStencilDescriptor::new();
    depth_desc.setDepthCompareFunction(MTLCompareFunction::Less);
    depth_desc.setDepthWriteEnabled(true);
    device
        .newDepthStencilStateWithDescriptor(&depth_desc)
        .ok_or_else(|| "failed to create depth stencil state".to_string())
}

// Read-only depth-stencil state: less-or-equal test, no write. Translucent
// passes (volumetric raymarch) bind this so they early-z against nearer
// opaque geometry without touching the depth buffer. A non-nil state is
// required: Metal's validation layer asserts on `setDepthStencilState(nil)`.
pub(crate) fn make_depth_state_read_only(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>, String> {
    let depth_desc = MTLDepthStencilDescriptor::new();
    depth_desc.setDepthCompareFunction(MTLCompareFunction::LessEqual);
    depth_desc.setDepthWriteEnabled(false);
    device
        .newDepthStencilStateWithDescriptor(&depth_desc)
        .ok_or_else(|| "failed to create read-only depth stencil state".to_string())
}
