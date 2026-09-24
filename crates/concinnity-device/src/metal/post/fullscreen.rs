//! Shared builders for fullscreen-triangle post-process passes. Every
//! screen-space effect (SSAO, SSR, SSGI, TAA, bloom, fog, RT reflections, the
//! final composite) draws one `[[vertex_id]]`-generated triangle into a single
//! color attachment with no vertex descriptor and no depth, differing only in
//! shader source, attachment format, and blend. These helpers fold that shared
//! pipeline-descriptor boilerplate into one place so each effect file keeps only
//! what is unique to it.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::error::{RenderError, RenderResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlendFactor, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLDevice as _, MTLFunction,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction,
    MTLTexture,
};

use crate::metal::builtin_shaders::{FULLSCREEN_VERT, ShaderProgram, entry_function};
use crate::metal::context::MtlContext;
use crate::metal::encode::RenderEncode;
use crate::metal::pass_timing::PassId;

// Blend configuration for a fullscreen pass's single color attachment.
#[derive(Clone, Copy)]
pub(crate) enum FullscreenBlend {
    // No blending; the fragment output replaces the destination. Used by every
    // pass that writes a fresh target (SSAO kernel/blur, SSR resolve, TAA
    // resolve, the SSGI gather, the bloom prefilter/downsample, the composite).
    Replace,
    // Additive accumulation (`src·1 + dst·1`). Used where a pass layers an
    // extra term onto content it loaded: the bloom upsample chain and the
    // SSGI composite.
    Additive,
    // Premultiplied "over" (`src·1 + dst·(1 − srcA)`): the fragment already
    // folded coverage into its color, so the source factor is `One`. Used by
    // the volumetric-fog composite.
    PremultipliedOver,
}

// Build a render pipeline state for a fullscreen-triangle post pass: the two
// stage functions, a single color attachment at `format` with the requested
// `blend`, single-sample, no vertex descriptor, no depth. `label` names the
// fragment in a pipeline-create error.
fn build_fullscreen_pipeline_from(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    vert_fn: &ProtocolObject<dyn MTLFunction>,
    frag_fn: &ProtocolObject<dyn MTLFunction>,
    label: &str,
    format: MTLPixelFormat,
    blend: FullscreenBlend,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(vert_fn));
    desc.setFragmentFunction(Some(frag_fn));
    desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setPixelFormat(format);
        match blend {
            FullscreenBlend::Replace => ca.setBlendingEnabled(false),
            FullscreenBlend::Additive => {
                ca.setBlendingEnabled(true);
                ca.setSourceRGBBlendFactor(MTLBlendFactor::One);
                ca.setDestinationRGBBlendFactor(MTLBlendFactor::One);
                ca.setSourceAlphaBlendFactor(MTLBlendFactor::One);
                ca.setDestinationAlphaBlendFactor(MTLBlendFactor::One);
            }
            FullscreenBlend::PremultipliedOver => {
                ca.setBlendingEnabled(true);
                ca.setSourceRGBBlendFactor(MTLBlendFactor::One);
                ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                ca.setSourceAlphaBlendFactor(MTLBlendFactor::One);
                ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            }
        }
    }

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| RenderError::ShaderCompile(format!("{label} pipeline: {e:?}")))
}

// Build a fullscreen-triangle pipeline whose fragment comes from a
// single-source program, paired with the shared `fullscreen_vertex`. The two
// stages come from separate libraries because a fragment variant declares only
// the resources it binds, so each variant is its own metallib while the vertex
// is compiled once for all of them.
pub(in crate::metal) fn build_fullscreen_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    fragment: &ShaderProgram,
    format: MTLPixelFormat,
    blend: FullscreenBlend,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert_fn = entry_function(device, &FULLSCREEN_VERT, hot_reload)?;
    let frag_fn = entry_function(device, fragment, hot_reload)?;
    build_fullscreen_pipeline_from(device, &vert_fn, &frag_fn, fragment.label, format, blend)
}

// Bind `sampler` to fragment sampler slots `first..first + count`. A post
// pass's source occupies a texture and a sampler at the same index, so a pass
// sampling N textures through one sampler state binds it N times. A pass that
// samples through two sampler states (SSR: the screen sources through one, the
// cubemaps through another) calls this once per contiguous run.
pub(in crate::metal) fn set_fragment_sampler_range(
    enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
    sampler: &ProtocolObject<dyn objc2_metal::MTLSamplerState>,
    first: usize,
    count: usize,
) {
    for i in first..first + count {
        enc.set_fragment_sampler(sampler, i);
    }
}

// Where a fullscreen pass sits within an effect's GPU-timing span. Most
// effects are a single encoder (`Whole`); bloom and SSGI span several, so they
// mark the start sample on the first encoder and the end sample on the last.
#[derive(Clone, Copy)]
pub(crate) enum PassTimer {
    // Record no timing sample on this pass.
    None,
    // The effect's only encoder: record both its start and end samples here.
    Whole(PassId),
    // The first encoder of a multi-encoder effect: record the start sample.
    First(PassId),
    // The last encoder of a multi-encoder effect: record the end sample.
    Last(PassId),
}

// The per-pass setup a fullscreen-triangle encode needs: the color target it
// writes, that attachment's load action, where the pass sits in the GPU-timing
// span, the pipeline it runs, and the encoder debug label.
pub(in crate::metal) struct FullscreenPass<'a> {
    pub target: &'a ProtocolObject<dyn MTLTexture>,
    pub load: MTLLoadAction,
    pub timer: PassTimer,
    pub pipeline: &'a ProtocolObject<dyn MTLRenderPipelineState>,
    pub label: &'a str,
}

// Run one fullscreen-triangle pass: open a single-attachment render encoder on
// `pass.target` (with the given `pass.load` action and an always-`Store`),
// attach GPU timing per `pass.timer`, set `pass.pipeline`, let `bind` set the
// pass's fragment inputs, draw the `[[vertex_id]]` triangle, and end encoding.
// Centralizes the encoder open / draw / close skeleton every screen-space
// effect repeats so each `encode_*` supplies only its unique bindings.
//
// A free function over the timing resources rather than a method on the
// context, so the shared post-pass seam (post/post_device.rs) can drive it from
// a device value assembled at init, before a context exists.
pub(in crate::metal) fn encode_fullscreen_pass(
    cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    timing: Option<&crate::metal::pass_timing::PassTimingResources>,
    pass: FullscreenPass,
    bind: impl FnOnce(&ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>),
) -> RenderResult<()> {
    let FullscreenPass {
        target,
        load,
        timer,
        pipeline,
        label,
    } = pass;
    let desc = MTLRenderPassDescriptor::new();
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setTexture(Some(target));
        ca.setLoadAction(load);
        ca.setStoreAction(MTLStoreAction::Store);
    }
    if let Some(t) = timing {
        match timer {
            PassTimer::None => {}
            PassTimer::Whole(id) => t.attach_render(&desc, id),
            PassTimer::First(id) => t.attach_render_first(&desc, id),
            PassTimer::Last(id) => t.attach_render_last(&desc, id),
        }
    }
    let enc = cmd_buf
        .renderCommandEncoderWithDescriptor(&desc)
        .ok_or_else(|| RenderError::Other(format!("failed to get {label} encoder")))?;
    enc.set_pipeline(pipeline);
    bind(&enc);
    // SAFETY: the fullscreen triangle's three vertices are generated from `[[vertex_id]]` in
    // the shader, so the draw reads no vertex buffer.
    unsafe {
        enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
    }
    enc.endEncoding();
    Ok(())
}

impl MtlContext {
    // `encode_fullscreen_pass` against this context's own timing resources.
    pub(in crate::metal) fn fullscreen_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        pass: FullscreenPass,
        bind: impl FnOnce(&ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>),
    ) -> RenderResult<()> {
        encode_fullscreen_pass(cmd_buf, self.diagnostics.pass_timing.as_ref(), pass, bind)
    }
}
