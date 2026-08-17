// src/metal/post/fullscreen.rs
//
// Shared builders for fullscreen-triangle post-process passes. Every
// screen-space effect (SSAO, SSR, SSGI, TAA, bloom, fog, RT reflections, the
// final composite) draws one `[[vertex_id]]`-generated triangle into a single
// colour attachment with no vertex descriptor and no depth, differing only in
// shader source, attachment format, and blend. These helpers fold that shared
// pipeline-descriptor boilerplate into one place so each effect file keeps only
// what is unique to it.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlendFactor, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLDevice as _, MTLLibrary as _,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction,
    MTLTexture,
};

use crate::metal::context::MtlContext;
use crate::metal::pass_timing::PassId;
use crate::metal::pipeline::ns_str;
use crate::metal::slang_shaders::{FULLSCREEN_VERT, SlangLib};

// Blend configuration for a fullscreen pass's single colour attachment.
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
    // folded coverage into its colour, so the source factor is `One`. Used by
    // the volumetric-fog composite.
    PremultipliedOver,
}

// Build a render pipeline state for a fullscreen-triangle post pass: the two
// named functions from `library`, a single colour attachment at `format` with
// the requested `blend`, single-sample, no vertex descriptor, no depth. The
// pipeline-create error is tagged with `fragment_name` so a failure points at
// the exact entry point.
pub(crate) fn build_fullscreen_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    library: &ProtocolObject<dyn objc2_metal::MTLLibrary>,
    vertex_name: &str,
    fragment_name: &str,
    format: MTLPixelFormat,
    blend: FullscreenBlend,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_fullscreen_pipeline_split(
        device,
        FullscreenStages {
            vertex_library: library,
            vertex_name,
            fragment_library: library,
            fragment_name,
        },
        format,
        blend,
    )
}

// The two stages of a fullscreen pass, each with the library it comes from.
// The single-source passes take their vertex from one shared `fullscreen.slang`
// library and their fragment from the effect's own, so the pair cannot be named
// by one library plus two function names.
pub(crate) struct FullscreenStages<'a> {
    pub vertex_library: &'a ProtocolObject<dyn objc2_metal::MTLLibrary>,
    pub vertex_name: &'a str,
    pub fragment_library: &'a ProtocolObject<dyn objc2_metal::MTLLibrary>,
    pub fragment_name: &'a str,
}

// `build_fullscreen_pipeline` with the stages drawn from separate libraries.
pub(crate) fn build_fullscreen_pipeline_split(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    stages: FullscreenStages,
    format: MTLPixelFormat,
    blend: FullscreenBlend,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let FullscreenStages {
        vertex_library,
        vertex_name,
        fragment_library,
        fragment_name,
    } = stages;
    let vert_fn = vertex_library
        .newFunctionWithName(&ns_str(vertex_name))
        .ok_or_else(|| format!("{} not found", vertex_name))?;
    let frag_fn = fragment_library
        .newFunctionWithName(&ns_str(fragment_name))
        .ok_or_else(|| format!("{} not found", fragment_name))?;

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
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
        .map_err(|e| format!("failed to create {} pipeline: {:?}", fragment_name, e))
}

// Build a fullscreen-triangle pipeline whose fragment comes from a
// single-source `.slang` program, paired with the one shared
// `fullscreen_vertex`. The two stages come from separate libraries because a
// fragment variant declares only the resources it binds, so each variant is its
// own metallib while the vertex is compiled once for all of them.
pub(in crate::metal) fn build_slang_fullscreen_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    fragment: &SlangLib,
    format: MTLPixelFormat,
    blend: FullscreenBlend,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let vert = FULLSCREEN_VERT.library(device, hot_reload)?;
    let frag = fragment.library(device, hot_reload)?;
    build_fullscreen_pipeline_split(
        device,
        FullscreenStages {
            vertex_library: &vert,
            vertex_name: "fullscreen_vertex",
            fragment_library: &frag,
            fragment_name: fragment.entries[0],
        },
        format,
        blend,
    )
}

// Bind `sampler` to fragment sampler slots `first..first + count`. The
// single-source post passes declare their inputs as combined texture-samplers,
// which slangc lowers to a texture and a sampler at the same index, so a pass
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
        unsafe { enc.setFragmentSamplerState_atIndex(Some(sampler), i) };
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

// The per-pass setup a fullscreen-triangle encode needs: the colour target it
// writes, that attachment's load action, where the pass sits in the GPU-timing
// span, the pipeline it runs, and the encoder debug label.
pub(in crate::metal) struct FullscreenPass<'a> {
    pub target: &'a ProtocolObject<dyn MTLTexture>,
    pub load: MTLLoadAction,
    pub timer: PassTimer,
    pub pipeline: &'a ProtocolObject<dyn MTLRenderPipelineState>,
    pub label: &'a str,
}

impl MtlContext {
    // Run one fullscreen-triangle pass: open a single-attachment render encoder
    // on `pass.target` (with the given `pass.load` action and an always-`Store`),
    // attach GPU timing per `pass.timer`, set `pass.pipeline`, let `bind` set the
    // pass's fragment inputs, draw the `[[vertex_id]]` triangle, and end
    // encoding. Centralises the encoder open / draw / close skeleton every
    // screen-space effect repeats so each `encode_*` supplies only its unique
    // bindings.
    pub(in crate::metal) fn fullscreen_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        pass: FullscreenPass,
        bind: impl FnOnce(&ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>),
    ) -> Result<(), String> {
        let FullscreenPass {
            target,
            load,
            timer,
            pipeline,
            label,
        } = pass;
        let desc = MTLRenderPassDescriptor::new();
        unsafe {
            let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(target));
            ca.setLoadAction(load);
            ca.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.pass_timing {
            match timer {
                PassTimer::None => {}
                PassTimer::Whole(id) => t.attach_render(&desc, id),
                PassTimer::First(id) => t.attach_render_first(&desc, id),
                PassTimer::Last(id) => t.attach_render_last(&desc, id),
            }
        }
        let enc = cmd_buf
            .renderCommandEncoderWithDescriptor(&desc)
            .ok_or_else(|| format!("failed to get {} encoder", label))?;
        enc.setRenderPipelineState(pipeline);
        bind(&enc);
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        enc.endEncoding();
        Ok(())
    }
}
