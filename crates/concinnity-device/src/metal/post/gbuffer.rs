//! The unified geometry G-buffer pre-pass. One jittered traversal of the cull
//! records (static + instanced + skinned) writes the view-space normal + linear
//! depth, perceptual roughness, and screen-space motion vector that SSR, SSAO,
//! SSGI, RT reflections, TAA, and the MetalFX upscaler all consume, replacing
//! the three separate SSR / SSAO / velocity pre-passes that each re-rasterized
//! the same geometry. Pipeline, targets, and the encoder live together so the
//! effect is a single unit the other backends can mirror.
#![deny(unsafe_op_in_unsafe_fn)]

use crate::metal::depth::CLEAR_DEPTH;
use crate::metal::error::allocation_failed;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shader_programs::metal::prepass_buffers::{
    DRAW_ARGS as PREPASS_DRAW_ARGS_INDEX, PREV_MODELS as PREPASS_PREV_MODELS_INDEX,
    VIEW as PREPASS_VIEW_INDEX,
};
use concinnity_core::render::uniforms::{GBufferView, ViewUniforms};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer as _, MTLDevice as _, MTLLoadAction, MTLPixelFormat,
    MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLStoreAction, MTLTexture, MTLTextureUsage, MTLVertexDescriptor,
    MTLVertexFormat, MTLVertexStepFunction,
};

use crate::metal::context::MtlContext;
use crate::metal::descriptors::{TextureDesc, VertexAttr, VertexLayout, vertex_descriptor};
use crate::metal::encode::RenderEncode;
use crate::metal::scoped_encoder::ScopedEncoder;

// All unified-G-buffer pre-pass state grouped into one unit: the shared
// targets (normal+depth / roughness / velocity / sampleable depth) and the
// model-history kernel, both `Some` when any consumer (SSR / SSGI / RT / SSAO /
// TAA / upscaler) is on. The pipelines that fill the targets are each shader
// bucket's own (`BucketPipelines::prepass`), so the pre-pass runs the vertex
// hook the main pass does.
pub(crate) struct GBufferState {
    pub targets: Option<GBufferTargets>,
    // Snapshot kernel filling this frame's model-history ring slot from the
    // object buffer, for the next frame's motion vectors.
    pub history_pipeline:
        Option<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>>,
}

// Targets

// The pre-pass's feature-owned target: its depth attachment, and only that.
//
// The three color channels (`gbuffer_normal_depth` / `_roughness` /
// `_velocity`) are pool-owned and read back by label through
// `MtlContext::gbuffer_*`, so nothing here holds them -- a pool rebuild repacks
// every slot, and a cached handle would point into memory that now belongs to
// another resource. The depth stays feature-owned to match DirectX and Vulkan
// (there it cannot be pooled: a shader-readable depth target needs a typeless
// resource format `PixelFormat` cannot express).
//
// `Some` when any consumer (SSR, SSGI, RT, SSAO, TAA, or the upscaler) is
// active -- the same gate the pool is built under -- and rebuilt with the HDR
// targets on resize, so no dimensions are stored here.
pub(crate) struct GBufferTargets {
    // `Depth32Float`, single-sample: the pre-pass z-buffer. Unlike the old
    // per-pass prepass depths this is `ShaderRead | RenderTarget` and stored,
    // because the MetalFX upscaler samples it (`setDepthTexture`). The main pass
    // keeps its own MSAA depth; Hi-Z still reduces that, not this.
    pub depth: Retained<ProtocolObject<dyn MTLTexture>>,
}

// Create or recreate the pre-pass depth attachment at `width`x`height`. The
// color channels come from the transient pool, which the caller must have
// built (or rebuilt) at the same extent first.
pub(crate) fn create_gbuffer_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
) -> RenderResult<GBufferTargets> {
    let desc = TextureDesc {
        format: MTLPixelFormat::Depth32Float,
        width: width.max(1) as usize,
        height: height.max(1) as usize,
        // Sampleable (MetalFX reads it), unlike the old prepass depths.
        usage: MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0),
        ..Default::default()
    }
    .build();
    let depth = device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| allocation_failed("G-buffer depth texture"))?;
    Ok(GBufferTargets { depth })
}

// Pipeline

// The color target formats, in attachment order: view normal + linear depth,
// perceptual roughness, screen-space motion.
pub(in crate::metal) const GBUFFER_FORMATS: [MTLPixelFormat; 3] = [
    MTLPixelFormat::RGBA16Float,
    MTLPixelFormat::R8Unorm,
    MTLPixelFormat::RG16Float,
];

// Two-stream vertex descriptor for the G-buffer pre-pass. Stream 0 (buffer 1) is
// the main pass's 56-byte `Vertex` (pos / normal / tangent / color / uv), which
// the vertex hook reads in full; stream 1 (buffer 2) is the PREVIOUS vertex
// position (attribute 5), read from a second buffer the encoder binds (the same
// static VB for the prefix -> zero per-vertex motion, the previous-frame
// deformed buffer for the skinned tail -> per-vertex skin motion). Stream 1's
// stride is the full `Vertex` so the cull-baked `base_vertex` indexes it
// identically to stream 0.
fn gbuffer_prepass_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    let attr = |index, format, offset, buffer_index| VertexAttr {
        index,
        format,
        offset,
        buffer_index,
    };
    let layout = |buffer_index| VertexLayout {
        buffer_index,
        stride: std::mem::size_of::<Vertex>(),
        step: MTLVertexStepFunction::PerVertex,
    };
    vertex_descriptor(
        &[
            attr(0, MTLVertexFormat::Float3, 0, 1),  // pos
            attr(1, MTLVertexFormat::Float3, 12, 1), // normal
            attr(2, MTLVertexFormat::Float3, 24, 1), // tangent
            attr(3, MTLVertexFormat::Float3, 36, 1), // color
            attr(4, MTLVertexFormat::Float2, 48, 1), // uv
            attr(5, MTLVertexFormat::Float3, 0, 2),  // prev pos
        ],
        &[layout(1), layout(2)],
    )
}

// Build one shader bucket's G-buffer pre-pass pipeline over its
// `vertex_prepass_bindless` + `fragment_prepass_bindless` pair: the three
// single-sample MRT targets plus a `Depth32Float` z-buffer, the two-stream
// vertex descriptor, and `supportIndirectCommandBuffers` so it can execute the
// bucket's cull-produced indirect command buffer.
pub(crate) fn build_gbuffer_prepass_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    vert_fn: &ProtocolObject<dyn objc2_metal::MTLFunction>,
    frag_fn: &ProtocolObject<dyn objc2_metal::MTLFunction>,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert_desc = gbuffer_prepass_vertex_descriptor();
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(vert_fn));
    desc.setFragmentFunction(Some(frag_fn));
    desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let targets = desc.colorAttachments();
        for (i, format) in GBUFFER_FORMATS.iter().enumerate() {
            let target = targets.objectAtIndexedSubscript(i);
            target.setPixelFormat(*format);
            target.setBlendingEnabled(false);
        }
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    desc.setSupportIndirectCommandBuffers(true);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| {
            RenderError::ShaderCompile(format!(
                "failed to create G-buffer pre-pass pipeline: {e:?}"
            ))
        })
}

// Encoder

// The view blocks the pre-pass binds: its own (motion matrices and the
// previous clock), the main pass's, which the vertex hook positions the surface
// through, and the raymarch block the SDF volumes march through, with the
// frustum they are culled against.
#[derive(Clone, Copy)]
pub(in crate::metal) struct GbufferPrepassViews<'a> {
    pub gbuffer: &'a GBufferView,
    pub main: &'a ViewUniforms,
    pub raymarch: &'a crate::metal::raymarch::RaymarchView,
    pub frustum: &'a concinnity_core::gfx::frustum::Frustum,
}

// The GPU-driven per-frame buffers the G-buffer pre-pass consumes: the
// cull-produced object records, the parallel previous-frame model matrices, and
// the current + previous-frame deformed skinned vertices. `None` for a world
// with nothing in the cull records, which draws no geometry here.
#[derive(Clone, Copy)]
pub(in crate::metal) struct GbufferGpuBuffers<'a> {
    pub object_buffer: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // The material parameter table a vertex hook may read.
    pub material_params: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // The bindless texture argument buffer the fragment samples the normal,
    // ORM and albedo maps through.
    pub bindless_tex_args: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // The model-history slot the PREVIOUS frame's snapshot filled.
    pub prev_model_buffer: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // This frame's draw args, read by the pre-pass only for `NO_HISTORY`.
    pub draw_args_buffer: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // The model-history slots this frame's snapshot fills: this frame's alone
    // in steady state, every slot on the frame a rebuild primes the ring.
    pub history_targets: &'a [Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>],
    pub deformed_current: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    pub deformed_prev: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // This frame's grass inputs, when the grass kernel ran.
    pub grass: Option<&'a crate::metal::grass::GrassFrame>,
}

impl MtlContext {
    // Whether anything reprojects through the pre-pass's motion channel this
    // frame: TAA, the MetalFX upscaler, or the SSGI accumulation.
    pub(in crate::metal) fn reads_motion(&self) -> bool {
        self.taa.enabled
            || self.upscale.scaler.is_some()
            || (self.ssgi.pass.is_some() && self.ssgi.settings.is_some_and(|s| s.contributes()))
    }

    // Encode the unified G-buffer pre-pass: one jittered traversal of the cull
    // records writing view-space normal + linear depth at color(0), perceptual
    // roughness at color(1), and screen-space motion at color(2), with a
    // sampleable `Depth32Float` z-buffer. Replaces the separate SSR / SSAO /
    // velocity pre-passes; runs before the main pass so the SSAO kernel and main
    // pass can read its output.
    //
    // Always writes all three color targets (the geometry traversal dominates,
    // so the extra R8 + RG16 stores are negligible). `velocity_active` selects
    // whether the static prev-model + skinned prev-pose come from last frame
    // (true) or collapse to the current frame (false): when false the motion
    // channel is a harmless zero that no consumer reads.
    pub(in crate::metal) fn encode_gbuffer_prepass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        views: GbufferPrepassViews<'_>,
        gpu: GbufferGpuBuffers,
        velocity_active: bool,
    ) -> RenderResult<u32> {
        let Some(targets) = &self.gbuffer.targets else {
            return Ok(0);
        };
        // The color channels are pool-owned; the pool is built under the same
        // gate as `targets`, so all three are present whenever it is. A missing
        // one means the pool and the feature disagree about that gate, which
        // would otherwise show up as a pre-pass rendering into nothing.
        let (normal_depth, roughness, velocity) = match (
            self.gbuffer_normal_depth(),
            self.gbuffer_roughness(),
            self.gbuffer_velocity(),
        ) {
            (Some(n), Some(r), Some(v)) => (n, r, v),
            _ => {
                return Err(RenderError::Other(
                    "G-buffer pre-pass: the transient pool is missing a color channel; \
                     its build gate disagrees with the pre-pass's"
                        .to_string(),
                ));
            }
        };

        let desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca0 = desc.colorAttachments().objectAtIndexedSubscript(0);
            ca0.setTexture(Some(normal_depth));
            ca0.setLoadAction(MTLLoadAction::Clear);
            ca0.setStoreAction(MTLStoreAction::Store);
            // Cleared alpha 0 marks "no geometry" for the SSR/SSAO/RT consumers.
            ca0.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });
            let ca1 = desc.colorAttachments().objectAtIndexedSubscript(1);
            ca1.setTexture(Some(roughness));
            ca1.setLoadAction(MTLLoadAction::Clear);
            ca1.setStoreAction(MTLStoreAction::Store);
            // Background roughness 1.0 -> non-reflective, so the border emits no SSR.
            ca1.setClearColor(MTLClearColor {
                red: 1.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });
            let ca2 = desc.colorAttachments().objectAtIndexedSubscript(2);
            ca2.setTexture(Some(velocity));
            ca2.setLoadAction(MTLLoadAction::Clear);
            ca2.setStoreAction(MTLStoreAction::Store);
            // Zero motion for the cleared background.
            ca2.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });
            let da = desc.depthAttachment();
            da.setTexture(Some(targets.depth.as_ref()));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setClearDepth(CLEAR_DEPTH);
            // Stored (not DontCare): the MetalFX upscaler samples this depth.
            da.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&desc, crate::metal::pass_timing::PassId::GBufferPrepass);
        }
        // Kept past the encode below, which consumes `gpu`: the snapshot that
        // fills the next frame's history runs after this frame has read the
        // previous one, so a single frame in flight reads before it overwrites.
        let snapshot = gpu.object_buffer.cloned();
        let history_targets = gpu.history_targets;
        let draw_calls = {
            let enc = ScopedEncoder::new(
                cmd_buf
                    .renderCommandEncoderWithDescriptor(&desc)
                    .ok_or_else(|| {
                        RenderError::Other("failed to get G-buffer pre-pass encoder".to_string())
                    })?,
                ns_string!("g-buffer prepass"),
            );

            // The encoder above cleared all four attachments, so a world with
            // nothing in the cull records still leaves the consumers a clean
            // "no geometry" G-buffer to read.
            let grass = gpu.grass;
            let mut draws =
                self.encode_gbuffer_prepass_gpu_driven(&enc, views, gpu, velocity_active);
            if let Some(frame) = grass {
                draws += self.encode_grass_prepass(&enc, frame, views.main, views.gbuffer);
            }
            draws += self.encode_raymarch_prepass(&enc, views.raymarch, views.frustum);
            // The sky keeps the "no geometry" depth and roughness and adds the
            // camera's motion where nothing was drawn.
            if self.draws_sky(self.state.view.mode) {
                self.encode_sky_velocity(&enc, views.gbuffer);
            }
            draws
        };
        if let Some(objects) = snapshot.as_ref() {
            self.encode_model_history(cmd_buf, objects, history_targets, self.cull_count())?;
        }
        Ok(draw_calls)
    }

    // GPU-driven G-buffer pre-pass: draw the SAME per-frame indirect command
    // set the bindless main pass executes, each shader bucket's ICB under that
    // bucket's pre-pass pipeline, so a world Shader's vertex hook places its
    // depth and motion as it places its shading. Mirrors
    // `execute_bindless_static_icb`'s two-range split -- the static + instance +
    // chunk prefix `[0, skinned_record_base())` over the static VB, then the
    // folded skinned tail `[skinned_record_base(), cull_count())` over the
    // deformed VB + skinned IB -- but reuses the PHASE-1 `cull.icbs` (the
    // pre-pass runs before Cull2/Main2, so phase-1 coverage is the natural
    // source; the camera frustum is identical to the main pass, so no extra
    // cull dispatch is needed). The previous vertex position rides a second
    // vertex stream (binding 2): the static VB for the prefix (prev_pos ==
    // cur_pos -> model-delta motion), the previous-frame deformed buffer for the
    // skinned tail (per-vertex skin motion). Returns the indirect draw count.
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        views: GbufferPrepassViews<'_>,
        gpu: GbufferGpuBuffers,
        velocity_active: bool,
    ) -> u32 {
        use objc2_metal::{MTLRenderStages, MTLResourceUsage};
        use std::sync::atomic::Ordering;
        let GbufferGpuBuffers {
            object_buffer,
            material_params,
            bindless_tex_args,
            prev_model_buffer,
            draw_args_buffer,
            history_targets: _,
            deformed_current,
            deformed_prev,
            grass: _,
        } = gpu;
        let (
            Some(default),
            Some(object_buffer),
            Some(tex_args),
            Some(prev_models),
            Some(draw_args),
        ) = (
            self.cull.main_pipeline.as_ref(),
            object_buffer,
            bindless_tex_args,
            prev_model_buffer,
            draw_args_buffer,
        )
        else {
            return 0;
        };
        if self.cull.icbs.is_empty() {
            return 0;
        }
        enc.set_depth_stencil(&self.targets.depth_state);
        // The main pass's view block, records and parameter table at their
        // main-pass slots, the pre-pass's own view block at buffer(3), model
        // history at buffer(17) and draw args at buffer(18), and the two vertex
        // streams at buffer(1) and buffer(2). The ICB commands inherit these
        // bindings; the cull baked base_instance = record id. The prefix binds
        // the static VB to BOTH streams (prev_pos == cur_pos), so its motion is
        // the model delta.
        enc.set_vertex_value(views.main, 0);
        enc.set_vertex_value(views.gbuffer, PREPASS_VIEW_INDEX);
        enc.set_fragment_value(views.gbuffer, PREPASS_VIEW_INDEX);
        enc.set_vertex_buffer(object_buffer, 0, 9);
        enc.set_fragment_buffer(object_buffer, 0, 9);
        if let Some(params) = material_params {
            enc.set_vertex_buffer(
                params,
                0,
                crate::metal::material_params::MATERIAL_PARAMS_BUFFER_INDEX,
            );
        }
        enc.set_vertex_buffer(prev_models, 0, PREPASS_PREV_MODELS_INDEX);
        enc.set_vertex_buffer(draw_args, 0, PREPASS_DRAW_ARGS_INDEX);
        enc.set_fragment_buffer(
            tex_args,
            0,
            crate::metal::context::BINDLESS_TEXTURE_ARG_BUFFER_INDEX,
        );
        if let Some(sampler_args) = &self.arg_buffers.bindless_sampler_args {
            enc.set_fragment_buffer(
                sampler_args,
                0,
                crate::metal::context::BINDLESS_SAMPLER_ARG_BUFFER_INDEX,
            );
        }
        self.use_bindless_textures(enc);
        enc.set_vertex_buffer(&self.scene.vertex_buffer, 0, 1);
        enc.set_vertex_buffer(&self.scene.vertex_buffer, 0, 2);

        let counts = self.draw_record_counts();
        let mut draw_calls = 0u32;

        // Static + instance + chunk prefix: static u32 IB resident. Each
        // bucket's ICB executes under that bucket's pre-pass pipeline; together
        // the buckets cover the whole record range exactly once. A bucket the
        // main pass skips (Shader not resident) is skipped here too, so depth
        // and velocity never carry geometry the color pass leaves out.
        if let Some(prefix) = counts.prefix(0) {
            enc.useResource_usage_stages(
                ProtocolObject::from_ref(&*self.scene.index_buffer),
                MTLResourceUsage::Read,
                MTLRenderStages::Vertex,
            );
            let range = crate::metal::context::ns_range(prefix);
            // A bucket without a pre-pass pipeline (one failed to build) is
            // skipped too: it shades, but adds nothing to the G-buffer.
            for (b, icb) in self.cull.icbs.iter().enumerate() {
                let pipelines = match b {
                    0 => Some(default),
                    b => self.cull.world_pipelines.get(b),
                };
                let Some(prepass) = pipelines.and_then(|p| p.prepass.as_ref()) else {
                    continue;
                };
                enc.set_pipeline(prepass);
                // SAFETY: the prefix spans the static + instance + chunk command
                // slots; every reused main ICB is sized for `counts.total`.
                unsafe {
                    enc.executeCommandsInBuffer_withRange(icb, range);
                }
                draw_calls += 1;
            }
        }

        // Folded skinned tail: current deformed at stream 0, previous-frame
        // deformed at stream 1. Until the deformed ring is primed (frame 0 /
        // after a rebuild), or with velocity inactive / a single frame in
        // flight, bind the CURRENT buffer as the previous one -> zero skinned
        // motion (no garbage motion vector from an unposed prior slot).
        if let (Some(deformed), Some(tail), Some(prepass)) = (
            deformed_current,
            counts.skinned_tail(0),
            default.prepass.as_ref(),
        ) {
            let prev = if velocity_active
                && self.frames_in_flight >= 2
                && self.skinned.deformed_primed.load(Ordering::Relaxed)
            {
                deformed_prev.unwrap_or(deformed)
            } else {
                deformed
            };
            // Skinned records are always bucket 0.
            enc.set_pipeline(prepass);
            enc.set_vertex_buffer(deformed, 0, 1);
            enc.set_vertex_buffer(prev, 0, 2);
            if let Some(skinned_ib) = self.skinned.index_buffer.as_ref() {
                enc.useResource_usage_stages(
                    ProtocolObject::from_ref(&**skinned_ib),
                    MTLResourceUsage::Read,
                    MTLRenderStages::Vertex,
                );
            }
            // SAFETY: the tail spans the folded skinned command slots.
            unsafe {
                enc.executeCommandsInBuffer_withRange(
                    &self.cull.icbs[0],
                    crate::metal::context::ns_range(tail),
                );
            }
            draw_calls += 1;
            // The current deformed slot now holds a valid pose, so next frame's
            // previous-frame read is well-defined. Relaxed: the only other access
            // is the next frame's same-pass load, ordered by the render-graph
            // scope join between frames; no other pass touches this flag.
            self.skinned.deformed_primed.store(true, Ordering::Relaxed);
        }
        draw_calls
    }
}
