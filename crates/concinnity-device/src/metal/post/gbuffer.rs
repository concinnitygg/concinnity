//! The unified geometry G-buffer pre-pass. One jittered traversal of the cull
//! records (static + instanced + skinned) writes the view-space normal + linear
//! depth, perceptual roughness, and screen-space motion vector that SSR, SSAO,
//! SSGI, RT reflections, TAA, and the `MetalFX` upscaler all consume, replacing
//! the three separate SSR / SSAO / velocity pre-passes that each re-rasterized
//! the same geometry. Pipeline, targets, and the encoder live together so the
//! effect is a single unit the other backends can mirror.
#![deny(unsafe_op_in_unsafe_fn)]

use crate::metal::depth::CLEAR_DEPTH;
use crate::metal::error::allocation_failed;
use concinnity_engine::gfx::mesh_payload::Vertex;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::shader::programs::abi::main::metal as main;
use concinnity_engine::render::uniforms::{GBufferView, ViewUniforms};
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

// The G-buffer pre-pass's targets and model-history kernel, `Some` when any
// consumer is on. Each shader bucket has its own pre-pass pipeline, so the
// vertex hook runs here too.
pub(crate) struct GBufferState {
    pub targets: Option<GBufferTargets>,
    // Snapshot kernel filling this frame's model-history ring slot from the
    // object buffer, for the next frame's motion vectors.
    pub history_pipeline:
        Option<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>>,
}

// Targets

// The pre-pass's depth attachment. The color channels are pool-owned and read
// back by label, since a pool rebuild repacks every slot. Rebuilt with the HDR
// targets on resize.
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
// perceptual roughness + specular F0, screen-space motion.
pub(in crate::metal) const GBUFFER_FORMATS: [MTLPixelFormat; 3] = [
    MTLPixelFormat::RGBA16Float,
    MTLPixelFormat::RGBA8Unorm,
    MTLPixelFormat::RG16Float,
];

// Two-stream vertex descriptor: stream 0 (buffer 1) is the full `Vertex`;
// stream 1 (buffer 2) is the previous position at the same stride, so the
// cull-baked `base_vertex` indexes both.
fn gbuffer_prepass_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    let attr = |index, format, offset, buffer_index| VertexAttr {
        index,
        format,
        offset,
        buffer_index,
    };
    let layout = |buffer_index| VertexLayout {
        buffer_index,
        stride: size_of::<Vertex>(),
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

// One bucket's pre-pass pipeline: three MRT targets plus `Depth32Float`, the
// two-stream descriptor, and ICB support.
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
    pub frustum: &'a concinnity_engine::gfx::frustum::Frustum,
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

// Whether a G-buffer pass starts the G-buffer afresh or draws over what the
// pre-pass left.
#[derive(Clone, Copy)]
enum GbufferLoad {
    Clear,
    Load,
}

impl GbufferLoad {
    // The timing slot the pass records into.
    fn timing(self) -> crate::metal::pass_timing::PassId {
        match self {
            Self::Clear => crate::metal::pass_timing::PassId::GBufferPrepass,
            Self::Load => crate::metal::pass_timing::PassId::GBufferLate,
        }
    }
}

// The cull lists a G-buffer pass draws: each bucket's ICB and where each
// stream's execution range sits in them.
#[derive(Clone, Copy)]
struct PrepassLists<'a> {
    icbs: &'a [Retained<ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>],
    regions: crate::metal::cull::BucketRegions<'a>,
}

impl MtlContext {
    // Encode the G-buffer pre-pass: normal and linear depth, roughness and
    // motion, plus a sampleable depth, before the main pass. Without velocity
    // the previous model and pose collapse to this frame's, so motion is zero.
    pub(in crate::metal) fn encode_gbuffer_prepass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        views: GbufferPrepassViews<'_>,
        gpu: GbufferGpuBuffers<'_>,
        velocity_active: bool,
    ) -> RenderResult<u32> {
        let Some(desc) = self.gbuffer_pass_desc(GbufferLoad::Clear)? else {
            return Ok(0);
        };
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
            let lists = PrepassLists {
                icbs: &self.cull.icbs,
                regions: self.main_regions(),
            };
            let mut draws =
                self.encode_gbuffer_prepass_gpu_driven(&enc, views, gpu, velocity_active, lists);
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
        Ok(draw_calls)
    }

    // Snapshot this frame's models into the history slots `gpu` names, for the
    // next frame's motion. It runs after the last G-buffer pass that reads the
    // previous frame's slot, so a single frame in flight reads it before it is
    // overwritten.
    pub(in crate::metal) fn encode_model_history_snapshot(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        gpu: GbufferGpuBuffers<'_>,
    ) -> RenderResult<()> {
        let Some(objects) = gpu.object_buffer else {
            return Ok(());
        };
        self.encode_model_history(cmd_buf, objects, gpu.history_targets, self.cull_count())
    }

    // Encode the late G-buffer pass: the late lists two-pass occlusion drew in
    // `MainLate`, drawn over the pre-pass's G-buffer so the readers after the
    // main pass see every surface it shows.
    pub(in crate::metal) fn encode_gbuffer_late(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        views: GbufferPrepassViews<'_>,
        gpu: GbufferGpuBuffers<'_>,
        velocity_active: bool,
    ) -> RenderResult<u32> {
        if self.cull.icbs_late.is_empty() {
            return Ok(0);
        }
        let Some(desc) = self.gbuffer_pass_desc(GbufferLoad::Load)? else {
            return Ok(0);
        };
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get late G-buffer encoder".to_string())
                })?,
            ns_string!("g-buffer late"),
        );
        let lists = PrepassLists {
            icbs: &self.cull.icbs_late,
            regions: crate::metal::cull::BucketRegions {
                ranges: self.cull.ranges_late.as_deref(),
                ..self.main_regions()
            },
        };
        Ok(self.encode_gbuffer_prepass_gpu_driven(&enc, views, gpu, velocity_active, lists))
    }

    // The pass descriptor over the G-buffer's four attachments, cleared to "no
    // geometry" or loaded, all stored. `None` without the targets.
    fn gbuffer_pass_desc(
        &self,
        load: GbufferLoad,
    ) -> RenderResult<Option<Retained<MTLRenderPassDescriptor>>> {
        let Some(targets) = &self.gbuffer.targets else {
            return Ok(None);
        };
        // The color channels are pool-owned; the pool is built under the same
        // gate as `targets`, so all three are present whenever it is. A missing
        // one means the pool and the feature disagree about that gate, which
        // would otherwise show up as a pre-pass rendering into nothing.
        let (Some(normal_depth), Some(roughness), Some(velocity)) = (
            self.gbuffer_normal_depth(),
            self.gbuffer_roughness(),
            self.gbuffer_velocity(),
        ) else {
            return Err(RenderError::Other(
                "G-buffer pre-pass: the transient pool is missing a color channel; \
                     its build gate disagrees with the pre-pass's"
                    .to_string(),
            ));
        };
        let action = match load {
            GbufferLoad::Clear => MTLLoadAction::Clear,
            GbufferLoad::Load => MTLLoadAction::Load,
        };
        let clear = |red| MTLClearColor {
            red,
            green: 0.0,
            blue: 0.0,
            alpha: 0.0,
        };
        // Alpha 0 marks "no geometry" for the SSR/SSAO/RT consumers, roughness
        // 1.0 leaves the background non-reflective, and the background has no
        // motion.
        let channels = [
            (normal_depth, clear(0.0)),
            (roughness, clear(1.0)),
            (velocity, clear(0.0)),
        ];
        let desc = MTLRenderPassDescriptor::new();
        for (i, (texture, color)) in channels.into_iter().enumerate() {
            // SAFETY: slots 0 to 2 are within the descriptor's eight color attachments.
            let ca = unsafe { desc.colorAttachments().objectAtIndexedSubscript(i) };
            ca.setTexture(Some(texture));
            ca.setLoadAction(action);
            ca.setStoreAction(MTLStoreAction::Store);
            ca.setClearColor(color);
        }
        let da = desc.depthAttachment();
        da.setTexture(Some(targets.depth.as_ref()));
        da.setLoadAction(action);
        da.setClearDepth(CLEAR_DEPTH);
        // Stored (not DontCare): the MetalFX upscaler samples this depth.
        da.setStoreAction(MTLStoreAction::Store);
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&desc, load.timing());
        }
        Ok(Some(desc))
    }

    // Execute each bucket's ICB in `lists` under its pre-pass pipeline, split
    // like `execute_bindless_static_icb`: the static prefix with the static VB
    // on both streams, then the skinned tail with the previous deformed buffer
    // on stream 1. Returns the draws issued.
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        views: GbufferPrepassViews<'_>,
        gpu: GbufferGpuBuffers<'_>,
        velocity_active: bool,
        lists: PrepassLists<'_>,
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
        let Some(first_icb) = lists.icbs.first() else {
            return 0;
        };
        enc.set_depth_stencil(&self.targets.depth_state);
        // The main pass's view, records and parameter table at their slots, the
        // pre-pass view at buffer 3, model history at 17, draw args at 18, and
        // the vertex streams at 1 and 2. ICB commands inherit these.
        enc.set_vertex_value(views.main, main::VIEW.index);
        enc.set_vertex_value(views.gbuffer, main::GBUFFER_VIEW.index);
        enc.set_fragment_value(views.gbuffer, main::GBUFFER_VIEW.index);
        enc.set_vertex_buffer(object_buffer, 0, main::OBJECTS.index);
        enc.set_fragment_buffer(object_buffer, 0, main::OBJECTS.index);
        if let Some(params) = material_params {
            enc.set_vertex_buffer(
                params,
                0,
                crate::metal::material_params::MATERIAL_PARAMS_BUFFER_INDEX,
            );
        }
        enc.set_vertex_buffer(prev_models, 0, main::PREV_MODELS.index);
        enc.set_vertex_buffer(draw_args, 0, main::DRAW_ARGS.index);
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
        enc.set_vertex_buffer(&self.scene.vertex_buffer, 0, main::MESH_VERTICES.index);

        let counts = self.draw_record_counts();
        let regions = lists.regions;
        let mut draw_calls = 0u32;

        // The prefix per bucket; a bucket the main pass skips is skipped here
        // too.
        if counts.prefix(0).is_some() {
            enc.useResource_usage_stages(
                ProtocolObject::from_ref(&*self.scene.index_buffer),
                MTLResourceUsage::Read,
                MTLRenderStages::Vertex,
            );
            // A bucket without a pre-pass pipeline (one failed to build) is
            // skipped too: it shades, but adds nothing to the G-buffer.
            for (b, icb) in lists.icbs.iter().enumerate() {
                if !regions.draws(b) {
                    continue;
                }
                let pipelines = match b {
                    0 => Some(default),
                    b => self.cull.world_pipelines.get(b),
                };
                let Some(prepass) = pipelines.and_then(|p| p.prepass.as_ref()) else {
                    continue;
                };
                enc.set_pipeline(prepass);
                draw_calls += crate::metal::draw::main::execute_list(
                    enc,
                    icb,
                    regions,
                    concinnity_core::math::saturating_u32(b),
                );
            }
        }

        // Until the ring is primed, or without velocity, bind the current
        // deformed buffer as the previous one so motion is zero.
        if let (Some(deformed), Some(_), Some(prepass)) = (
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
            draw_calls += crate::metal::draw::main::execute_list(
                enc,
                first_icb,
                regions,
                concinnity_engine::gfx::render_types::SKINNED_STREAM,
            );
            // The current deformed slot now holds a valid pose, so next frame's
            // previous-frame read is well-defined. Relaxed: the only other access
            // is the next frame's same-pass load, ordered by the render-graph
            // scope join between frames; no other pass touches this flag.
            self.skinned.deformed_primed.store(true, Ordering::Relaxed);
        }
        draw_calls
    }
}
