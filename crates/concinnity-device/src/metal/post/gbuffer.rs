// src/metal/post/gbuffer.rs
//
// The unified geometry G-buffer pre-pass. One jittered traversal of the visible
// set (static + instanced + skinned) writes the view-space normal + linear
// depth, perceptual roughness, and screen-space motion vector that SSR, SSAO,
// SSGI, RT reflections, TAA, and the MetalFX upscaler all consume, replacing
// the three separate SSR / SSAO / velocity pre-passes that each re-rasterized
// the same geometry. Pipelines, targets, and the encoder live together so the
// effect is a single unit the other backends can mirror.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer as _, MTLDevice as _, MTLLoadAction, MTLPixelFormat,
    MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLStoreAction, MTLTexture, MTLTextureUsage, MTLVertexDescriptor,
    MTLVertexFormat, MTLVertexStepFunction,
};

use crate::gfx::mesh_payload::Vertex;

use crate::metal::context::MtlContext;
use crate::metal::descriptors::{TextureDesc, VertexAttr, VertexLayout, vertex_descriptor};
use crate::metal::encode::RenderEncode;
use crate::metal::scoped_encoder::ScopedEncoder;
use crate::metal::slang_shaders::{self, SlangLib};
use crate::metal::uniforms::SsrPrepassMat;
use concinnity_render::uniforms::GBufferModel;
use concinnity_render::uniforms::GBufferView;

// All unified-G-buffer pre-pass state grouped into one unit: the shared
// targets (normal+depth / roughness / velocity / sampleable depth) plus the
// per-geometry-kind pipelines. `targets`/`prepass_pipeline` are `Some` when
// any consumer (SSR / SSGI / RT / SSAO / TAA / upscaler) is on;
// `instanced_pipeline` only when the world has GPU-instanced clusters; and
// `skinned_pipeline` is filled later by `upload_skinned` (80-byte layout).
pub(crate) struct GBufferState {
    pub targets: Option<GBufferTargets>,
    pub prepass_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub instanced_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub skinned_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // GPU-driven bindless variant: one unified pipeline that draws the
    // SAME per-frame indirect command set the bindless main pass executes, so the
    // G-buffer feeder goes fully GPU-driven for static / instanced / chunk /
    // skinned geometry (no CPU draw loop). `Some` only when the world is bindless
    // and a G-buffer consumer is active (`bindless && targets.is_some()`);
    // non-bindless / custom-shader worlds leave it `None` and keep the legacy
    // per-geometry-kind CPU loops above. Rebuilt by `reload_shaders`.
    pub bindless_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
}

// Targets

// The pre-pass's feature-owned target: its depth attachment, and only that.
//
// The three colour channels (`gbuffer_normal_depth` / `_roughness` /
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
// colour channels come from the transient pool, which the caller must have
// built (or rebuilt) at the same extent first.
pub(crate) fn create_gbuffer_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
) -> Result<GBufferTargets, String> {
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
        .ok_or("failed to create G-buffer depth texture")?;
    Ok(GBufferTargets { depth })
}

// Pipeline

// Build one G-buffer pre-pass pipeline for the given single-source vertex
// variant (`GBUFFER_PREPASS_VERT{,_INSTANCED,_SKINNED}`). All three share
// `GBUFFER_PREPASS_FRAG` and render to three single-sample MRT targets
// (`RGBA16Float` normal+depth, `R8Unorm` roughness, `RG16Float` velocity) plus
// a `Depth32Float` z-buffer. `vert_desc` selects the static (56-byte) or
// skinned (80-byte) vertex layout. Each entry compiles to its own metallib, so
// the two stages come from separate libraries and pair by semantic.
pub(crate) fn build_gbuffer_prepass_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    vertex: &SlangLib,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let vert_fn = slang_shaders::entry_function(device, vertex, hot_reload)?;
    let frag_fn =
        slang_shaders::entry_function(device, &slang_shaders::GBUFFER_PREPASS_FRAG, hot_reload)?;

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let ca0 = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca0.setPixelFormat(MTLPixelFormat::RGBA16Float);
        ca0.setBlendingEnabled(false);
        let ca1 = desc.colorAttachments().objectAtIndexedSubscript(1);
        ca1.setPixelFormat(MTLPixelFormat::R8Unorm);
        ca1.setBlendingEnabled(false);
        let ca2 = desc.colorAttachments().objectAtIndexedSubscript(2);
        ca2.setPixelFormat(MTLPixelFormat::RG16Float);
        ca2.setBlendingEnabled(false);
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create G-buffer pre-pass pipeline: {:?}", e))
}

// Two-stream vertex descriptor for the GPU-driven bindless G-buffer pipeline.
// Stream 0 (buffer 1) is the standard 56-byte `Vertex` (pos / normal / tangent /
// colour / uv) the cull-baked indirect commands draw; stream 1 (buffer 2) is the
// PREVIOUS vertex position (attribute 5), read from a second buffer the encoder
// binds (the same static VB for the prefix -> zero per-vertex motion, the
// previous-frame deformed buffer for the skinned tail -> per-vertex skin motion).
// Stream 1 reads only position at offset 0; its stride is the full 56-byte
// `Vertex` so the cull-baked `base_vertex` indexes it identically to stream 0.
pub(crate) fn gbuffer_bindless_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    // Stream 0 (buffer 1): the attributes the bindless VS reads (pos, normal,
    // colour for the skybox sentinel). Tangent/uv are unused by the G-buffer.
    // Stream 1 (buffer 2): previous vertex position only.
    vertex_descriptor(
        &[
            VertexAttr {
                index: 0,
                format: MTLVertexFormat::Float3,
                offset: 0,
                buffer_index: 1,
            }, // pos
            VertexAttr {
                index: 1,
                format: MTLVertexFormat::Float3,
                offset: 12,
                buffer_index: 1,
            }, // normal
            VertexAttr {
                index: 3,
                format: MTLVertexFormat::Float3,
                offset: 36,
                buffer_index: 1,
            }, // color
            VertexAttr {
                index: 5,
                format: MTLVertexFormat::Float3,
                offset: 0,
                buffer_index: 2,
            }, // prev pos
        ],
        &[
            VertexLayout {
                buffer_index: 1,
                stride: std::mem::size_of::<Vertex>(),
                step: MTLVertexStepFunction::PerVertex,
            },
            VertexLayout {
                buffer_index: 2,
                stride: std::mem::size_of::<Vertex>(),
                step: MTLVertexStepFunction::PerVertex,
            },
        ],
    )
}

// Build the GPU-driven bindless G-buffer pre-pass pipeline:
// `gbuffer_prepass_vertex_bindless` + `gbuffer_prepass_fragment_bindless`, the
// same 3-MRT + Depth32 targets as the legacy pipeline, but with the two-stream
// vertex descriptor and `supportIndirectCommandBuffers` so it can execute the
// shared cull-produced indirect command buffer. Reads each record's model +
// roughness from the GpuObjectData buffer by `[[base_instance]]`.
pub(crate) fn build_gbuffer_bindless_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let vert_fn = slang_shaders::entry_function(
        device,
        &slang_shaders::GBUFFER_PREPASS_VERT_BINDLESS,
        hot_reload,
    )?;
    let frag_fn = slang_shaders::entry_function(
        device,
        &slang_shaders::GBUFFER_PREPASS_FRAG_BINDLESS,
        hot_reload,
    )?;

    let vert_desc = gbuffer_bindless_vertex_descriptor();
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let ca0 = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca0.setPixelFormat(MTLPixelFormat::RGBA16Float);
        ca0.setBlendingEnabled(false);
        let ca1 = desc.colorAttachments().objectAtIndexedSubscript(1);
        ca1.setPixelFormat(MTLPixelFormat::R8Unorm);
        ca1.setBlendingEnabled(false);
        let ca2 = desc.colorAttachments().objectAtIndexedSubscript(2);
        ca2.setPixelFormat(MTLPixelFormat::RG16Float);
        ca2.setBlendingEnabled(false);
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    desc.setSupportIndirectCommandBuffers(true);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create G-buffer bindless pipeline: {:?}", e))
}

// Encoder

// The CPU draw-loop inputs the legacy (non-bindless) G-buffer pre-pass walks:
// the visible set, the camera position for the LOD pick, the prepared instanced
// clusters, and the skinned pose pair (current + previous-frame joint palettes;
// the previous drives skinned motion vectors).
pub(in crate::metal) struct GbufferSceneInputs<'a> {
    pub visible: &'a [u32],
    pub cam_pos: [f32; 3],
    pub prepared_instances: &'a super::super::instanced::PreparedInstances,
    pub cur_joint_bufs: &'a [Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>],
    pub prev_joint_bufs: &'a [Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>],
}

// The GPU-driven per-frame buffers the bindless G-buffer pre-pass consumes: the
// cull-produced object records, the parallel previous-frame model matrices, and
// the current + previous-frame deformed skinned vertices. All `Some` together on
// the bindless path; the legacy path leaves them `None` and takes the CPU loop.
#[derive(Clone, Copy)]
pub(in crate::metal) struct GbufferGpuBuffers<'a> {
    pub object_buffer: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    pub prev_model_buffer: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    pub deformed_current: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    pub deformed_prev: Option<&'a Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
}

impl MtlContext {
    // Encode the unified G-buffer pre-pass: one jittered traversal of the
    // visible set (static, GPU-instanced, then skinned) writing view-space
    // normal + linear depth at color(0), perceptual roughness at color(1), and
    // screen-space motion at color(2), with a sampleable `Depth32Float`
    // z-buffer. Replaces the separate SSR / SSAO / velocity pre-passes; runs
    // before the main pass so the SSAO kernel and main pass can read its output.
    //
    // Always writes all three color targets (the geometry traversal dominates,
    // so the extra R8 + RG16 stores are negligible). `velocity_active` selects
    // whether the static prev-model + skinned prev-pose come from last frame
    // (true) or collapse to the current frame (false): when false the motion
    // channel is a harmless zero that no consumer reads.
    pub(in crate::metal) fn encode_gbuffer_prepass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        view: &GBufferView,
        scene: GbufferSceneInputs,
        gpu: GbufferGpuBuffers,
        velocity_active: bool,
    ) -> Result<u32, String> {
        let GbufferSceneInputs {
            visible,
            cam_pos,
            prepared_instances,
            cur_joint_bufs,
            prev_joint_bufs,
        } = scene;
        let (targets, static_ps) = match (&self.gbuffer.targets, &self.gbuffer.prepass_pipeline) {
            (Some(t), Some(p)) => (t, p),
            _ => return Ok(0),
        };
        // The colour channels are pool-owned; the pool is built under the same
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
                return Err(
                    "G-buffer pre-pass: the transient pool is missing a colour channel; \
                     its build gate disagrees with the pre-pass's"
                        .to_string(),
                );
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
            da.setClearDepth(1.0);
            // Stored (not DontCare): the MetalFX upscaler samples this depth.
            da.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.pass_timing {
            t.attach_render(&desc, crate::metal::pass_timing::PassId::GBufferPrepass);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&desc)
                .ok_or("failed to get G-buffer pre-pass encoder")?,
            "g-buffer prepass",
        );

        // GPU-driven path: when the world is bindless (the cull
        // produced an object buffer) and the bindless G-buffer pipeline exists,
        // draw the SAME per-frame indirect command set the main pass executes,
        // with no CPU draw loop. Mirrors the main pass's two-range ICB split.
        if self.gbuffer.bindless_pipeline.is_some() && gpu.object_buffer.is_some() {
            let draws = self.encode_gbuffer_prepass_gpu_driven(&enc, view, gpu, velocity_active);
            return Ok(draws);
        }

        enc.set_pipeline(static_ps);
        enc.set_depth_stencil(&self.depth_state);
        enc.set_vertex_value(view, 0);
        enc.set_vertex_buffer(&self.vertex_buffer, 0, 1);

        // Static geometry: model (cur + prev for motion) at vertex(2), roughness
        // at fragment(0). prev_model collapses to cur when velocity is inactive.
        let mut draws = self.draw_static_objects(&enc, visible, cam_pos, |enc, obj, idx| {
            let model = GBufferModel {
                cur_model: obj.model,
                prev_model: if velocity_active {
                    self.prev_draw_models[idx]
                } else {
                    obj.model
                },
            };
            let mat = SsrPrepassMat {
                roughness: obj.material.roughness,
                _pad: [0.0; 3],
            };
            enc.set_vertex_value(&model, 2);
            enc.set_fragment_value(&mat, 0);
        });

        // GPU-instanced clusters: instance transforms are immutable, so binding
        // the same buffer as cur + prev (`bind_prev`) yields zero motion;
        // roughness is cluster-wide.
        if let Some(inst_ps) = &self.gbuffer.instanced_pipeline
            && !prepared_instances.clusters.is_empty()
        {
            enc.set_pipeline(inst_ps);
            draws +=
                self.draw_prepared_instances(&enc, prepared_instances, true, |enc, cluster| {
                    let mat = SsrPrepassMat {
                        roughness: cluster.material.roughness,
                        _pad: [0.0; 3],
                    };
                    enc.set_fragment_value(&mat, 0);
                });
        }

        // Skinned meshes: current pose at buffer(8), previous at buffer(9) (falls
        // back to the current pose when no previous buffer exists -> zero motion).
        // The model matrix is static, so cur == prev model.
        if let (Some(skinned_ps), Some(svb), Some(sib)) = (
            &self.gbuffer.skinned_pipeline,
            &self.skinned.vertex_buffer,
            &self.skinned.index_buffer,
        ) && !self.skinned.draw_objects.is_empty()
        {
            enc.set_pipeline(skinned_ps);
            enc.set_vertex_buffer(svb, 0, 1);
            draws += self.draw_skinned_objects(&enc, sib, cam_pos, |enc, obj, i| {
                let model = GBufferModel {
                    cur_model: obj.model,
                    prev_model: obj.model,
                };
                let mat = SsrPrepassMat {
                    roughness: obj.material.roughness,
                    _pad: [0.0; 3],
                };
                let prev = prev_joint_bufs.get(i).unwrap_or(&cur_joint_bufs[i]);
                enc.set_vertex_value(&model, 2);
                enc.set_fragment_value(&mat, 0);
                enc.set_vertex_buffer(&cur_joint_bufs[i], 0, 8);
                enc.set_vertex_buffer(prev, 0, 9);
            });
        }
        Ok(draws)
    }

    // GPU-driven G-buffer pre-pass: draw the SAME per-frame indirect
    // command set the bindless main pass executes, with the unified bindless
    // G-buffer pipeline. Mirrors `execute_bindless_static_icb`'s two-range split
    // -- the static + instance + chunk prefix `[0, skinned_record_base())` over
    // the static VB, then the folded skinned tail `[skinned_record_base(),
    // cull_count())` over the deformed VB + skinned u16 IB -- but reuses the
    // PHASE-1 `cull.icb` (the pre-pass runs before Cull2/Main2, so phase-1
    // coverage is the natural source; the camera frustum is identical to the main
    // pass, so no extra cull dispatch is needed). The previous vertex position
    // rides a second vertex stream (binding 2): the static VB for the prefix
    // (prev_pos == cur_pos -> model-delta motion), the previous-frame deformed
    // buffer for the skinned tail (per-vertex skin motion). Returns the indirect
    // draw count (0-2).
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view: &GBufferView,
        gpu: GbufferGpuBuffers,
        velocity_active: bool,
    ) -> u32 {
        use objc2_metal::{MTLRenderStages, MTLResourceUsage};
        use std::sync::atomic::Ordering;
        let GbufferGpuBuffers {
            object_buffer,
            prev_model_buffer,
            deformed_current,
            deformed_prev,
        } = gpu;
        let (Some(pipeline), Some(object_buffer), Some(prev_models)) = (
            self.gbuffer.bindless_pipeline.as_ref(),
            object_buffer,
            prev_model_buffer,
        ) else {
            return 0;
        };
        if self.cull.icbs.is_empty() {
            return 0;
        }
        enc.set_pipeline(pipeline);
        enc.set_depth_stencil(&self.depth_state);
        // GBufferView (vbuf 0), current vertex stream (vbuf 1), previous
        // vertex stream (vbuf 2), object records (vbuf 9), prev_model parallel
        // buffer (vbuf 10). The ICB commands inherit these bindings; the cull
        // baked base_instance = record id, so the VS reads objects[id].model
        // + prev_models[id]. The prefix binds the static VB to BOTH streams
        // (prev_pos == cur_pos), so its motion is purely the model delta.
        enc.set_vertex_value(view, 0);
        enc.set_vertex_buffer(object_buffer, 0, 9);
        enc.set_vertex_buffer(prev_models, 0, 10);
        enc.set_vertex_buffer(&self.vertex_buffer, 0, 1);
        enc.set_vertex_buffer(&self.vertex_buffer, 0, 2);

        let counts = self.draw_record_counts();
        let mut draw_calls = 0u32;

        // Static + instance + chunk prefix: static u32 IB resident.
        if let Some(prefix) = counts.prefix(0) {
            enc.useResource_usage_stages(
                ProtocolObject::from_ref(&*self.index_buffer),
                MTLResourceUsage::Read,
                MTLRenderStages::Vertex,
            );
            let range = crate::metal::context::ns_range(prefix);
            // The pre-pass writes normals/depth/velocity under its single
            // engine pipeline, so every bucket's ICB executes with the same
            // PSO; together the buckets cover the whole record range exactly
            // once. A bucket the main pass skips (Shader not resident) is
            // skipped here too, so depth and velocity never carry geometry the
            // colour pass leaves out.
            for (b, icb) in self.cull.icbs.iter().enumerate() {
                if !self.world_shader_resident(b) {
                    continue;
                }
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
        if let (Some(deformed), Some(tail)) = (deformed_current, counts.skinned_tail(0)) {
            let prev = if velocity_active
                && self.frames_in_flight >= 2
                && self.skinned.deformed_primed.load(Ordering::Relaxed)
            {
                deformed_prev.unwrap_or(deformed)
            } else {
                deformed
            };
            enc.set_vertex_buffer(deformed, 0, 1);
            enc.set_vertex_buffer(prev, 0, 2);
            if let Some(skinned_ib) = self.skinned.index_buffer.as_ref() {
                enc.useResource_usage_stages(
                    ProtocolObject::from_ref(&**skinned_ib),
                    MTLResourceUsage::Read,
                    MTLRenderStages::Vertex,
                );
            }
            // Skinned records are always bucket 0.
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

    // Build the per-frame `prev_model` buffer for the GPU-driven G-buffer pass:
    // one column-major `float4x4` per cull record, indexed
    // identically to `build_object_buffer` (static + chunks + clones, then
    // instances, then skinned). The G-buffer VS reads it at `[[base_instance]]`
    // to derive per-object motion. Returns `None` when there is no static
    // geometry. Rebuilt every frame: the static + chunk region follows last
    // frame's model (or the current model when velocity is inactive), the
    // instance region is the immutable instance transforms (camera-only motion),
    // and the skinned region is the current model (per-vertex skin motion comes
    // from the previous-frame deformed buffer, not the model matrix).
    pub(in crate::metal) fn build_gbuffer_prev_models(
        &mut self,
        ring_slot: usize,
        velocity_active: bool,
    ) -> Result<Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>, String> {
        if self.draw_objects.is_empty() {
            return Ok(None);
        }
        let mut models = std::mem::take(&mut self.prev_model_scratch);
        models.clear();
        // Static + chunks + clones: index-parallel to build_object_buffer's
        // draw_objects loop. `velocity_active` gates last-frame vs current model
        // (current -> zero model-delta motion, a harmless zero no consumer reads).
        for (i, obj) in self.draw_objects.iter().enumerate() {
            models.push(if velocity_active {
                self.prev_draw_models[i]
            } else {
                obj.model
            });
        }
        // Instances: transforms are immutable, so cur == prev (camera-only motion).
        if self.n_instances > 0 {
            models.extend(self.instance_records.iter().map(|r| r.model));
        }
        // Skinned: the model matrix is static (cur == prev); per-vertex motion
        // comes from the previous-frame deformed buffer, not the model.
        if self.n_skinned > 0 {
            models.extend(self.skinned.draw_objects.iter().map(|o| o.model));
        }
        let result = self.prev_model_ring.write(
            &self.device,
            ring_slot,
            crate::metal::context::bytes_of_slice(&models),
        );
        self.prev_model_scratch = models;
        result.map(Some)
    }
}
