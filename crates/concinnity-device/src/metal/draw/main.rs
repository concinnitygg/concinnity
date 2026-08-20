// src/metal/draw/main.rs
//
// Main pass (off-screen HDR + 4x MSAA). Renders the visible scene into the
// HDR colour + depth attachments, multisample-resolving to `hdr_resolve`.
// Has three geometry paths:
//
//   * Bindless static path: GPU-driven, issued through `cull_icb` in a single
//     executeCommandsInBuffer. Used when the world's fragment shader provides
//     `fragment_main_bindless` (main.metal) and the static draw list is
//     non-empty -- `object_buffer` / `bindless_tex_args` are `Some` in that
//     case.
//   * Legacy static path: per-draw bindings, walks the `visible` list. Used
//     by shaders without a bindless entry point (custom shaders).
//   * Instanced clusters + skinned meshes: drawn after the static path, with
//     their own pipelines rebound.
//
// The three sub-paths are encoded in fixed (static, instanced, skinned)
// order on a single `MTLRenderCommandEncoder`. Two `MTLParallelRenderCommandEncoder`
// attempts (one full-fat with parallel encoders on every shadow cascade,
// one scoped to just this main pass with 3 sub-encoders + no pass-timing)
// both tripped G14X (M2/M3 Pro/Max class) into an abort inside
// `IOGPUMetalCommandBufferStorageAllocResourceAtIndex`: the crash window
// scaled with parallel-encoder usage rate (~20 s with shadow split, ~90 s
// with main-only) but never went away. The mechanism appears fundamentally
// incompatible with our usage on this hardware / macOS 26.4 combo.
// The per-path helpers below stay split out so a future CPU-parallel
// strategy (per-pass command buffers committed through events, or
// data-parallel draw-record prep) can plug in without re-deriving the
// dispatch shape.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLClearColor, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLLoadAction,
    MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLStoreAction,
};

use crate::gfx::render_types::ShadowUniforms;
use crate::metal::context::{BINDLESS_TEXTURE_ARG_BUFFER_INDEX, MtlContext};
use crate::metal::scoped_encoder::ScopedEncoder;
use crate::metal::uniforms::ModelUniforms;
use concinnity_render::uniforms::ViewUniforms;

// Camera state a main-pass encode builds its ViewUniforms from. `view` is
// `self.view_matrix` for the on-screen main pass (and its phase-2 sibling) but
// the caller's face view matrix for a reflection-probe face capture, so it
// travels with the rest of the camera state rather than being read off `self`.
#[derive(Clone, Copy)]
pub(in crate::metal) struct MainPassCamera {
    pub elapsed: f32,
    pub vp: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub cam_pos: [f32; 3],
}

// The GPU-driven buffers the bindless static path consumes. All three are `Some`
// together exactly when the bindless cull path is active; the legacy per-draw
// path leaves them `None` and walks the CPU `visible` list. `counts` is the
// record set those buffers were built for, which the indirect-draw ranges
// address; it is the live draw list for a frame pass and an earlier snapshot for
// the reflection-probe bake.
#[derive(Clone, Copy)]
pub(in crate::metal) struct GpuFrameBuffers<'a> {
    pub object_buffer: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub bindless_tex_args: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub deformed_skinned: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub counts: crate::metal::context::DrawRecordCounts,
}

// The scene draw inputs the main pass walks: the CPU visible set (legacy path),
// the prepared instanced clusters, and the per-skinned-mesh joint palettes.
#[derive(Clone, Copy)]
pub(in crate::metal) struct DrawInputs<'a> {
    pub visible: &'a [u32],
    pub prepared_instances: &'a super::super::instanced::PreparedInstances,
    pub skinned_joint_bufs: &'a [Retained<ProtocolObject<dyn MTLBuffer>>],
}

// The reflection-probe face attachments `encode_main_into_face` renders into
// instead of the HDR targets: a square MSAA colour + depth, resolving colour
// into `resolve`.
#[derive(Clone, Copy)]
pub(in crate::metal) struct FaceTargets<'a> {
    pub color_msaa: &'a ProtocolObject<dyn objc2_metal::MTLTexture>,
    pub depth_msaa: &'a ProtocolObject<dyn objc2_metal::MTLTexture>,
    pub resolve: &'a ProtocolObject<dyn objc2_metal::MTLTexture>,
}

impl MtlContext {
    // 1.0 when a reflection-resolve pass (SSR resolve or RT reflections) will
    // composite over this frame's HDR target, else 0.0. Mirrors the
    // `scene_input` gate in draw/mod.rs (both resolves write `ssr.targets.output`
    // and the graph picks RT over SSR). The forward shader reads it from
    // `ViewUniforms.reflections_enabled` to hand glossy specular to that resolve.
    fn reflection_resolve_active(&self) -> f32 {
        if self.ssr.settings.is_some() || self.rt.accel.is_some() {
            1.0
        } else {
            0.0
        }
    }

    // The frame's unlit flag for ViewUniforms, from the viewport view mode.
    fn shade_mode(&self) -> f32 {
        if self.view_mode == concinnity_core::gfx::view_modes::ViewMode::Unlit {
            1.0
        } else {
            0.0
        }
    }

    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    pub(in crate::metal) fn encode_main_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        camera: MainPassCamera,
        draw_inputs: DrawInputs,
        gpu: GpuFrameBuffers,
        world_hidden: bool,
    ) -> Result<u32, String> {
        let MainPassCamera {
            elapsed,
            vp,
            view: _,
            cam_pos,
        } = camera;
        let DrawInputs {
            visible,
            prepared_instances,
            skinned_joint_bufs,
        } = draw_inputs;
        // Only `object_buffer` gates the descriptor's store action below; the
        // rest of `gpu` travels intact into `encode_main_static_into`.
        let object_buffer = gpu.object_buffer;
        // Build the HDR render pass descriptor. Colour writes into the MSAA
        // attachment and resolves into the single-sample target at end-of-pass;
        // depth lives entirely on the MSAA attachment and is discarded unless
        // a later pass (projected decals, volumetric fog) needs to sample it.
        // Under two-pass occlusion the phase-2 main pass (`Main2`) loads this
        // pass's MSAA colour to draw the disoccluded geometry on top, so the
        // MSAA samples must be stored, not just resolved away. `Main2`
        // performs the final resolve. Without two-pass we resolve-and-discard
        // the MSAA samples as before. The decision mirrors the graph's two-pass
        // gating (`two_pass_occlusion` config AND the bindless cull path active
        // this frame, i.e. an `object_buffer` exists).
        let store_msaa_color = self.cull.two_pass_occlusion
            && object_buffer.is_some()
            && self.cull.pipeline_phase2.is_some();
        let main_pass_desc = MTLRenderPassDescriptor::new();
        let [r, g, b, a] = self.clear_color;
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = main_pass_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_color.as_ref()));
            ca.setResolveTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Clear);
            ca.setStoreAction(if store_msaa_color {
                MTLStoreAction::StoreAndMultisampleResolve
            } else {
                MTLStoreAction::MultisampleResolve
            });
            ca.setClearColor(MTLClearColor {
                red: r as f64,
                green: g as f64,
                blue: b as f64,
                alpha: a as f64,
            });

            let da = main_pass_desc.depthAttachment();
            da.setTexture(Some(self.hdr_targets.depth.as_ref()));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setClearDepth(1.0);
            // Always resolve depth into the single-sample
            // `hdr_targets.depth_resolve` sibling. This is the canonical
            // post-rasterise scene depth that the post chain consumes:
            // raymarch writes hit depth into it; water / decal / fog
            // sample it (single-sample is enough since they only ever
            // read sample 0 anyway). `Sample0` filter matches the
            // existing MSAA-sample-0 read pattern bit-for-bit. The MSAA
            // attachment also stays alive (`StoreAndMultisampleResolve`):
            // the raymarch fragment shader samples it as a read-only
            // snapshot to drive the cone-march early-out without
            // aliasing the writable depth target.
            da.setResolveTexture(Some(self.hdr_targets.depth_resolve.as_ref()));
            da.setDepthResolveFilter(objc2_metal::MTLMultisampleDepthResolveFilter::Sample0);
            da.setStoreAction(MTLStoreAction::StoreAndMultisampleResolve);
        }

        if let Some(t) = &self.pass_timing {
            t.attach_render(&main_pass_desc, super::super::pass_timing::PassId::Main);
        }
        // ScopedEncoder ends the main HDR pass (MSAA resolves into hdr_resolve)
        // and pops the debug group when it drops at end of scope.
        let encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&main_pass_desc)
                .ok_or("failed to get render encoder")?,
            "main pass",
        );
        // Wireframe view: fill mode is encoder state that indirect commands
        // inherit, so the one call covers the ICB sub-paths too.
        if self.view_mode == concinnity_core::gfx::view_modes::ViewMode::Wireframe {
            encoder.setTriangleFillMode(objc2_metal::MTLTriangleFillMode::Lines);
        }

        let view_uniforms = ViewUniforms {
            vp,
            view: self.view_matrix,
            elapsed,
            reflections_enabled: self.reflection_resolve_active(),
            cam_pos,
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
            shade_mode: self.shade_mode(),
            _end_pad: 0.0,
        };

        // While the world is hidden behind an opaque menu, the pass stops at the
        // descriptor's Clear load action: skip every geometry sub-path so even a
        // non-bindless skinned world (whose draw does not consult the now-empty
        // visible / instance / object inputs) renders nothing behind the menu.
        // A scene-less world (no main pipeline) takes the same bare-clear shape
        // every frame.
        if world_hidden || self.pipeline_state.is_none() {
            return Ok(0);
        }

        // Main camera: bind the per-cluster light lists once for every sub-path.
        self.bind_clusters(&encoder, true);
        let count_static = self.encode_main_static_into(
            &encoder,
            &view_uniforms,
            cam_pos,
            visible,
            gpu,
            // Main pass: the main cull ICB (no override).
            None,
        );
        let count_instanced =
            self.encode_main_instanced_into(&encoder, &view_uniforms, prepared_instances);
        let count_skinned =
            self.encode_main_skinned_into(&encoder, &view_uniforms, cam_pos, skinned_joint_bufs);

        Ok(count_static + count_instanced + count_skinned)
    }

    // Render the main pass into one reflection-probe cube face instead of the
    // HDR targets. A thin sibling of `encode_main_pass`: same three geometry
    // sub-paths and shared bindings, but the render-pass descriptor points at a
    // square MSAA colour + depth (resolving colour into `face_resolve`), and the
    // view + view-projection are the caller's face matrices (not `self.*`), so
    // the capture never disturbs the frame's camera state. Depth is cleared and
    // discarded -- the probe consumes only the resolved colour. Driven by
    // `capture_reflection_probe` (metal/probe.rs), once at first frame.
    pub(in crate::metal) fn encode_main_into_face(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        face_targets: FaceTargets,
        camera: MainPassCamera,
        draw_inputs: DrawInputs,
        gpu: GpuFrameBuffers,
        // Bindless ICB to execute instead of the main cull's. The planar mirror
        // render passes its slot's mirror ICB (culled against the reflected
        // frustum); the probe capture passes `None` (reuses the main cull ICB).
        // Only consulted on the bindless static path; the legacy fallback uses
        // `visible` regardless.
        icb_override: Option<&ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>,
    ) -> Result<u32, String> {
        let FaceTargets {
            color_msaa: face_color_msaa,
            depth_msaa: face_depth_msaa,
            resolve: face_resolve,
        } = face_targets;
        let MainPassCamera {
            elapsed,
            vp,
            view,
            cam_pos,
        } = camera;
        let DrawInputs {
            visible,
            prepared_instances,
            skinned_joint_bufs,
        } = draw_inputs;
        let desc = MTLRenderPassDescriptor::new();
        let [r, g, b, a] = self.clear_color;
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(face_color_msaa));
            ca.setResolveTexture(Some(face_resolve));
            ca.setLoadAction(MTLLoadAction::Clear);
            ca.setStoreAction(MTLStoreAction::MultisampleResolve);
            ca.setClearColor(MTLClearColor {
                red: r as f64,
                green: g as f64,
                blue: b as f64,
                alpha: a as f64,
            });

            let da = desc.depthAttachment();
            da.setTexture(Some(face_depth_msaa));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setClearDepth(1.0);
            da.setStoreAction(MTLStoreAction::DontCare);
        }

        let encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&desc)
                .ok_or("failed to get probe render encoder")?,
            "probe face",
        );

        let view_uniforms = ViewUniforms {
            vp,
            view,
            elapsed,
            // Probe-face bake: no reflection resolve runs over the probe cube, so
            // the forward probe specular is the only reflection source here. Keep
            // it (0.0) so captured glossy surfaces are not flattened.
            reflections_enabled: 0.0,
            cam_pos,
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
            // A probe capture is always lit, whatever the viewport shows.
            shade_mode: 0.0,
            _end_pad: 0.0,
        };

        // Planar / probe re-render: the main camera's cluster grid does not match
        // this viewpoint, so iterate every local light instead of the clusters.
        self.bind_clusters(&encoder, false);
        let count_static = self.encode_main_static_into(
            &encoder,
            &view_uniforms,
            cam_pos,
            visible,
            gpu,
            icb_override,
        );
        let count_instanced =
            self.encode_main_instanced_into(&encoder, &view_uniforms, prepared_instances);
        let count_skinned =
            self.encode_main_skinned_into(&encoder, &view_uniforms, cam_pos, skinned_joint_bufs);

        Ok(count_static + count_instanced + count_skinned)
    }

    // Phase-2 main pass for two-pass occlusion (`Main2`). Loads (does not
    // clear) the HDR colour + depth that `encode_main_pass` (phase 1) wrote
    // and re-runs the bindless indirect draw through `cull_icb_2` (the phase-2
    // cull's output), depth-compositing the disoccluded geometry with phase 1.
    // Folded instances AND folded skinned objects ride the unified cull buffers
    // as cullable records, so both are Hi-Z-culled through both phases: drawn in
    // phase 1 when visible, and redrawn here only when phase 1 Hi-Z-occluded
    // them and the rebuilt pyramid (Cull2) disoccludes them. The shared
    // `execute_bindless_static_icb` issues the static+instance range then the
    // skinned tail of `cull_icb_2` (the phase-2 cull resets every already-drawn
    // slot, so nothing double-draws). Resolves colour + depth at end-of-pass so
    // the post-decoration stack reads the combined result. A no-op (returns 0)
    // when there is nothing to redraw: two-pass off, no bindless geometry, or
    // the phase-2 ICB was not built.
    pub(in crate::metal) fn encode_main_pass_phase2(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        camera: MainPassCamera,
        gpu: GpuFrameBuffers,
    ) -> Result<u32, String> {
        let MainPassCamera {
            elapsed,
            vp,
            view: _,
            cam_pos,
        } = camera;
        let GpuFrameBuffers {
            object_buffer,
            bindless_tex_args,
            deformed_skinned,
            counts,
        } = gpu;
        let (Some(obj_buf), Some(tex_args), false) = (
            object_buffer,
            bindless_tex_args,
            self.cull.icbs_2.is_empty(),
        ) else {
            return Ok(0);
        };

        // Load the phase-1 MSAA colour + depth (phase 1 stored them under
        // two-pass), draw the disoccluded geometry on top, and resolve both at
        // end-of-pass: this resolve is the one the post stack consumes.
        let main_pass_desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = main_pass_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_color.as_ref()));
            ca.setResolveTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::StoreAndMultisampleResolve);

            let da = main_pass_desc.depthAttachment();
            da.setTexture(Some(self.hdr_targets.depth.as_ref()));
            da.setLoadAction(MTLLoadAction::Load);
            da.setResolveTexture(Some(self.hdr_targets.depth_resolve.as_ref()));
            da.setDepthResolveFilter(objc2_metal::MTLMultisampleDepthResolveFilter::Sample0);
            da.setStoreAction(MTLStoreAction::StoreAndMultisampleResolve);
        }

        if let Some(t) = &self.pass_timing {
            t.attach_render(&main_pass_desc, super::super::pass_timing::PassId::Main2);
        }
        let encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&main_pass_desc)
                .ok_or("failed to get render encoder")?,
            "main2 pass",
        );
        if self.view_mode == concinnity_core::gfx::view_modes::ViewMode::Wireframe {
            encoder.setTriangleFillMode(objc2_metal::MTLTriangleFillMode::Lines);
        }

        let view_uniforms = ViewUniforms {
            vp,
            view: self.view_matrix,
            elapsed,
            reflections_enabled: self.reflection_resolve_active(),
            cam_pos,
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
            shade_mode: self.shade_mode(),
            _end_pad: 0.0,
        };
        self.bind_main_pass_shared(&encoder, &view_uniforms);
        // Main2 is the same main camera as phase 1, so it reads the clusters too.
        self.bind_clusters(&encoder, true);
        let draw_calls = self.execute_bindless_static_icb(
            &encoder,
            obj_buf,
            tex_args,
            &self.cull.icbs_2,
            deformed_skinned,
            counts,
        );

        Ok(draw_calls)
    }

    // Issue the GPU-driven bindless pass: bind the per-object data +
    // bindless-texture argument buffer, declare the index buffer + sampled
    // textures resident (they are reached only through indirect commands /
    // the argument buffer, never bound on the encoder), then execute the `icb`.
    // Shared by the phase-1 main pass (`icb = cull_icb`) and the phase-2 main
    // pass (`icb = cull_icb_2`) so both issue the bindless geometry identically.
    //
    // The ICB is split into two `executeCommandsInBuffer` ranges because Metal
    // bakes the index buffer into each indirect command: the prefix records are
    // static + instances (static u32 index buffer, static vertex buffer already
    // bound at binding 1 by `bind_main_pass_shared`); the tail records are the
    // folded skinned ones, which the cull kernel encoded against the skinned
    // u16 index buffer and which read the compute-deformed vertices, so this
    // rebinds the deformed buffer at binding 1 for that range (inherited by the
    // ICB commands). Both ranges come from `counts`, the record set the caller's
    // buffers were built and culled for, so a snapshot capture never addresses
    // slots the live draw list grew after its ICB was sized. Returns the number
    // of indirect draws issued (1 or 2).
    fn execute_bindless_static_icb(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        obj_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
        tex_args: &Retained<ProtocolObject<dyn MTLBuffer>>,
        icbs: &[Retained<ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>],
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        counts: crate::metal::context::DrawRecordCounts,
    ) -> u32 {
        use objc2_metal::{MTLRenderStages, MTLResourceUsage};
        // Object records (binding 9) + bindless textures (binding 7) are shared
        // by both ranges: the skinned records live in the same object buffer and
        // sample the same flat pool. The object id reaches the shader via each
        // command's [[base_instance]].
        // SAFETY: every resource bound here is owned by `self` and outlives the encoder, at the
        // buffer/texture indices the shaders declare.
        unsafe {
            enc.setVertexBuffer_offset_atIndex(Some(obj_buf), 0, 9);
            enc.setFragmentBuffer_offset_atIndex(Some(obj_buf), 0, 9);
            enc.setFragmentBuffer_offset_atIndex(
                Some(tex_args),
                0,
                BINDLESS_TEXTURE_ARG_BUFFER_INDEX,
            );
            // The engine sampler block (single-source main program only;
            // world-authored fragments use inline samplers and ignore the
            // slot).
            if let Some(sampler_args) = &self.bindless_sampler_args {
                enc.setFragmentBuffer_offset_atIndex(
                    Some(sampler_args),
                    0,
                    crate::metal::context::BINDLESS_SAMPLER_ARG_BUFFER_INDEX,
                );
            }
        }
        self.use_bindless_textures(enc);

        let mut draw_calls = 0u32;

        // Static + instance prefix, once per shader bucket: bucket 0 executes
        // under the pipeline `bind_main_pass_shared` bound, each later bucket
        // under its material shader's pipeline. The cull kernel wrote every
        // record's command into exactly one bucket's ICB, so the ranges never
        // double-draw. The static u32 index buffer is referenced only inside the
        // indirect commands (not bound), so make it resident; the static vertex
        // buffer is already bound at binding 1.
        if let Some(prefix) = counts.prefix(0) {
            enc.useResource_usage_stages(
                ProtocolObject::from_ref(&*self.index_buffer),
                MTLResourceUsage::Read,
                MTLRenderStages::Vertex,
            );
            let range = crate::metal::context::ns_range(prefix);
            for (b, icb) in icbs.iter().enumerate() {
                if b > 0 {
                    // A bucket whose Shader is not resident yet (its scene has
                    // not pinned) has no pipeline: skip it until warmup builds
                    // one rather than drawing it with the wrong program.
                    let Some(pso) = self.world_pipeline(b) else {
                        continue;
                    };
                    enc.setRenderPipelineState(pso);
                }
                // SAFETY: the prefix spans the static + instance command slots
                // (`ensure_icb_capacity` sized every ICB for `counts.total`).
                unsafe {
                    enc.executeCommandsInBuffer_withRange(icb, range);
                }
                draw_calls += 1;
            }
            // Restore the default pipeline for the skinned tail below (and for
            // the caller's subsequent sub-paths, which re-bind anyway).
            if icbs.len() > 1
                && let Some(ps) = &self.pipeline_state
            {
                enc.setRenderPipelineState(ps);
            }
        }

        // Skinned tail: bind the deformed vertex buffer (inherited by the ICB
        // commands) and make the skinned u16 index buffer resident (the cull
        // kernel baked it into these commands). `deformed_skinned` is `Some`
        // exactly when the fold is active, which is also when the tail is
        // non-empty. Skinned draws always render under the world default shader,
        // so only bucket 0's ICB executes here.
        if let (Some(deformed), Some(icb0), Some(tail)) =
            (deformed_skinned, icbs.first(), counts.skinned_tail(0))
        {
            // SAFETY: every resource bound here is owned by `self` and outlives the encoder, at the
            // buffer/texture indices the shaders declare.
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(deformed), 0, 1);
            }
            if let Some(skinned_ib) = self.skinned.index_buffer.as_ref() {
                enc.useResource_usage_stages(
                    ProtocolObject::from_ref(&**skinned_ib),
                    MTLResourceUsage::Read,
                    MTLRenderStages::Vertex,
                );
            }
            // SAFETY: the tail spans the folded skinned command slots.
            unsafe {
                enc.executeCommandsInBuffer_withRange(icb0, crate::metal::context::ns_range(tail));
            }
            draw_calls += 1;
        }
        draw_calls
    }

    // Apply the bindings every main-pass sub-path needs (view uniforms,
    // lights, shadow + IBL + SSAO bindings, the shared vertex buffer at
    // binding 1). With the serial encoder this is re-applied on each
    // sub-path entry (duplicating some state writes) so the per-path
    // helpers stay shape-compatible with a future parallel-encoder retry.
    // Returns false without touching the encoder when the world has no main
    // pipeline (no 3D scene content); the caller then skips its draws.
    // Bind the clustered-lighting inputs the forward pass reads: the params at
    // fragment buffer(11) + the per-cluster light-index list at buffer(12). Bound
    // once per pass (the value is pass-level, shared by every geometry sub-path)
    // on the shared encoder. `clustered` = true for the main camera (binds the
    // live params); false for the planar / probe re-renders, which shade from a
    // viewpoint the main camera's grid does not match and so fall back to
    // iterating every local light (use_clusters cleared).
    fn bind_clusters(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        clustered: bool,
    ) {
        let cluster_params = if clustered {
            self.cluster_params
        } else {
            crate::gfx::render_types::ClusterParams {
                use_clusters: 0,
                ..self.cluster_params
            }
        };
        // SAFETY: each `setBytes` pointer is derived from a live borrow with that type's `size_of`
        // as the length, and every bound resource outlives the encoder; the indices are the slots
        // the shaders declare.
        unsafe {
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&cluster_params).cast(),
                std::mem::size_of::<crate::gfx::render_types::ClusterParams>(),
                11,
            );
            enc.setFragmentBuffer_offset_atIndex(Some(&self.light_cull.cluster_buffer), 0, 12);
        }
    }

    fn bind_main_pass_shared(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view_uniforms: &ViewUniforms,
    ) -> bool {
        let Some(pipeline_state) = &self.pipeline_state else {
            return false;
        };
        enc.setRenderPipelineState(pipeline_state);
        enc.setDepthStencilState(Some(&self.depth_state));

        // SAFETY: each pointer is derived from the live `view_uniforms` borrow with its `size_of`
        // as the length, and every bound resource outlives the encoder; the indices are the slots
        // the shaders declare.
        unsafe {
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(view_uniforms).cast(),
                std::mem::size_of::<ViewUniforms>(),
                0,
            );
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(view_uniforms).cast(),
                std::mem::size_of::<ViewUniforms>(),
                0,
            );
            enc.setVertexBuffer_offset_atIndex(Some(&self.vertex_buffer), 0, 1);
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&self.light_uniforms).cast(),
                std::mem::size_of::<crate::gfx::render_types::LightUniforms>(),
                4,
            );
            // Local-light storage buffer at fragment buffer(8). Encoder-bound
            // buffers are inherited by the ICB-executed bindless draws (the same
            // way the object buffer at binding 9 is), so this single bind covers
            // the legacy and GPU-driven paths and the planar / probe re-renders.
            enc.setFragmentBuffer_offset_atIndex(Some(&self.local_light_buffer), 0, 8);
            // Per-slice spot shadow projections at fragment buffer(13), inherited
            // by the ICB draws like the local lights above. The matching depth
            // array is a discrete texture on the legacy path only; the bindless
            // path reaches it through the BindlessTextures argument buffer,
            // because discrete texture bindings break ICB compatibility.
            enc.setFragmentBuffer_offset_atIndex(Some(&self.spot_shadow_buffer), 0, 13);
            // Rect area-light extents at fragment buffer(14), indexed by
            // GpuLight.data_index. Inherited by the ICB draws like the buffers
            // above, so this one bind covers every main-pass variant.
            enc.setFragmentBuffer_offset_atIndex(Some(&self.area_light_buffer), 0, 14);
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&self.shadow_uniforms).cast(),
                std::mem::size_of::<ShadowUniforms>(),
                5,
            );
            enc.setFragmentTexture_atIndex(Some(self.shadow_map.as_ref()), 2);
            enc.setFragmentTexture_atIndex(Some(self.spot_shadow_map.as_ref()), 14);
            // Area-light LTC tables. The cube sampler at sampler(2) is a plain
            // linear clamp-to-edge, which is what the lookup wants, so no extra
            // sampler slot is needed.
            enc.setFragmentTexture_atIndex(Some(self.ltc_matrix_texture.as_ref()), 15);
            enc.setFragmentTexture_atIndex(Some(self.ltc_magnitude_texture.as_ref()), 16);
            enc.setFragmentSamplerState_atIndex(Some(self.shadow_sampler.as_ref()), 1);
            // IBL bindings: irradiance + prefilter cubes at texture(3) / texture(4)
            // and a shared linear-clamp sampler at sampler(2). Always bound; the
            // shader uses prefilter_mip_count == 0 to detect the fallback case.
            enc.setFragmentTexture_atIndex(Some(self.env_map.irradiance.as_ref()), 3);
            enc.setFragmentTexture_atIndex(Some(self.env_map.prefilter.as_ref()), 4);
            enc.setFragmentSamplerState_atIndex(Some(self.cube_sampler.as_ref()), 2);
            // SSAO occlusion at texture(5): the blurred AO when SSAO is on,
            // else a 1x1 white texture so shade_surface samples a constant 1.0
            // and the ambient term is left untouched. (The bindless static
            // pass instead reaches it through the BindlessTextures argument
            // buffer; see build_bindless_texture_args.)
            enc.setFragmentTexture_atIndex(Some(self.ao_output_texture()), 5);
            // Reflection-probe cube array at texture(6 .. 6+MAX_PROBES): the legacy
            // path now selects + blends per-surface from the same probe set as the
            // bindless path (which reaches the cubes through the BindlessTextures arg
            // buffer instead, ICB-incompatible discrete binds being the reason).
            // probe_cube_or_sky returns the sky prefilter for unbaked slots, so all
            // MAX_PROBES are always valid. Skybox + diffuse keep texture 3/4.
            for i in 0..concinnity_render::uniforms::MAX_PROBES {
                enc.setFragmentTexture_atIndex(Some(self.probe_cube_or_sky(i)), 6 + i);
            }
            // Reflection-probe set (count + per-probe parallax boxes) at fragment
            // buffer(6) (a buffer slot, distinct from the texture(6) array). `EMPTY`
            // until a bake; the shader weights every box covering the surface.
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&self.probe_set).cast(),
                std::mem::size_of::<concinnity_render::uniforms::ProbeSet>(),
                6,
            );
        }
        true
    }

    // Encode the static-geometry sub-path: either the bindless GPU-driven
    // ICB execution or the legacy per-draw loop, depending on which path
    // the world's pipeline opted into.
    fn encode_main_static_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view_uniforms: &ViewUniforms,
        cam_pos: [f32; 3],
        visible: &[u32],
        gpu: GpuFrameBuffers,
        // ICB to execute instead of the main cull's `self.cull.icb`: the planar
        // mirror render passes its slot's mirror ICB (culled against the
        // reflected frustum) here; the main + probe paths pass `None` to use the
        // main ICB.
        icb_override: Option<&ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>,
    ) -> u32 {
        let GpuFrameBuffers {
            object_buffer,
            bindless_tex_args,
            deformed_skinned,
            counts,
        } = gpu;
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("main static"));
        if !self.bind_main_pass_shared(enc, view_uniforms) {
            enc.popDebugGroup();
            return 0;
        }

        let last_tex = self.textures.len().saturating_sub(1);
        let mut draw_calls: u32 = 0;

        // The planar mirror override is a single command stream executed under
        // the encoder's default pipeline; the main + probe paths execute the
        // per-bucket main cull set.
        let override_holder;
        let bucket_icbs: &[Retained<_>] = match icb_override {
            Some(icb) => {
                override_holder = [objc2::Message::retain(icb)];
                &override_holder
            }
            None => &self.cull.icbs,
        };
        if let (Some(obj_buf), Some(tex_args), false) =
            (object_buffer, bindless_tex_args, bucket_icbs.is_empty())
        {
            // Bindless static pass, GPU-driven. Shared with the phase-2 main
            // pass under two-pass occlusion: see `execute_bindless_static_icb`.
            draw_calls += self.execute_bindless_static_icb(
                enc,
                obj_buf,
                tex_args,
                bucket_icbs,
                deformed_skinned,
                counts,
            );
        } else {
            // Legacy static pass: rebind model/material/textures per draw.
            // Used by shaders without a `fragment_main_bindless` entry point
            // (custom shaders). The shared helper owns the
            // visible/resident filter, the camera-distance LOD pick (matching the
            // bindless path's GpuDrawArgs selection), and the indexed draw.
            draw_calls += self.draw_static_objects(enc, visible, cam_pos, |enc, obj, _| {
                let model_uniforms = ModelUniforms { model: obj.model };
                let slot = obj.texture_slot.min(last_tex);
                // SAFETY: each pointer is derived from a live borrow with that type's `size_of` as
                // the length, and the textures are owned by `self`; `slot` was clamped to
                // `last_tex`.
                unsafe {
                    // model matrix at vertex buffer(2)
                    enc.setVertexBytes_length_atIndex(
                        std::ptr::NonNull::from(&model_uniforms).cast(),
                        std::mem::size_of::<ModelUniforms>(),
                        2,
                    );
                    // material at fragment buffer(3)
                    enc.setFragmentBytes_length_atIndex(
                        std::ptr::NonNull::from(&obj.material).cast(),
                        std::mem::size_of::<crate::gfx::render_types::MaterialUniforms>(),
                        3,
                    );
                    // albedo at texture(0), normal map at texture(1)
                    enc.setFragmentTexture_atIndex(Some(self.textures[slot].as_ref()), 0);
                    enc.setFragmentTexture_atIndex(
                        Some(self.normal_pool_texture(obj.normal_map_slot)),
                        1,
                    );
                    enc.setFragmentSamplerState_atIndex(Some(&self.sampler), 0);
                }
            });
        }
        enc.popDebugGroup();
        draw_calls
    }

    // Encode the instanced-cluster sub-path. One drawIndexedInstanced per
    // cluster*LOD bucket, after a cluster-wide frustum/distance cull.
    fn encode_main_instanced_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view_uniforms: &ViewUniforms,
        prepared: &super::super::instanced::PreparedInstances,
    ) -> u32 {
        let Some(inst_ps) = self.instanced_pipeline_state.clone() else {
            return 0;
        };
        if prepared.clusters.is_empty() {
            return 0;
        }
        // When the bindless static pass is active with build-time geometry, the
        // instances were folded into its cull buffers and drawn by the static
        // ICB (`execute_bindless_static_icb` over `cull_count()`), so the legacy
        // per-cluster main draw would double-draw them. This condition equals
        // `object_buffer.is_some()` (bindless && static geometry present),
        // mirroring DX/VK's `&& !use_bindless`. Gate only the MAIN draw: the
        // SSR / SSAO / velocity pre-passes + shadow still consume `prepared`
        // through the legacy instanced path. Instance-
        // only worlds (no static geometry) keep the legacy draw here.
        if self.bindless && !self.draw_objects.is_empty() {
            return 0;
        }
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("main instanced"));
        // Share the same view / lights / shadow / IBL / SSAO bindings as the
        // static path. The pipeline override below swaps to the instanced PSO.
        if !self.bind_main_pass_shared(enc, view_uniforms) {
            enc.popDebugGroup();
            return 0;
        }
        enc.setRenderPipelineState(&inst_ps);

        let last_tex = self.textures.len().saturating_sub(1);

        // Per cluster: bind material (fragment buffer(3)) + albedo / normal
        // textures, shared across the cluster's LOD buckets. The shared helper
        // owns the cull / LOD-bucket / instance-buffer / draw loop.
        let draw_calls = self.draw_prepared_instances(enc, prepared, false, |enc, cluster| {
            // SAFETY: each pointer is derived from a live borrow and paired with that type's
            // `size_of` as the length, so the encoder copies exactly the bytes it was handed; the
            // buffer indices are the slots the shaders declare.
            unsafe {
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&cluster.material).cast(),
                    std::mem::size_of::<crate::gfx::render_types::MaterialUniforms>(),
                    3,
                );
            }
            let slot = cluster.texture_slot.min(last_tex);
            // SAFETY: `slot` was clamped to `last_tex`, so it indexes `self.textures`; the normal
            // pool always returns a live texture, and the sampler is owned by `self`.
            unsafe {
                enc.setFragmentTexture_atIndex(Some(self.textures[slot].as_ref()), 0);
                enc.setFragmentTexture_atIndex(
                    Some(self.normal_pool_texture(cluster.normal_map_slot)),
                    1,
                );
                enc.setFragmentSamplerState_atIndex(Some(&self.sampler), 0);
            }
        });

        // Restore the regular pipeline state so a future addition to the
        // main pass starts from the same shape the static path left it in.
        if let Some(pipeline_state) = &self.pipeline_state {
            enc.setRenderPipelineState(pipeline_state);
        }
        enc.popDebugGroup();
        draw_calls
    }

    // Encode the skinned-mesh sub-path. Linear-blend-skinned geometry,
    // drawn last in the main pass.
    fn encode_main_skinned_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view_uniforms: &ViewUniforms,
        cam_pos: [f32; 3],
        skinned_joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) -> u32 {
        let (Some(skinned_ps), Some(svb), Some(sib)) = (
            self.skinned.pipeline_state.clone(),
            self.skinned.vertex_buffer.clone(),
            self.skinned.index_buffer.clone(),
        ) else {
            return 0;
        };
        if self.skinned.draw_objects.is_empty() {
            return 0;
        }
        // When the GPU-driven skinned fold is active (bindless + static
        // geometry), skinned objects were pre-skinned into the deformed buffer
        // and drawn by the bindless ICB's skinned tail, so this legacy VS-skinned
        // draw would double-draw them. `n_skinned > 0` is set in `upload_skinned`
        // under exactly that condition, mirroring the instanced gate. A pure-
        // skinned or non-bindless world keeps `n_skinned == 0` and draws here.
        if self.n_skinned > 0 {
            return 0;
        }
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("main skinned"));
        if !self.bind_main_pass_shared(enc, view_uniforms) {
            enc.popDebugGroup();
            return 0;
        }
        enc.setRenderPipelineState(&skinned_ps);
        // SAFETY: every resource bound here is owned by `self` and outlives the encoder, at the
        // buffer/texture indices the shaders declare.
        unsafe {
            enc.setVertexBuffer_offset_atIndex(Some(&svb), 0, 1);
        }

        let last_tex = self.textures.len().saturating_sub(1);

        // The shared helper owns the visible filter, the skinned-camera-distance
        // LOD pick, and the u16 indexed draw; the closure binds this mesh's model,
        // joint palette, material, and textures.
        let draw_calls = self.draw_skinned_objects(enc, &sib, cam_pos, |enc, obj, i| {
            let model_uniforms = ModelUniforms { model: obj.model };
            let slot = obj.texture_slot.min(last_tex);
            // Morph bindings for the VS: the delta buffer at 9 and the
            // per-draw params + weights at 10. Objects without morph targets
            // bind the shared VB as a dummy the shader never reads
            // (`target_count == 0`).
            let morph = self.skinned.morphs.get(i).and_then(|m| m.as_ref());
            let mut morph_params = crate::metal::uniforms::VsMorphParams {
                vertex_base: obj.vertex_base as u32,
                vertex_count: obj.vertex_count as u32,
                target_count: morph.map_or(0, |m| m.target_count),
                _pad: 0,
                weights: [0.0; crate::metal::uniforms::MAX_MORPH_TARGETS],
            };
            if let Some(w) = self.skinned.morph_weights.get(i) {
                for (dst, src) in morph_params.weights.iter_mut().zip(w.iter()) {
                    *dst = *src;
                }
            }
            // SAFETY: each pointer is derived from a live borrow with that type's `size_of` as the
            // length, and the joint / morph-delta buffers outlive the encoder.
            unsafe {
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&model_uniforms).cast(),
                    std::mem::size_of::<ModelUniforms>(),
                    2,
                );
                enc.setVertexBuffer_offset_atIndex(Some(&skinned_joint_bufs[i]), 0, 8);
                let delta_buf = morph.map_or(svb.as_ref(), |m| m.buffer.as_ref());
                enc.setVertexBuffer_offset_atIndex(Some(delta_buf), 0, 9);
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&morph_params).cast(),
                    std::mem::size_of::<crate::metal::uniforms::VsMorphParams>(),
                    10,
                );
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&obj.material).cast(),
                    std::mem::size_of::<crate::gfx::render_types::MaterialUniforms>(),
                    3,
                );
                enc.setFragmentTexture_atIndex(Some(self.textures[slot].as_ref()), 0);
                enc.setFragmentTexture_atIndex(
                    Some(self.normal_pool_texture(obj.normal_map_slot)),
                    1,
                );
                enc.setFragmentSamplerState_atIndex(Some(&self.sampler), 0);
            }
        });
        enc.popDebugGroup();
        draw_calls
    }
}
