//! Main pass (off-screen HDR + 4x MSAA). Renders the visible scene into the
//! HDR color + depth attachments, multisample-resolving to `hdr_resolve`.
//!
//! One geometry path: the GPU-driven bindless pass, issued through the cull's
//! indirect command buffers. Static objects, folded instances and the folded
//! skinned tail are all cull records, so the encode is one prefix range per
//! shader bucket plus the skinned tail. `object_buffer` / `bindless_tex_args`
//! are `Some` exactly when the world has something to draw.
//!
//! The pass is encoded on a single `MTLRenderCommandEncoder`. Two
//! `MTLParallelRenderCommandEncoder` attempts (one full-fat with parallel
//! encoders on every shadow cascade, one scoped to just this main pass with 3
//! sub-encoders + no pass-timing) both tripped G14X (M2/M3 Pro/Max class) into
//! an abort inside `IOGPUMetalCommandBufferStorageAllocResourceAtIndex`: the
//! crash window scaled with parallel-encoder usage rate (~20 s with shadow
//! split, ~90 s with main-only) but never went away. The mechanism appears
//! fundamentally incompatible with our usage on this hardware / macOS 26.4
//! combo.
use concinnity_engine::gfx::render_types;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::planar_reflection::PixelRect;
use concinnity_engine::render::post::reflection_path::ReflectionPath;
use concinnity_engine::render::probe::radiance::{FaceBackground, FaceStorage};
use concinnity_engine::render::shadow::probe::FaceShadow;
use concinnity_engine::render::uniforms::{DaylightBlock, FaceCamera, StandIns, ViewUniforms};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBuffer, MTLClearColor, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLLoadAction,
    MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLScissorRect, MTLStoreAction,
};

use crate::metal::context::{BINDLESS_TEXTURE_ARG_BUFFER_INDEX, MtlContext};
use crate::metal::depth::CLEAR_DEPTH;
use crate::metal::encode::RenderEncode;
use crate::metal::pass_timing::PassTimer;
use crate::metal::scoped_encoder::ScopedEncoder;
use concinnity_engine::render::shader::programs::abi::main::metal as main;

// Camera state a main-pass encode builds its ViewUniforms from. `view` is
// `self.view_matrix` for the on-screen main pass (and its late sibling) but
// the caller's face view matrix for a reflection-probe face capture, so it
// travels with the rest of the camera state rather than being read off `self`.
#[derive(Clone, Copy)]
pub(in crate::metal) struct MainPassCamera {
    pub elapsed: f32,
    pub vp: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub cam_pos: [f32; 3],
}

// The GPU-driven buffers the main pass consumes, `Some` together when the cull
// ran. `counts` is the record set they were built for: the live list for a
// frame pass, a snapshot for the probe bake.
#[derive(Clone, Copy)]
pub(in crate::metal) struct GpuFrameBuffers<'a> {
    pub object_buffer: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub material_params: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub bindless_tex_args: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub deformed_skinned: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub counts: crate::metal::context::DrawRecordCounts,
    // The regions the ICBs were encoded with.
    pub regions: crate::metal::cull::BucketRegions<'a>,
}

// What the main pass draws besides the surfaces: nothing at all behind an
// opaque menu, and the grass field when its kernel ran this frame.
#[derive(Clone, Copy)]
pub(in crate::metal) struct MainPassExtras<'a> {
    pub world_hidden: bool,
    pub grass: Option<&'a crate::metal::grass::GrassFrame>,
}

// Off-screen attachments `encode_main_into_face` renders into. At one sample
// `color_msaa` is `None` and the pass draws straight into `resolve`.
#[derive(Clone, Copy)]
pub(in crate::metal) struct FaceTargets<'a> {
    pub color_msaa: Option<&'a ProtocolObject<dyn objc2_metal::MTLTexture>>,
    pub depth: &'a ProtocolObject<dyn objc2_metal::MTLTexture>,
    pub resolve: &'a ProtocolObject<dyn objc2_metal::MTLTexture>,
    // Array slice of `resolve` this face resolves into. The probe capture passes
    // the cube face it is rendering; the planar mirror's target is a plain 2D
    // texture, so it passes 0.
    pub resolve_slice: usize,
}

// A light-cluster grid: the params a fragment places itself in the grid with
// and the per-cluster light lists and probe masks binned for them.
#[derive(Clone, Copy)]
pub(in crate::metal) struct ClusterGrid<'a> {
    pub params: &'a render_types::ClusterParams,
    pub lists: &'a ProtocolObject<dyn MTLBuffer>,
}

// How an off-camera render into `FaceTargets` is drawn: the ICB to execute,
// the clip rectangle (`None` for the whole target), the cluster grid (`None`
// shades every local light and probe), and its timing span position.
#[derive(Clone, Copy)]
pub(in crate::metal) struct FacePass<'a> {
    pub icb_override: Option<&'a ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>,
    pub scissor: Option<PixelRect>,
    pub clusters: Option<ClusterGrid<'a>>,
    pub timer: PassTimer,
    // The scale the face's radiance is stored at.
    pub storage: FaceStorage,
    // The per-point sky the face shades by.
    pub daylight: &'a DaylightBlock,
    // What the face's ground stands in for that the face does not draw.
    pub stand_ins: StandIns,
    // What the face shows behind its geometry.
    pub background: FaceBackground,
    // The sun shadow the face shades with.
    pub shadow: FaceShadow,
}

impl FacePass<'_> {
    // The whole target through the main cull ICB, unclustered and untimed, at
    // the probes' storage scale: the probe capture.
    pub(in crate::metal) const PROBE: FacePass<'static> = FacePass {
        icb_override: None,
        scissor: None,
        clusters: None,
        timer: PassTimer::None,
        storage: FaceStorage::PROBE,
        daylight: &DaylightBlock::OFF,
        stand_ins: StandIns::None,
        background: FaceBackground::Sky,
        shadow: FaceShadow::Frame,
    };
}

impl MtlContext {
    // Which reflection stages run this frame, from the authored SSR and the
    // live RT pass and BVH. This backend reads the composite's output only while
    // a resolve writes it, so the RT node never runs without a BVH.
    pub(in crate::metal) fn reflection_path(&self) -> ReflectionPath {
        ReflectionPath::new(
            self.ssr.settings.is_some(),
            self.rt.settings.is_some(),
            self.rt.accel.is_some(),
        )
    }

    // The frame's unlit flag for ViewUniforms, from the viewport view mode.
    fn shade_mode(&self) -> f32 {
        if self.state.view.mode == concinnity_engine::gfx::view_modes::ViewMode::Unlit {
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
        gpu: GpuFrameBuffers<'_>,
        extras: MainPassExtras<'_>,
    ) -> RenderResult<u32> {
        let MainPassExtras {
            world_hidden,
            grass,
        } = extras;
        // Only `object_buffer` gates the descriptor's store action below; the
        // rest of `gpu` travels intact into `encode_main_static_into`.
        let object_buffer = gpu.object_buffer;
        // Under two-pass occlusion `MainLate` loads the MSAA color, so it is
        // stored and the late pass resolves; otherwise resolve and discard.
        let store_msaa_color = self.cull.two_pass_occlusion
            && object_buffer.is_some()
            && self.cull.pipeline_late.is_some();
        let main_pass_desc = MTLRenderPassDescriptor::new();
        let [r, g, b, a] = self.pre_exposure.clear_color(self.state.view.clear_color);
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = main_pass_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.targets.hdr.color_attachment()));
            ca.setLoadAction(MTLLoadAction::Clear);
            if self.targets.hdr.multisampled() {
                ca.setResolveTexture(Some(self.targets.hdr.hdr_resolve.as_ref()));
                ca.setStoreAction(if store_msaa_color {
                    MTLStoreAction::StoreAndMultisampleResolve
                } else {
                    MTLStoreAction::MultisampleResolve
                });
            } else {
                // The color attachment *is* the spine, so there is nothing to
                // resolve and the pass just stores what it drew.
                ca.setStoreAction(MTLStoreAction::Store);
            }
            ca.setClearColor(MTLClearColor {
                red: f64::from(r),
                green: f64::from(g),
                blue: f64::from(b),
                alpha: f64::from(a),
            });

            let da = main_pass_desc.depthAttachment();
            da.setTexture(Some(self.targets.hdr.depth_attachment()));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setClearDepth(CLEAR_DEPTH);
            // `depth_resolve` is the scene depth the post chain reads. With MSAA
            // it resolves with `Sample0`, keeping the MSAA depth for Hi-Z;
            // without MSAA it is the attachment.
            if self.targets.hdr.multisampled() {
                da.setResolveTexture(Some(self.targets.hdr.depth_resolve.as_ref()));
                da.setDepthResolveFilter(objc2_metal::MTLMultisampleDepthResolveFilter::Sample0);
                da.setStoreAction(MTLStoreAction::StoreAndMultisampleResolve);
            } else {
                da.setStoreAction(MTLStoreAction::Store);
            }
        }

        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&main_pass_desc, super::super::pass_timing::PassId::Main);
        }
        // ScopedEncoder ends the main HDR pass (MSAA resolves into hdr_resolve)
        // and pops the debug group when it drops at end of scope.
        let encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&main_pass_desc)
                .ok_or_else(|| RenderError::Other("failed to get render encoder".to_string()))?,
            ns_string!("main pass"),
        );
        // Wireframe view: fill mode is encoder state that indirect commands
        // inherit, so the one call covers the ICB sub-paths too.
        if self.state.view.mode == concinnity_engine::gfx::view_modes::ViewMode::Wireframe {
            encoder.setTriangleFillMode(objc2_metal::MTLTriangleFillMode::Lines);
        }

        let view_uniforms = self.main_view_uniforms(&camera);

        // While the world is hidden behind an opaque menu, the pass stops at the
        // descriptor's Clear load action. A scene-less world (no main pipeline)
        // takes the same bare-clear shape every frame.
        if world_hidden || self.cull.main_pipeline.is_none() {
            return Ok(0);
        }

        // Main camera: bind the per-cluster light lists once for the pass.
        self.bind_clusters(&encoder, Some(self.main_cluster_grid()));
        let mut draw_calls = self.encode_main_geometry_into(
            &encoder,
            &view_uniforms,
            gpu,
            // Main pass: the main cull ICB (no override).
            None,
            FaceShadow::Frame,
        );
        // The blades shade on the lights, shadows and argument buffers the
        // surfaces just bound, which they bind only with an indirect stream to
        // execute.
        let surfaces_bound = gpu.object_buffer.is_some()
            && gpu.bindless_tex_args.is_some()
            && !self.cull.icbs.is_empty();
        if let (Some(frame), true) = (grass, surfaces_bound) {
            draw_calls += self.encode_grass_main(&encoder, frame, &view_uniforms);
        }
        // The sky lands behind the early pass's geometry. Under two-pass occlusion
        // `MainLate` then draws the disoccluded rest over it, which wins the depth
        // test because the sky writes none.
        if self.draws_sky(self.state.view.mode) {
            self.encode_sky(&encoder, &view_uniforms);
        }
        Ok(draw_calls)
    }

    // Render the main pass into one probe cube face or mirror target with the
    // caller's matrices, leaving the frame's camera state alone. Depth is
    // discarded.
    pub(in crate::metal) fn encode_main_into_face(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        face_targets: FaceTargets<'_>,
        camera: MainPassCamera,
        gpu: GpuFrameBuffers<'_>,
        pass: FacePass<'_>,
    ) -> RenderResult<u32> {
        let FaceTargets {
            color_msaa: face_color_msaa,
            depth: face_depth,
            resolve: face_resolve,
            resolve_slice,
        } = face_targets;
        let MainPassCamera {
            elapsed,
            vp,
            view,
            cam_pos,
        } = camera;
        let desc = MTLRenderPassDescriptor::new();
        let [r, g, b, a] = pass
            .background
            .clear_color(pass.storage, self.state.view.clear_color);
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
            if let Some(msaa) = face_color_msaa {
                ca.setTexture(Some(msaa));
                ca.setResolveTexture(Some(face_resolve));
                ca.setResolveSlice(resolve_slice);
                ca.setStoreAction(MTLStoreAction::MultisampleResolve);
            } else {
                ca.setTexture(Some(face_resolve));
                ca.setSlice(resolve_slice);
                ca.setStoreAction(MTLStoreAction::Store);
            }
            ca.setLoadAction(MTLLoadAction::Clear);
            ca.setClearColor(MTLClearColor {
                red: f64::from(r),
                green: f64::from(g),
                blue: f64::from(b),
                alpha: f64::from(a),
            });

            let da = desc.depthAttachment();
            da.setTexture(Some(face_depth));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setClearDepth(CLEAR_DEPTH);
            da.setStoreAction(MTLStoreAction::DontCare);
        }
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render_timer(&desc, pass.timer);
        }

        let encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get probe render encoder".to_string())
                })?,
            ns_string!("probe face"),
        );
        if let Some(r) = pass.scissor {
            encoder.setScissorRect(MTLScissorRect {
                x: r.x as usize,
                y: r.y as usize,
                width: r.width as usize,
                height: r.height as usize,
            });
        }

        let view_uniforms = ViewUniforms::face(FaceCamera {
            vp,
            view,
            cam_pos,
            elapsed,
            prefilter_mip_count: self.scene.env_map.prefilter_mip_count as f32,
            sky_rot: self.state.view.sky_rot,
            irradiance: self.irradiance_lane(),
            pre_exposure: pass.storage.scale(),
            daylight: *pass.daylight,
            tile_morph: self.state.tile_morph_view(),
        })
        .with_stand_ins(pass.stand_ins);

        self.bind_clusters(&encoder, pass.clusters);
        let draw_calls = self.encode_main_geometry_into(
            &encoder,
            &view_uniforms,
            gpu,
            pass.icb_override,
            pass.shadow,
        );
        // A face is always rendered lit, whatever the viewport shows.
        if pass.background.draws_sky()
            && self.draws_sky(concinnity_engine::gfx::view_modes::ViewMode::Lit)
        {
            self.encode_sky(&encoder, &view_uniforms);
        }
        Ok(draw_calls)
    }

    // Late main pass for two-pass occlusion: load the early color and depth and
    // execute `cull_icb_late`, then resolve. Returns 0 with nothing to redraw.
    pub(in crate::metal) fn encode_main_pass_late(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        camera: MainPassCamera,
        gpu: GpuFrameBuffers<'_>,
    ) -> RenderResult<u32> {
        let GpuFrameBuffers {
            object_buffer,
            bindless_tex_args,
            ..
        } = gpu;
        let (Some(obj_buf), Some(tex_args), false) = (
            object_buffer,
            bindless_tex_args,
            self.cull.icbs_late.is_empty(),
        ) else {
            return Ok(0);
        };

        // Load the early MSAA color + depth (the early pass stored them under
        // two-pass), draw the disoccluded geometry on top, and resolve both at
        // end-of-pass: this resolve is the one the post stack consumes.
        let main_pass_desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = main_pass_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.targets.hdr.color_attachment()));
            ca.setLoadAction(MTLLoadAction::Load);

            let da = main_pass_desc.depthAttachment();
            da.setTexture(Some(self.targets.hdr.depth_attachment()));
            da.setLoadAction(MTLLoadAction::Load);

            if self.targets.hdr.multisampled() {
                ca.setResolveTexture(Some(self.targets.hdr.hdr_resolve.as_ref()));
                ca.setStoreAction(MTLStoreAction::StoreAndMultisampleResolve);
                da.setResolveTexture(Some(self.targets.hdr.depth_resolve.as_ref()));
                da.setDepthResolveFilter(objc2_metal::MTLMultisampleDepthResolveFilter::Sample0);
                da.setStoreAction(MTLStoreAction::StoreAndMultisampleResolve);
            } else {
                ca.setStoreAction(MTLStoreAction::Store);
                da.setStoreAction(MTLStoreAction::Store);
            }
        }

        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&main_pass_desc, super::super::pass_timing::PassId::MainLate);
        }
        let encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&main_pass_desc)
                .ok_or_else(|| RenderError::Other("failed to get render encoder".to_string()))?,
            ns_string!("main late pass"),
        );
        if self.state.view.mode == concinnity_engine::gfx::view_modes::ViewMode::Wireframe {
            encoder.setTriangleFillMode(objc2_metal::MTLTriangleFillMode::Lines);
        }

        let view_uniforms = self.main_view_uniforms(&camera);
        self.bind_main_pass_shared(&encoder, &view_uniforms, FaceShadow::Frame);
        // MainLate is the same main camera as the early pass, so it reads the clusters too.
        self.bind_clusters(&encoder, Some(self.main_cluster_grid()));
        // The late pass's lists, in its own ICBs.
        let gpu = GpuFrameBuffers {
            regions: crate::metal::cull::BucketRegions {
                ranges: self.cull.ranges_late.as_deref(),
                ..gpu.regions
            },
            ..gpu
        };
        let draw_calls = self.execute_bindless_static_icb(
            &encoder,
            obj_buf,
            tex_args,
            &self.cull.icbs_late,
            gpu,
        );

        Ok(draw_calls)
    }

    // Issue the bindless pass: bind the object data and texture argument
    // buffer, declare what the ICB reaches resident, and execute `icb` in two
    // ranges, since each command bakes its index buffer: the static + instance
    // prefix, then the skinned tail over the deformed vertices. Ranges come from
    // `counts`. Returns the draws issued.
    fn execute_bindless_static_icb(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        obj_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
        tex_args: &Retained<ProtocolObject<dyn MTLBuffer>>,
        icbs: &[Retained<ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>],
        gpu: GpuFrameBuffers<'_>,
    ) -> u32 {
        use objc2_metal::{MTLRenderStages, MTLResourceUsage};
        let GpuFrameBuffers {
            material_params,
            deformed_skinned,
            counts,
            regions,
            ..
        } = gpu;
        // Object records (binding 9) + bindless textures (binding 7) are shared
        // by both ranges: the skinned records live in the same object buffer and
        // sample the same flat pool. The object id reaches the shader via each
        // command's [[base_instance]].
        enc.set_vertex_buffer(obj_buf, 0, main::OBJECTS.index);
        enc.set_fragment_buffer(obj_buf, 0, main::OBJECTS.index);
        // The material parameter table either hook may read.
        if let Some(params) = material_params {
            let slot = crate::metal::material_params::MATERIAL_PARAMS_BUFFER_INDEX;
            enc.set_vertex_buffer(params, 0, slot);
            enc.set_fragment_buffer(params, 0, slot);
        }
        enc.set_fragment_buffer(tex_args, 0, BINDLESS_TEXTURE_ARG_BUFFER_INDEX);
        // The engine sampler block (single-source main program only;
        // world-authored fragments use inline samplers and ignore the
        // slot).
        if let Some(sampler_args) = &self.arg_buffers.bindless_sampler_args {
            enc.set_fragment_buffer(
                sampler_args,
                0,
                crate::metal::context::BINDLESS_SAMPLER_ARG_BUFFER_INDEX,
            );
        }
        self.use_bindless_textures(enc);

        let mut draw_calls = 0u32;

        // The prefix, once per shader bucket under its pipeline. The static
        // index buffer is reached only through the ICB, so declare it resident.
        if counts.prefix(0).is_some() {
            enc.useResource_usage_stages(
                ProtocolObject::from_ref(&*self.scene.index_buffer),
                MTLResourceUsage::Read,
                MTLRenderStages::Vertex,
            );
            for (b, icb) in icbs.iter().enumerate() {
                if !regions.draws(b) {
                    continue;
                }
                if b > 0 {
                    // A bucket whose Shader is not resident yet (its scene has
                    // not pinned) has no pipeline: skip it until warmup builds
                    // one rather than drawing it with the wrong program.
                    let Some(pipelines) = self.cull.world_pipelines.get(b) else {
                        continue;
                    };
                    enc.set_pipeline(&pipelines.main);
                }
                draw_calls +=
                    execute_list(enc, icb, regions, concinnity_core::math::saturating_u32(b));
            }
            // Restore the default pipeline for the skinned tail below (and for
            // the caller's subsequent sub-paths, which re-bind anyway).
            if icbs.len() > 1
                && let Some(pipelines) = &self.cull.main_pipeline
            {
                enc.set_pipeline(&pipelines.main);
            }
        }

        // The skinned tail: bind the deformed buffer and declare the skinned
        // index buffer resident. Only bucket 0 executes.
        if let (Some(deformed), Some(icb0), Some(_)) =
            (deformed_skinned, icbs.first(), counts.skinned_tail(0))
        {
            enc.set_vertex_buffer(deformed, 0, 1);
            if let Some(skinned_ib) = self.skinned.index_buffer.as_ref() {
                enc.useResource_usage_stages(
                    ProtocolObject::from_ref(&**skinned_ib),
                    MTLResourceUsage::Read,
                    MTLRenderStages::Vertex,
                );
            }
            draw_calls += execute_list(enc, icb0, regions, render_types::SKINNED_STREAM);
        }
        draw_calls
    }

    // The view block the on-screen main pass shades through, which the G-buffer
    // pre-pass binds too so a vertex hook places a surface identically in both.
    pub(in crate::metal) fn main_view_uniforms(&self, camera: &MainPassCamera) -> ViewUniforms {
        let blend = self.state.planner.blend();
        ViewUniforms {
            vp: camera.vp,
            view: self.state.view.matrix,
            elapsed: camera.elapsed,
            reflection: 0.0,
            cam_pos: camera.cam_pos,
            prefilter_mip_count: self.scene.env_map.prefilter_mip_count as f32,
            shade_mode: self.shade_mode(),
            ambient_occlusion: blend.occlusion,
            sky_rot: self.state.view.sky_rot,
            irradiance: self.irradiance_lane(),
            pre_exposure: self.pre_exposure.value(),
            screen_bounce: blend.bounce,
            stand_ins: 0.0,
            daylight: self.state.view.daylight,
            tile_morph: self.state.tile_morph_view(),
            reflection_scope: blend.reflections,
        }
    }

    // The main camera's cluster grid, binned by the frame's `LightCull` node.
    pub(in crate::metal) fn main_cluster_grid(&self) -> ClusterGrid<'_> {
        ClusterGrid {
            params: &self.cluster_params,
            lists: &self.light_cull.cluster_buffer,
        }
    }

    // Bind the clustered-lighting params (buffer 11) and lists (buffer 12).
    // Without a grid `use_clusters` is cleared.
    fn bind_clusters(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        grid: Option<ClusterGrid<'_>>,
    ) {
        let (params, lists) = match grid {
            Some(grid) => (*grid.params, grid.lists),
            None => (
                render_types::ClusterParams {
                    use_clusters: 0,
                    ..self.cluster_params
                },
                &*self.light_cull.cluster_buffer,
            ),
        };
        enc.set_fragment_value(&params, main::CLUSTER.index);
        enc.set_fragment_buffer(lists, 0, main::CLUSTER_LIST.index);
    }

    // Apply the state every main-pass encode needs. Only buffers, since ICB
    // draws inherit buffer bindings but reach textures through the argument
    // buffers. Returns false without a main pipeline.
    fn bind_main_pass_shared(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view_uniforms: &ViewUniforms,
        shadow: FaceShadow,
    ) -> bool {
        let Some(pipelines) = &self.cull.main_pipeline else {
            return false;
        };
        enc.set_pipeline(&pipelines.main);
        enc.set_depth_stencil(&self.targets.depth_state);

        enc.set_vertex_value(view_uniforms, main::VIEW.index);
        enc.set_fragment_value(view_uniforms, main::VIEW.index);
        enc.set_vertex_buffer(&self.scene.vertex_buffer, 0, 1);
        enc.set_vertex_buffer(&self.scene.vertex_buffer, 0, main::MESH_VERTICES.index);
        enc.set_fragment_value(self.state.lights.uniforms(), main::LIGHTS.index);
        enc.set_fragment_value(&self.canopy, main::CANOPY.index);
        if let Some(tables) = self.canopy_tables.buffer() {
            enc.set_fragment_buffer(tables, 0, crate::metal::canopy::CANOPY_TABLES_BUFFER_INDEX);
        }
        // The local lights. Encoder-bound buffers are inherited by the
        // ICB-executed bindless draws (the same way the object buffer is), so
        // this single bind covers the main pass and the planar / probe
        // re-renders.
        enc.set_fragment_buffer(&self.scene.local_light_buffer, 0, main::LOCAL_LIGHTS.index);
        // Spot shadow projections; the depth array travels in the argument
        // buffer.
        enc.set_fragment_buffer(&self.spot_shadow.buffer, 0, main::SPOT_SHADOWS.index);
        // Rect area-light extents, indexed by
        // GpuLight.data_index. Inherited by the ICB draws like the buffers
        // above, so this one bind covers every main-pass variant.
        enc.set_fragment_buffer(&self.scene.area_light_buffer, 0, main::AREA_LIGHTS.index);
        enc.set_fragment_value(
            self.state.planner.shadow.face_uniforms(shadow),
            main::SHADOW.index,
        );
        // The reflection-probe count and one parallax record per baked probe;
        // the cube array rides the texture argument buffer. The shader weights
        // every box covering the surface.
        self.probe_bindings().bind(
            enc,
            crate::metal::probe_set::ProbeSlots {
                set: main::PROBE_SET.index,
                records: main::PROBE_RECORDS.index,
                cubes: None,
                cluster: None,
            },
        );
        true
    }

    // Encode the frame's geometry: the GPU-driven bindless pass, one indirect
    // range per shader bucket plus the folded skinned tail. Returns 0 without
    // touching the encoder when the world has nothing to draw this frame.
    fn encode_main_geometry_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view_uniforms: &ViewUniforms,
        gpu: GpuFrameBuffers<'_>,
        // ICB to execute instead of the main cull's `self.cull.icb`: the planar
        // mirror render passes its slot's mirror ICB (culled against the
        // reflected frustum) here; the main + probe paths pass `None` to use the
        // main ICB.
        icb_override: Option<&ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>,
        shadow: FaceShadow,
    ) -> u32 {
        let GpuFrameBuffers {
            object_buffer,
            bindless_tex_args,
            ..
        } = gpu;
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("main geometry"));
        if !self.bind_main_pass_shared(enc, view_uniforms, shadow) {
            enc.popDebugGroup();
            return 0;
        }

        // The planar mirror override is a single command stream executed under
        // the encoder's default pipeline; the main + probe paths execute the
        // per-bucket main cull set.
        let override_holder;
        let bucket_icbs: &[Retained<_>] = match icb_override {
            Some(icb) => {
                // Every record of the mirror stream draws under one program,
                // which must shade the far bands the grass variants otherwise do.
                if let Some(superset) = self.cull.mirror_superset_pipeline() {
                    enc.set_pipeline(superset);
                }
                override_holder = [objc2::Message::retain(icb)];
                &override_holder
            }
            None => &self.cull.icbs,
        };
        let mut draw_calls: u32 = 0;
        if let (Some(obj_buf), Some(tex_args), false) =
            (object_buffer, bindless_tex_args, bucket_icbs.is_empty())
        {
            // Shared with the late main pass under two-pass occlusion: see
            // `execute_bindless_static_icb`.
            draw_calls +=
                self.execute_bindless_static_icb(enc, obj_buf, tex_args, bucket_icbs, gpu);
        }
        enc.popDebugGroup();
        draw_calls
    }
}

// Execute stream `stream`'s list of `icb` at the execution range the cull
// wrote for it, returning the indirect executes issued: none before the ICBs
// are first built.
pub(in crate::metal) fn execute_list(
    enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
    icb: &ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>,
    regions: crate::metal::cull::BucketRegions<'_>,
    stream: u32,
) -> u32 {
    let Some(ranges) = regions.ranges else {
        return 0;
    };
    // SAFETY: the cull that encoded `icb` wrote this stream's range at its
    // offset in `ranges`, a list no longer than the stream's region, which
    // `ensure_icb_capacity` sized `icb` to hold; both buffers outlive the
    // command buffer this encoder records into.
    unsafe {
        enc.executeCommandsInBuffer_indirectBuffer_indirectBufferOffset(
            icb,
            ranges,
            crate::metal::cull::BucketRegions::range_offset(stream),
        );
    }
    1
}
