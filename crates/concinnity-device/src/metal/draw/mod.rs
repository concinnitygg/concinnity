//! `MtlContext::draw_frame` -- the per-frame orchestration. The per-pass GPU
//! encoders live in sibling files:
//!
//!   shadow.rs    cascaded shadow map (depth-only, one render pass per cascade)
//!   main.rs      main HDR pass, GPU-driven bindless geometry
//!   composite.rs ACES tonemap + FXAA composite + text overlay
//!
//! Other passes (SSAO, SSR pre + resolve, decals, fog, velocity, TAA, bloom,
//! auto-exposure) live in their own files at the `metal/` level alongside
//! `decal.rs`, `fog.rs`, `post.rs`, etc., and are invoked through the
//! `self.encode_*` methods defined there.
#![deny(unsafe_op_in_unsafe_fn)]

mod composite;
// pub(in crate::metal) so the render-graph executor, planar mirror, and probe
// bake can name the shared main-pass param structs defined here.
pub(in crate::metal) mod main;
mod shadow;
mod spot_shadow;
mod stages;

use concinnity_core::gfx::render_types;
use concinnity_core::render::backend::FrameParams;
use concinnity_core::render::error;
use concinnity_core::render::lights;
use concinnity_core::render::model_history::HistoryMode;
use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::post::rt_reflections::RtParamsInputs;
use concinnity_core::render::render_graph;
use concinnity_core::transform::mat4_inverse;
use concinnity_core::transform::mat4_mul;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandQueue as _};

use self::stages::{AcquiredFrame, FrameProjection, GraphInputArgs, PresentFrame};
use super::context::MtlContext;
use super::graph_exec::GraphFrameParams;
use concinnity_core::render::uniforms::metal::*;

impl MtlContext {
    // Pump the NSEvent queue and encode one frame to the GPU.
    //
    // NSEvent processing must happen here rather than in the run loop because
    // window close, resize, and key events are only delivered after NSApp
    // dequeues them. CFRunLoopRunInMode alone does not dispatch NSEvents.
    //
    // The whole frame runs inside a fresh autorelease pool. The render loop is
    // a tight Rust loop with no Cocoa run-loop pool of its own, so without this
    // the autoreleased per-frame Metal objects (command buffers, which retain
    // every resource they reference until they are released, plus encoders,
    // descriptors, and `NSArray`s) would accumulate in the never-drained outer
    // pool. That keeps each frame's transient buffers / acceleration structures
    // alive even after they are replaced on the context, so
    // `device.currentAllocatedSize()` climbs every frame (faster at higher FPS)
    // until unified memory is exhausted and the GPU faults / the host panics.
    // Draining per frame frees each frame's command buffers once the GPU
    // retires them, bounding VRAM to the work actually in flight.
    pub(crate) fn draw_frame(&mut self, params: FrameParams<'_>) -> error::RenderResult<()> {
        objc2::rc::autoreleasepool(|_| self.draw_frame_inner(params))
    }

    fn draw_frame_inner(&mut self, params: FrameParams<'_>) -> error::RenderResult<()> {
        let FrameParams {
            elapsed,
            fov_y_radians,
            near,
            far,
            cam_pos,
            text_calls,
            lines,
            world_hidden,
            view_mode,
            show,
            sky_rot,
        } = params;
        let mtm = objc2::MainThreadMarker::new().ok_or_else(|| {
            error::RenderError::Other("draw_frame must be called from the main thread".into())
        })?;
        // Snapped for the pass encoders (wireframe fill mode, unlit shading,
        // the composite's channel visualization + depth normalization).
        self.view.mode = view_mode;
        self.view.far = far;
        self.view.sky_rot = sky_rot;

        let pass_timing_slot = self.begin_frame_stats();
        if !self.pump_window_events(mtm) {
            return Ok(());
        }
        let Some(AcquiredFrame {
            frame_slot,
            drawable,
            frame_id,
            ring_slot,
        }) = self.acquire_frame(world_hidden)
        else {
            return Ok(());
        };

        let cmd_buf = self
            .hw
            .command_queue
            .commandBuffer()
            .ok_or_else(|| error::RenderError::Other("failed to get command buffer".into()))?;

        self.service_background_work(elapsed, ring_slot);

        // Transient per-frame GPU buffers holding each skinned object's joint
        // matrices. Built once and reused across the shadow cascades and the
        // main pass. Empty when no SkinnedMesh is in the world.
        let skinned_joint_bufs = self.build_joint_buffers(ring_slot)?;

        // Per-object morph weights for the skinned fold, from the same ring slot.
        let skinned_morph_weight_bufs = self.build_morph_weight_buffers(ring_slot)?;

        let aspect = self.update_shadow_schedule(cam_pos, fov_y_radians, near, far);

        // Main pass prep: resize off-screen targets.
        // Resize the HDR targets if the drawable size changed (window resize
        // or initial layout). The drawable was just refreshed by window.view.draw().
        let draw_size = self.window().view.drawableSize();
        // Geometry-less worlds keep their off-screen targets pinned at 1x1
        // (see MtlContext::new); the composite pass still uses the full drawable.
        let (want_w, want_h) = if self.targets.geometry_less {
            (1, 1)
        } else {
            (
                draw_size.width.max(1.0) as u32,
                draw_size.height.max(1.0) as u32,
            )
        };
        self.resize_targets_if_needed(want_w, want_h)?;

        // Render resolution: where the 3D scene + most post passes draw.
        // Equals `want_w/h` (the drawable size) when no upscaler is active;
        // otherwise it's smaller, so the upscaler reconstructs back up to
        // drawable size.
        let render_w = self.targets.hdr.width;
        let render_h = self.targets.hdr.height;

        let FrameProjection {
            proj,
            vp,
            inv_vp,
            frustum,
        } = self.frame_projection(fov_y_radians, aspect, near, far, render_w, render_h);

        // The probe cube handles, for every pass that samples the set. Built
        // ahead of the bindless prep below and outside its world-hidden gate:
        // the transparent and post passes read the set without a static draw
        // list of their own, and a slot left holding last frame's ring buffer
        // would outlive the frame that wrote it.
        self.probe.cube_args = Some(self.build_probe_cube_args(ring_slot)?);
        // The residency sets the argument buffers' contents need. Refreshed
        // before any pass encodes, and a no-op on a frame whose textures are
        // unchanged, which is every frame between a stream-in or a bake.
        self.refresh_probe_cube_residency();
        self.refresh_bindless_residency();

        // While the world is hidden behind an opaque menu, the surviving Main
        // pass is fed an empty scene -- no bindless object / cull / texture
        // buffers, no instanced clusters, and no acceleration-structure refresh
        // -- so it runs as a bare clear that the opaque overlay then covers. The
        // masked graph drops every other world pass, so none of this work would
        // be consumed anyway.
        // The GPU-driven G-buffer pre-pass both fills and reads the model-history
        // ring. With no consumer of motion, or with the pre-pass not running,
        // the ring goes stale, so the draw-args build marks every record
        // `NO_HISTORY` and the tracker re-primes when the pre-pass returns.
        let history_live = !world_hidden
            && (self.taa.enabled || self.upscale.scaler.is_some())
            && self.gbuffer.targets.is_some()
            && self.gbuffer.bindless_pipeline.is_some();
        let (object_buffer, cull_draw_args, bindless_tex_args) = if world_hidden {
            (None, None, None)
        } else {
            // Per-frame GPU buffer prep for the bindless path.
            // The object data + indirect-args + bindless texture argbuf are
            // all per-frame Metal buffers the bindless Main pass + Cull
            // compute pass consume. They must outlive the command buffer,
            // hence the bindings kept here through to `cmd_buf.commit()`.
            let object_buffer = if self.cull.bindless {
                self.build_object_buffer(ring_slot)?
            } else {
                None
            };
            let cull_draw_args = if object_buffer.is_some() {
                let draw_args = self.build_draw_args_buffer(
                    cam_pos,
                    ring_slot,
                    if history_live {
                        HistoryMode::Track
                    } else {
                        HistoryMode::Stale
                    },
                )?;
                if draw_args.is_some() {
                    self.ensure_icb_capacity(self.cull_count())?;
                    // GPU-driven cascaded shadow: size the per-cascade
                    // shadow ICB to NUM_SHADOW_CASCADES * cull_count. A no-op when
                    // the shadow-bindless path is inactive (no shadow cull encoder).
                    self.ensure_shadow_icb_capacity(self.cull_count())?;
                    // Per-planar-slot mirror cull ICBs: one per distinct reflection
                    // plane, each sized to cull_count. A no-op (clears the slots) when
                    // the world has no planar set (RT on, or no flat reflectors).
                    let mirror_slots = self
                        .planar_reflection
                        .as_ref()
                        .map(|s| s.planes.len())
                        .unwrap_or(0);
                    self.ensure_mirror_icb_capacity(mirror_slots, self.cull_count())?;
                }
                draw_args
            } else {
                None
            };
            let bindless_tex_args = if object_buffer.is_some() {
                self.build_bindless_texture_args(ring_slot)?
            } else {
                None
            };
            // Asynchronous reflection-probe bake, capture half: submit one cube
            // face, or start or hand off a capture. Every face samples through
            // this frame's texture arguments, so it never reads a texture that
            // streaming has since replaced.
            self.advance_probe_capture(elapsed, near, far, bindless_tex_args.as_ref());
            // Keep the RT acceleration structure current with this frame's
            // transforms before any pass reads `rt_accel`. The default `Auto` mode
            // rebuilds the TLAS only when a participating prop actually moved; a
            // fully static scene pays just a matrix compare here. Non-fatal: a
            // transient rebuild failure keeps last frame's BVH rather than stopping
            // the renderer.
            self.rt_dynamic_update(
                super::raytrace::RtFrame {
                    id: frame_id,
                    ring_slot,
                },
                &skinned_joint_bufs,
            );

            (object_buffer, cull_draw_args, bindless_tex_args)
        };

        // Per-frame pass uniforms hoisted upfront.
        // Every pass that needs a struct of per-frame params builds its
        // uniforms here so a single GraphFrameParams below can carry
        // the union into `execute_graph`.
        let ssao_params = self
            .ssao
            .settings
            .map(|settings| settings.params(fov_y_radians, aspect));
        let ssr_params = self.ssr.settings.map(|settings| {
            let v = self.view.matrix;
            let inv_view_rot = [
                [v[0][0], v[1][0], v[2][0], 0.0],
                [v[0][1], v[1][1], v[2][1], 0.0],
                [v[0][2], v[1][2], v[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let prefilter_mip_count = self.scene.env_map.prefilter_mip_count as f32;
            settings.params(
                fov_y_radians,
                aspect,
                inv_view_rot,
                cam_pos,
                prefilter_mip_count,
                sky_rot,
            )
        });
        let ssgi_params = self
            .ssgi
            .settings
            .map(|settings| settings.params(fov_y_radians, aspect));
        // RT-reflection params: built only when the acceleration structure is
        // live (so they stay in lockstep with `rt_reflections_enabled`). Carries
        // the camera-to-world transform + sun the kernel shades hits with, like
        // SSR's params plus the world-space camera + sun.
        let rt_reflection_params =
            self.rt
                .settings
                .filter(|_| self.rt.accel.is_some())
                .map(|settings| {
                    let v = self.view.matrix;
                    let inv_view_rot = [
                        [v[0][0], v[1][0], v[2][0], 0.0],
                        [v[0][1], v[1][1], v[2][1], 0.0],
                        [v[0][2], v[1][2], v[2][2], 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ];
                    let prefilter_mip_count = self.scene.env_map.prefilter_mip_count as f32;
                    let sun = &self.light_uniforms.directional[0];
                    let sun_color = [
                        sun.color[0] * sun.intensity,
                        sun.color[1] * sun.intensity,
                        sun.color[2] * sun.intensity,
                    ];
                    settings.params(RtParamsInputs {
                        fov_y_radians,
                        aspect,
                        inv_view_rot,
                        cam_pos,
                        sun_dir: sun.direction,
                        sun_color,
                        prefilter_mip_count,
                        sky_rot,
                    })
                });
        // The live settings, dropped when the medium cannot affect the frame (a
        // zero density integrates to a transparent black over the whole volume).
        // One source for the two param blocks and the graph gate below, so
        // `GraphFrameParams`'s "Some only when the Fog pass is in the graph"
        // contract holds.
        let fog_settings = self.fog.settings.filter(|s| s.contributes());
        let fog_params = fog_settings.map(|fog| {
            // Sun = the first directional light; falls back to the
            // LightUniforms::DEFAULT direction if the world declared none.
            let sun = &self.light_uniforms.directional[0];
            let sun_color = [
                sun.color[0] * sun.intensity,
                sun.color[1] * sun.intensity,
                sun.color[2] * sun.intensity,
            ];
            // Fog renders into hdr_resolve, which is render-resolution
            // when the upscaler is on. The fog shader uses the viewport
            // to reconstruct world position from screen UV, so it must
            // match the actual render target's pixel grid.
            let viewport = [render_w as f32, render_h as f32];
            // Reconstruct the froxel volume with the UN-jittered view-projection.
            // Fog is volumetric, so its screen-space contribution does not follow
            // the surface motion vectors TAA reprojects by. Feeding it the jittered
            // inv_vp shifts the whole volume sub-pixel every frame; on a large
            // smooth low-contrast surface, where the fog is the dominant
            // high-frequency signal, TAA cannot reconcile that per-frame shift with
            // the jitter-free history, so the fog flickers (a moving moire). The
            // un-jittered inv_vp keeps the volume stable frame to frame; its offset
            // versus the jittered depth buffer is far below the coarse froxel grid.
            let fog_inv_vp = mat4_inverse(mat4_mul(proj, self.view.matrix));
            fog.params(fog_inv_vp, cam_pos, sun.direction, sun_color, viewport)
        });
        // FogFroxel volume extras: view matrix + volume dimensions + near/far
        // so the compute kernel can place each froxel in world-space and the
        // fragment shader can map a scene depth into the volume's Z axis.
        let fog_froxel_params = fog_settings.map(|fog| render_types::FogFroxelParams {
            view: self.view.matrix,
            froxel_dims: [
                render_graph::FOG_FROXEL_X,
                render_graph::FOG_FROXEL_Y,
                render_graph::FOG_FROXEL_Z,
            ],
            _pad_align: 0,
            z_near: near.max(1e-3),
            z_far: fog.max_distance,
            _pad: [0.0; 2],
        });
        // Clustered light-binning params (main camera). The compute pass reads
        // these to build each cluster's world-space AABB (un-jittered inverse VP
        // + camera forward, matching the fog froxel convention) and the forward
        // pass reads the grid dims / depth range / screen size to place a
        // fragment. `use_clusters` is set only when the world has local lights
        // (the pipeline is built iff so) and at least one is still live;
        // otherwise the forward pass brute-forces an empty list and the LightCull
        // graph node is omitted, so a list the skipped pass did not write is never
        // read. Stored on self so the shared main-pass bind can push it; a local
        // copy feeds the LightCull arm.
        let clustered = lights::clustered_lighting_active(
            self.light_cull.pipeline.is_some(),
            self.light_uniforms.num_local_lights,
        );
        let cluster_inv_vp = mat4_inverse(mat4_mul(proj, self.view.matrix));
        self.cluster_params = render_types::ClusterParams {
            inv_view_proj: cluster_inv_vp,
            cam_pos,
            z_near: near.max(1e-3),
            view_forward: [
                -self.view.matrix[0][2],
                -self.view.matrix[1][2],
                -self.view.matrix[2][2],
            ],
            z_far: far,
            grid_x: render_types::CLUSTER_GRID_X,
            grid_y: render_types::CLUSTER_GRID_Y,
            grid_z: render_types::CLUSTER_GRID_Z,
            num_lights: self.light_uniforms.num_local_lights.max(0) as u32,
            screen_w: render_w as f32,
            screen_h: render_h as f32,
            use_clusters: u32::from(clustered),
            _pad: 0,
        };
        let cluster_params = self.cluster_params;
        // Velocity (motion vectors in the G-buffer pre-pass) is needed whenever
        // temporal reconstruction runs: that's TAA or the MetalFX upscaler.
        let velocity_active = self.taa.enabled || self.upscale.scaler.is_some();
        let vel_uniforms = if velocity_active {
            Some(VelocityUniforms {
                jittered_vp: vp,
                cur_vp: mat4_mul(proj, self.view.matrix),
                prev_vp: self.prev_view_proj,
            })
        } else {
            None
        };
        // `scene_input` is the engine-owned texture the post-decoration stack
        // treats as the pre-TAA scene: `ssr_targets.output` when a reflection
        // path is live, else the raw `hdr_resolve`.
        //
        // `output` is the *composited* scene, not the reflection. Both the SSR
        // and the RT resolve write radiance into `ssr_targets.reflection`, then
        // call the shared `encode_reflection_composite`, which blends that over
        // `hdr_resolve` into `output`. Worth stating precisely: the DirectX
        // equivalent split the two apart and left its upscaler reading the
        // radiance buffer as if it were the scene.
        //
        // `scene_color` is what Bloom + Composite read:
        //   - the upscaler's output (drawable-res) when MetalFX is on,
        //   - the TAA resolve target when TAA is on,
        //   - otherwise just the pre-TAA scene (no temporal stage).
        let scene_input = if self.ssr.settings.is_some() || self.rt.accel.is_some() {
            self.ssr
                .targets
                .as_ref()
                .ok_or_else(|| {
                    error::RenderError::Other("reflections enabled but SSR targets missing".into())
                })?
                .output
                .clone()
        } else {
            self.targets.hdr.hdr_resolve.clone()
        };
        let scene_color = if let Some(u) = &self.upscale.scaler {
            u.output.clone()
        } else if let Some(out) = self.taa.output() {
            out.clone()
        } else {
            scene_input.clone()
        };

        // The transparent pass runs when any translucent producer is live.
        // Drives both the graph-input gate (whether the slot is inserted) and
        // the `scene_pre_taa` supply below (the pass reads + writes it). With
        // SSR off `scene_input` aliases `hdr_resolve`, which is the correct
        // RMW target: the transparent encoder blits a scene copy first, so the
        // self-read for refraction is safe.
        let transparent_active = (self.water.pipeline.is_some()
            && self.water.surfaces.iter().any(|s| s.visible))
            || (self.glass.pipeline.is_some() && self.glass.panels.iter().any(|p| p.visible))
            || self.mesh_glass_visible();

        // Line pipeline: built on the first frame that publishes lines,
        // so the graph gate below can see it live this same frame.
        self.ensure_line_pipeline(!lines.is_empty());
        // This frame's ribbon geometry, written into this slot's persistent
        // vertex buffer up front for the same reason the text geometry is: the
        // line pass encodes through `&self`, so it cannot upload for itself.
        self.upload_lines(ring_slot, lines)?;

        // Single render-graph dispatch for the full frame.
        // The merged graph contains every Metal pass that once ran
        // inline through `draw_frame`. The compile pass derives
        // execution order, per-pass barriers, and resource lifetimes
        // from the RAW + WAW + WAR edges over the version-chained
        // read / write declarations in `build_frame_graph`. Composite
        // is the presenter and runs last; the drawable is fetched at
        // frame start above and stays alive through `presentDrawable`
        // below.
        let graph_inputs = self.frame_graph_inputs(GraphInputArgs {
            bindless_cull_enabled: object_buffer.is_some() && cull_draw_args.is_some(),
            velocity_active,
            fog_settings: fog_settings.as_ref(),
            transparent_active,
            lines,
            world_hidden,
            clustered,
            view_mode,
            show,
        });
        // Reuse the cached compiled graph when this frame's inputs match the
        // ones it was built from (the common case: graph topology changes only
        // when a feature toggles or a target resizes). Taken out of the cache so
        // the later `&mut self` execute_graph does not conflict with a borrow of
        // it; put back after execution. A mismatch (or a cold cache) rebuilds.
        let graph = match self.draw.graph_cache.take() {
            Some((cached_inputs, cached_graph)) if cached_inputs == graph_inputs => cached_graph,
            // A graph the core refuses to build is a topology mistake, not a
            // device failure.
            _ => render_graph::build_frame_graph(&graph_inputs)
                .map_err(|e| error::RenderError::Other(format!("frame graph: {e}")))?,
        };
        // This frame's skinned deformed-vertex buffer (skinned fold), cloned into
        // a local so `params` owns a handle rather than borrowing `self.skinned`
        // across the `&mut self` execute_graph call (every other GraphFrameParams
        // buffer is likewise a local). `Some` only when the fold is active
        // (draw.n_skinned > 0, set in upload_skinned under bindless + static geometry);
        // the Cull pass writes it via encode_main_skin and the Main / Main2
        // skinned ICB tail binds it.
        let deformed_this_frame = if self.draw.n_skinned > 0 {
            self.skinned.deformed.get(ring_slot).cloned()
        } else {
            None
        };
        // The previous frame's deformed slot (one behind in the ring), read by
        // the GPU-driven G-buffer skinned tail for per-vertex skin motion. The
        // priming gate (`deformed_primed`) covers the unposed first frame.
        let deformed_prev_frame = if self.draw.n_skinned > 0 {
            let prev_slot = (ring_slot + self.frames_in_flight - 1) % self.frames_in_flight;
            self.skinned.deformed.get(prev_slot).cloned()
        } else {
            None
        };
        // Model-history ring slots for the GPU-driven G-buffer pass: the one the
        // previous frame's snapshot filled, which this frame reprojects through,
        // and the one(s) this frame's snapshot fills. Both are bound whenever the
        // pre-pass runs, motion consumer or not -- the pass still writes the
        // normals and depth every screen-space consumer reads. Priming writes
        // every slot, so the first pre-pass after a rebuild reads this frame's
        // models rather than an unwritten buffer.
        let (prev_model_buffer, history_targets) = if object_buffer.is_some()
            && self.gbuffer.targets.is_some()
            && self.gbuffer.bindless_pipeline.is_some()
        {
            let bytes = self.cull_count() * std::mem::size_of::<[[f32; 4]; 4]>();
            let prime = self.model_history.take_prime();
            let read_slot = (ring_slot + self.frames_in_flight - 1) % self.frames_in_flight;
            let mut targets = Vec::new();
            if prime {
                for slot in 0..self.frames_in_flight {
                    targets.push(
                        self.rings
                            .model_history
                            .slot(&self.hw.device, slot, bytes)?,
                    );
                }
            } else {
                targets.push(
                    self.rings
                        .model_history
                        .slot(&self.hw.device, ring_slot, bytes)?,
                );
            }
            let read = self
                .rings
                .model_history
                .slot(&self.hw.device, read_slot, bytes)?;
            (Some(read), targets)
        } else {
            (None, Vec::new())
        };
        // This frame's HUD text geometry, written into this slot's persistent
        // upload buffer up front so the composite pass binds sub-ranges of one
        // buffer instead of minting a pair per label mid-encode. Done here, past
        // the frames-in-flight fence, so overwriting the slot cannot race a GPU
        // read of the frame that last used it.
        self.text
            .upload
            .upload(&self.hw.device, ring_slot, text_calls)?;

        let params = GraphFrameParams {
            cmd_buf: &cmd_buf,
            cam_pos,
            skinned_joint_bufs: &skinned_joint_bufs,
            skinned_morph_weight_bufs: &skinned_morph_weight_bufs,
            scene_color: Some(&scene_color),
            text_calls,
            ring_slot,
            world_hidden,
            elapsed,
            vp,
            inv_vp,
            frustum: &frustum,
            object_buffer: object_buffer.as_ref(),
            bindless_tex_args: bindless_tex_args.as_ref(),
            deformed_skinned: deformed_this_frame.as_ref(),
            deformed_prev: deformed_prev_frame.as_ref(),
            prev_model_buffer: prev_model_buffer.as_ref(),
            history_targets: &history_targets,
            draw_args_buffer: cull_draw_args.as_ref(),
            vel_uniforms: vel_uniforms.as_ref(),
            scene_pre_taa: if self.taa.enabled
                || self.upscale.scaler.is_some()
                || transparent_active
            {
                Some(&scene_input)
            } else {
                None
            },
            ssr_params: ssr_params.as_ref(),
            fog_params: fog_params.as_ref(),
            fog_froxel_params: fog_froxel_params.as_ref(),
            cluster_params: if clustered {
                Some(&cluster_params)
            } else {
                None
            },
            ssao_params: ssao_params.as_ref(),
            ssgi_params: ssgi_params.as_ref(),
            rt_reflection_params: rt_reflection_params.as_ref(),
        };
        // The frame's completion join. Every command buffer the frame submits
        // registers a part before it is committed and arrives from its
        // completion handler; the last arrival resolves the frame's per-pass
        // timings and releases the frame-in-flight slot. The join is built
        // before submission because the graph executor attaches the async
        // queue's parts, and the submission token holds it open until this
        // frame has finished recording -- including the error paths, where the
        // token's Drop is the only arrival.
        //
        // Per-pass timings resolve here rather than from the presenting command
        // buffer's own handler because the async queue's terminal pass
        // (`HizFinal`) is not an ancestor of the composite: its sample-buffer
        // slots would still be a few frames stale when the composite retires.
        let gpu_time = std::sync::Arc::clone(&self.diagnostics.gpu_time_us);
        let pass_times = std::sync::Arc::clone(&self.diagnostics.pass_times_us);
        // The composite command buffer's own GPU span, published by its handler
        // as the fallback whole-frame time when per-pass timing is unavailable.
        let composite_span_us = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        // Which passes actually ran this frame. The sample buffer is reused
        // across frames and never cleared, so a pass absent this frame (e.g.
        // every world pass behind an opaque menu) would otherwise resolve to
        // its last run's stale timestamps; the resolve zeroes those slots. Read
        // on this thread once the frame is recorded, which the submission token
        // orders before any arrival can run the completion work.
        let active_mask = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let pass_buffer = self
            .diagnostics
            .pass_timing
            .as_ref()
            .map(|p| super::pass_timing::SendableSampleBuf(p.buffer_for(pass_timing_slot)));
        let (join, submission_token) = frame_slot.into_join({
            let composite_span_us = std::sync::Arc::clone(&composite_span_us);
            let active_mask = std::sync::Arc::clone(&active_mask);
            Box::new(move || {
                use std::sync::atomic::Ordering as AtomicOrdering;
                let mut frame_us = composite_span_us.load(AtomicOrdering::Relaxed);
                if let Some(buf) = &pass_buffer {
                    let mask = active_mask.load(AtomicOrdering::Relaxed);
                    let per_pass = super::pass_timing::resolve(&buf.0, mask);
                    for (slot, micros) in pass_times.iter().zip(per_pass.iter()) {
                        slot.store(*micros, AtomicOrdering::Relaxed);
                    }
                    // First pass start to last pass end, across both queues:
                    // the frame's GPU span, which the per-pass times cannot be
                    // summed into once passes overlap.
                    if let Some(span_us) = super::pass_timing::frame_span_us(&buf.0, mask) {
                        frame_us = span_us;
                    }
                }
                gpu_time.store(frame_us, AtomicOrdering::Relaxed);
            })
        });

        let submission = self.execute_graph(&graph, &params, &join)?;
        if let Some(timing) = self.diagnostics.pass_timing.as_ref() {
            active_mask.store(timing.attached_mask(), std::sync::atomic::Ordering::Relaxed);
        }
        // Cache the compiled graph under this frame's inputs so the next frame
        // with matching inputs skips the rebuild.
        self.draw.graph_cache = Some((graph_inputs, graph));

        self.submit_and_present(PresentFrame {
            cmd_buf,
            drawable,
            join,
            composite_span_us,
            pending_terminal: submission.pending_terminal,
            submission_token,
        });
        self.advance_temporal_state(velocity_active, proj);

        Ok(())
    }

    // Update the RT acceleration structure to this frame's transforms. The
    // per-frame skinned path (`update_rt_skinned` -> `rebuild_skinned`) keeps the
    // persistent static/cluster BLAS and rebuilds only the skinned BLAS + TLAS +
    // geometry table from the current pose; the non-skinned `rebuild_tlas` path
    // and the one-time seed rebuild the TLAS (or the whole BVH). All paths
    // allocate fresh and retire the outgoing structures through a deferred-free
    // pool keyed on `frame_id`, so a prior in-flight frame keeps reading the old
    // structures. The skinned skin-compute + BLAS/TLAS build are committed without
    // waiting and ordered against the trace by same-queue commit order (both cmd
    // bufs are committed here, before the trace cmd buf in `execute_graph`, on the
    // shared queue). A no-op when RT is off or the scene is static (`Off`).
    //
    // `Auto` (the default) rebuilds the TLAS only when a participating
    // transform actually changed; `Rebuild` / `Tlas` force their work every
    // frame and exist only as diagnostics.
    // Keep the RT acceleration structure current with this frame's transforms
    // and skinned pose. Non-fatal: a per-frame rebuild can fail transiently
    // (e.g. a momentary acceleration-structure allocation hiccup under the
    // per-frame skinned rebuild), and a reflection-BVH update failure must
    // never stop the whole renderer. On failure the previous frame's BVH is
    // kept (the reflection is at most one frame stale, imperceptible) and the
    // failure is logged once per streak (and once on recovery), not at frame
    // rate. The actual work is in `rt_dynamic_update_inner`.
    fn rt_dynamic_update(
        &mut self,
        frame: super::raytrace::RtFrame,
        joint_buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) {
        match self.rt_dynamic_update_inner(frame, joint_buffers) {
            Ok(()) => {
                if self.rt.update_failed {
                    tracing::info!("ray-traced reflections: BVH update recovered");
                    self.rt.update_failed = false;
                }
            }
            Err(e) => {
                if !self.rt.update_failed {
                    tracing::warn!(
                        "ray-traced reflections: keeping last frame's BVH, update failed: {e}"
                    );
                    self.rt.update_failed = true;
                }
            }
        }
    }

    fn rt_dynamic_update_inner(
        &mut self,
        frame: super::raytrace::RtFrame,
        joint_buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) -> error::RenderResult<()> {
        use concinnity_core::render::rt_geom::RtDynamicMode;
        let frame_id = frame.id;
        if !self.rt.dynamic_mode.is_dynamic() {
            return Ok(());
        }
        // RT reflections are not enabled this run (no settings, or the GPU lacks
        // ray tracing): there is no BVH to keep current, and a lingering topology
        // flag must not trigger a build. Clear it and bail.
        if self.rt.settings.is_none() {
            self.rt.topology_dirty = false;
            return Ok(());
        }
        let albedo_count = self.scene.textures.len();

        // Free resources parked by prior skinned rebuilds that the frames-in-
        // flight fence now guarantees no in-flight frame can still read.
        let depth = self.frames_in_flight;
        if let Some(accel) = self.rt.accel.as_mut() {
            accel.retire_completed(frame_id, depth);
        }

        // Did a streamed chunk, cloned prop, or participation-changing material
        // edit alter the RT-relevant draw set since the last update? Consume the
        // flag; the BLAS topology must be refreshed below rather than ignored (the
        // `Auto` dirty check only watches the transforms of the prior set).
        let topology_changed = std::mem::take(&mut self.rt.topology_dirty);

        // Skinned meshes deform every frame, so their BLAS (baked from the posed
        // vertices) must be rebuilt each frame: a TLAS-only rebuild can't
        // re-skin. But the static + cluster BLAS never change under a rigid
        // transform, so only the skinned tail (+ TLAS + geometry table) needs
        // rebuilding. `rebuild_skinned` does exactly that, keeping the persistent
        // static BLAS; a full `rebuild_rt_accel` is used only to seed the BVH the
        // first frame after `upload_skinned` (the init build is static-only) or
        // when the `Rebuild` diagnostic forces a from-scratch build every frame.
        let has_skinned = self.rt.skinned_geometry
            && !self.skinned.slots.draw_objects.is_empty()
            && self.rt.pipelines.skin.is_some();
        if has_skinned {
            if self.rt.accel.is_none() || self.rt.dynamic_mode == RtDynamicMode::Rebuild {
                return self.rebuild_rt_accel(albedo_count);
            }
            // Fold any added/removed draw geometry into the static head (BLAS only,
            // async), then the skinned path rebuilds the TLAS + table over the
            // refreshed head + the fresh skinned tail.
            if topology_changed {
                self.refresh_rt_topology(albedo_count, false, frame_id)?;
            }
            return self.update_rt_skinned(albedo_count, frame, joint_buffers);
        }
        // No skinned geometry.
        if self.rt.accel.is_none() {
            // A topology change can introduce the first participating geometry
            // (e.g. the first streamed chunk in a world that began empty): seed
            // the BVH from scratch. Otherwise nothing to keep current.
            if topology_changed {
                return self.rebuild_rt_accel(albedo_count);
            }
            return Ok(());
        }
        // The `Rebuild` diagnostic rebuilds every BLAS every frame, which already
        // absorbs any topology change.
        if self.rt.dynamic_mode == RtDynamicMode::Rebuild {
            return self.rebuild_rt_accel(albedo_count);
        }
        if topology_changed {
            // Incrementally refresh the draw-object BLAS head AND rebuild the TLAS
            // over the refreshed set, all async on one command buffer (the
            // transform dirty check only sees the prior set, so the rebuild is
            // forced). `build_tlas = true` does the TLAS inline -- no separate
            // `rebuild_rt_tlas` follow-up.
            self.refresh_rt_topology(albedo_count, true, frame_id)?;
            if self.rt.accel.as_ref().is_some_and(|a| a.is_empty()) {
                // The refresh removed the last draw + cluster geometry; drop the
                // BVH so a later add re-seeds it instead of building a degenerate
                // zero-instance TLAS.
                self.rt.accel = None;
            }
            return Ok(());
        }
        match self.rt.dynamic_mode {
            RtDynamicMode::Auto => {
                // Cheap shared-borrow dirty check; rebuild only if something moved.
                let dirty = self
                    .rt
                    .accel
                    .as_ref()
                    .expect("rt_accel is Some (checked above)")
                    .transforms_dirty(&self.draw.objects);
                if dirty {
                    self.rebuild_rt_tlas(albedo_count)?;
                }
            }
            RtDynamicMode::Tlas => self.rebuild_rt_tlas(albedo_count)?,
            // Handled above / filtered out by the `is_dynamic` guard.
            RtDynamicMode::Rebuild | RtDynamicMode::Off => {}
        }
        Ok(())
    }

    // Incrementally refresh the RT draw-object BLAS head to match the current
    // draw set (added/removed chunks, cloned props, participation-changing
    // material edits), reusing every unchanged BLAS, async. `build_tlas` also
    // rebuilds the TLAS + geometry table inline (the no-skinned path); when clear,
    // the caller's `rebuild_skinned` rebuilds the TLAS over the refreshed head +
    // skinned tail. Borrows the accel mutably while reading the device / queue /
    // shared buffers / draw list, so the cheap handles are cloned and the draw
    // list is lifted out (an O(1) `Vec` swap) to keep the borrows disjoint, then
    // restored.
    fn refresh_rt_topology(
        &mut self,
        albedo_count: usize,
        build_tlas: bool,
        frame_id: u64,
    ) -> error::RenderResult<()> {
        let device = self.hw.device.clone();
        let queue = self.hw.command_queue.clone();
        let vbuf = self.scene.vertex_buffer.retained();
        let ibuf = self.scene.index_buffer.retained();
        let exclude_seethrough = self.seethrough_meshes_enabled();
        let draw_objects = std::mem::take(&mut self.draw.objects);
        let res = self
            .rt
            .accel
            .as_mut()
            .expect("rt_accel is Some (checked by caller)")
            .refresh_static_topology(
                super::raytrace::RtGpu {
                    device: &device,
                    command_queue: &queue,
                    frames_in_flight: self.frames_in_flight,
                },
                super::raytrace::RtStaticGeometry {
                    vertex_buffer: &vbuf,
                    index_buffer: &ibuf,
                },
                &draw_objects,
                super::raytrace::RtTextureCounts { albedo_count },
                super::raytrace::RtTopologyRefreshOptions {
                    exclude_seethrough,
                    build_tlas,
                    frame_id,
                },
            );
        self.draw.objects = draw_objects;
        res
    }

    // Full BVH rebuild (fresh BLAS + TLAS + table) from the current draw list,
    // instanced clusters, and skinned pose. The proven hazard-free path (fresh
    // allocations): used by the `Rebuild` diagnostic mode and, every frame, by
    // any scene with skinned geometry (its deformed vertices change per frame).
    // Replaces `rt_accel` only on a successful non-empty build, so a transient
    // failure or an emptied scene leaves the previous BVH in place. The
    // immutable borrows of `self` all end when the build returns, before the
    // assignment, so there is no aliasing.
    pub(in crate::metal) fn rebuild_rt_accel(
        &mut self,
        albedo_count: usize,
    ) -> error::RenderResult<()> {
        use super::raytrace::SkinnedRtInputs;
        let skinned = match (
            &self.skinned.vertex_buffer,
            &self.skinned.index_buffer,
            &self.rt.pipelines.skin,
        ) {
            (Some(svb), Some(sib), Some(pipe))
                if !self.skinned.slots.draw_objects.is_empty() && self.rt.skinned_geometry =>
            {
                Some(SkinnedRtInputs {
                    objects: &self.skinned.slots.draw_objects,
                    vertex_buffer: svb,
                    index_buffer: sib,
                    joint_matrices: &self.skinned.slots.joint_matrices,
                    skin_pipeline: pipe.as_ref(),
                })
            }
            _ => None,
        };
        let built = super::raytrace::build_rt_accel(
            super::raytrace::RtGpu {
                device: &self.hw.device,
                command_queue: &self.hw.command_queue,
                frames_in_flight: self.frames_in_flight,
            },
            super::raytrace::RtStaticGeometry {
                vertex_buffer: &self.scene.vertex_buffer,
                index_buffer: &self.scene.index_buffer,
            },
            super::raytrace::RtSceneGeometry {
                draw_objects: &self.draw.objects,
                clusters: &self.instanced.clusters,
            },
            super::raytrace::RtTextureCounts { albedo_count },
            skinned,
            self.seethrough_meshes_enabled(),
        )?;
        if let Some(accel) = built {
            self.rt.accel = Some(accel);
        }
        Ok(())
    }

    // Per-frame skinned RT update: rebuild only the skinned BLAS + TLAS +
    // geometry table (keeping the persistent static/cluster BLAS) from the
    // current pose and transforms. The accel is borrowed mutably while the
    // skinned inputs are borrowed immutably, so the cheap handles are cloned and
    // the draw list is lifted out (an O(1) `Vec` swap) to keep the borrows
    // disjoint, then restored. A no-op (keeps last frame's BVH) if the required
    // skinned resources are missing.
    fn update_rt_skinned(
        &mut self,
        albedo_count: usize,
        frame: super::raytrace::RtFrame,
        joint_buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) -> error::RenderResult<()> {
        use super::raytrace::SkinnedRtInputs;
        let device = self.hw.device.clone();
        let queue = self.hw.command_queue.clone();
        let frames_in_flight = self.frames_in_flight;
        let (Some(svb), Some(sib), Some(pipe)) = (
            self.skinned.vertex_buffer.clone(),
            self.skinned.index_buffer.clone(),
            self.rt.pipelines.skin.clone(),
        ) else {
            return Ok(());
        };
        let draw_objects = std::mem::take(&mut self.draw.objects);
        let skinned = SkinnedRtInputs {
            objects: &self.skinned.slots.draw_objects,
            vertex_buffer: &svb,
            index_buffer: &sib,
            joint_matrices: &self.skinned.slots.joint_matrices,
            skin_pipeline: pipe.as_ref(),
        };
        let res = self
            .rt
            .accel
            .as_mut()
            .expect("rt_accel is Some (checked by caller)")
            .rebuild_skinned(
                super::raytrace::RtGpu {
                    device: &device,
                    command_queue: &queue,
                    frames_in_flight,
                },
                &draw_objects,
                skinned,
                joint_buffers,
                super::raytrace::RtTextureCounts { albedo_count },
                frame,
            );
        self.draw.objects = draw_objects;
        res
    }

    // Rebuild just the TLAS + geometry table (fresh allocations, static BLAS)
    // from the current draw-object transforms. `rebuild_tlas` borrows the accel
    // mutably while reading the device / queue / draw list, so clone the two
    // cheap handles and lift the draw list out (an O(1) `Vec` swap) to keep the
    // borrows from aliasing, then put the draw list back.
    fn rebuild_rt_tlas(&mut self, albedo_count: usize) -> error::RenderResult<()> {
        let device = self.hw.device.clone();
        let queue = self.hw.command_queue.clone();
        let draw_objects = std::mem::take(&mut self.draw.objects);
        let res = self
            .rt
            .accel
            .as_mut()
            .expect("rt_accel is Some (checked by caller)")
            .rebuild_tlas(&device, &queue, &draw_objects, albedo_count);
        self.draw.objects = draw_objects;
        res
    }

    // Rebuild the off-screen render targets whose footprint follows the
    // drawable size: HDR color + depth + resolve, bloom chain, TAA history +
    // velocity (when TAA is on), SSAO targets (when SSAO is on), SSR targets
    // (when SSR is on). Called at the top of every frame; only the targets
    // that actually changed dimensions are recreated.
    //
    // When MetalFX upscaling is on, "render resolution" (where the 3D scene
    // draws) is `output * upscale_scale`, smaller than the drawable. Bloom
    // and the MetalFX output texture stay at the drawable (output)
    // resolution so the final composite reads cleanly into the swapchain.
    fn resize_targets_if_needed(&mut self, want_w: u32, want_h: u32) -> error::RenderResult<()> {
        // A MetalFX scaler is bound to one (input, output) size pair at
        // construction, so a changed output needs a fresh instance. Rebuild
        // before deriving the render resolution below, which reads the sizes
        // this decides. Reset the temporal history on the first frame after, so
        // the new scaler does not pull from a stale buffer.
        if let Some(u) = self.upscale.scaler.as_ref()
            && (want_w != u.output_width || want_h != u.output_height)
        {
            self.upscale.scaler = Some(super::post::MetalFXUpscaler::new(
                &self.hw.device,
                want_w,
                want_h,
                self.upscale.scale,
            )?);
            self.upscale
                .reset_pending
                .store(true, std::sync::atomic::Ordering::Release);
        }

        // Output (drawable) dimensions are the `want_w/h` arg; the render
        // dimensions are the scaler's own input size, taken from it rather than
        // recomputed. `upscale.scale` is a rounded ratio (`input / output`), so
        // multiplying it back out can land a pixel low -- at 2048x1536 the
        // scaler declares a 1024-row input and the arithmetic yields 1023, and
        // MetalFX asserts that the content exceeds the texture. The scaler is
        // the authority on the size it was built for.
        let (render_w, render_h) = match self.upscale.scaler.as_ref() {
            Some(u) => (u.input_width.max(1), u.input_height.max(1)),
            None => (want_w, want_h),
        };

        let render_changed =
            render_w != self.targets.hdr.width || render_h != self.targets.hdr.height;
        if render_changed {
            self.targets.hdr = super::texture::create_hdr_targets(
                &self.hw.device,
                render_w,
                render_h,
                self.targets.hdr.sample_count,
            )?;
        }
        // The planar reflection targets are render-resolution (they re-render the
        // scene from the mirrored camera at the same resolution the reflectors
        // sample). The plane set carries over; only the targets are reallocated.
        if render_changed && let Some(set) = self.planar_reflection.as_ref() {
            let planes = set.planes.clone();
            self.planar_reflection = Some(super::planar::create_planar_set(
                &self.hw.device,
                render_w,
                render_h,
                self.targets.hdr.sample_count,
                &planes,
            )?);
        }
        // The bloom chain reads `scene_color`: at drawable size when the
        // upscaler runs, otherwise at native (= render) resolution. Sized
        // off `want_w/h` either way.
        let bloom_changed =
            want_w != self.targets.bloom.width || want_h != self.targets.bloom.height;
        // Whether the unified G-buffer pre-pass runs, derived once so the pool
        // and the pre-pass's own depth target below cannot disagree about it.
        // Same expression as `EffectSettings::gbuffer_needed`.
        let needs_gbuffer = self.ssr.settings.is_some()
            || self.ssgi.settings.is_some()
            || self.rt.settings.is_some()
            || self.ssao.settings.is_some()
            || self.taa.enabled
            || self.upscale.scaler.is_some();
        // The transient pool backs `ao_output` and the G-buffer channels (all
        // render-resolution) plus `bloom_top` (half output-resolution), so
        // either extent moving invalidates it. The bloom chain then rebuilds
        // around the pool's fresh top mip. Nothing else caches a pooled handle:
        // the per-frame bindless argument buffer re-encodes `ao_output` itself
        // and every G-buffer consumer fetches its channel by label at encode
        // time, which is what makes a rebuild's slot repack harmless here.
        if render_changed || bloom_changed {
            self.targets.transient_pool.rebuild(
                &self.hw.device,
                &super::transient_pool::transient_slots(
                    self.ssao.settings.is_some(),
                    needs_gbuffer,
                    (render_w, render_h),
                    (want_w, want_h),
                )?,
            )?;
            self.targets.bloom = super::post::create_bloom_targets(
                &self.hw.device,
                want_w,
                want_h,
                self.targets.transient_pool.bloom_top()?,
            )?;
        }
        // The TAA history + velocity buffers are render-resolution. Stale
        // history can't be reprojected into the new resolution, so mark
        // it invalid: the next frame passes straight through and
        // accumulation restarts.
        if render_changed && let Some(mut taa) = self.taa.pass.take() {
            let r = taa.resize(
                &self.post_device(),
                PostExtent {
                    width: render_w,
                    height: render_h,
                },
            );
            self.taa.pass = Some(taa);
            r?;
        }
        // The SSAO kernel's raw-occlusion target is render-resolution. Its depth
        // + normal input now comes from the unified G-buffer pre-pass (below),
        // so SSAO owns no G-buffer of its own; its blurred output is the pool's
        // `ao_output`, rebuilt above.
        if render_changed && self.ssao.settings.is_some() {
            self.ssao.targets = Some(super::post::create_ssao_targets(
                &self.hw.device,
                render_w,
                render_h,
            )?);
        }
        // The SSR resolve-output target is render-resolution. Rebuilt when SSR,
        // SSGI, *or* RT reflections are on (RT reuses `ssr_targets.output`). The
        // acceleration structure is resolution-independent, so it is not
        // rebuilt here.
        if render_changed
            && (self.ssr.settings.is_some()
                || self.ssgi.settings.is_some()
                || self.rt.settings.is_some())
        {
            self.ssr.targets = Some(super::post::create_ssr_targets(
                &self.hw.device,
                render_w,
                render_h,
                self.ssr.blur_scale,
            )?);
        }
        // The pre-pass's depth attachment is render-resolution and stays
        // feature-owned; its three color channels were rebuilt with the pool
        // above. Same gate, so the two halves of the pre-pass's targets are
        // always present or absent together.
        if render_changed && needs_gbuffer {
            self.gbuffer.targets = Some(super::post::create_gbuffer_targets(
                &self.hw.device,
                render_w,
                render_h,
            )?);
        }
        // The SSGI gather target is render-resolution scaled by `gi_scale`
        // (the composite bilateral-upsamples it back to full resolution).
        if render_changed && let Some(mut ssgi) = self.ssgi.pass.take() {
            let r = ssgi.resize(
                &self.post_device(),
                PostExtent {
                    width: render_w,
                    height: render_h,
                },
            );
            self.ssgi.pass = Some(ssgi);
            r?;
        }
        // The Hi-Z pyramid matches the render (depth) resolution. Rebuild it
        // and mark it invalid so the next cull dispatch ignores the now-stale
        // pyramid (the projection coordinates were generated at the old
        // resolution); the next frame's build refills it.
        if render_changed && let Some(hiz) = self.cull.hiz.as_mut() {
            hiz.resize_to(&self.hw.device, render_w, render_h)?;
            self.cull.hiz_valid = false;
        }
        Ok(())
    }
}
