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
mod pass_uniforms;
mod shadow;
mod spot_shadow;
mod stages;

use concinnity_core::render::backend::FrameParams;
use concinnity_core::render::error;
use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::reactive_mask::ReactiveMaskPlan;
use concinnity_core::render::render_graph;
use concinnity_core::render::rt_accel::RtUpdate;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandQueue as _};

use self::pass_uniforms::{PassUniformArgs, PassUniforms};
use self::stages::{
    AcquiredFrame, FrameProjection, GraphInputArgs, HistoryBuffers, PresentFrame, SceneBufferArgs,
    SceneBuffers,
};
use super::context::MtlContext;
use super::graph_exec::GraphFrameParams;

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
            view_distance,
            cam_pos,
            text_calls,
            lines,
            world_hidden,
            view_mode,
            show,
            sky_rot,
            history_reset,
            rebase,
            grass_benders,
        } = params;
        let mtm = objc2::MainThreadMarker::new().ok_or_else(|| {
            error::RenderError::Other("draw_frame must be called from the main thread".into())
        })?;
        // Snapped for the pass encoders (wireframe fill mode, unlit shading,
        // the composite's channel visualization + depth normalization).
        self.state.view.mode = view_mode;
        self.state.view.near = near;
        self.state.view.view_distance = view_distance;
        self.state.view.sky_rot = sky_rot;
        self.apply_pending_rebuilds()?;
        // A moved world's decals move with it whatever happens to history.
        if let Some(rebase) = &rebase {
            self.decal.set.rebase(rebase);
        }
        if history_reset {
            self.reset_temporal_history();
        } else if let Some(rebase) = rebase {
            self.rebase_temporal_history(&rebase);
        }

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

        let aspect = self.update_shadow_schedule(cam_pos, fov_y_radians, near, view_distance);

        let (render_w, render_h) = self.resize_frame_targets()?;

        let FrameProjection {
            proj,
            vp,
            inv_vp,
            frustum,
        } = self.frame_projection(
            fov_y_radians,
            aspect,
            near,
            view_distance,
            render_w,
            render_h,
        );

        let texture_signature = self.refresh_probe_records_and_residency(ring_slot)?;

        let SceneBuffers {
            object_buffer,
            material_params,
            cull_draw_args,
            bindless_tex_args,
        } = self.build_scene_buffers(SceneBufferArgs {
            ring_slot,
            frame_id,
            cam_pos,
            elapsed,
            world_hidden,
            skinned_joint_bufs: &skinned_joint_bufs,
            texture_signature,
        })?;

        let PassUniforms {
            ssao_params,
            ssr_params,
            ssgi_params,
            rt_reflection_params,
            fog_settings,
            fog_params,
            fog_froxel_params,
            clustered,
            cluster_params,
            velocity_active,
            gbuffer_view,
            scene_input,
            scene_color,
            transparent_active,
        } = self.frame_pass_uniforms(PassUniformArgs {
            fov_y_radians,
            aspect,
            near,
            view_distance,
            cam_pos,
            sky_rot,
            proj,
            vp,
            render_w,
            render_h,
            elapsed,
        })?;

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
        let graph = match self.graph_cache.take() {
            Some((cached_inputs, cached_graph)) if cached_inputs == graph_inputs => cached_graph,
            // A graph the core refuses to build is a topology mistake, not a
            // device failure.
            _ => render_graph::build_frame_graph(&graph_inputs)
                .map_err(|e| error::RenderError::Other(format!("frame graph: {e}")))?,
        };
        // The grass kernel's inputs, advanced only on a frame whose graph runs it,
        // so the draw-argument slot it fills alternates frame to frame.
        let grass_frame = (graph_inputs.grass_enabled && !graph_inputs.world_hidden)
            .then(|| {
                self.prepare_grass_frame(crate::metal::grass::GrassRequest {
                    cam_pos,
                    vp: gbuffer_view.cur_vp,
                    elapsed,
                    cast: graph_inputs.grass_shadow_enabled,
                    benders: grass_benders,
                })
            })
            .flatten();
        let HistoryBuffers {
            deformed_this_frame,
            deformed_prev_frame,
            prev_model_buffer,
            history_targets,
        } = self.build_history_buffers(ring_slot, object_buffer.is_some())?;
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
            planar: self
                .planar_reflection
                .as_ref()
                .map(|set| set.layout.frame_plan(vp))
                .unwrap_or_default(),
            object_buffer: object_buffer.as_ref(),
            material_params: material_params.as_ref(),
            bindless_tex_args: bindless_tex_args.as_ref(),
            deformed_skinned: deformed_this_frame.as_ref(),
            deformed_prev: deformed_prev_frame.as_ref(),
            prev_model_buffer: prev_model_buffer.as_ref(),
            history_targets: &history_targets,
            draw_args_buffer: cull_draw_args.as_ref(),
            gbuffer_view: &gbuffer_view,
            velocity_active,
            grass: grass_frame.as_ref(),
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
            reactive: ReactiveMaskPlan::of(&graph_inputs),
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
        self.graph_cache = Some((graph_inputs, graph));

        self.submit_and_present(PresentFrame {
            cmd_buf,
            drawable,
            join,
            composite_span_us,
            pending_terminal: submission.pending_terminal,
            submission_token,
        });
        self.advance_temporal_state(
            velocity_active,
            concinnity_core::render::view_history::ViewFrame {
                vp: concinnity_core::transform::mat4_mul(proj, self.state.view.matrix),
                elapsed,
                cam_pos,
            },
        );

        Ok(())
    }

    // Keep the RT acceleration structure current with this frame's transforms
    // and skinned pose, following the shared update plan (`rt_accel`): refresh
    // the draw BLAS head when the participating set changed, then re-skin,
    // rebuild the TLAS, or keep it. The skinned skin-compute + BLAS/TLAS build
    // and a topology refresh are committed without waiting and ordered against
    // the trace by same-queue commit order (both cmd bufs are committed here,
    // before the trace cmd buf in `execute_graph`, on the shared queue); the
    // static TLAS rebuild commits and waits. Outgoing structures are retired
    // through a deferred-free pool keyed on the frame id.
    //
    // Non-fatal: a per-frame rebuild can fail transiently (e.g. a momentary
    // acceleration-structure allocation hiccup under the per-frame skinned
    // rebuild), and a reflection-BVH update failure must never stop the whole
    // renderer. On failure the previous frame's BVH is kept (the reflection is
    // at most one frame stale, imperceptible) and the failure is logged once per
    // streak (and once on recovery), not at frame rate.
    fn rt_dynamic_update(
        &mut self,
        frame: super::raytrace::RtFrame,
        joint_buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) {
        // Consumed whether or not anything below runs: a change the BVH cannot
        // follow (RT off, or a mode that never updates) is not owed later.
        let topology_dirty = std::mem::take(&mut self.state.gpu_dirty.rt_topology);
        self.rt.collect_retired(frame.id, self.frames_in_flight);
        if self.rt.settings.is_none() {
            return;
        }
        let result = self.rt_dynamic_update_inner(frame, joint_buffers, topology_dirty);
        crate::rt_report::report_rt_update(&mut self.rt.update_streak, result);
    }

    fn rt_dynamic_update_inner(
        &mut self,
        frame: super::raytrace::RtFrame,
        joint_buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        topology_dirty: bool,
    ) -> error::RenderResult<RtUpdate> {
        use concinnity_core::render::rt_accel::{RtStep, seed_wanted};
        let mode = self.rt.dynamic_mode;
        let albedo_count = self.scene.textures.len();
        let skinned_present = self.rt.skinned_geometry
            && !self.state.skinned.draw_objects.is_empty()
            && self.rt.pipelines.skin.is_some()
            && self.skinned.vertex_buffer.is_some()
            && self.skinned.index_buffer.is_some();

        let Some(accel) = self.rt.accel.as_mut() else {
            if seed_wanted(mode, topology_dirty, skinned_present) {
                self.rebuild_rt_accel(albedo_count)?;
            }
            return Ok(RtUpdate::Done);
        };
        // Free resources parked by prior rebuilds that the frames-in-flight fence
        // now guarantees no in-flight frame can still read.
        accel.retire_completed(frame.id, self.frames_in_flight);
        accel.set_albedo_count(albedo_count);
        let skinned_objects = skinned_present.then_some(self.state.skinned.draw_objects.as_slice());
        let Some(plan) = accel.book_mut().plan(mode, topology_dirty, skinned_objects) else {
            return Ok(RtUpdate::Done);
        };

        // A failed refresh keeps the last head, is planned again next frame, and
        // still lets the step below run.
        let mut refreshed = Ok(());
        if let Some(refresh) = plan.refresh {
            let shape = super::raytrace::RefreshShape {
                skinned_follows: plan.skinned,
                skinned_present,
            };
            refreshed = self.refresh_rt_topology(refresh, shape, frame.id);
            if refreshed.is_err()
                && let Some(accel) = self.rt.accel.as_mut()
            {
                accel.book_mut().owe_refresh();
            }
        }

        let Some(accel) = self.rt.accel.as_mut() else {
            return refreshed.map(|()| RtUpdate::Done);
        };
        let step = accel
            .book_mut()
            .next_step(mode, &plan, &self.state.draw.objects);
        let nothing_can_rejoin = accel.is_empty() && !skinned_present;
        let stepped = match step {
            RtStep::Keep => Ok(RtUpdate::Done),
            // Nothing static is left and no skinned geometry can rejoin: stop
            // publishing the skinned tail, which leaves the BVH spent.
            RtStep::Tlas if nothing_can_rejoin => {
                accel.release_skinned_tail(frame.id);
                Ok(RtUpdate::Done)
            }
            RtStep::Tlas => self.rebuild_rt_tlas(frame.id).map(|()| RtUpdate::Done),
            RtStep::Skinned => {
                self.update_rt_skinned(frame, joint_buffers, plan.full_skinned_build)
            }
        };
        // A refresh on the skinned path leaves the TLAS to the skinned step, so a
        // step that did not build it keeps the refresh owed.
        if plan.refresh.is_some()
            && plan.skinned
            && !matches!(stepped, Ok(RtUpdate::Done))
            && let Some(accel) = self.rt.accel.as_mut()
        {
            accel.book_mut().owe_refresh();
        }
        // The last draw + cluster geometry is gone and no skinned geometry can
        // rejoin (`EmptyHead::Drop`): drop the BVH so a later add re-seeds it.
        if self
            .rt
            .accel
            .as_ref()
            .is_some_and(|a| a.is_spent(skinned_present))
        {
            self.rt.retire_accel(frame.id);
        }
        refreshed.and(stepped)
    }

    // Bring the RT draw-object BLAS head in line with the current draw set
    // (added/removed chunks, cloned props, participation-changing material
    // edits), async. On the static path the TLAS + geometry table are rebuilt
    // over the refreshed head inline; on the skinned path the caller's
    // `rebuild_skinned` builds the TLAS over the refreshed head + skinned tail.
    // The device / queue / shared buffers are cheap handles, cloned so the accel
    // can be borrowed mutably beside the draw list.
    fn refresh_rt_topology(
        &mut self,
        mode: concinnity_core::render::rt_accel::RefreshMode,
        shape: super::raytrace::RefreshShape,
        frame_id: u64,
    ) -> error::RenderResult<()> {
        let device = self.hw.device.clone();
        let queue = self.hw.command_queue.clone();
        let vbuf = self.scene.vertex_buffer.retained();
        let ibuf = self.scene.index_buffer.retained();
        let exclude_seethrough = self.seethrough_meshes_enabled();
        let Some(accel) = self.rt.accel.as_mut() else {
            return Ok(());
        };
        accel.refresh_static_topology(
            super::raytrace::RtGpu {
                device: &device,
                command_queue: &queue,
                frames_in_flight: self.frames_in_flight,
            },
            super::raytrace::RtStaticGeometry {
                vertex_buffer: &vbuf,
                index_buffer: &ibuf,
            },
            &self.state.draw.objects,
            super::raytrace::RtTopologyRefreshOptions {
                exclude_seethrough,
                mode,
                shape,
                frame_id,
            },
        )
    }

    // Full BVH rebuild (fresh BLAS + TLAS + table) from the current draw list,
    // instanced clusters, and skinned pose: seeds a BVH where there was none, and
    // replaces one whose shared geometry buffers were rebuilt underneath it.
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
                if !self.state.skinned.draw_objects.is_empty() && self.rt.skinned_geometry =>
            {
                Some(SkinnedRtInputs {
                    objects: &self.state.skinned.draw_objects,
                    vertex_buffer: svb,
                    index_buffer: sib,
                    joint_matrices: &self.state.skinned.joint_matrices,
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
                draw_objects: &self.state.draw.objects,
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
    // current pose and transforms. Skipped, keeping last frame's BVH, if the
    // required skinned resources are missing.
    fn update_rt_skinned(
        &mut self,
        frame: super::raytrace::RtFrame,
        joint_buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        full_build: bool,
    ) -> error::RenderResult<RtUpdate> {
        use super::raytrace::SkinnedRtInputs;
        let device = self.hw.device.clone();
        let queue = self.hw.command_queue.clone();
        let frames_in_flight = self.frames_in_flight;
        let (Some(svb), Some(sib), Some(pipe), Some(accel)) = (
            self.skinned.vertex_buffer.as_ref(),
            self.skinned.index_buffer.as_ref(),
            self.rt.pipelines.skin.as_ref(),
            self.rt.accel.as_mut(),
        ) else {
            return Ok(RtUpdate::Skipped);
        };
        let skinned = SkinnedRtInputs {
            objects: &self.state.skinned.draw_objects,
            vertex_buffer: svb,
            index_buffer: sib,
            joint_matrices: &self.state.skinned.joint_matrices,
            skin_pipeline: pipe.as_ref(),
        };
        accel.rebuild_skinned(
            super::raytrace::RtGpu {
                device: &device,
                command_queue: &queue,
                frames_in_flight,
            },
            &self.state.draw.objects,
            skinned,
            joint_buffers,
            frame,
            full_build,
        )
    }

    // Rebuild just the TLAS + geometry table (fresh allocations, static BLAS)
    // from the current draw-object transforms.
    fn rebuild_rt_tlas(&mut self, frame_id: u64) -> error::RenderResult<()> {
        let device = self.hw.device.clone();
        let queue = self.hw.command_queue.clone();
        let Some(accel) = self.rt.accel.as_mut() else {
            return Ok(());
        };
        accel.rebuild_tlas(&device, &queue, &self.state.draw.objects, frame_id)
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
            self.upscale.reset.rebuilt();
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
        // The planar mirror targets follow the render resolution (scaled by the
        // mirror resolution). The layout carries over; only the targets are
        // reallocated.
        if render_changed && let Some(set) = self.planar_reflection.as_mut() {
            set.resize(&self.hw.device, (render_w, render_h))?;
        }
        // The bloom chain reads `scene_color`: at drawable size when the
        // upscaler runs, otherwise at native (= render) resolution. Sized
        // off `want_w/h` either way.
        let bloom_changed = (want_w, want_h) != self.targets.output;
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
        // either extent moving invalidates it. Nothing caches a pooled handle:
        // the per-frame bindless argument buffer re-encodes `ao_output` itself
        // and every other consumer fetches its texture by label at encode time,
        // which is what makes a rebuild's slot repack harmless here.
        if render_changed || bloom_changed {
            self.targets.transient_pool.rebuild(
                &self.hw.device,
                &render_graph::plan_pool_slots(
                    render_graph::PoolGates {
                        ssao: self.ssao.settings.is_some(),
                        gbuffer: needs_gbuffer,
                    },
                    (render_w, render_h),
                    (want_w, want_h),
                )?,
            )?;
        }
        if bloom_changed {
            if let Some(mut bloom) = self.bloom.take() {
                let r = bloom.resize(
                    &self.post_device(),
                    PostExtent {
                        width: want_w,
                        height: want_h,
                    },
                );
                self.bloom = Some(bloom);
                r?;
            }
            // Recorded only once the chain matches, so a failed resize retries.
            self.targets.output = (want_w, want_h);
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
        // + normal input comes from the unified G-buffer pre-pass (below), and
        // its blurred output is the pool's `ao_output`, rebuilt above.
        let render_extent = PostExtent {
            width: render_w,
            height: render_h,
        };
        if render_changed && let Some(mut ssao) = self.ssao.pass.take() {
            let r = ssao.resize(&self.post_device(), render_extent);
            self.ssao.pass = Some(ssao);
            r?;
        }
        // The reflection target and the composite's output and blur are
        // render-resolution. The reflection target exists when SSR, SSGI, *or*
        // RT reflections are on (RT writes it too). The acceleration structure
        // is resolution-independent, so it is not rebuilt here.
        if render_changed && self.ssr.reflection.is_some() {
            self.ssr.reflection = Some(super::post::create_reflection_target(
                &self.hw.device,
                render_w,
                render_h,
                self.ssr.trace_scale,
            )?);
        }
        if render_changed && let Some(mut composite) = self.ssr.composite.take() {
            let r = composite.resize(&self.post_device(), render_extent);
            self.ssr.composite = Some(composite);
            r?;
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
        // The SSGI trace targets are render-resolution scaled by `gi_scale`
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
        self.sync_glass_reflection_target(render_w, render_h)
    }
}
