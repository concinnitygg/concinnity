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
use concinnity_core::render::render_graph;
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

        let (render_w, render_h) = self.resize_frame_targets()?;

        let FrameProjection {
            proj,
            vp,
            inv_vp,
            frustum,
        } = self.frame_projection(fov_y_radians, aspect, near, far, render_w, render_h);

        self.refresh_argument_buffers(ring_slot)?;

        let SceneBuffers {
            object_buffer,
            cull_draw_args,
            bindless_tex_args,
        } = self.build_scene_buffers(SceneBufferArgs {
            ring_slot,
            frame_id,
            cam_pos,
            elapsed,
            near,
            far,
            world_hidden,
            skinned_joint_bufs: &skinned_joint_bufs,
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
            vel_uniforms,
            scene_input,
            scene_color,
            transparent_active,
        } = self.frame_pass_uniforms(PassUniformArgs {
            fov_y_radians,
            aspect,
            near,
            far,
            cam_pos,
            sky_rot,
            proj,
            vp,
            render_w,
            render_h,
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
        let graph = match self.draw.graph_cache.take() {
            Some((cached_inputs, cached_graph)) if cached_inputs == graph_inputs => cached_graph,
            // A graph the core refuses to build is a topology mistake, not a
            // device failure.
            _ => render_graph::build_frame_graph(&graph_inputs)
                .map_err(|e| error::RenderError::Other(format!("frame graph: {e}")))?,
        };
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
                self.ssr.scales,
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
        self.sync_glass_reflection_target(render_w, render_h)
    }
}
