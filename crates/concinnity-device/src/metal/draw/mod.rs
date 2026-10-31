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
mod scene_textures;
mod shadow;
mod spot_shadow;
mod stages;

use concinnity_core::math::{SaturatingFrom, narrow_f32};
use concinnity_engine::profile::FrameSpans;
use concinnity_engine::render::backend::FrameParams;
use concinnity_engine::render::environment::grass::lod as grass_lod;
use concinnity_engine::render::frame::plan::{FrameDevice, FramePlan, camera_inputs, drive_frame};
use concinnity_engine::render::post::device::PostExtent;
use concinnity_engine::render::rt_accel::RtUpdate;
use concinnity_engine::render::scene_state::SceneState;
use concinnity_engine::render::{backend::error, render_graph};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue as _};

use self::scene_textures::SceneTextures;
use self::stages::{
    AcquiredFrame, HistoryBuffers, PlanArgs, PresentFrame, SceneBufferArgs, SceneBuffers,
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
    #[expect(
        clippy::large_types_passed_by_value,
        reason = "keeps the `RenderBackend::draw_frame` signature it serves"
    )]
    pub(crate) fn draw_frame(&mut self, params: FrameParams<'_>) -> error::RenderResult<()> {
        objc2::rc::autoreleasepool(|_| {
            drive_frame(self, params, concinnity_host::clock::monotonic_micros)
        })
    }
}

// What `begin` acquired for a frame: its in-flight slot and drawable, the
// command buffer the presenter records into, and the per-frame buffers built
// ahead of the plan.
pub(crate) struct MtlFrame {
    acquired: AcquiredFrame,
    pass_timing_slot: usize,
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    skinned_joint_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    skinned_morph_weight_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    render_size: [u32; 2],
}

impl FrameDevice for MtlContext {
    type Frame = MtlFrame;

    fn frame_scene(&mut self) -> &mut SceneState {
        &mut self.state
    }

    fn draws_atmosphere(&self) -> bool {
        self.atmosphere.is_some()
    }

    fn prepare(&mut self, params: &FrameParams<'_>) -> error::RenderResult<()> {
        self.frame_capture.arm(params.capture);
        self.apply_pending_rebuilds()?;
        // A moved world's decals and grass move with it whatever happens to
        // history.
        if let Some(rebase) = &params.rebase {
            self.decal.set.rebase(rebase);
            self.rebase_grass(rebase);
            self.rebase_water(rebase);
            self.particle.system.rebase(rebase);
        }
        Ok(())
    }

    // The TAA and SSGI rings, and MetalFX's history on its next encode.
    fn reset_history(&mut self) {
        if let Some(taa) = self.taa.pass.as_mut() {
            taa.reset_history();
        }
        if let Some(ssgi) = self.ssgi.pass.as_mut() {
            ssgi.reset_history();
        }
        if let Some(disocclusion) = self.disocclusion.as_mut() {
            disocclusion.reset_history();
        }
        self.upscale.reset.request();
    }

    fn begin(&mut self, params: &FrameParams<'_>) -> error::RenderResult<Option<MtlFrame>> {
        let mtm = objc2::MainThreadMarker::new().ok_or_else(|| {
            error::RenderError::Other("draw_frame must be called from the main thread".into())
        })?;
        let pass_timing_slot = self.begin_frame_stats();
        if !self.pump_window_events(mtm) {
            return Ok(None);
        }
        // Nothing a minimized window draws can be seen; park the frame until
        // it is restored.
        if self.window().appkit.window_minimized() {
            return Ok(None);
        }
        let Some(acquired) = self.acquire_frame(params.world_hidden) else {
            return Ok(None);
        };
        let ring_slot = acquired.ring_slot;

        let cmd_buf = self
            .hw
            .command_queue
            .commandBuffer()
            .ok_or_else(|| error::RenderError::Other("failed to get command buffer".into()))?;

        self.service_background_work(narrow_f32(params.elapsed), ring_slot);

        // Transient per-frame GPU buffers holding each skinned object's joint
        // matrices. Built once and reused across the shadow cascades and the
        // main pass. Empty when no SkinnedMesh is in the world.
        let skinned_joint_bufs = self.build_joint_buffers(ring_slot)?;

        // Per-object morph weights for the skinned fold, from the same ring slot.
        let skinned_morph_weight_bufs = self.build_morph_weight_buffers(ring_slot)?;

        let (render_w, render_h) = self.resize_frame_targets()?;
        if let Some(grass) = self.grass.as_mut() {
            grass.resize(&self.hw.device, [render_w, render_h])?;
        }
        self.grass_lod = self.grass.as_ref().map(|g| {
            g.field
                .lod_view(grass_lod::focal_length(render_h, params.fov_y_radians))
        });

        // Line pipeline: built on the first frame that publishes lines, so the
        // plan's graph gate sees it live this same frame.
        self.ensure_line_pipeline(!params.lines.is_empty());

        Ok(Some(MtlFrame {
            acquired,
            pass_timing_slot,
            cmd_buf,
            skinned_joint_bufs,
            skinned_morph_weight_bufs,
            render_size: [render_w, render_h],
        }))
    }

    fn plan(
        &mut self,
        frame: &MtlFrame,
        params: &FrameParams<'_>,
    ) -> error::RenderResult<FramePlan> {
        self.plan_frame(PlanArgs {
            camera: camera_inputs(&self.state, params),
            elapsed: narrow_f32(params.elapsed),
            render_size: frame.render_size,
            lines: params.lines,
            world_hidden: params.world_hidden,
            prop_probe: params.prop_probe.is_some(),
        })
    }

    fn execute(
        &mut self,
        frame: MtlFrame,
        plan: &FramePlan,
        frame_params: &FrameParams<'_>,
    ) -> error::RenderResult<()> {
        let FrameParams {
            elapsed: elapsed_f64,
            water_time,
            cam_pos,
            text_calls,
            lines,
            world_hidden,
            grass_benders,
            water_stirs,
            particle_bursts,
            input_sampled_us,
            ..
        } = *frame_params;
        let elapsed = narrow_f32(elapsed_f64);
        let MtlFrame {
            acquired:
                AcquiredFrame {
                    frame_slot,
                    drawable,
                    frame_id,
                    ring_slot,
                },
            pass_timing_slot,
            cmd_buf,
            skinned_joint_bufs,
            skinned_morph_weight_bufs,
            ..
        } = frame;

        self.prepare_prop_probe(frame_params.prop_probe, plan.inputs.prop_probe_enabled);
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
            records: plan.records,
            skinned_joint_bufs: &skinned_joint_bufs,
            texture_signature,
        })?;

        let SceneTextures {
            scene_input,
            scene_color,
        } = self.scene_textures()?;

        self.prepare_atmosphere(frame_params.atmosphere.as_ref(), plan.atmosphere)?;
        self.prepare_water_wake(cam_pos, water_time, water_stirs);

        // This frame's ribbon geometry, written into this slot's persistent
        // vertex buffer up front for the same reason the text geometry is: the
        // line pass encodes through `&self`, so it cannot upload for itself.
        self.upload_lines(ring_slot, lines)?;

        let hand_off = self
            .grass
            .as_ref()
            .zip(self.grass_lod.as_ref())
            .map(|(grass, lod)| (lod, grass.crossfade_phase()));
        self.canopy = self
            .canopy_field
            .as_ref()
            .map_or_else(Default::default, |field| {
                self.state.canopy_params(field, hand_off, water_time)
            });
        self.canopy_tables.update(
            &self.hw.device,
            self.canopy_field.as_ref().map_or(&[], |f| f.tables()),
        )?;
        // The grass kernel's inputs, advanced only on a frame whose graph runs it,
        // so the draw-argument slot it fills alternates frame to frame.
        let grass_started = std::time::Instant::now();
        let grass_frame = (plan.inputs.grass_enabled && !plan.inputs.world_hidden)
            .then(|| {
                self.prepare_grass_frame(crate::metal::grass::GrassRequest {
                    cam_pos,
                    vp: plan.camera.cur_vp,
                    elapsed,
                    cast: plan.inputs.grass_shadow_enabled,
                    benders: grass_benders,
                    ring_slot,
                })
            })
            .flatten();
        self.state.spans.set(FrameSpans {
            grass_us: concinnity_host::clock::micros_since(grass_started),
            ..self.state.spans.get()
        });
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
            particle_bursts,
            cam_pos,
            skinned_joint_bufs: &skinned_joint_bufs,
            skinned_morph_weight_bufs: &skinned_morph_weight_bufs,
            scene_color: Some(&scene_color),
            text_calls,
            ring_slot,
            world_hidden,
            elapsed: elapsed_f64,
            water_time,
            vp: plan.camera.vp,
            inv_vp: plan.camera.inv_vp,
            frustum: &plan.camera.frustum,
            planar: self
                .planar_reflection
                .as_ref()
                .map(|set| set.layout.frame_plan(plan.camera.vp))
                .unwrap_or_default(),
            object_buffer: object_buffer.as_ref(),
            material_params: material_params.as_ref(),
            bindless_tex_args: bindless_tex_args.as_ref(),
            deformed_skinned: deformed_this_frame.as_ref(),
            deformed_prev: deformed_prev_frame.as_ref(),
            prev_model_buffer: prev_model_buffer.as_ref(),
            history_targets: &history_targets,
            draw_args_buffer: cull_draw_args.as_ref(),
            gbuffer_view: &plan.gbuffer_view,
            velocity_active: plan.motion_read,
            grass: grass_frame.as_ref(),
            scene_pre_taa: if self.taa.enabled
                || self.upscale.scaler.is_some()
                || plan.inputs.transparent_enabled
                || plan.inputs.clouds_enabled
            {
                Some(&scene_input)
            } else {
                None
            },
            ssr_params: plan.effects.ssr.as_ref(),
            fog_params: plan.fog.as_ref().map(|f| &f.params),
            fog_froxel_params: plan.fog.as_ref().map(|f| &f.froxel),
            cluster_params: plan.clustered.then_some(&plan.cluster),
            ssao_params: plan.effects.ssao.as_ref(),
            ssgi_params: plan.effects.ssgi.as_ref(),
            rt_reflection_params: plan.effects.rt.as_ref(),
            reactive: plan.reactive,
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
        let pass_async = std::sync::Arc::clone(&self.diagnostics.pass_async_mask);
        let gpu_overlap = std::sync::Arc::clone(&self.diagnostics.gpu_overlap_us);
        let gpu_gap_beside_async = std::sync::Arc::clone(&self.diagnostics.gpu_gap_beside_async_us);
        let gpu_done = std::sync::Arc::clone(&self.diagnostics.gpu_done);
        let span_end = std::sync::Arc::clone(&self.diagnostics.gpu_span_end_ns);
        let async_mask = plan.graph.async_pass_mask();
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
                let mut overlap_us = 0;
                let mut gap_beside_async_us = 0;
                if let Some(buf) = &pass_buffer {
                    let mask = active_mask.load(AtomicOrdering::Relaxed);
                    let timings = super::pass_timing::resolve_frame(&buf.0, mask, async_mask);
                    let per_slot = timings.as_ref().map(|t| &t.per_slot);
                    for (i, slot) in pass_times.iter().enumerate() {
                        slot.store(per_slot.map_or(0, |p| p[i]), AtomicOrdering::Relaxed);
                    }
                    pass_async.store(async_mask, AtomicOrdering::Relaxed);
                    let micros = |ns: u64| u32::saturating_from(ns / 1000);
                    // First graphics-queue pass start to last end, clipped at
                    // the previous frame's end: what this frame added to the
                    // GPU timeline, which the per-pass times cannot be summed
                    // into once passes overlap.
                    if let Some(t) = &timings
                        && let Some((start, end)) = t.span
                    {
                        let previous = span_end.fetch_max(end, AtomicOrdering::Relaxed);
                        let span =
                            concinnity_engine::render::backend::pass_timing::FrameSpan::after(
                                start, end, previous,
                            );
                        frame_us = micros(span.own);
                        overlap_us = micros(span.overlap);
                        gap_beside_async_us = micros(t.gap_beside_async);
                    }
                }
                gpu_time.store(frame_us, AtomicOrdering::Relaxed);
                gpu_overlap.store(overlap_us, AtomicOrdering::Relaxed);
                gpu_gap_beside_async.store(gap_beside_async_us, AtomicOrdering::Relaxed);
                // The last part of the frame just retired, so this is when its
                // GPU work ended on the host clock.
                if let (Some(input_sampled_us), Ok(mut slot)) = (input_sampled_us, gpu_done.lock())
                {
                    *slot = Some(concinnity_engine::profile::GpuDone {
                        input_sampled_us,
                        done_us: concinnity_host::clock::monotonic_micros(),
                    });
                }
            })
        });

        let submission = self.execute_graph(&plan.graph, &params, &join)?;
        if let Some(timing) = self.diagnostics.pass_timing.as_ref() {
            active_mask.store(timing.attached_mask(), std::sync::atomic::Ordering::Relaxed);
        }
        let started = std::time::Instant::now();
        self.submit_and_present(PresentFrame {
            cmd_buf,
            drawable,
            join,
            composite_span_us,
            pending_terminal: submission.pending_terminal,
            submission_token,
        });
        self.diagnostics.frame_stats.submit_present_us =
            concinnity_host::clock::micros_since(started);

        Ok(())
    }

    // What the TAA and SSGI accumulations wrote is next frame's history.
    fn finish(&mut self, plan: &FramePlan) {
        if plan.motion_read
            && let Some(taa) = self.taa.pass.as_mut()
        {
            taa.advance();
        }
        if let Some(ssgi) = self.ssgi.pass.as_mut() {
            ssgi.advance();
        }
        if let Some(disocclusion) = self.disocclusion.as_mut() {
            disocclusion.advance();
        }
    }
}

impl MtlContext {
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
        use concinnity_engine::render::rt_accel::{RtDriveInputs, drive_update, seed_wanted};
        let mode = self.rt.dynamic_mode;
        let exclude_seethrough = self.seethrough_meshes_enabled();
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
        let skinned = match (
            self.skinned.vertex_buffer.as_ref(),
            self.skinned.index_buffer.as_ref(),
            self.rt.pipelines.skin.as_ref(),
        ) {
            (Some(vertex_buffer), Some(index_buffer), Some(pipe)) => {
                Some(super::raytrace::SkinnedRtInputs {
                    objects: &self.state.skinned.draw_objects,
                    vertex_buffer,
                    index_buffer,
                    joint_matrices: &self.state.skinned.joint_matrices,
                    skin_pipeline: pipe.as_ref(),
                })
            }
            _ => None,
        };
        let mut steps = super::raytrace::MtlRtSteps {
            accel,
            gpu: super::raytrace::RtGpu {
                device: &self.hw.device,
                command_queue: &self.hw.command_queue,
                frames_in_flight: self.frames_in_flight,
            },
            geometry: super::raytrace::RtStaticGeometry {
                vertex_buffer: &self.scene.vertex_buffer,
                index_buffer: &self.scene.index_buffer,
            },
            draw_objects: &self.state.draw.objects,
            exclude_seethrough,
            frame,
            joint_buffers,
            skinned,
        };
        let result = drive_update(
            &mut steps,
            RtDriveInputs {
                mode,
                topology_dirty,
                draw_objects: &self.state.draw.objects,
                skinned: skinned_objects,
                now_us: concinnity_host::clock::monotonic_micros,
                spans: &self.state.spans,
            },
        );
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
        result
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
                    render_graph::PoolGates::REACHABLE,
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
        // above.
        if render_changed && self.gbuffer.targets.is_some() {
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
        if render_changed && let Some(mut disocclusion) = self.disocclusion.take() {
            let r = disocclusion.resize(
                &self.post_device(),
                PostExtent {
                    width: render_w,
                    height: render_h,
                },
            );
            self.disocclusion = Some(disocclusion);
            r?;
        }
        // The Hi-Z pyramid matches the render (depth) resolution. Rebuild it
        // and mark it invalid so the next cull dispatch ignores the now-stale
        // pyramid (the projection coordinates were generated at the old
        // resolution); the next frame's build refills it.
        if render_changed && let Some(hiz) = self.cull.hiz.as_mut() {
            hiz.resize_to(&self.hw.device, render_w, render_h)?;
            self.state.planner.invalidate_occlusion();
        }
        self.sync_glass_reflection_target(render_w, render_h)
    }
}
