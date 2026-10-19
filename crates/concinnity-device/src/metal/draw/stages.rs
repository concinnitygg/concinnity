//! The stages `MtlContext::draw_frame` runs in order around its render-graph
//! dispatch.

use concinnity_core::gfx::render_types::LineVertex;
use concinnity_core::profile;
use concinnity_core::render::error;
use concinnity_core::render::frame_plan::{
    CameraInputs, EffectInputs, FramePlan, PlanInputs, SurfaceInputs,
};
use concinnity_core::render::model_history::HistoryMode;
use concinnity_core::render::reactive_mask::ReactiveReader;
use concinnity_core::render::render_graph::{FrameContent, ResidentFeatures, ResidentUpscaler};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLDevice as _};
use objc2_quartz_core::CAMetalDrawable;

use crate::metal::context::MtlContext;
use crate::metal::frame_pacing::{FrameJoin, FrameSlot, SubmissionToken};

// A frame past the frames-in-flight fence with a drawable to present into.
pub(super) struct AcquiredFrame {
    pub(super) frame_slot: FrameSlot,
    pub(super) drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
    pub(super) frame_id: u64,
    pub(super) ring_slot: usize,
}

// The per-frame inputs `plan_frame` plans from.
pub(super) struct PlanArgs<'a> {
    pub(super) camera: CameraInputs,
    pub(super) elapsed: f32,
    pub(super) render_size: [u32; 2],
    pub(super) lines: &'a [LineVertex],
    pub(super) world_hidden: bool,
}

// The recorded frame's presenting command buffer and its completion join.
pub(super) struct PresentFrame {
    pub(super) cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub(super) drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
    pub(super) join: std::sync::Arc<FrameJoin>,
    pub(super) composite_span_us: std::sync::Arc<std::sync::atomic::AtomicU32>,
    pub(super) pending_terminal: Option<u64>,
    pub(super) submission_token: SubmissionToken,
}

// The per-frame inputs the scene buffer builds read.
pub(super) struct SceneBufferArgs<'a> {
    pub(super) ring_slot: usize,
    pub(super) frame_id: u64,
    pub(super) cam_pos: [f32; 3],
    pub(super) elapsed: f32,
    pub(super) world_hidden: bool,
    // The cull records the plan packs this frame, and their history mode.
    pub(super) records: Option<HistoryMode>,
    pub(super) skinned_joint_bufs: &'a [Retained<ProtocolObject<dyn MTLBuffer>>],
    // This frame's `bindless_texture_signature`.
    pub(super) texture_signature: u64,
}

// This frame's bindless-path buffers. All three are `None` while the world is
// hidden, and individually `None` when the path that fills them is inactive.
pub(super) struct SceneBuffers {
    pub(super) object_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(super) material_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(super) cull_draw_args: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(super) bindless_tex_args: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

// The motion-history buffers the GPU-driven G-buffer pre-pass binds.
pub(super) struct HistoryBuffers {
    pub(super) deformed_this_frame: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(super) deformed_prev_frame: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(super) prev_model_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(super) history_targets: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl MtlContext {
    pub(super) fn begin_frame_stats(&mut self) -> usize {
        // Reset this frame's render stats; the draw counters below accumulate
        // into `diagnostics.frame_stats`, and `render_stats()` reports them (plus the GPU
        // frame time) to the profiler overlay.
        let counts = concinnity_core::render::object_counts::object_counts(
            self.state.draw.objects.len(),
            self.instanced.clusters.iter().map(|c| c.instances.len()),
            self.state.skinned.draw_objects.iter().map(|o| o.visible),
        );
        self.diagnostics.frame_stats = profile::RenderStats {
            objects: counts.objects,
            skinned_visible: counts.skinned_visible,
            // On Apple Silicon's unified memory this is the Metal device's
            // allocation within system RAM.
            vram_bytes: self.hw.device.currentAllocatedSize() as u64,
            transient_pool_bytes: self.targets.transient_pool.heap_bytes(),
            ..profile::RenderStats::default()
        };

        // Rotate the per-frame sample-buffer slot if per-pass GPU timing is
        // available. Every `diagnostics.pass_timing.attach_*` call this frame writes
        // into the same slot's buffer; the completion handler resolves it
        // after the frame retires.
        self.diagnostics.pass_timing.as_mut().map_or(
            0,
            super::super::pass_timing::PassTimingResources::begin_frame,
        )
    }

    // Returns false once the window has closed.
    pub(super) fn pump_window_events(&mut self, mtm: objc2::MainThreadMarker) -> bool {
        // Drain all pending NSEvents so the window stays responsive. The
        // preview tab leaves `pump_events` false so the host owns event
        // delivery (pumping there would dequeue mouse clicks meant for the
        // tab bar before they reach their targets); the windowed CLI path
        // and the blocking-in-view play path opt in.
        if self.window().appkit.pump_events() {
            self.window_mut().appkit.pump_ns_events(mtm);
            if self.window_closed() {
                return false;
            }
        }

        // Converge the display on the fullscreen state: hold the chosen mode
        // while the window is fullscreen, restore the desktop mode otherwise.
        // Runs off the delegate-tracked flag so OS-driven fullscreen exits
        // (green traffic-light button, Mission Control) restore too. Cheap
        // when nothing changed.
        self.window_mut().appkit.reconcile_display_mode();
        true
    }

    // Reserve the next frame's slot ahead of its draw: blocks until the GPU
    // retires an older frame. A minimized window draws nothing, so it reserves
    // nothing; a slot already reserved is kept.
    pub(crate) fn prepare_frame(&mut self) {
        if self.prepared_slot.is_none() && !self.window().appkit.window_minimized() {
            self.prepared_slot = Some(self.frame_pacing.acquire());
        }
    }

    // Pump the window's pending events outside the draw.
    pub(crate) fn pump_events(&mut self) {
        if let Some(mtm) = objc2::MainThreadMarker::new() {
            self.pump_window_events(mtm);
        }
    }

    // None when no drawable is ready, which skips the frame.
    pub(super) fn acquire_frame(&mut self, world_hidden: bool) -> Option<AcquiredFrame> {
        // Frames-in-flight gate: block until the GPU has retired an older frame
        // so the CPU never queues more than `frames_in_flight` frames ahead,
        // bounding how many sets of per-frame transient buffers pile up. Taken
        // here, before drawable prep and all per-frame buffer building, unless
        // `prepare_frame` already reserved it (that wait is the caller's). The slot
        // is handed to the frame command buffer's completion handler below
        // (released on GPU retirement); if this frame is abandoned before commit
        // (the drawable isn't ready, or a `?` fails mid-encode) `frame_slot`
        // drops and releases the slot synchronously, keeping the count balanced.
        // Both blocking calls this frame makes are measured into `gpu_wait`
        // and published on the frame's stats: this one and the drawable
        // acquire below. They are wall time inside `draw_frame`, which the
        // engine times its graphics system around, so a GPU-bound frame is
        // only distinguishable from a CPU-bound one with this reading.
        let mut gpu_wait = crate::gpu_wait::GpuWait::none();
        let frame_slot = match self.prepared_slot.take() {
            Some(slot) => slot,
            None => gpu_wait.measure(|| self.frame_pacing.acquire()),
        };

        // Asynchronous reflection-probe bake, prefiltering half: convolve one mip of
        // a finished capture or install its cube, ahead of the argument buffers
        // below so an installed cube is sampled this frame. The capture half runs
        // once those buffers exist. Runs AFTER `acquire()` so the bake's retire-pool
        // collection sees a fence-consistent frame id. Skipped while the world is
        // hidden: a probe bake feeds reflections no pass will sample this frame.
        if !world_hidden {
            self.advance_probe_prefilter();
        }

        // tell MTKView to prepare its drawable for this frame, then take it.
        // `currentDrawable` blocks on the drawable pool when every drawable is
        // still with the compositor, which is the display-paced half of the
        // frame's GPU wait.
        let drawable = gpu_wait.measure(|| {
            self.window().view.draw();
            self.window().view.currentDrawable()
        });
        self.diagnostics.frame_stats.gpu_wait_us = gpu_wait.micros();
        // drawable not yet available -- skip this frame silently
        let drawable = drawable?;

        // This frame's transient-buffer ring slot. The fence guarantees the
        // frame that last used `frame_ring_index - frames_in_flight` has retired
        // on the GPU, so overwriting this slot's buffers can't race an in-flight
        // read. Advanced once per built frame; skipped frames (no drawable) bail
        // above without consuming a slot.
        let frame_id = self.frame_ring_index;
        let ring_slot = (frame_id % self.frames_in_flight as u64) as usize;
        self.frame_ring_index = frame_id.wrapping_add(1);

        // Hand back the pooled ranges whose retire frame has passed and release
        // any heap left holding nothing. Ticked here, past the fence, so a range
        // is only reused once every frame that could reference it has retired.
        self.hw.allocator.begin_frame();
        Some(AcquiredFrame {
            frame_slot,
            drawable,
            frame_id,
            ring_slot,
        })
    }

    // Rebuilds requested since last frame: hot-reloaded shaders.
    pub(super) fn apply_pending_rebuilds(&mut self) -> error::RenderResult<()> {
        // Shader hot-reload: if the debug `reload-shaders` command set the
        // flag, rebuild every built-in pipeline from disk-resident source
        // before the frame's passes start using them. The flag is cleared
        // regardless of outcome so a failed rebuild (typo in a shader edit)
        // doesn't loop, and the previous pipelines stay live so the session
        // keeps rendering; only a device failure propagates. In-flight command
        // buffers retain the pipelines they encoded, so no GPU drain is needed.
        if self.hot_reload.requested() {
            self.hot_reload.clear();
            match self.reload_shaders() {
                Ok(()) => tracing::info!("hot-reload: shader pipelines rebuilt"),
                Err(e) if e.is_device_failure() => return Err(e),
                Err(e) => tracing::error!("hot-reload: shader rebuild failed: {}", e),
            }
        }
        Ok(())
    }

    pub(super) fn service_background_work(&mut self, elapsed: f32, ring_slot: usize) {
        // Update auto-exposure from the previous frame's GPU-measured average
        // log-luminance, *before* any pass reads `self.post_process.exposure`
        // (the bloom prefilter and composite both consume it). A no-op when
        // auto-exposure is disabled -- the static authored EV then drives the
        // exposure multiplier unchanged.
        self.update_auto_exposure(elapsed, ring_slot);
        self.pre_exposure.advance(self.post_process.exposure);
    }

    // What this backend has built, for the frame's graph gates.
    pub(super) fn resident_features(&self) -> ResidentFeatures {
        ResidentFeatures {
            hdr_width: self.targets.hdr.width,
            hdr_height: self.targets.hdr.height,
            hdr_sample_count: self.targets.hdr.sample_count,
            shadow: self.shadow.enabled,
            shadow_map_size: self.shadow.map_size,
            cull: self.cull.bindless,
            two_pass: self.cull.two_pass_occlusion,
            hiz: self.cull.hiz.is_some(),
            bloom_chain: self.bloom.is_some(),
            upscaler: self.upscale.scaler.as_ref().map(|s| ResidentUpscaler {
                reads_reactive: ReactiveReader::MetalFx.reads() && s.reactive,
            }),
            particles: self.particle.pipelines.is_some(),
            fog: self.fog.pipeline.is_some(),
            decals: self.decal.pipeline.is_some(),
            lines: self.lines.pipeline.is_some(),
            atmosphere: self.atmosphere.is_some(),
            rt_transparent: self.transparent_rt_pipelines_ready(),
            taa: self.taa.enabled,
            model_history: self.gbuffer.targets.is_some() && self.cull.main_pipeline.is_some(),
            reflection_composite: self.ssr.composite.is_some(),
        }
    }

    // What the world puts on screen this frame, for the frame's graph gates.
    pub(super) fn frame_content(&self, world_hidden: bool, lines: &[LineVertex]) -> FrameContent {
        FrameContent {
            world_hidden,
            cull_records: self.cull_count(),
            shadowed_spot_count: self.spot_shadow.count,
            bloom_intensity: self.post_process.bloom_intensity,
            bvh_live: self.rt.accel.is_some(),
            emitters: !self.particle.system.is_empty(),
            fog: self.fog.settings.is_some_and(|s| s.contributes()),
            decals: !self.decal.set.is_empty(),
            visible_surfaces: (self.water.pipeline.is_some()
                && self.water.surfaces.iter().any(|s| s.visible))
                || (self.glass.pipeline.is_some() && self.glass.panels.iter().any(|p| p.visible)),
            visible_see_through: self.see_through_mesh_visible(),
            planar_targets: self
                .planar_reflection
                .as_ref()
                .is_some_and(|s| !s.targets.is_empty()),
            water_planar_slot: self.water_planar_slot_live(),
            lines: !lines.is_empty(),
            visible_volumes: self.raymarch.volumes.iter().any(|v| v.visible),
            atmosphere: self.atmosphere.as_ref().is_some_and(|a| a.params.is_some()),
            water_sim: self.water_sim_live(),
            grass: self.grass.is_some(),
            ssgi_motion: self.ssgi.pass.is_some()
                && self.ssgi.settings.is_some_and(|s| s.contributes()),
        }
    }

    // Plan this frame in core over what this backend holds, then put what the
    // encoders read from `&self` in place: the spot slices, the cull's view and
    // the cluster grid, and the jitter MetalFX is dispatched with.
    pub(super) fn plan_frame(&mut self, args: PlanArgs<'_>) -> error::RenderResult<FramePlan> {
        let PlanArgs {
            camera,
            elapsed,
            render_size,
            lines,
            world_hidden,
        } = args;
        let resident = self.resident_features();
        let content = self.frame_content(world_hidden, lines);
        let surface = SurfaceInputs {
            prefilter_mip_count: self.scene.env_map.prefilter_mip_count as f32,
            irradiance: self.irradiance_lane(),
            screen_bounce: self.screen_bounce(),
        };
        let inputs = PlanInputs {
            camera,
            elapsed,
            render_size,
            resident,
            content,
            upscaler_jitter: None,
            lights: &self.lights,
            probes: self.probe.book.records(),
            fog: self.fog.settings,
            pre_exposure: self.pre_exposure.value(),
            surface,
            effects: EffectInputs {
                ssao: self.ssao.settings,
                ssr: self.ssr.settings,
                ssgi: self.ssgi.settings.zip(
                    self.ssgi
                        .pass
                        .as_ref()
                        .map(concinnity_core::render::post::ssgi::SsgiPass::frame),
                ),
                rt: self.rt.settings,
                history_scale: self.pre_exposure.history_scale(),
            },
        };
        let plan = self.state.planner.plan(&self.state.view, &inputs)?;
        self.spot_shadow.render_mask = self.state.planner.shadow.spot_mask;
        self.cull.cur_view_proj = plan.camera.cur_vp;
        self.cluster_params = plan.cluster;
        // MetalFX expects the jitter in render pixels; TAA reads it from the
        // projection itself.
        if self.upscale.scaler.is_some() {
            let [x, y] = plan.camera.jitter_px;
            self.upscale
                .jitter
                .store(x, y, std::sync::atomic::Ordering::Release);
        }
        Ok(plan)
    }

    pub(super) fn submit_and_present(&mut self, frame: PresentFrame) {
        let PresentFrame {
            cmd_buf,
            drawable,
            join,
            composite_span_us,
            pending_terminal,
            submission_token,
        } = frame;
        let texture = drawable.texture();
        self.reserve_frame_capture(&texture);
        self.record_frame_capture(&cmd_buf, &texture);
        cmd_buf.presentDrawable(ProtocolObject::from_ref(&*drawable));

        // Retain this drawable's color texture so the headless `screenshot`
        // command can blit the last presented frame back to the host. Only
        // under `hot_reload` (the `cn debug` path that runs the debug endpoint able
        // to request a capture, and the only path where the MTKView has
        // `framebufferOnly` switched off so this texture is blit-readable);
        // production keeps this `None`. Reading it next frame is safe: the
        // composite pass that wrote it committed earlier on the same queue, so
        // same-queue FIFO order guarantees it is fully rendered, and a
        // read-only blit may run alongside the compositor's scan-out.
        if self.capture {
            self.last_present_texture = Some(texture);
        }

        // The presenting command buffer's own completion handler. It reports
        // the buffer's fault status and publishes its GPU span as the
        // whole-frame fallback, then arrives at the frame's completion join;
        // the per-pass resolve and the frame-slot release belong to the join's
        // last arrival, since the async queue may still be running.
        {
            let device_error = std::sync::Arc::clone(&self.diagnostics.device_error);
            let part = std::sync::Arc::clone(&join);
            join.add_part();
            let handler = block2::RcBlock::new(
                move |cb: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                    // SAFETY: Metal hands the completion handler a live command buffer, and the
                    // borrow does not escape the block.
                    let cb = unsafe { cb.as_ref() };
                    crate::metal::fault_log::report_fault(cb, "frame render");
                    use objc2_metal::MTLCommandBufferStatus;
                    if cb.status() == MTLCommandBufferStatus::Error {
                        // Classify and park the first failure for the next
                        // draw_frame to report across the backend boundary.
                        let classified = match cb.error() {
                            Some(e) => crate::metal::error::classify_ns_error(&e),
                            None => error::RenderError::Other(
                                "frame command buffer faulted without an error object".to_string(),
                            ),
                        };
                        if let Ok(mut slot) = device_error.lock()
                            && slot.is_none()
                        {
                            *slot = Some(classified);
                        }
                    }
                    // This buffer is one slice of a multi-buffer, two-queue
                    // frame, so its own span under-reports the frame; it is only
                    // the fallback for a device with no per-pass timing.
                    // GPUStartTime / GPUEndTime are valid only inside the handler.
                    let span = cb.GPUEndTime() - cb.GPUStartTime();
                    composite_span_us.store(
                        (span * 1.0e6).clamp(0.0, f64::from(u32::MAX)) as u32,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    // Fires on success and on GPU fault alike, so the join can
                    // never leak the frame's slot.
                    part.arrive();
                },
            );
            // SAFETY: addCompletedHandler copies the block (Block_copy), so
            // the RcBlock is free to drop when this scope ends.
            unsafe {
                cmd_buf.addCompletedHandler(block2::RcBlock::as_ptr(&handler));
            }
        }

        cmd_buf.commit();
        // The graphics queue's frame terminal rides that buffer, so the next
        // frame may only wait on it now that it has been committed.
        if let Some(value) = pending_terminal {
            self.record_graph_terminal(value);
        }
        // Recording is done: release the join's submission part so the frame
        // can complete once every command buffer has retired.
        drop(submission_token);
    }

    // Resizes the off-screen targets to this frame's drawable and returns the
    // render resolution the scene and post passes draw at.
    pub(super) fn resize_frame_targets(&mut self) -> error::RenderResult<(u32, u32)> {
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
        Ok((render_w, render_h))
    }

    // This frame's probe records, plus the residency the bindless block's
    // contents need. Returns the block's texture signature, which the block's
    // own write reuses.
    pub(super) fn refresh_probe_records_and_residency(
        &mut self,
        ring_slot: usize,
    ) -> error::RenderResult<u64> {
        // The probe records, for every pass that samples the set. Built ahead
        // of `build_scene_buffers` and outside its world-hidden gate: the
        // transparent and post passes read the set without a static draw list
        // of their own, and a slot left holding last frame's ring buffer would
        // outlive the frame that wrote it.
        self.probe.records_buf = Some(self.build_probe_records(ring_slot)?);
        // The residency the bindless block's contents need. Refreshed before
        // any pass encodes, and a no-op on a frame whose textures are
        // unchanged, which is every frame between a stream-in or a bake.
        let sig = self.bindless_texture_signature();
        self.refresh_bindless_residency(sig);
        Ok(sig)
    }

    // The per-frame GPU buffers the world's passes consume, plus the probe
    // capture and acceleration-structure refresh that ride the same gate.
    pub(super) fn build_scene_buffers(
        &mut self,
        args: SceneBufferArgs<'_>,
    ) -> error::RenderResult<SceneBuffers> {
        let SceneBufferArgs {
            ring_slot,
            frame_id,
            cam_pos,
            elapsed,
            world_hidden,
            records,
            skinned_joint_bufs,
            texture_signature,
        } = args;
        // While the world is hidden behind an opaque menu, the surviving Main
        // pass is fed an empty scene -- no bindless object / cull / texture
        // buffers, no instanced clusters, and no acceleration-structure refresh
        // -- so it runs as a bare clear that the opaque overlay then covers. The
        // masked graph drops every other world pass, so none of this work would
        // be consumed anyway.
        let (object_buffer, material_params, cull_draw_args, bindless_tex_args) = if world_hidden {
            (None, None, None, None)
        } else {
            // Per-frame GPU buffer prep for the bindless path.
            // The object data + indirect-args + bindless texture argbuf are
            // all per-frame Metal buffers the bindless Main pass + Cull
            // compute pass consume. They must outlive the command buffer,
            // hence the bindings handed back to the caller, which holds them
            // through to `cmd_buf.commit()`.
            let records = match records {
                Some(history) => self.build_record_buffers(ring_slot, cam_pos, history)?,
                None => None,
            };
            let (object_buffer, draw_args) = match records {
                Some(records) => {
                    self.diagnostics.frame_stats.packing_fan_out = records.packing;
                    (Some(records.objects), Some(records.draw_args))
                }
                None => (None, None),
            };
            let material_params = match object_buffer {
                Some(_) => Some(
                    self.rings
                        .material_params
                        .buffer(&self.hw.device, ring_slot)?,
                ),
                None => None,
            };
            if draw_args.is_some() {
                self.ensure_icb_capacity(self.cull_count())?;
                // GPU-driven shadow views: size the cascade ICB to
                // NUM_SHADOW_CASCADES * cull_count and the spot ICB to one
                // region per slice. A no-op when the shadow-bindless path is
                // inactive (no shadow cull pipeline).
                self.ensure_shadow_icb_capacity(self.cull_count())?;
                // Per-planar-slot mirror cull ICBs: one per distinct reflection
                // plane, each sized to cull_count. A no-op (clears the slots) when
                // the world has no planar set (RT on, or no flat reflectors).
                let mirror_slots = self
                    .planar_reflection
                    .as_ref()
                    .map_or(0, |s| s.layout.planes().len());
                self.ensure_mirror_icb_capacity(mirror_slots, self.cull_count())?;
            }
            let cull_draw_args = draw_args;
            let bindless_tex_args = if object_buffer.is_some() {
                self.build_bindless_texture_args(ring_slot, texture_signature)?
            } else {
                None
            };
            // Asynchronous reflection-probe bake, capture half: submit one cube
            // face, or start or hand off a capture. Every face samples through
            // this frame's texture arguments, so it never reads a texture that
            // streaming has since replaced.
            self.advance_probe_capture(&crate::metal::probe::ProbeFrame {
                elapsed,
                tex_args: bindless_tex_args.as_ref(),
            });
            // Keep the RT acceleration structure current with this frame's
            // transforms before any pass reads `rt_accel`. The default `Auto` mode
            // rebuilds the TLAS only when a participating prop actually moved; a
            // fully static scene pays just a matrix compare here. Non-fatal: a
            // transient rebuild failure keeps last frame's BVH rather than stopping
            // the renderer.
            self.rt_dynamic_update(
                crate::metal::raytrace::RtFrame {
                    id: frame_id,
                    ring_slot,
                },
                skinned_joint_bufs,
            );

            (
                object_buffer,
                material_params,
                cull_draw_args,
                bindless_tex_args,
            )
        };
        Ok(SceneBuffers {
            object_buffer,
            material_params,
            cull_draw_args,
            bindless_tex_args,
        })
    }

    // The skinned deformed-vertex and model-history ring slots the GPU-driven
    // G-buffer pre-pass reads and writes.
    pub(super) fn build_history_buffers(
        &mut self,
        ring_slot: usize,
        object_buffer_live: bool,
    ) -> error::RenderResult<HistoryBuffers> {
        // This frame's skinned deformed-vertex buffer (skinned fold), cloned into
        // a local so `params` owns a handle rather than borrowing `self.skinned`
        // across the `&mut self` execute_graph call (every other GraphFrameParams
        // buffer is likewise a local). `Some` only when the fold is active
        // (draw.n_skinned > 0, set in upload_skinned under bindless + static geometry);
        // the Cull pass writes it via encode_main_skin and the Main / Main2
        // skinned ICB tail binds it.
        let deformed_this_frame = if self.state.draw.n_skinned > 0 {
            self.skinned.deformed.get(ring_slot).cloned()
        } else {
            None
        };
        // The previous frame's deformed slot (one behind in the ring), read by
        // the GPU-driven G-buffer skinned tail for per-vertex skin motion. The
        // priming gate (`deformed_primed`) covers the unposed first frame.
        let deformed_prev_frame = if self.state.draw.n_skinned > 0 {
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
        let (prev_model_buffer, history_targets) = if object_buffer_live
            && self.gbuffer.targets.is_some()
            && self.cull.main_pipeline.is_some()
        {
            let bytes = self.cull_count() * size_of::<[[f32; 4]; 4]>();
            let prime = self.state.model_history.get_mut().take_prime();
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
        Ok(HistoryBuffers {
            deformed_this_frame,
            deformed_prev_frame,
            prev_model_buffer,
            history_targets,
        })
    }
}
