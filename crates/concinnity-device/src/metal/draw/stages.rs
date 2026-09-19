//! The stages `MtlContext::draw_frame` runs in order around its render-graph
//! dispatch.

use concinnity_core::gfx::frustum::Frustum;
use concinnity_core::gfx::jitter;
use concinnity_core::gfx::projection::perspective_rh;
use concinnity_core::gfx::render_types::{self, LineVertex};
use concinnity_core::gfx::view_modes::{ShowFlags, ViewMode};
use concinnity_core::profile;
use concinnity_core::render::csm;
use concinnity_core::render::error;
use concinnity_core::render::model_history::HistoryMode;
use concinnity_core::render::render_graph::{self, FrameGraphInputs};
use concinnity_core::render::volumetric_fog::FogSettings;
use concinnity_core::transform::mat4_inverse;
use concinnity_core::transform::mat4_mul;
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

// The camera projection, its jittered view-projection, and what derives from it.
pub(super) struct FrameProjection {
    pub(super) proj: [[f32; 4]; 4],
    pub(super) vp: [[f32; 4]; 4],
    pub(super) inv_vp: [[f32; 4]; 4],
    pub(super) frustum: Frustum,
}

// The per-frame locals `frame_graph_inputs` gates passes on.
pub(super) struct GraphInputArgs<'a> {
    pub(super) bindless_cull_enabled: bool,
    pub(super) velocity_active: bool,
    pub(super) fog_settings: Option<&'a FogSettings>,
    pub(super) transparent_active: bool,
    pub(super) lines: &'a [LineVertex],
    pub(super) world_hidden: bool,
    pub(super) clustered: bool,
    pub(super) view_mode: ViewMode,
    pub(super) show: ShowFlags,
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
    pub(super) near: f32,
    pub(super) far: f32,
    pub(super) world_hidden: bool,
    pub(super) skinned_joint_bufs: &'a [Retained<ProtocolObject<dyn MTLBuffer>>],
}

// This frame's bindless-path buffers. All three are `None` while the world is
// hidden, and individually `None` when the path that fills them is inactive.
pub(super) struct SceneBuffers {
    pub(super) object_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
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
        let counts = crate::object_counts::object_counts(
            self.draw.objects.len(),
            self.instanced.clusters.iter().map(|c| c.instances.len()),
            self.skinned.slots.draw_objects.iter().map(|o| o.visible),
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
        self.diagnostics
            .pass_timing
            .as_mut()
            .map(|p| p.begin_frame())
            .unwrap_or(0)
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

    // None when no drawable is ready, which skips the frame.
    pub(super) fn acquire_frame(&mut self, world_hidden: bool) -> Option<AcquiredFrame> {
        // Frames-in-flight gate: block until the GPU has retired an older frame
        // so the CPU never queues more than `frames_in_flight` frames ahead,
        // bounding how many sets of per-frame transient buffers pile up. Taken
        // here, before drawable prep and all per-frame buffer building. The slot
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
        let frame_slot = gpu_wait.measure(|| self.frame_pacing.acquire());

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
        self.window_mut().was_visible = true;

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
        // Shader hot-reload: if either the filesystem watcher or the debug
        // `reload-shaders` command set the flag, rebuild every built-in
        // pipeline from disk-resident source before the frame's passes start
        // using them. The flag is cleared regardless of outcome so a failed
        // rebuild (typo in a shader edit) doesn't loop, and the previous
        // pipelines stay live so the session keeps rendering; only a device
        // failure propagates. In-flight command buffers retain the pipelines
        // they encoded, so no GPU drain is needed.
        if self.shader_reload_requested() {
            self.clear_shader_reload_flag();
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
    }

    // Picks this frame's shadow cascades and spot slices; returns the cascade aspect.
    pub(super) fn update_shadow_schedule(
        &mut self,
        cam_pos: [f32; 3],
        fov_y_radians: f32,
        near: f32,
        far: f32,
    ) -> f32 {
        // Compute per-frame cascade VPs + splits from current camera + light.
        // The aspect/near/far are taken from the same params used by the main
        // perspective below so cascades match the visible camera frustum.
        let cascade_aspect = {
            let s = self.window().view.drawableSize();
            if s.height == 0.0 {
                1.0
            } else {
                (s.width / s.height) as f32
            }
        };
        if self.shadow.pipeline_state.is_some() {
            let fresh = csm::compute_shadow_uniforms(csm::ShadowUniformInputs {
                view: self.view.matrix,
                cam_pos,
                fov_y_rad: fov_y_radians,
                aspect: cascade_aspect,
                near,
                shadow_distance: (self.shadow.distance as f32).min(far),
                light_dir_to_source: self.shadow.light_dir,
                shadow_map_size: self.shadow.map_size,
                active_cascades: self.shadow.cascades,
            });
            // Pick this frame's cascades and refresh only their VPs; cascades
            // skipped this frame keep the VP their slice was rendered with so
            // the Main pass samples each slice consistently. Splits depend only
            // on the camera near/far range (not position), so always take fresh.
            let mask = self.next_shadow_cascade_mask();
            self.shadow.render_mask = mask;
            self.shadow.uniforms.cascade_splits = fresh.cascade_splits;
            self.shadow.uniforms.active_cascades = fresh.active_cascades;
            for i in 0..render_types::NUM_SHADOW_CASCADES {
                if mask & (1u32 << i) != 0 {
                    self.shadow.uniforms.light_vps[i] = fresh.light_vps[i];
                }
            }
        }

        // Spot shadow slices refresh on their own prime-then-round-robin clock;
        // their projections are static, so only the depth contents redraw.
        self.spot_shadow.render_mask = self.next_spot_shadow_mask();
        cascade_aspect
    }

    pub(super) fn frame_projection(
        &mut self,
        fov_y_radians: f32,
        aspect: f32,
        near: f32,
        far: f32,
        render_w: u32,
        render_h: u32,
    ) -> FrameProjection {
        // View-projection + GPU-driven cull.
        // The projection / jitter / VP are resolved here, ahead of the main
        // render encoder, because the cull compute pass needs the frustum
        // before the render pass begins.
        let proj = perspective_rh(fov_y_radians, aspect, near, far);
        // This frame's un-jittered VP, captured before the graph runs so the
        // two-pass phase-2 cull (`encode_cull_phase2`, dispatched inside
        // `execute_graph`) can project AABBs through it against the pyramid the
        // mid-frame `HizBuild` rebuilds from this frame's depth. The same value
        // becomes `cull_prev_view_proj` at end-of-frame for next frame's phase 1.
        self.cull.cur_view_proj = mat4_mul(proj, self.view.matrix);
        // When TAA or the MetalFX upscaler is on, offset the projection by
        // a sub-pixel Halton jitter so the temporal accumulator has fresh
        // sample positions each frame. The jitter is a pure NDC x/y shift,
        // so depth is unaffected. `proj[2][0/1]` are the z-coefficients of
        // clip x/y; subtracting the jitter there shifts post-divide NDC by
        // exactly the jitter amount (clip.w == -view_z). Pixel-space
        // jitter (`±0.5` per axis) is stashed for MetalFX, which expects
        // its input in pixel coords; TAA reads NDC directly.
        let needs_jitter = self.taa.enabled || self.upscale.scaler.is_some();
        let proj_render = if needs_jitter {
            let idx = self.taa.frame % 8 + 1;
            let jx_pix = jitter::radical_inverse(idx, 2) - 0.5;
            let jy_pix = jitter::radical_inverse(idx, 3) - 0.5;
            let jx = jx_pix * 2.0 / render_w as f32;
            let jy = jy_pix * 2.0 / render_h as f32;
            if self.upscale.scaler.is_some() {
                self.upscale
                    .jitter
                    .store(jx_pix, jy_pix, std::sync::atomic::Ordering::Release);
            }
            let mut p = proj;
            p[2][0] -= jx;
            p[2][1] -= jy;
            p
        } else {
            proj
        };
        let vp = mat4_mul(proj_render, self.view.matrix);
        // Inverse of the (jittered) view-projection, computed once here and
        // threaded through `GraphFrameParams` to every pass that reconstructs a
        // world-space position from depth (fog, decals, raymarch, transparent),
        // instead of each pass re-inverting `vp` independently.
        let inv_vp = mat4_inverse(vp);
        let frustum = Frustum::from_view_projection(vp);
        FrameProjection {
            proj,
            vp,
            inv_vp,
            frustum,
        }
    }

    // This frame's graph inputs: the passes the live resources and settings run,
    // masked by the viewport's view mode and show flags.
    pub(super) fn frame_graph_inputs(&self, args: GraphInputArgs<'_>) -> FrameGraphInputs {
        let GraphInputArgs {
            bindless_cull_enabled,
            velocity_active,
            fog_settings,
            transparent_active,
            lines,
            world_hidden,
            clustered,
            view_mode,
            show,
        } = args;
        let graph_inputs = FrameGraphInputs {
            shadow_enabled: self.shadow.pipeline_state.is_some(),
            shadow_map_size: self.shadow.map_size,
            hdr_width: self.targets.hdr.width,
            hdr_height: self.targets.hdr.height,
            hdr_sample_count: self.targets.hdr.sample_count,
            bindless_cull_enabled,
            auto_exposure_enabled: self.auto_exposure.pipelines.is_some(),
            // Gated on the pipelines existing: a scene-less world builds none
            // (its 1x1 bloom targets stay untouched black).
            bloom_enabled: self.post_process.bloom_intensity > 0.0
                && self.bloom_pipelines.is_some(),
            // Velocity runs whenever its targets exist: that's TAA on or
            // the upscaler on. The graph builder adds the Velocity pass
            // when this flag is true; TaaResolve / Upscale then declare a
            // read edge on it for ordering.
            velocity_enabled: velocity_active,
            taa_enabled: self.taa.enabled,
            ssr_enabled: self.ssr.settings.is_some(),
            particles_enabled: self.particle.pipelines.is_some()
                && !self.particle.records.is_empty()
                && !self.particle.emitter_state.is_empty(),
            fog_enabled: self.fog.pipeline.is_some() && fog_settings.is_some(),
            decals_enabled: self.decal.pipeline.is_some() && !self.decal.set.is_empty(),
            // The SSR depth + normal + roughness pre-pass also feeds SSGI and
            // the RT-reflection kernel, so it runs when SSR, SSGI, *or* RT
            // reflections are on (RT keys off the live acceleration structure).
            ssr_prepass_enabled: self.ssr.settings.is_some()
                || self.ssgi.settings.is_some()
                || self.rt.accel.is_some(),
            ssao_enabled: self.ssao.settings.is_some(),
            upscale_enabled: self.upscale.scaler.is_some(),
            // Transparent pass runs when at least one translucent producer
            // (`WaterSurface` or `GlassPanel`) exists; the executor
            // short-circuits an empty draw list, but gating here keeps the
            // graph builder from inserting the slot at all.
            transparent_enabled: transparent_active,
            // Lines run only on the frames a system published them (the
            // `cn editor` axes), and only once their pipeline is live: the
            // build above is lazy, so a shipped runtime never compiles it.
            lines_enabled: !lines.is_empty() && self.lines.pipeline.is_some(),
            // Raymarch runs when at least one `SdfVolume` is live; the
            // per-volume pipeline cache is populated in lockstep with
            // the volume vec at init. Tightened to the real `is_some()
            // && !empty()` predicate once the context fields land
            // alongside `encode_raymarch` (see metal/raymarch.rs).
            raymarch_enabled: !self.raymarch.volumes.is_empty(),
            // Two-pass Hi-Z occlusion. Resolved from
            // `PostProcessConfig.occlusion_two_pass` (and gated at init on the
            // bindless cull path existing). The graph builder further ANDs this
            // with `bindless_cull_enabled` for this frame, so a frame with no
            // static geometry simply runs single-pass. When on, the builder
            // inserts HizBuild → Cull2 → Main2 between Main and the post chain.
            two_pass_occlusion_enabled: self.cull.two_pass_occlusion,
            // The terminal Hi-Z build. Present whenever the GPU-cull path built a
            // pyramid: the frame ends by reducing its final depth into it for the
            // next frame's phase-1 occlusion test.
            hiz_build_enabled: self.cull.hiz.is_some(),
            // SSGI runs when `indirect_lighting: "ssgi"` resolved settings that
            // contribute: the composite scales by intensity, so zero would pay a
            // hemisphere ray-march to add nothing. The builder inserts the Ssgi
            // RMW pass after Raymarch on the hdr_resolve chain; the gather reads
            // the SSR pre-pass G-buffer (forced on above via
            // `ssr_prepass_enabled`).
            ssgi_enabled: self.ssgi.settings.is_some_and(|s| s.contributes()),
            // RT reflections run when the scene acceleration structure is live
            // (RT requested + GPU supports it + scene has geometry). The builder
            // inserts the RtReflections pass in the SsrResolve slot and, when
            // both are on, picks it over SsrResolve (RT takes precedence; SSR is
            // the cross-backend fallback).
            rt_reflections_enabled: self.rt.accel.is_some(),
            // Metal collapses the SSR / SSAO / velocity pre-passes into one
            // GBufferPrepass node; the other backends keep them separate.
            gbuffer_prepass_enabled: true,
            // An opaque menu backdrop hides the scene: the builder masks every
            // world pass off, collapsing to Main (a bare clear, fed the empty
            // scene above) -> Composite (presents the overlay).
            world_hidden,
            // Clustered light binning runs when the world has local lights (the
            // cull pipeline is built iff so). The builder inserts LightCull
            // before Main and Main reads its per-cluster list buffer.
            clustered_lighting_enabled: clustered,
            // Set by the view-mode mask below (occlusion view only).
            composite_reads_ao: false,
            shadowed_spot_count: self.spot_shadow.count,
            spot_shadow_slice_size: render_types::spot_shadow_slice_size(self.shadow.map_size),
        };
        // The viewport's view mode + show flags mask the seeded inputs (the
        // per-frame counterpart of the init-time trims); Lit with every flag
        // set is the identity, so a shipped runtime is unaffected.
        render_graph::apply_view(&graph_inputs, view_mode, show)
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
            self.last_present_texture = Some(drawable.texture());
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
                move |cb: std::ptr::NonNull<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>| {
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

    pub(super) fn advance_temporal_state(&mut self, velocity_active: bool, proj: [[f32; 4]; 4]) {
        // The Hi-Z reduction that feeds next frame's cull is the graph's terminal
        // `HizFinal` pass, so it has already been encoded. Advance the temporal
        // state it depends on: the pyramid is now valid for next frame's cull, and
        // the un-jittered VP captured at the top of the frame becomes the
        // projection that cull tests through (distinct from the velocity
        // pre-pass's `prev_view_proj`, which only advances when velocity runs).
        if self.cull.hiz.is_some() {
            self.cull.hiz_valid = true;
            self.cull.prev_view_proj = self.cull.cur_view_proj;
        }

        // Advance temporal state for the next frame whenever the velocity
        // pre-pass runs: that's TAA *or* the MetalFX upscaler. The
        // un-jittered VP becomes `prev_vp` so the velocity shader can
        // diff against it; the per-object transforms were snapshotted on the
        // GPU by the pre-pass's own history dispatch. TAA-specific bookkeeping
        // (history-target ping-pong) only runs when TAA itself is on.
        if velocity_active {
            self.prev_view_proj = mat4_mul(proj, self.view.matrix);
            self.taa.frame = self.taa.frame.wrapping_add(1);
            if let Some(taa) = self.taa.pass.as_mut() {
                taa.advance();
            }
        }
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

    // This frame's probe cube argument buffer, plus the residency every
    // argument buffer's contents need.
    pub(super) fn refresh_argument_buffers(&mut self, ring_slot: usize) -> error::RenderResult<()> {
        // The probe cube handles, for every pass that samples the set. Built
        // ahead of `build_scene_buffers` and outside its world-hidden gate:
        // the transparent and post passes read the set without a static draw
        // list of their own, and a slot left holding last frame's ring buffer
        // would outlive the frame that wrote it.
        self.probe.cube_args = Some(self.build_probe_cube_args(ring_slot)?);
        // The residency sets the argument buffers' contents need. Refreshed
        // before any pass encodes, and a no-op on a frame whose textures are
        // unchanged, which is every frame between a stream-in or a bake.
        self.refresh_probe_cube_residency();
        self.refresh_bindless_residency();
        Ok(())
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
            near,
            far,
            world_hidden,
            skinned_joint_bufs,
        } = args;
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
            // hence the bindings handed back to the caller, which holds them
            // through to `cmd_buf.commit()`.
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
                crate::metal::raytrace::RtFrame {
                    id: frame_id,
                    ring_slot,
                },
                skinned_joint_bufs,
            );

            (object_buffer, cull_draw_args, bindless_tex_args)
        };
        Ok(SceneBuffers {
            object_buffer,
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
        let (prev_model_buffer, history_targets) = if object_buffer_live
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
        Ok(HistoryBuffers {
            deformed_this_frame,
            deformed_prev_frame,
            prev_model_buffer,
            history_targets,
        })
    }
}
