//! The stages `MtlContext::draw_frame` runs in order around its render-graph
//! dispatch.

use concinnity_core::math::{SaturatingFrom, narrow_f32};
use concinnity_core::memory::InlineVec;
use concinnity_engine::profile;
use concinnity_engine::profile::FrameSpans;
use concinnity_engine::render::backend::{FrameParams, error};
use concinnity_engine::render::config::apply::SettingsTarget as _;
use concinnity_engine::render::frame::plan::{
    EffectInputs, FramePlan, PlanInputs, SurfaceInputs, camera_inputs,
};
use concinnity_engine::render::render_graph::{FrameContent, ResidentFeatures, ResidentUpscaler};
use concinnity_engine::render::temporal::model_history::HistoryMode;
use concinnity_engine::render::temporal::reactive_mask::ReactiveReader;
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

// The recorded frame's presenting command buffer and its completion join.
pub(super) struct PresentFrame {
    pub(super) cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub(super) drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
    pub(super) join: std::sync::Arc<FrameJoin>,
    pub(super) pending_terminal: Option<u64>,
    pub(super) submission_token: SubmissionToken,
}

// The per-frame inputs the scene buffer builds read.
#[derive(Clone, Copy)]
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
    pub(super) history_targets: InlineVec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl MtlContext {
    pub(super) fn begin_frame_stats(&mut self) -> usize {
        // Reset this frame's render stats; the draw counters below accumulate
        // into `diagnostics.frame_stats`, and `render_stats()` reports them (plus the GPU
        // frame time) to the profiler overlay.
        let counts = concinnity_engine::render::backend::object_counts::object_counts(
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

    // None when no drawable is ready, which skips the frame.
    pub(super) fn acquire_frame(&mut self, world_hidden: bool) -> Option<AcquiredFrame> {
        // Frames-in-flight gate, unless `prepare_frame` already reserved the
        // slot. The slot is released by the frame's completion handler, or on
        // drop if the frame is abandoned before commit. This wait and the
        // drawable acquire are measured into `gpu_wait`, which separates a
        // GPU-bound frame from a CPU-bound one.
        let mut gpu_wait = crate::gpu_wait::GpuWait::none();
        let frame_slot = match self.prepared_slot.take() {
            Some(slot) => slot,
            None => gpu_wait.measure(|| self.frame_pacing.acquire()),
        };

        // Probe bake prefiltering: convolve a mip or install a cube ahead of the
        // argument buffers, so an installed cube is sampled this frame. Skipped
        // while the world is hidden.
        if !world_hidden {
            self.advance_probe_prefilter();
        }

        // tell MTKView to prepare its drawable for this frame, then take it.
        // `currentDrawable` blocks on the drawable pool when every drawable is
        // still with the compositor, which is the display-paced half of the
        // frame's GPU wait.
        let drawable = gpu_wait.measure(|| {
            self.window_state().view.draw();
            self.window_state().view.currentDrawable()
        });
        self.diagnostics.frame_stats.gpu_wait_us = gpu_wait.micros();
        // drawable not yet available -- skip this frame silently
        let drawable = drawable?;

        // This frame's ring slot, safe to overwrite once the fence retired its
        // last user. Skipped frames do not consume one.
        let frame_id = self.frame_ring_index;
        let ring_slot = usize::saturating_from(frame_id % self.frames_in_flight as u64);
        self.frame_ring_index = frame_id.wrapping_add(1);
        // Settings-replaced feature state the fence now proves no frame reads.
        self.retired_effects
            .collect(frame_id, self.frames_in_flight as u64);

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
        // Shader hot reload: rebuild every built-in pipeline from source. The
        // flag clears either way and a failed rebuild keeps the old pipelines;
        // only a device failure propagates. In-flight buffers retain their
        // pipelines, so no drain.
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
        // Step auto-exposure before anything reads the exposure.
        self.update_auto_exposure(elapsed, ring_slot);
        self.pre_exposure.advance(self.post_process.exposure);
        if self.upscale.scaler.is_some() && self.upscale.exposure.advance(&self.pre_exposure) {
            self.upscale.reset.request();
        }
    }

    // What this backend has built, for the frame's graph gates.
    pub(super) fn resident_features(&self) -> ResidentFeatures {
        let maps = self.built_shadow_maps();
        ResidentFeatures {
            hdr_width: self.targets.hdr.width,
            hdr_height: self.targets.hdr.height,
            hdr_sample_count: self.targets.hdr.sample_count,
            shadow: maps.cascades > 0,
            shadow_map_size: maps.cascades,
            far_shadow_map_size: maps.far,
            cull: self.cull.bindless,
            two_pass: self.cull.two_pass_occlusion,
            hiz: self.cull.hiz.is_some(),
            shadow_receivers: self.shadow_receivers.is_some(),
            view_samples: self.view_samples.is_some(),
            bloom_chain: self.bloom.is_some(),
            upscaler: self.upscale.scaler.as_ref().map(|s| ResidentUpscaler {
                reads_reactive: ReactiveReader::MetalFx.reads() && s.reactive,
            }),
            particles: self.particle.pipelines.is_some(),
            fog: self.fog.pipeline.is_some(),
            decals: self.decal.pipeline.is_some(),
            lines: self.lines.pipeline.is_some(),
            atmosphere: self.atmosphere.is_some(),
            clouds: self.atmosphere.as_ref().is_some_and(|a| a.clouds.is_some()),
            rt_transparent: self.transparent_rt_pipelines_ready(),
            taa: self.taa.enabled,
            disocclusion: self.disocclusion.is_some(),
            model_history: self.gbuffer.targets.is_some() && self.cull.main_pipeline.is_some(),
            reflection_composite: self.ssr.composite.is_some(),
        }
    }

    // What the world puts on screen this frame, for the frame's graph gates.
    pub(super) fn frame_content(&self, params: &FrameParams<'_>) -> FrameContent {
        FrameContent {
            world_hidden: params.world_hidden,
            cull_records: self.cull_count(),
            shadowed_spot_count: self.spot_shadow.count,
            bloom_intensity: self.post_process.bloom_intensity,
            bvh_live: self.rt.accel.is_some(),
            emitters: !self.state.particles.is_empty(),
            late_emitters: self.state.particles.draws_after_transparent(),
            fog: self.state.fog.settings().is_some_and(|s| s.contributes()),
            decals: !self.state.decals.is_empty(),
            visible_surfaces: (self.water.pipeline.is_some()
                && self.water.surfaces.iter().any(|s| s.visible))
                || (self.glass.pipeline.is_some() && self.glass.panels.iter().any(|p| p.visible)),
            visible_see_through: self.see_through_mesh_visible(),
            planar_targets: self
                .planar_reflection
                .as_ref()
                .is_some_and(|s| !s.targets.is_empty()),
            water_planar_slot: self.water_planar_slot_live(),
            lines: !params.lines.is_empty(),
            visible_volumes: self.raymarch.visible_bounds(),
            atmosphere: self.atmosphere.as_ref().is_some_and(|a| a.params.is_some()),
            clouds: self
                .atmosphere
                .as_ref()
                .and_then(|a| a.clouds.as_ref())
                .is_some_and(super::super::clouds::CloudState::live),
            water_sim: self.water_sim_wave(),
            grass: self.grass.as_ref().map(|g| g.field.reach()),
            ssgi_motion: self.ssgi.pass.is_some()
                && self.ssgi.settings.is_some_and(|s| s.contributes()),
            prop_probe: params
                .prop_probe
                .filter(|_| self.prop_probe_supported())
                .map(|p| p.bounds()),
        }
    }

    // Plan this frame in the engine over what this backend holds, then put what the
    // encoders read from `&self` in place: the spot slices, the cull's view and
    // the cluster grid, and the jitter MetalFX is dispatched with.
    pub(super) fn plan_frame(
        &mut self,
        params: &FrameParams<'_>,
        render_size: [u32; 2],
    ) -> error::RenderResult<FramePlan> {
        let resident = self.resident_features();
        let content = self.frame_content(params);
        // MetalFX takes the 16-frame Halton cycle, which covers its output
        // pixels better than the TAA cycle.
        let upscaler_jitter = self
            .upscale
            .scaler
            .as_ref()
            .map(|_| concinnity_engine::gfx::jitter::offset(self.state.planner.jitter_phase()));
        let surface = SurfaceInputs {
            prefilter_mip_count: self.scene.env_map.prefilter_mip_count as f32,
            irradiance: self.irradiance_lane(),
            screen_bounce: self.screen_bounce(),
        };
        let inputs = PlanInputs {
            camera: camera_inputs(&self.state, params),
            elapsed: narrow_f32(params.elapsed),
            render_size,
            resident,
            content,
            upscaler_jitter,
            lights: &self.state.lights,
            probes: self.state.probes.records(),
            fog: self.state.fog.settings(),
            pre_exposure: self.pre_exposure,
            adaptation: self.auto_exposure.adaptation.as_ref(),
            surface,
            effects: EffectInputs {
                ssao: self.ssao.settings,
                ssr: self.ssr.settings,
                ssgi: self.ssgi.settings.zip(
                    self.ssgi
                        .pass
                        .as_ref()
                        .map(concinnity_engine::render::post::ssgi::SsgiPass::frame),
                ),
                rt: self.rt.settings,
            },
        };
        let plan = self.state.planner.plan(&self.state.view, &inputs)?;
        self.spot_shadow.render_mask = self.state.planner.shadow.spot_mask;
        self.cull.cur_view_proj = plan.camera.cur_vp;
        self.cluster_params = plan.cluster;
        // MetalFX expects the jitter in render pixels, measured from the sample
        // back to the pixel center; TAA reads it from the projection itself.
        if self.upscale.scaler.is_some() {
            let [x, y] = concinnity_engine::gfx::jitter::pixel_center_offset(plan.camera.jitter_px);
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
            pending_terminal,
            submission_token,
        } = frame;
        let texture = drawable.texture();
        self.reserve_frame_capture(&texture);
        self.record_frame_capture(&cmd_buf, &texture);
        cmd_buf.presentDrawable(ProtocolObject::from_ref(&*drawable));

        // Retain the drawable's texture under `hot_reload` so `screenshot` can
        // read the last presented frame. Queue order guarantees it is fully
        // rendered by the next frame.
        if self.capture {
            self.last_present_texture = Some(texture);
        }

        // The presenting buffer's handler reports faults and the whole-frame GPU
        // span, then arrives at the frame's completion join, whose last arrival
        // resolves timing and releases the slot.
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
                    part.composite_span_us.store(
                        u32::saturating_from((span * 1.0e6).clamp(0.0, f64::from(u32::MAX))),
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
        let draw_size = self.window_state().view.drawableSize();
        // Geometry-less worlds keep their off-screen targets pinned at 1x1
        // (see MtlContext::new); the composite pass still uses the full drawable.
        let (want_w, want_h) = if self.targets.geometry_less {
            (1, 1)
        } else {
            (
                u32::saturating_from(draw_size.width.max(1.0)),
                u32::saturating_from(draw_size.height.max(1.0)),
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
        // Probe records for every pass that samples the set, built outside the
        // world-hidden gate since transparent and post passes read them too.
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
        // A hidden world feeds Main an empty scene, so it runs as a bare clear
        // under the opaque overlay.
        let (object_buffer, material_params, cull_draw_args, bindless_tex_args) = if world_hidden {
            (None, None, None, None)
        } else {
            // Per-frame bindless buffers, handed back so the caller holds them
            // until commit.
            let packed = std::time::Instant::now();
            let records = match records {
                Some(history) => self.build_record_buffers(ring_slot, cam_pos, history)?,
                None => None,
            };
            self.state.spans.set(FrameSpans {
                record_pack_us: concinnity_host::clock::micros_since(packed),
                ..self.state.spans.get()
            });
            let (object_buffer, draw_args) = match records {
                Some(records) => {
                    self.diagnostics.frame_stats.packing_fan_out = records.packing;
                    (Some(records.objects), Some(records.draw_args))
                }
                None => (None, None),
            };
            let material_params = match object_buffer {
                Some(_) => Some(self.rings.material_params.buffer(
                    &self.hw.device,
                    &mut self.state.material_params,
                    ring_slot,
                )?),
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
                // The prop probe culls into one more slot after the planes'.
                let mirror_slots =
                    self.prop_probe_cull_slot() + usize::from(self.probe.prop.schedule.capturing());
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
            // Keep the acceleration structure current before any pass reads it.
            // A failed rebuild keeps last frame's BVH.
            let rt_started = std::time::Instant::now();
            self.rt_dynamic_update(
                crate::metal::raytrace::RtFrame {
                    id: frame_id,
                    ring_slot,
                },
                skinned_joint_bufs,
            );
            self.state.spans.set(FrameSpans {
                rt_accel_us: concinnity_host::clock::micros_since(rt_started),
                ..self.state.spans.get()
            });

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
        // This frame's deformed skinned buffer, cloned so `params` owns a
        // handle. `Some` only while the fold is active.
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
        // Model-history slots: the one last frame's snapshot filled and the
        // ones this frame fills. Bound whenever the pre-pass runs; priming fills
        // every slot.
        let (prev_model_buffer, history_targets) = if object_buffer_live
            && self.gbuffer.targets.is_some()
            && self.cull.main_pipeline.is_some()
        {
            let bytes = self.cull_count() * size_of::<[[f32; 4]; 4]>();
            let prime = self.state.model_history.get_mut().take_prime();
            let read_slot = (ring_slot + self.frames_in_flight - 1) % self.frames_in_flight;
            let mut targets = InlineVec::new();
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
            (None, InlineVec::new())
        };
        Ok(HistoryBuffers {
            deformed_this_frame,
            deformed_prev_frame,
            prev_model_buffer,
            history_targets,
        })
    }
}
