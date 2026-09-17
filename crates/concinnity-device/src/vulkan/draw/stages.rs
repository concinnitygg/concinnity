//! The stages `VkContext::draw_frame` runs in order around `record_frame`, and
//! the per-frame state `record_frame` advances around its graph dispatch.

use ash::vk;
use concinnity_core::components;
use concinnity_core::gfx::render_types;
use concinnity_core::profile;
use concinnity_core::profile::PassTiming;
use concinnity_core::render::csm;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::hdr_output;
use concinnity_core::render::pass_timing;

use super::upload_shadow_uniforms;
use crate::gpu_wait::GpuWait;
use crate::vulkan::context::VkContext;
use crate::vulkan::error::map_vk_result;

impl VkContext {
    // Rebuilds requested since last frame: wireframe pipelines and hot-reloaded shaders.
    pub(in crate::vulkan) fn apply_pending_rebuilds(&mut self) {
        // Vulkan polygon mode is pipeline state, so the wireframe view needs its
        // own main-pass pipelines; built here on the first wireframe frame.
        self.ensure_wireframe_pipelines();
        // Shader hot-reload: if either the filesystem watcher or the debug
        // `reload-shaders` command set the flag, rebuild every built-in
        // pipeline from disk-resident source before this frame's passes
        // start using them. The flag is cleared regardless of outcome so a
        // failed rebuild (typo in a shader edit) doesn't loop, and the
        // previous pipelines stay live so the session keeps rendering.
        // Wait for the GPU to drain first so swapping pipelines out from
        // under in-flight command buffers is safe. Mirrors the DirectX
        // `apply_pending_rebuilds`.
        if self.shader_reload_requested() {
            self.clear_shader_reload_flag();
            self.wait_idle();
            match self.reload_shaders() {
                Ok(()) => tracing::info!("hot-reload: shader pipelines rebuilt"),
                Err(e) => tracing::error!("hot-reload: shader rebuild failed: {}", e),
            }
        }
    }

    // Blocks on this frame slot's previous submission, then runs the ticks it gates.
    pub(in crate::vulkan) fn wait_frame_slot(&mut self, frame: usize) -> RenderResult<GpuWait> {
        // Wait for this frame's slot to finish. Measured, with the swapchain
        // acquire in `acquire_frame`, into the frame's `gpu_wait_us`: both block
        // the CPU on the GPU inside `draw_frame`, which the engine times its
        // graphics system around.
        let mut gpu_wait = crate::gpu_wait::GpuWait::none();
        gpu_wait
            .measure(|| {
                // SAFETY: the fence belongs to this frame slot and was created from this device; the
                // slice borrows it for the call.
                unsafe {
                    self.hw.device.wait_for_fences(
                        std::slice::from_ref(&self.frame_sync.in_flight[frame]),
                        true,
                        u64::MAX,
                    )
                }
            })
            .map_err(|e| map_vk_result(e, "wait fences"))?;

        // Streamed texture swaps: re-point this frame slot's bindless pool
        // copy at the swapped-in views (legal now -- the fence wait above
        // retired every command buffer that binds this slot's set), and free
        // the old images / upload transients this slot parked on its previous
        // trip (this slot's fence signaling also covers the older frames that
        // last sampled them, and every pool copy has been re-pointed since).
        self.apply_streamed_texture_rewrites(frame);

        // Reclaim this frame slot's shared post-pass descriptor sets. Here for
        // the same reason as the two ticks below: the fence wait above is what
        // makes reclaiming the previous pass's sets legal.
        self.post.arena.begin_frame(&self.hw.device, frame);

        // Tick the device allocator: destroy retired handles, reclaim retired
        // ranges, release empty blocks. Here because the fence wait above is
        // what guarantees a range freed `retire_depth` ticks ago is no longer
        // referenced.
        self.hw.alloc.begin_frame();
        // Same tick for the owned pipeline / layout / render-pass handles a
        // rebuild displaced, on the same reasoning.
        self.hw.device.begin_frame();

        // Periodic footprint readout, for measuring the pool under streaming
        // churn at scale. Inert unless debug logging is enabled.
        if self.stream.frame.is_multiple_of(1024) && tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!("device allocator: {}", self.hw.alloc.stats());
        }
        Ok(gpu_wait)
    }

    // Probe bake and auto-exposure steps that need the slot's GPU work retired.
    pub(in crate::vulkan) fn service_background_work(&mut self, elapsed: f32, frame: usize) {
        // Advance the staggered reflection-probe bake one step. Runs here -- after
        // this frame's slot fence wait, before `record_frame` -- so any cube it
        // installs (a binding-8 rewrite + `probe.set.count` bump) is picked up by this
        // frame's `record_frame` ProbeSet upload + rendering. Non-fatal.
        if let Err(e) = self.bake_pending_probes() {
            tracing::warn!("reflection probe bake step failed: {e}");
        }

        // Auto-exposure: step the EMA from a previous frame's GPU
        // measurement before any pipeline reads `post_process.exposure`.
        // The `wait_frame_slot` fence wait already gated the
        // GPU work that wrote this slot's readback, so the value is
        // committed. No-op when auto-exposure is disabled.
        self.update_auto_exposure(elapsed, frame);
    }

    // Whole-frame and per-pass GPU times this slot's last submission resolved.
    pub(in crate::vulkan) fn read_gpu_timings(
        &self,
        frame: usize,
    ) -> (u32, [PassTiming; profile::MAX_PASS_TIMINGS]) {
        let device = &self.hw.device;
        // GPU timing for the most-recently completed block on this frame slot:
        // the whole-frame pair plus one (start, end) pair per render pass. The
        // `wait_frame_slot` fence wait guarantees the previous trip's writes have
        // retired, so the available query results are committed. The block is read with
        // `WITH_AVAILABILITY` so a pass that did not run this trip (its slots were
        // reset but never written) reads back unavailable -> 0, without stalling
        // the host (no `WAIT`). Zero before a slot has been visited a second time.
        let empty_pass_times = [("", 0u32); profile::MAX_PASS_TIMINGS];
        if let Some(pool) = self.hw.timestamp_query_pool {
            // One [value, availability] pair per query slot (TYPE_64 +
            // WITH_AVAILABILITY -> two u64 per query; ash uses the element size as
            // the stride and the slice length as the query count).
            let mut results = vec![[0u64; 2]; pass_timing::SLOTS_PER_FRAME];
            // SAFETY: a property query on a live handle; it only reads.
            let res = unsafe {
                device.get_query_pool_results(
                    pool,
                    pass_timing::frame_block_base(frame),
                    &mut results,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                )
            };
            // WITH_AVAILABILITY fills the buffer + per-query availability bits and
            // returns SUCCESS; tolerate NOT_READY defensively (the buffer is still
            // written, and the availability bits gate every read).
            if matches!(res, Ok(()) | Err(vk::Result::NOT_READY)) {
                let period = self.hw.timestamp_period_ns;
                let pair_micros = |start_slot: usize, end_slot: usize| -> u32 {
                    let [s_val, s_avail] = results[start_slot];
                    let [e_val, e_avail] = results[end_slot];
                    if s_avail != 0 && e_avail != 0 && e_val > s_val && period > 0.0 {
                        let nanos = (e_val - s_val) as f64 * period as f64;
                        ((nanos / 1000.0) as u64).min(u32::MAX as u64) as u32
                    } else {
                        0
                    }
                };
                pass_timing::decode_frame_block(pair_micros)
            } else {
                (0, empty_pass_times)
            }
        } else {
            (0, empty_pass_times)
        }
    }

    // Publishes this frame's stats before recording; draw calls fill in after.
    pub(in crate::vulkan) fn begin_frame_stats(
        &self,
        gpu_wait: &GpuWait,
        (gpu_frame_us, pass_times_us): (u32, [PassTiming; profile::MAX_PASS_TIMINGS]),
    ) {
        // Reset this frame's render stats. `record_frame` accumulates
        // `draw_calls` through `inc_draw_calls` (interior-mutability since
        // the encoders run through `&self`); the rest is filled here from
        // context state.
        let counts = crate::object_counts::object_counts(
            self.draw.objects.len(),
            self.instanced.clusters.iter().map(|c| c.instances.len()),
            self.skinned.slots.draw_objects.iter().map(|o| o.visible),
        );
        let vram_bytes = self.query_vram_bytes();
        let transient_pool_bytes = self.targets.transient_pool.allocated_bytes();
        // Reset the parallel-safe draw-call accumulator for this frame; the
        // encoders fetch_add into it during recording and `record_frame`
        // drains it back into `frame_stats.draw_calls` once recording is done.
        self.draw_calls_accum
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.frame_stats.set(profile::RenderStats {
            draw_calls: 0,
            objects: counts.objects,
            skinned_visible: counts.skinned_visible,
            gpu_frame_us,
            // The fence wait alone so far; `acquire_frame` adds to it.
            gpu_wait_us: gpu_wait.micros(),
            vram_bytes,
            transient_pool_bytes,
            pass_times_us,
            // Adapted auto-exposure EV for the StatHud `EV` chip. `Some` only
            // when the world opted into auto-exposure (the EMA state is then
            // live); the static-exposure path leaves it `None` so the chip
            // stays blank. The value is the EV the most recent
            // `update_auto_exposure` EMA step settled on (the multiplier the
            // post stack pushes is `2^ev`). Mirrors `DxContext` / `MtlContext`.
            auto_exposure_ev: self.auto_exposure.state.as_ref().map(|s| s.current_ev),
            // EDR headroom for the StatHud `EDR x.X` chip, taken from the
            // `HdrOutputMode` resolved at init. `Some` only on the HDR path
            // (Vulkan has no portable max-EDR query, so the value is the
            // synthesized placeholder set in `init`); `None` on SDR blanks the
            // chip. Mirrors `DxContext` / `MtlContext::render_stats`.
            max_edr: match self.hw.hdr_mode {
                hdr_output::HdrOutputMode::Hdr { max_edr, .. } => Some(max_edr),
                hdr_output::HdrOutputMode::Sdr => None,
            },
            ..profile::RenderStats::default()
        });
    }

    // Acquires the swapchain image, or `None` when the swapchain was rebuilt instead.
    pub(in crate::vulkan) fn acquire_frame(
        &mut self,
        frame: usize,
        gpu_wait: &mut GpuWait,
    ) -> RenderResult<Option<u32>> {
        // Acquire swapchain image. Blocks when the presentation engine holds
        // every image, so it is the display-paced half of the frame's GPU wait.
        let acquire = gpu_wait.measure(|| {
            // SAFETY: `self.swapchain.handle` is the live swapchain and `image_available[frame]` is
            // an unsignaled semaphore from this device's own pool for this frame slot.
            unsafe {
                self.swapchain.loader.acquire_next_image(
                    self.swapchain.handle,
                    u64::MAX,
                    self.frame_sync.image_available[frame],
                    vk::Fence::null(),
                )
            }
        });
        // Fold the acquire into the reading `begin_frame_stats` published, which
        // the stats snapshot had already captured with the fence wait alone.
        let mut waited = self.frame_stats.get();
        waited.gpu_wait_us = gpu_wait.micros();
        self.frame_stats.set(waited);
        let image_index = match acquire {
            Ok((idx, suboptimal)) => {
                if suboptimal {
                    self.rebuild_swapchain()?;
                    return Ok(None);
                }
                idx
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.rebuild_swapchain()?;
                return Ok(None);
            }
            Err(e) => return Err(map_vk_result(e, "acquire swapchain image")),
        };

        let device = &self.hw.device;
        // SAFETY: the fence belongs to this frame slot and was just waited on, so it is signaled
        // and not in use by a pending submission.
        unsafe { device.reset_fences(std::slice::from_ref(&self.frame_sync.in_flight[frame])) }
            .map_err(|e| map_vk_result(e, "reset fences"))?;
        Ok(Some(image_index))
    }

    // Submits the recorded buffers and presents, rebuilding the swapchain when out of date.
    pub(in crate::vulkan) fn submit_and_present(
        &mut self,
        frame: usize,
        image_index: u32,
        submit_bufs: &[vk::CommandBuffer],
    ) -> RenderResult<()> {
        let device = &self.hw.device;
        // Submit the whole batch in one call: submission order = GPU order on
        // the single graphics queue. The render-finished semaphore is indexed
        // by swapchain image (not frame slot) so present never reuses one still
        // in flight.
        let wait_sems = [self.frame_sync.image_available[frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_sems = [self.frame_sync.render_finished[image_index as usize]];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(submit_bufs)
            .signal_semaphores(&signal_sems);
        // SAFETY: every command buffer in `submit_bufs` was ended and belongs to this frame slot,
        // the semaphores and fence were created from this device, and `submit_info` borrows all of
        // them for the call.
        unsafe {
            device
                .queue_submit(
                    self.hw.graphics_queue,
                    std::slice::from_ref(&submit_info),
                    self.frame_sync.in_flight[frame],
                )
                .map_err(|e| map_vk_result(e, "queue submit"))?;
        }

        // Present.
        let swapchains = [self.swapchain.handle];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_sems)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        // SAFETY: `present_info` borrows the swapchain, image index, and wait semaphore for the
        // call; the semaphore is signaled by the submission above.
        let present_result = unsafe {
            self.swapchain
                .loader
                .queue_present(self.hw.present_queue, &present_info)
        };
        if present_result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR) || present_result == Ok(true) {
            self.rebuild_swapchain()?;
        } else {
            present_result.map_err(|e| map_vk_result(e, "present"))?;
            // Record which swapchain image now holds a complete, presented frame
            // so the `screenshot` debug command can read it back.
            self.swapchain.last_present_index = Some(image_index);
        }

        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;
        Ok(())
    }

    // Cascade light VPs and splits for this camera, the shadow UBO upload, and the spot schedule.
    pub(super) fn update_shadow_schedule(
        &mut self,
        extent: vk::Extent2D,
        cam_pos: [f32; 3],
        fov_y_radians: f32,
        near: f32,
        far: f32,
        frame_idx: usize,
    ) {
        // Recompute cascade VPs + splits from the current camera + light, and
        // push the result to the shadow UBO so both passes see the same data.
        let cascade_aspect = if extent.height == 0 {
            1.0
        } else {
            extent.width as f32 / extent.height as f32
        };
        if self.shadow.pipeline.is_some() {
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
            // Advance the cascade schedule and refresh only this frame's
            // cascades' light VPs; skipped cascades keep the VP + depth their
            // slice was last rendered with, so the Main pass samples each cascade
            // consistently. Splits depend only on the camera range (not which
            // cascades render), so always refresh. encode_shadow_pass
            // re-rasterizes only the masked slices.
            let update = self.shadow.update;
            let mask = self
                .shadow
                .scheduler
                .next_mask(update, self.shadow.cascades);
            self.shadow.render_mask = mask;
            self.shadow.uniforms.cascade_splits = fresh.cascade_splits;
            self.shadow.uniforms.active_cascades = fresh.active_cascades;
            for i in 0..render_types::NUM_SHADOW_CASCADES {
                if mask & (1u32 << i) != 0 {
                    self.shadow.uniforms.light_vps[i] = fresh.light_vps[i];
                }
            }
            upload_shadow_uniforms(&self.shadow.ubos[frame_idx], &self.shadow.uniforms);
        }

        // Spot shadow refresh schedule. Prime-then-round-robin over the slices,
        // so N shadowed spots cost one extra depth render per frame rather than
        // N. No uniform refresh: the projections are static and were baked at
        // init. A no-op (mask stays 0) when the world has no shadowed spot.
        self.spot_shadow.advance(matches!(
            self.shadow.update,
            components::ShadowUpdate::EveryFrame
        ));
    }

    // History the next frame reads: TAA jitter and ring, G-buffer VP, Hi-Z VP and validity.
    pub(super) fn advance_temporal_state(&mut self, cur_vp: [[f32; 4]; 4]) {
        // The Hi-Z reduction that feeds next frame's cull is the graph's terminal
        // `HizFinal` pass, so it has already been recorded; `hiz_valid` only
        // tracks whether a pyramid at the current resolution now exists.

        // The cascade slices rest sampled (SHADER_READ_ONLY_OPTIMAL) between
        // frames; next frame's Shadow producer barrier (graph-driven) performs
        // the SHADER_READ_ONLY -> DEPTH_STENCIL_ATTACHMENT reset over every
        // cascade layer, so no inline end-of-frame restore is needed here.

        // Advance the TAA jitter sequence and the accumulation ring that
        // validates next frame's history. The motion-vector temporal state lives
        // on the unified G-buffer (advanced below); TAA only consumes its
        // velocity view.
        if let Some(taa) = &mut self.taa {
            taa.taa_frame = taa.taa_frame.wrapping_add(1);
            // Step the accumulation ring in lockstep: what this frame wrote is
            // next frame's history.
            taa.pass.advance();
        }

        // Advance the unified G-buffer's velocity-channel temporal state in
        // lockstep with TAA's: this frame's un-jittered VP becomes next frame's
        // `prev_vp`. The per-object half of the same history was snapshotted on
        // the GPU by the pre-pass's own dispatch. Owned by `GbufferResources` so
        // the motion vector works for any consumer (TAA or FSR), exactly
        // mirroring the TAA advance above.
        if let Some(gb) = &mut self.gbuffer {
            gb.prev_view_proj = cur_vp;
        }

        // Advance Hi-Z temporal state: this frame's un-jittered VP becomes next
        // frame's occlusion-test projection, and the pyramid the graph's
        // `HizFinal` pass just wrote is now valid for next frame's cull (kept
        // independent of TAA, which may be off while Hi-Z is on).
        if self.cull.hiz.is_some() {
            self.cull.hiz_prev_view_proj = cur_vp;
            self.cull.hiz_valid = true;
        }
    }

    pub(super) fn finish_frame_stats(&self) {
        // Drain the parallel-safe draw-call accumulator (bumped by every pass
        // encoder, including those fanned onto rayon workers) into this frame's
        // `frame_stats` for the profiler overlay. All recording is done by here.
        let mut stats = self.frame_stats.get();
        stats.draw_calls = self
            .draw_calls_accum
            .load(std::sync::atomic::Ordering::Relaxed);
        self.frame_stats.set(stats);
    }
}
