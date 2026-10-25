//! The stages `DxContext::draw_frame` runs in order around `record_frame`, and
//! the per-frame inputs `record_frame` derives before its graph dispatch.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::LineVertex;
use concinnity_engine::profile;
use concinnity_engine::profile::PassTiming;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::backend::pass_timing;
use concinnity_engine::render::frame::plan::{
    CameraInputs, EffectInputs, FramePlan, PlanInputs, SurfaceInputs,
};
use concinnity_engine::render::render_graph::{FrameContent, ResidentFeatures, ResidentUpscaler};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::WaitForSingleObject;
use windows::core::Interface;

use crate::directx::context::DxContext;

// The longest a frame start waits for DXGI to accept another present.
const FRAME_LATENCY_TIMEOUT_MS: u32 = 1000;
use crate::directx::error::{self, map_hresult};
use crate::gpu_wait::GpuWait;

// The per-frame inputs `plan_frame` plans from.
#[derive(Clone, Copy)]
pub(in crate::directx) struct PlanArgs<'a> {
    pub(in crate::directx) camera: CameraInputs,
    pub(in crate::directx) elapsed: f32,
    pub(in crate::directx) render_size: [u32; 2],
    pub(in crate::directx) lines: &'a [LineVertex],
    pub(in crate::directx) world_hidden: bool,
}

impl DxContext {
    // Rebuilds requested since last frame: wireframe PSOs, hot-reloaded shaders, resize.
    pub(in crate::directx) fn apply_pending_rebuilds(&mut self) -> RenderResult<()> {
        // D3D12 fill mode is pipeline state, so the wireframe view needs its own
        // main-pass PSOs; built here on the first wireframe frame so the `&self`
        // pass encoders can just read them.
        self.ensure_wireframe_pipelines();
        // Shader hot-reload: if the debug `reload-shaders` command set the flag, rebuild every
        // built-in PSO from disk-resident source before the frame's passes start using them. The
        // flag is cleared regardless of outcome so a failed rebuild (typo in a shader edit) doesn't
        // loop, and the previous pipelines stay live so the session keeps rendering; only a device
        // failure propagates. Wait for the GPU to drain first so swapping PSOs out from under
        // in-flight command lists is safe.
        if self.hot_reload.requested() {
            self.hot_reload.clear();
            self.wait_idle();
            match self.reload_shaders() {
                Ok(()) => tracing::info!("hot-reload: shader pipelines rebuilt"),
                Err(e) if e.is_device_failure() => return Err(e),
                Err(e) => tracing::error!("hot-reload: shader rebuild failed: {}", e),
            }
        }

        // Window resize: rebuild the swapchain back-buffers, the HDR / depth
        // scene targets, the bloom mip chain, and the TAA / SSAO / SSR
        // resource sets at the new size. A no-op when the size hasn't
        // changed; skips the rebuild (and the frame) when the window is
        // minimized so we never present 0×0. Failures other than a device
        // failure are logged; the client keeps trying on subsequent frames.
        match self.maybe_handle_resize() {
            Ok(()) => Ok(()),
            Err(e) if e.is_device_failure() => Err(e),
            Err(e) => {
                tracing::error!("D3D12 resize failed: {e}");
                Ok(())
            }
        }
    }

    // Block on the next frame slot ahead of the draw: DXGI's frame-latency
    // waitable, then the slot's fence. The draw's own wait then returns at once.
    pub(crate) fn prepare_frame(&mut self) {
        if self.window_minimized() {
            return;
        }
        let mut gpu_wait = GpuWait::none();
        if self
            .block_on_frame_slot(self.current_frame, &mut gpu_wait)
            .is_ok()
        {
            self.frame_latency_reserved = true;
        }
    }

    // Pump the window's messages outside the draw. The closed flag is sticky,
    // so the frame's own poll still reports a close.
    pub(crate) fn pump_events(&mut self) {
        self.window_closed();
    }

    // Wait until DXGI may queue another present, unless `prepare_frame` already
    // did, then until `frame`'s previous submission has finished.
    fn block_on_frame_slot(&mut self, frame: usize, gpu_wait: &mut GpuWait) -> RenderResult<()> {
        // Start the frame only once DXGI may queue its present. Bounded, so a
        // display that stops consuming presents cannot wedge the loop.
        if !std::mem::take(&mut self.frame_latency_reserved) {
            gpu_wait.measure(|| {
                // SAFETY: the handle came from this context's swapchain and lives as long as the
                // context, and the wait borrows nothing else.
                unsafe {
                    WaitForSingleObject(self.swapchain.frame_latency, FRAME_LATENCY_TIMEOUT_MS)
                }
            });
        }
        // SAFETY: the fence and the event were created from this device and are live for the call.
        let completed = unsafe { self.frame_sync.fence.GetCompletedValue() };
        if self.frame_sync.fence_values[frame] > completed {
            // SAFETY: the fence and the event were created from this device and are live for the
            // call.
            unsafe {
                self.frame_sync.fence.SetEventOnCompletion(
                    self.frame_sync.fence_values[frame],
                    self.frame_sync.fence_event,
                )
            }
            .map_err(|e| map_hresult(e.code(), "SetEventOnCompletion"))?;
            gpu_wait.measure(|| {
                // SAFETY: the event handle was created in `DxContext::new` and lives as long as
                // the context, and the wait borrows nothing else.
                unsafe { WaitForSingleObject(self.frame_sync.fence_event, u32::MAX) }
            });
        }
        Ok(())
    }

    // Blocks on this frame slot's previous submission, then runs the ticks it gates.
    pub(in crate::directx) fn wait_frame_slot(&mut self, frame: usize) -> RenderResult<GpuWait> {
        // Wait for this frame slot's previous work to finish before reusing it.
        // Measured, with the `Present` in `submit_and_present`, into the frame's `gpu_wait_us`:
        // both block the CPU on the GPU inside `draw_frame`, which the engine
        // times its graphics system around.
        let mut gpu_wait = GpuWait::none();
        self.block_on_frame_slot(frame, &mut gpu_wait)?;

        // Streamed texture swaps: re-point this frame's flat-pool SRV copy at
        // the swapped-in resources (legal now -- the fence wait retired
        // every list that binds this copy), and release the old resources /
        // upload transients whose covering fence has signaled.
        self.apply_streamed_texture_rewrites(frame);

        // Tick the placement pool: the same fence wait retired every list that
        // could still reference a range freed `frames + 1` ticks ago, so those
        // bytes become placeable again here.
        self.hw.alloc.begin_frame();
        // Same tick for the post block's freed descriptor slots and the feature
        // state a settings change replaced.
        self.post.begin_frame();
        self.retired_effects.tick();
        // Same tick for the staged geometry writes' ring and copy lists.
        self.geometry_uploads.get_mut().begin_frame();

        // Periodic footprint readout, for measuring the pool under streaming
        // churn at scale. Inert unless debug logging is enabled.
        if self.stream.frame.is_multiple_of(1024) && tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!("device allocator: {}", self.hw.alloc.stats());
        }
        Ok(gpu_wait)
    }

    // Probe bake and auto-exposure steps that need the slot's GPU work retired.
    pub(in crate::directx) fn service_background_work(&mut self, elapsed: f32, frame: usize) {
        // Advance the staggered reflection-probe bake. Called after the frame-slot
        // fence wait (so any in-flight capture resources are safe to recycle) and
        // before the frame's passes record. Non-fatal: a failure is logged and the
        // frame proceeds with whatever probes have baked.
        self.bake_pending_probes(elapsed);

        // Auto-exposure EMA step. The `wait_frame_slot` fence wait ensured the GPU work
        // that wrote this slot's readback buffer has completed, so the read
        // is race-free. Must happen *before* the bloom prefilter / composite
        // consume `self.post_process.exposure`. No-op when auto-exposure is
        // disabled.
        self.update_auto_exposure(elapsed, frame);
        self.pre_exposure.advance(self.post_process.exposure);
    }

    // Whole-frame and per-pass GPU times this slot's last submission resolved.
    pub(in crate::directx) fn read_gpu_timings(
        &self,
        frame: usize,
    ) -> (u32, [PassTiming; profile::MAX_PASS_TIMINGS]) {
        // Pull the most recently completed GPU times for this slot. The fence
        // wait in `wait_frame_slot` already ensured the GPU work
        // that wrote this slot's readback bytes has retired, so the
        // persistently-mapped pointer reflects fully committed pairs (one ring
        // cycle stale by construction). Zero before the slot has been visited
        // a second time (the readback buffer starts zero-initialized). Inactive
        // passes keep the frame-start timestamp in both slots (see the pre-init
        // loop in `record_frame`), so they read 0 us.
        if !self.timestamps.readback_ptr.is_null() && self.timestamps.frequency > 0 {
            // SAFETY: `readback_ptr` is the persistently-mapped base of a READBACK buffer
            // sized for one block of SLOTS_PER_FRAME u64s per frame in flight (see
            // build_timestamp_resources). The `wait_frame_slot` fence wait ensures this block's writes
            // have retired.
            let block_base = unsafe {
                self.timestamps
                    .readback_ptr
                    .add(frame * pass_timing::SLOTS_PER_FRAME)
            };
            let frequency = self.timestamps.frequency;
            let ticks_to_micros = |ticks: u64| -> u32 {
                u32::saturating_from(
                    (ticks.saturating_mul(1_000_000) / frequency).min(u64::from(u32::MAX)),
                )
            };
            pass_timing::decode_frame_block(|start_slot, end_slot| {
                // SAFETY: `decode_frame_block` only passes slots below `SLOTS_PER_FRAME`, so
                // `start_slot` stays inside this frame's block of the READBACK buffer.
                let start_ptr = unsafe { block_base.add(start_slot) };
                // SAFETY: `start_ptr` is inside the block, whose writes the `wait_frame_slot`
                // fence wait retired.
                let ts_start = unsafe { start_ptr.read() };
                // SAFETY: `end_slot` is also below `SLOTS_PER_FRAME`, so it stays in the block.
                let end_ptr = unsafe { block_base.add(end_slot) };
                // SAFETY: `end_ptr` is inside that same retired block.
                let ts_end = unsafe { end_ptr.read() };
                if ts_end > ts_start {
                    ticks_to_micros(ts_end - ts_start)
                } else {
                    0
                }
            })
        } else {
            (0, [("", 0); profile::MAX_PASS_TIMINGS])
        }
    }

    // When the slot's last frame's GPU work ended on the host clock: its
    // whole-frame end timestamp, placed through a fresh queue clock
    // calibration. `None` when the frame's input time is unknown, its timestamp
    // is not in yet, or the queue cannot calibrate.
    pub(in crate::directx) fn read_gpu_done(&self, frame: usize) -> Option<profile::GpuDone> {
        let input_sampled_us = self.timestamps.input_sampled_us[frame]?;
        if self.timestamps.readback_ptr.is_null() || self.timestamps.frequency == 0 {
            return None;
        }
        let (_, end_slot) = pass_timing::whole_frame_pair(frame);
        // SAFETY: `readback_ptr` is the persistently-mapped base of a READBACK buffer holding
        // every frame slot's block, and `end_slot` lies inside this frame's block.
        let end_ptr = unsafe { self.timestamps.readback_ptr.add(end_slot as usize) };
        // SAFETY: `end_ptr` is inside the mapping, and the `wait_frame_slot` fence wait retired
        // the write.
        let end_tick = unsafe { end_ptr.read() };
        if end_tick == 0 {
            return None;
        }
        let (mut gpu_now, mut cpu_now) = (0u64, 0u64);
        // SAFETY: a clock query on the live command queue; it writes only the two out-params.
        if unsafe {
            self.hw
                .alloc
                .queue()
                .GetClockCalibration(&raw mut gpu_now, &raw mut cpu_now)
        }
        .is_err()
        {
            return None;
        }
        Some(profile::GpuDone {
            input_sampled_us,
            done_us: pass_timing::gpu_end_on_host(
                end_tick,
                gpu_now,
                1e9 / self.timestamps.frequency as f64,
                concinnity_host::clock::monotonic_micros(),
            ),
        })
    }

    // Publishes this frame's stats before recording; draw calls fill in after.
    pub(in crate::directx) fn begin_frame_stats(
        &self,
        gpu_wait: GpuWait,
        (gpu_frame_us, pass_times_us): (u32, [PassTiming; profile::MAX_PASS_TIMINGS]),
        gpu_done: Option<profile::GpuDone>,
    ) {
        // Reset this frame's render stats. `record_frame` accumulates
        // `draw_calls` through `inc_draw_calls` (an interior-mutability path
        // because the encoders run through `&self`); the rest is filled here
        // from context state.
        let counts = concinnity_engine::render::backend::object_counts::object_counts(
            self.state.draw.objects.len(),
            self.instanced.clusters.iter().map(|c| c.instances.len()),
            self.state.skinned.draw_objects.iter().map(|o| o.visible),
        );

        // Reset the parallel-encoder draw-call accumulator so this frame's
        // encoders bump from zero. Drained back into `diagnostics.frame_stats.draw_calls`
        // after `record_frame` returns (the actual encoding fan-out happens
        // inside it).
        self.diagnostics
            .draw_calls_accum
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.diagnostics.frame_stats.set(profile::RenderStats {
            draw_calls: 0,
            objects: counts.objects,
            skinned_visible: counts.skinned_visible,
            gpu_frame_us,
            gpu_done,
            // The fence wait alone so far; `submit_and_present` adds the `Present`.
            gpu_wait_us: gpu_wait.micros(),
            vram_bytes: self.query_vram_bytes(),
            transient_pool_bytes: self.targets.transient_pool.allocated_bytes(),
            pass_times_us,
            // EMA-adapted exposure value, surfaced to the StatHud `EV ±X.XX`
            // chip. `None` when the world stayed on the authored static
            // exposure; the chip blanks itself in that case. Mirrors
            // `MtlContext::render_stats`.
            auto_exposure_ev: self
                .auto_exposure
                .adaptation
                .as_ref()
                .map(concinnity_engine::gfx::auto_exposure::ExposureAdaptation::current_ev),
            // Captured from the resolved `HdrOutputMode` at init. `None` on
            // the SDR path (chip blanks). Mirrors `MtlContext::render_stats`.
            max_edr: self.hw.max_edr(),
            ..profile::RenderStats::default()
        });
    }

    pub(in crate::directx) fn finish_frame_stats(&self) {
        // Drain the parallel-encoder draw-call accumulator into this
        // frame's `diagnostics.frame_stats.draw_calls`. The accumulator was reset
        // to 0 in `begin_frame_stats` and bumped by each `inc_draw_calls` call site
        // (potentially from worker threads) during `record_frame`.
        let mut s = self.diagnostics.frame_stats.get();
        s.draw_calls = self
            .diagnostics
            .draw_calls_accum
            .load(std::sync::atomic::Ordering::Relaxed);
        self.diagnostics.frame_stats.set(s);
    }

    pub(in crate::directx) fn close_end_list(&self, frame: usize) -> RenderResult<()> {
        let end_cmd = &self.commands.end_command_lists[frame];
        // 4. Timestamp the end of GPU work and resolve this frame's
        //    entire block (whole-frame pair + every per-pass pair the
        //    workers wrote) into the matching slice of the readback
        //    buffer. Resolves are cmd-list ops so they precede `Close`.
        if let (Some(heap), Some(readback)) = (
            self.timestamps.query_heap.as_ref(),
            self.timestamps.readback.as_ref(),
        ) {
            let (_, end_slot) = pass_timing::whole_frame_pair(frame);
            // SAFETY: the end list is recording, and `end_slot` is inside the timestamp heap this
            // context owns.
            unsafe { end_cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end_slot) };
            let block_base = pass_timing::frame_block_base(frame);
            let slot_count = u32::saturating_from(pass_timing::SLOTS_PER_FRAME);
            let readback_offset = pass_timing::frame_readback_byte_offset(frame);
            // SAFETY: the frame's block of `SLOTS_PER_FRAME` queries lands at its own offset in
            // the readback buffer, which is sized for every frame in flight.
            unsafe {
                end_cmd.ResolveQueryData(
                    heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    block_base,
                    slot_count,
                    &**readback,
                    readback_offset,
                );
            }
        }

        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { end_cmd.Close() }.map_err(|e| map_hresult(e.code(), "end cmd close"))?;
        Ok(())
    }

    pub(in crate::directx) fn submit_and_present(
        &mut self,
        start_cmd: &ID3D12GraphicsCommandList,
        pass_cmd_lists: &[ID3D12GraphicsCommandList],
        back_idx: usize,
        frame: usize,
        mut gpu_wait: GpuWait,
    ) -> RenderResult<()> {
        // 5. Submit everything in topological order: [start, per-pass...,
        //    end]. Single ExecuteCommandLists call → the GPU executes
        //    them in submission order; the queue is serial on a single
        //    DIRECT command queue, so this guarantees pass ordering.
        let mut submission = std::mem::take(&mut self.submission);
        submission.clear();
        let end_cmd = &self.commands.end_command_lists[frame];
        let start_handle: ID3D12CommandList = start_cmd
            .cast()
            .map_err(|e| map_hresult(e.code(), "start cmd cast"))?;
        submission.push(Some(start_handle));
        for cl in pass_cmd_lists {
            let h: ID3D12CommandList = cl
                .cast()
                .map_err(|e| map_hresult(e.code(), "per-pass cmd cast"))?;
            submission.push(Some(h));
        }
        let end_handle: ID3D12CommandList = end_cmd
            .cast()
            .map_err(|e| map_hresult(e.code(), "end cmd cast"))?;
        submission.push(Some(end_handle));
        // SAFETY: every command list in the submission is live and closed, and the slice outlives
        // the call.
        unsafe { self.hw.command_queue.ExecuteCommandLists(&submission) };
        submission.clear();
        self.submission = submission;

        // Present. Sync interval 1 locks to the display refresh (vsync); 0 runs
        // uncapped. The tearing present flag is required (and only valid) at
        // sync interval 0 on a swapchain created with ALLOW_TEARING, so gate it
        // on the current interval too -- `set_vsync` flips the interval at
        // runtime, and ALLOW_TEARING with interval >= 1 is an invalid Present.
        let present_flags =
            if self.swapchain.present_sync_interval == 0 && self.swapchain.allow_tearing {
                DXGI_PRESENT_ALLOW_TEARING
            } else {
                DXGI_PRESENT(0)
            };
        // At sync interval 1 this blocks on the display refresh once the
        // present queue is full, so it is the display-paced half of the frame's
        // GPU wait.
        let present_result = gpu_wait.measure(|| {
            // SAFETY: the swapchain is live, and `Present` takes no borrowed state beyond the
            // interval and flags.
            unsafe {
                self.swapchain
                    .handle
                    .Present(self.swapchain.present_sync_interval, present_flags)
            }
        });
        // Fold the present into the reading `begin_frame_stats` published, which the stats
        // snapshot had already captured with the fence wait alone.
        {
            let mut waited = self.diagnostics.frame_stats.get();
            waited.gpu_wait_us = gpu_wait.micros();
            self.diagnostics.frame_stats.set(waited);
        }
        if let Err(e) = present_result.ok() {
            self.flush_validation();
            // SAFETY: a property query on a live COM object; it only reads.
            let reason = unsafe { self.hw.device.GetDeviceRemovedReason() };
            return Err(error::classify_present_failure(
                e.code(),
                reason.err().map_or(windows::core::HRESULT(0), |r| r.code()),
            ));
        }
        // Record the buffer just shown so a headless `screenshot` captures the
        // on-screen image (the next `GetCurrentBackBufferIndex` already advanced
        // past it).
        self.swapchain.last_present_index = Some(back_idx);

        // Advance fence. The signaled value must be globally unique across
        // slots so each slot's wait-before-reuse only observes completion of
        // its own prior submission.
        let next_val = self.frame_sync.next_fence_value.get();
        self.frame_sync.next_fence_value.set(next_val + 1);
        self.frame_sync.fence_values[frame] = next_val;
        // SAFETY: the fence and the event were created from this device and are live for the call.
        unsafe {
            self.hw
                .command_queue
                .Signal(&self.frame_sync.fence, next_val)
        }
        .map_err(|e| map_hresult(e.code(), "Signal"))?;

        self.current_frame = (self.current_frame + 1) % self.hw.frames();
        Ok(())
    }

    // What this backend has built, for the frame's graph gates.
    fn resident_features(&self, render_size: [u32; 2]) -> ResidentFeatures {
        ResidentFeatures {
            hdr_width: render_size[0],
            hdr_height: render_size[1],
            hdr_sample_count: self.targets.hdr.msaa_samples,
            shadow: !self.shadow.dsvs.is_empty(),
            shadow_map_size: self.shadow.map_size,
            cull: self.cull.main_bindless_pso.is_some(),
            two_pass: self.two_pass_occlusion_built(),
            hiz: self.cull.hiz.is_some(),
            bloom_chain: self.bloom.is_some(),
            upscaler: self.upscale.backend.as_ref().map(|u| ResidentUpscaler {
                reads_reactive: u.reactive_reader().reads(),
            }),
            particles: self.particle.resources.is_some(),
            fog: self.fog.resources.is_some(),
            // `DecalState` is built at init unconditionally so a runtime
            // `add_decal` works from a world that declared none.
            decals: self.decal.state.is_some(),
            lines: self.lines.resources.is_some(),
            atmosphere: self.atmosphere.is_some(),
            rt_transparent: self
                .transparent
                .as_ref()
                .is_some_and(super::super::transparent::TransparentResources::rt_pipelines_ready),
            taa: self.taa.is_some(),
            model_history: self.gbuffer.is_some() && self.cull.model_history_pso.is_some(),
            reflection_composite: self.reflection_composite.is_some(),
        }
    }

    // What the world puts on screen this frame, for the frame's graph gates.
    fn frame_content(&self, lines: &[LineVertex], world_hidden: bool) -> FrameContent {
        FrameContent {
            world_hidden,
            cull_records: self.state.draw.record_count(),
            shadowed_spot_count: self.spot_shadow.count(),
            bloom_intensity: self.post_process.bloom_intensity,
            bvh_live: self.rt.accel.is_some(),
            emitters: !self.particle.system.is_empty(),
            late_emitters: self.particle.system.draws_after_transparent(),
            fog: self.fog.settings.is_some_and(|s| s.contributes()),
            decals: !self.decal.set.is_empty(),
            visible_surfaces: self
                .transparent
                .as_ref()
                .is_some_and(super::super::transparent::TransparentResources::any_visible),
            visible_see_through: self.see_through_mesh_visible(),
            planar_targets: self
                .planar_reflection
                .as_ref()
                .is_some_and(super::super::planar::PlanarReflectionSet::has_targets),
            water_planar_slot: self.transparent.as_ref().is_some_and(
                super::super::transparent::TransparentResources::water_planar_slot_live,
            ),
            lines: !lines.is_empty(),
            visible_volumes: self.raymarch_enabled(),
            atmosphere: self.atmosphere.as_ref().is_some_and(|a| a.frame.is_some()),
            water_sim: self.water_sim_live(),
            grass: self.grass.is_some(),
            ssgi_motion: self.ssgi.as_ref().is_some_and(|s| s.settings.contributes()),
        }
    }

    // Plan this frame in core over what this backend holds, then put what the
    // encoders read from `&self` in place: the spot slices and the jitter the
    // upscaler is dispatched with.
    pub(in crate::directx) fn plan_frame(&mut self, args: PlanArgs<'_>) -> RenderResult<FramePlan> {
        let PlanArgs {
            camera,
            elapsed,
            render_size,
            lines,
            world_hidden,
        } = args;
        let resident = self.resident_features(render_size);
        let content = self.frame_content(lines, world_hidden);
        // The upscaler prescribes its jitter sequence (FSR queries its FFX-tuned
        // offsets; DLSS / XeSS use the shared Halton-2/3) at the plan's phase.
        // FFX returns it in input-pixel coordinates, the Halton path's units.
        let phase = self.state.planner.jitter_phase();
        let upscaler_jitter = self
            .upscale
            .backend
            .as_ref()
            .map(|up| up.jitter_offset(phase));
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
            upscaler_jitter,
            lights: &self.uniforms.lights,
            probes: self.probe.book.records(),
            fog: self.fog.settings,
            pre_exposure: self.pre_exposure.value(),
            surface,
            effects: EffectInputs {
                ssao: self.ssao.resources.as_ref().map(|s| s.settings),
                ssr: self
                    .ssr
                    .as_ref()
                    .and_then(|s| s.resolve.as_ref())
                    .map(|r| r.settings),
                ssgi: self
                    .ssgi
                    .as_ref()
                    .map(|s| s.settings)
                    .zip(self.ssgi_frame()),
                rt: self.rt_reflections.as_ref().map(|r| r.settings),
                history_scale: self.pre_exposure.history_scale(),
            },
        };
        let plan = self.state.planner.plan(&self.state.view, &inputs)?;
        self.spot_shadow.render_mask = self.state.planner.shadow.spot_mask;
        if self.upscale.backend.is_some() {
            self.upscale.jitter.set(plan.camera.jitter_px);
        }
        Ok(plan)
    }
}
