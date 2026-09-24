//! The stages `DxContext::draw_frame` runs in order around `record_frame`, and
//! the per-frame inputs `record_frame` derives before its graph dispatch.

use concinnity_core::components;
use concinnity_core::gfx::jitter;
use concinnity_core::gfx::projection::perspective_rh;
use concinnity_core::gfx::render_types::{self, LineVertex};
use concinnity_core::profile;
use concinnity_core::profile::PassTiming;
use concinnity_core::render::csm;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::pass_timing;
use concinnity_core::render::render_graph::{self, FrameGraphInputs};
use concinnity_core::transform::mat4_mul;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::WaitForSingleObject;
use windows::core::Interface;

use crate::directx::context::{DxContext, FRAMES};
use crate::directx::error::{self, map_hresult};
use crate::gpu_wait::GpuWait;

// The camera projection and the view-projections derived from it.
pub(super) struct FrameProjection {
    pub(super) proj: [[f32; 4]; 4],
    // Un-jittered view-projection, for motion vectors and next frame's cull.
    pub(super) cur_vp: [[f32; 4]; 4],
    // Jittered view-projection the scene passes rasterize with.
    pub(super) vp_mat: [[f32; 4]; 4],
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
        if self.shader_reload_requested() {
            self.clear_shader_reload_flag();
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

    // Blocks on this frame slot's previous submission, then runs the ticks it gates.
    pub(in crate::directx) fn wait_frame_slot(&mut self, frame: usize) -> RenderResult<GpuWait> {
        // Wait for this frame slot's previous work to finish before reusing it.
        // Measured, with the `Present` in `submit_and_present`, into the frame's `gpu_wait_us`:
        // both block the CPU on the GPU inside `draw_frame`, which the engine
        // times its graphics system around.
        let mut gpu_wait = crate::gpu_wait::GpuWait::none();
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
            .map_err(|e| error::map_hresult(e.code(), "SetEventOnCompletion"))?;
            gpu_wait.measure(|| {
                // SAFETY: the event handle was created in `DxContext::new` and lives as long as
                // the context, and the wait borrows nothing else.
                unsafe { WaitForSingleObject(self.frame_sync.fence_event, u32::MAX) }
            });
        }

        // Streamed texture swaps: re-point this frame's flat-pool SRV copy at
        // the swapped-in resources (legal now -- the fence wait retired
        // every list that binds this copy), and release the old resources /
        // upload transients whose covering fence has signaled.
        self.apply_streamed_texture_rewrites(frame);

        // Tick the placement pool: the same fence wait retired every list that
        // could still reference a range freed `FRAMES + 1` ticks ago, so those
        // bytes become placeable again here.
        self.hw.alloc.begin_frame();

        // Periodic footprint readout, for measuring the pool under streaming
        // churn at scale. Inert unless debug logging is enabled.
        if self.stream.frame.is_multiple_of(1024) && tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!("device allocator: {}", self.hw.alloc.stats());
        }
        Ok(gpu_wait)
    }

    // Probe bake and auto-exposure steps that need the slot's GPU work retired.
    pub(in crate::directx) fn service_background_work(
        &mut self,
        elapsed: f32,
        near: f32,
        far: f32,
        frame: usize,
    ) {
        // Advance the staggered reflection-probe bake. Called after the frame-slot
        // fence wait (so any in-flight capture resources are safe to recycle) and
        // before the frame's passes record. Non-fatal: a failure is logged and the
        // frame proceeds with whatever probes have baked.
        if let Err(e) = self.bake_pending_probes(near, far) {
            tracing::warn!("reflection probe bake step failed: {e}");
        }

        // Auto-exposure EMA step. The `wait_frame_slot` fence wait ensured the GPU work
        // that wrote this slot's readback buffer has completed, so the read
        // is race-free. Must happen *before* the bloom prefilter / composite
        // consume `self.post_process.exposure`. No-op when auto-exposure is
        // disabled.
        self.update_auto_exposure(elapsed, frame);
    }

    // Whole-frame and per-pass GPU times this slot's last submission resolved.
    pub(in crate::directx) fn read_gpu_timings(
        &self,
        frame: usize,
    ) -> (u32, [PassTiming; profile::MAX_PASS_TIMINGS]) {
        // Pull the most recently completed GPU times for this slot. The fence
        // wait in `wait_frame_slot` already ensured the GPU work
        // that wrote this slot's readback bytes has retired, so the
        // persistently-mapped pointer reflects fully committed pairs (`FRAMES`
        // frames stale by construction). Zero before the slot has been visited
        // a second time (the readback buffer starts zero-initialized). Inactive
        // passes keep the frame-start timestamp in both slots (see the pre-init
        // loop in `record_frame`), so they read 0 us.
        if !self.timestamps.readback_ptr.is_null() && self.timestamps.frequency > 0 {
            // SAFETY: `readback_ptr` is the persistently-mapped base of a READBACK buffer
            // sized for FRAMES blocks of SLOTS_PER_FRAME u64s each (see
            // build_timestamp_resources). The `wait_frame_slot` fence wait ensures this block's writes
            // have retired.
            let block_base = unsafe {
                self.timestamps
                    .readback_ptr
                    .add(frame * pass_timing::SLOTS_PER_FRAME)
            };
            let frequency = self.timestamps.frequency;
            let ticks_to_micros = |ticks: u64| -> u32 {
                (ticks.saturating_mul(1_000_000) / frequency).min(u32::MAX as u64) as u32
            };
            pass_timing::decode_frame_block(|start_slot, end_slot| {
                // SAFETY: `decode_frame_block` only passes slots below `SLOTS_PER_FRAME`, so
                // both reads stay inside this frame's block of the READBACK buffer, whose
                // writes the `wait_frame_slot` fence wait retired.
                let (ts_start, ts_end) = unsafe {
                    (
                        block_base.add(start_slot).read(),
                        block_base.add(end_slot).read(),
                    )
                };
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

    // Publishes this frame's stats before recording; draw calls fill in after.
    pub(in crate::directx) fn begin_frame_stats(
        &self,
        gpu_wait: &GpuWait,
        (gpu_frame_us, pass_times_us): (u32, [PassTiming; profile::MAX_PASS_TIMINGS]),
    ) {
        // Reset this frame's render stats. `record_frame` accumulates
        // `draw_calls` through `inc_draw_calls` (an interior-mutability path
        // because the encoders run through `&self`); the rest is filled here
        // from context state.
        let counts = crate::object_counts::object_counts(
            self.draw.objects.len(),
            self.instanced.clusters.iter().map(|c| c.instances.len()),
            self.skinned.slots.draw_objects.iter().map(|o| o.visible),
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
            // The fence wait alone so far; `submit_and_present` adds the `Present`.
            gpu_wait_us: gpu_wait.micros(),
            vram_bytes: self.query_vram_bytes(),
            transient_pool_bytes: self.targets.transient_pool.allocated_bytes(),
            pass_times_us,
            // EMA-adapted exposure value, surfaced to the StatHud `EV ±X.XX`
            // chip. `None` when the world stayed on the authored static
            // exposure; the chip blanks itself in that case. Mirrors
            // `MtlContext::render_stats`.
            auto_exposure_ev: self.auto_exposure.state.as_ref().map(|s| s.current_ev),
            // Captured from the resolved `HdrOutputMode` at init. `None` on
            // the SDR path (chip blanks). Mirrors `MtlContext::render_stats`.
            max_edr: self.hw.max_edr(),
            ..profile::RenderStats::default()
        });
    }

    pub(in crate::directx) fn update_shadow_schedule(
        &mut self,
        cam_pos: [f32; 3],
        fov_y_radians: f32,
        near: f32,
        far: f32,
    ) {
        // Cascaded-shadow update policy. Advance the round-robin schedule, then
        // refresh only this frame's cascades' light VPs (splits always refresh).
        // Skipped cascades keep the VP + depth their slice was last rendered
        // with, so the Main pass samples each cascade consistently. record_frame
        // uploads the merged `self.shadow.uniforms` to this frame's shadow UBO,
        // and encode_shadow_pass re-rasterizes only the masked slices. Mirrors
        // Metal; no-op (mask stays 0, uniforms stay empty) when shadows are off.
        if !self.shadow.dsvs.is_empty() {
            let aspect = self.targets.extent.render_width.max(1) as f32
                / self.targets.extent.render_height.max(1) as f32;
            let fresh = csm::compute_shadow_uniforms(csm::ShadowUniformInputs {
                view: self.view.matrix,
                cam_pos,
                fov_y_rad: fov_y_radians,
                aspect,
                near,
                shadow_distance: (self.shadow.distance as f32).min(far),
                light_dir_to_source: self.shadow.light_dir,
                shadow_map_size: self.shadow.map_size,
                active_cascades: self.shadow.cascades,
            });
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
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                end_cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end_slot);
                end_cmd.ResolveQueryData(
                    heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    pass_timing::frame_block_base(frame),
                    pass_timing::SLOTS_PER_FRAME as u32,
                    &**readback,
                    pass_timing::frame_readback_byte_offset(frame),
                );
            }
        }

        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { end_cmd.Close() }.map_err(|e| error::map_hresult(e.code(), "end cmd close"))?;
        Ok(())
    }

    pub(in crate::directx) fn submit_and_present(
        &mut self,
        start_cmd: ID3D12GraphicsCommandList,
        pass_cmd_lists: &[ID3D12GraphicsCommandList],
        back_idx: usize,
        frame: usize,
        mut gpu_wait: GpuWait,
    ) -> RenderResult<()> {
        let end_cmd = &self.commands.end_command_lists[frame];
        // 5. Submit everything in topological order: [start, per-pass...,
        //    end]. Single ExecuteCommandLists call → the GPU executes
        //    them in submission order; the queue is serial on a single
        //    DIRECT command queue, so this guarantees pass ordering.
        let mut submission: Vec<Option<ID3D12CommandList>> =
            Vec::with_capacity(2 + pass_cmd_lists.len());
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
                reason
                    .err()
                    .map(|r| r.code())
                    .unwrap_or(windows::core::HRESULT(0)),
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
        .map_err(|e| error::map_hresult(e.code(), "Signal"))?;

        self.current_frame = (self.current_frame + 1) % FRAMES;
        Ok(())
    }

    // Seeds the shared frame-graph builder from this frame's live feature state.
    pub(super) fn frame_graph_inputs(
        &self,
        width: u32,
        height: u32,
        bindless_cull_enabled: bool,
        clustered: bool,
        lines: &[LineVertex],
        world_hidden: bool,
    ) -> FrameGraphInputs {
        // Per-frame seed inputs for the shared backend-agnostic frame builder
        // ([gfx/render_graph/frame.rs](../../gfx/render_graph/frame.rs)).
        // Every backend (Metal / Vulkan / DirectX) now drives the same builder.
        // FSR3 owns its own temporal accumulation, so when the upscaler
        // is built the engine bypasses the TAA pass entirely. The G-buffer
        // pre-pass still runs; FSR consumes its motion vectors.
        let upscale_on = self.upscale.backend.is_some();
        let taa_on = self.taa.is_some() && !upscale_on;
        let seed_inputs = FrameGraphInputs {
            shadow_enabled: !self.shadow.dsvs.is_empty(),
            shadow_map_size: self.shadow.map_size,
            hdr_width: width,
            hdr_height: height,
            hdr_sample_count: self.targets.hdr.msaa_samples,
            bindless_cull_enabled,
            bloom_enabled: self.post_process.bloom_intensity > 0.0,
            velocity_enabled: taa_on || upscale_on,
            taa_enabled: taa_on,
            // Only the SSR *resolve* is gated here; `self.ssr` is also `Some`
            // for a SSGI-only world (which reuses the pre-pass G-buffer), so
            // key off the resolve half rather than the bundle's presence.
            ssr_enabled: self.ssr.as_ref().is_some_and(|s| s.resolve.is_some()),
            // The SSR depth + normal pre-pass feeds SSR resolve *and* SSGI, so
            // `SsrResources` (and thus this flag) is on whenever either is.
            ssr_prepass_enabled: self.ssr.is_some(),
            auto_exposure_enabled: self.auto_exposure.resources.is_some(),
            particles_enabled: self.particle.resources.is_some()
                && !self.particle.records.is_empty(),
            // Gated on the resources (built at init when the world declared a
            // VolumetricFog) and on live settings that can affect the frame, so
            // runtime `update_fog_settings(None)` -- or an authored zero density,
            // which integrates to a transparent black over the whole volume --
            // drops the FogFroxel + Fog passes from the graph entirely. Mirrors
            // Vulkan + Metal; without the settings half a settings-None frame would still
            // emit the (bailing) Fog pass, and the graph-driven froxel-volume
            // consumer barrier would transition the volume with no encoder to
            // reset it.
            fog_enabled: self.fog.resources.is_some()
                && self.fog.settings.is_some_and(|s| s.contributes()),
            // `DecalState` is built at init unconditionally so a runtime
            // `add_decal` works from a world that declared none, so the
            // resources half alone is always true. The live half drops the
            // pass (and its depth-read transition) from the graph until a
            // decal exists. Mirrors Vulkan + Metal.
            decals_enabled: self.decal.state.is_some() && !self.decal.set.is_empty(),
            ssao_enabled: self.ssao.resources.is_some(),
            // FSR3 upscaling (runs at native resolution as a TAA
            // replacement). `Some` only when the FFX DLL loaded;
            // otherwise the renderer silently falls back to TAA-or-none.
            // When on, the engine sets `taa_enabled = false` below
            // (FSR's temporal accumulation supersedes TAA) and
            // `velocity_enabled = true` (FSR needs motion vectors).
            upscale_enabled: self.upscale.backend.is_some(),
            // Generic translucent pass: on when the world declared visible
            // `GlassPanel` or `WaterSurface`. The shared builder then seeds the
            // Transparent node and the executor draws every record back-to-front
            // over the post-SSR scene.
            transparent_enabled: self.transparent_enabled(),
            // Raymarched SDF volumes. Gated on the resources existing and a
            // currently visible volume.
            raymarch_enabled: self.raymarch_enabled(),
            // Two-pass Hi-Z occlusion: inserts HizBuild / Cull2 / Main2 after
            // Main when the world requested `occlusion_two_pass` and the bindless
            // GPU-cull path + phase-2 pipeline are live. `two_pass_occlusion_active`
            // is the single gate the executor's phase-2 arms + the Main resolve
            // skip share, so the graph shape matches what the executor dispatches.
            two_pass_occlusion_enabled: self.two_pass_occlusion_active(),
            // The terminal Hi-Z build. Present whenever the GPU-cull path built a
            // pyramid: the frame ends by reducing its final depth into it for the
            // next frame's phase-1 occlusion test.
            hiz_build_enabled: self.cull.hiz.is_some(),
            // Screen-space global illumination: inserts the `Ssgi` RMW node
            // after `Raymarch` and before `Decals`. On when the world selected
            // `indirect_lighting: ssgi` (which also forces the SSR pre-pass on
            // above so the gather has a G-buffer).
            ssgi_enabled: self.ssgi.as_ref().is_some_and(|s| s.settings.contributes()),
            // Hardware ray-traced reflections (DXR inline `RayQuery`). On when
            // the world authored `ray_traced_reflections`, the GPU supports the
            // DXR tier, and the DXC compile + acceleration-structure build
            // succeeded (`rt_reflections` + `rt.accel` both live). The shared
            // builder then seeds `RtReflections` in the SsrResolve slot and omits
            // `SsrResolve`; otherwise it falls back to SSR.
            rt_reflections_enabled: self.rt_reflections_active(),
            // One jittered traversal writes normal+depth, roughness, and motion
            // for every screen-space consumer, replacing the separate SSR /
            // SSAO / velocity geometry pre-passes. On whenever the G-buffer
            // resources exist (any of SSR / SSGI / SSAO / TAA / FSR enabled).
            gbuffer_prepass_enabled: self.gbuffer.is_some(),
            // An opaque menu backdrop hides the scene: the shared builder masks
            // every world pass off, collapsing to Main (a bare clear, fed the
            // empty scene below) -> Composite (presents the overlay).
            world_hidden,
            // Clustered light binning, sharing the gate that sets `use_clusters`
            // below: the pipeline is built only for a world with local lights,
            // and the live count has to still be non-zero. Otherwise the forward
            // pass brute-forces an empty light list.
            clustering_enabled: clustered,
            // Zero drops the SpotShadow node and its imported array from the
            // graph entirely, which is the common case (no shadow-casting spot).
            shadowed_spot_count: self.spot_shadow.count(),
            spot_shadow_slice_size: self.spot_shadow.slice_size,
            // Lines run only on the frames a system published them (the
            // `cn editor` axes), and only once their resources are live: the
            // build is lazy, so a shipped runtime never compiles them.
            lines_enabled: !lines.is_empty() && self.lines.resources.is_some(),
            // Set by the view-mode mask below (occlusion view only).
            composite_reads_ao: false,
        };
        // The viewport's view mode + show flags mask the seeded inputs (the
        // per-frame counterpart of the init-time trims); Lit with every flag
        // set is the identity, so a shipped runtime is unaffected.
        render_graph::apply_view(&seed_inputs, self.view.mode, self.view.show)
    }

    // Camera projection, jittered for TAA or the upscaler's phase sequence.
    pub(super) fn frame_projection(
        &self,
        fov_y_radians: f32,
        aspect: f32,
        near: f32,
        far: f32,
        width: u32,
        height: u32,
    ) -> FrameProjection {
        // Compute the camera VPs the main + velocity passes consume.
        let proj = perspective_rh(fov_y_radians, aspect, near, far);
        // Un-jittered camera VP, fed to the velocity pre-pass so the stored
        // motion vector is free of the sub-pixel projection jitter.
        let cur_vp = mat4_mul(proj, self.view.matrix);
        // When TAA is on, offset the projection by a sub-pixel Halton jitter so
        // the accumulation has fresh sample positions each frame. The jitter is
        // applied to the z-coefficients of clip x/y, so subtracting it shifts
        // post-divide NDC by exactly the jitter amount (clip.w == -view_z) and
        // leaves depth untouched. Mirrors the jitter in vulkan/draw.rs.
        //
        // When FSR3 upscale is on instead, the sub-pixel offset comes
        // from FFX's prescribed phase sequence (tuned to FSR's temporal
        // kernel), not Halton-2/3; Halton phases would mis-align with
        // FSR's accumulation and produce blur or ghosting. The offset
        // is queried from the upscaler once per frame and stashed in
        // `upscale_jitter` so the Upscale arm of the executor sees the
        // same value the projection was jittered with.
        let render_proj = match (&self.upscale.backend, &self.taa) {
            (Some(up), _) => {
                // FFX returns jitter in input-pixel coordinates
                // (each axis roughly [-0.5, 0.5]). The projection
                // offset is `(2 * jitter / extent)` in NDC, same
                // conversion as the Halton path below.
                let frame_idx = self.taa.as_ref().map(|t| t.frame.get()).unwrap_or(0);
                let [jx_px, jy_px] = up.jitter_offset(frame_idx);
                self.upscale.jitter.set([jx_px, jy_px]);
                let jx = jx_px * 2.0 / width.max(1) as f32;
                let jy = jy_px * 2.0 / height.max(1) as f32;
                let mut p = proj;
                p[2][0] -= jx;
                p[2][1] -= jy;
                p
            }
            (None, Some(taa)) => {
                let idx = taa.frame.get() % 8 + 1;
                let jx = (jitter::radical_inverse(idx, 2) - 0.5) * 2.0 / width.max(1) as f32;
                let jy = (jitter::radical_inverse(idx, 3) - 0.5) * 2.0 / height.max(1) as f32;
                let mut p = proj;
                p[2][0] -= jx;
                p[2][1] -= jy;
                p
            }
            (None, None) => proj,
        };
        let vp_mat = mat4_mul(render_proj, self.view.matrix);
        FrameProjection {
            proj,
            cur_vp,
            vp_mat,
        }
    }

    // History the next frame reads: Hi-Z validity, cull VP, TAA jitter, G-buffer VP.
    pub(super) fn advance_temporal_state(&self, cur_vp: [[f32; 4]; 4]) {
        // The Hi-Z reduction that feeds next frame's cull is the graph's
        // terminal `HizFinal` pass, so it has already been recorded; `hiz_valid`
        // only tracks whether a pyramid at the current resolution now exists.
        if self.cull.hiz.is_some() {
            self.cull.hiz_valid.set(true);
        }
        // Capture the un-jittered view-projection for the next frame's cull
        // dispatch. Stored regardless of whether Hi-Z is on so the matrix
        // is always current when it later gets switched on by a hot-reload
        // or a re-init.
        self.cull.prev_view_proj.set(cur_vp);

        // The HDR targets are graph resources: `emit_graph_restores` already
        // returned each to its resting state on the "end" cmd list, which is
        // where the MSAA-off spine's PIXEL_SHADER_RESOURCE -> RENDER_TARGET
        // reset now comes from.

        // The shadow map rests sampled between frames; next frame's Shadow
        // producer barrier (graph-driven) performs the PIXEL_SHADER_RESOURCE ->
        // DEPTH_WRITE reset, so no inline end-of-frame restore is needed.

        // Advance the TAA jitter sequence (which also validates history for the
        // next frame). TAA-specific, so gated on `self.taa`.
        if let Some(taa) = &self.taa {
            taa.frame.set(taa.frame.get().wrapping_add(1));
        }
        // Snapshot this frame's un-jittered VP so next frame's G-buffer pre-pass
        // can derive motion vectors. The per-draw half of the same history was
        // snapshotted on the GPU by the pre-pass's own dispatch. Owned by the
        // G-buffer (decoupled from TAA, so FSR-without-engine-TAA also gets
        // correct motion).
        if let Some(gb) = &self.gbuffer {
            *gb.prev_view_proj.borrow_mut() = cur_vp;
        }
    }
}
