// Frame recording for the Vulkan backend, and the passes it records.
// Encodes the shadow pass, main scene pass, and post stack into a single
// command buffer. Called from VkContext::draw_frame each tick. Every
// pass dispatches through the render-graph executor; see
// `vulkan/graph_exec.rs` and this directory's `composite.rs`.

mod composite;
mod main;
mod shadow;
pub(in crate::vulkan) mod spot_shadow;
mod stages;

use ash::vk;
use concinnity_core::gfx::lod;
use concinnity_core::gfx::render_types::{LightUniforms, ShadowUniforms};
use concinnity_core::math::{SaturatingFrom, narrow_f32};
use concinnity_core::profile::FanOutTiming;
use concinnity_core::render::backend::FrameParams;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::frame_plan::{
    CameraInputs, EffectInputs, FrameDevice, FramePlan, PlanInputs, SurfaceInputs, camera_inputs,
};
use concinnity_core::render::model_history::HistoryMode;
use concinnity_core::render::pass_timing;
use concinnity_core::render::render_graph::{FrameContent, ResidentFeatures, ResidentUpscaler};
use concinnity_core::render::scene_state::SceneState;
// `ViewUniforms` (the std140 main-pass `ViewBlock` UBO) is a GPU-free layout
// struct that lives in `core::render`; re-export it so
// `crate::vulkan::draw::ViewUniforms` is unchanged for the passes that fill it.
pub(in crate::vulkan) use concinnity_core::render::uniforms::ViewUniforms;

use super::context::VkContext;
use super::graph_exec::GraphFrameParams;

impl VkContext {
    // Where this frame's draws growing grass route: the grass twins, and where
    // the blades start handing the field to the canopy.
    fn grass_route(&self) -> concinnity_core::render::record_pack::GrassRoute {
        concinnity_core::render::record_pack::GrassRoute {
            twins: self.cull.world_pipelines.grass_twins(),
            far_start: self.grass_lod.map_or(0.0, |l| l.fade_start()),
        }
    }

    // Pack this frame's bindless cull records into frame `frame_idx`'s
    // `GpuObjectData` and `GpuDrawArgs` storage buffers, indexed by object id,
    // and bring its material parameter table up to date. Returns the static
    // draws' packing time for the frame report; a no-op when bindless is off.
    fn build_record_buffers(
        &mut self,
        frame_idx: usize,
        cam_pos: [f32; 3],
        history: HistoryMode,
    ) -> FanOutTiming {
        if let Some(params) = self.cull.material_params.as_mut() {
            params.upload(frame_idx);
        }
        let (Some(objects), Some(args)) = (
            self.cull.object_buffers.get(frame_idx),
            self.cull.draw_args_buffers.get(frame_idx),
        ) else {
            return FanOutTiming::default();
        };
        let packing = self.build_records_into(objects, args, cam_pos, history);
        self.patch_instance_lod_into(args, cam_pos);
        packing
    }

    // Overwrite the instance tail's `GpuDrawArgs` with each instance's active
    // LOD slice, for the clusters that declare alternates. The init-time fill
    // wrote every cluster's base slice into every frame's buffer, so this is a
    // no-op for a world without alternates and touches only the instances of
    // the clusters that have them. Not part of `build_records_into`: the probe
    // bake shares that body and deliberately leaves its instance tail zeroed,
    // which disables instances in a bake.
    fn patch_instance_lod_into(&self, buf: &super::allocator::PooledBuffer, cam_pos: [f32; 3]) {
        use concinnity_core::gfx::render_types::{GpuDrawArgs, draw_args_flags};
        if !self.instanced.any_lod {
            return;
        }
        let stride = size_of::<GpuDrawArgs>();
        let base = self.state.draw.n_objects;
        lod::for_each_instance_lod(
            &self.instanced.clusters,
            cam_pos,
            |record, index_offset, index_count| {
                let rec = GpuDrawArgs {
                    index_count: u32::saturating_from(index_count),
                    index_offset: u32::saturating_from(index_offset),
                    base_vertex: 0,
                    flags: draw_args_flags(true, true, true),
                };
                buf.write_val((base + record) * stride, &rec);
            },
        );
    }

    // Write the bindless `GpuObjectData` and GPU-cull `GpuDrawArgs` records --
    // static, streamed-chunk and spawned-clone, and skinned-tail, each draw's
    // active-LOD slice picked by distance from `cam_pos` -- into `objects` and
    // `args`. The pool indices address the shared handle-indexed texture pool:
    // albedo = `texture_slot`, normal = the normal map's own handle (or the
    // flat-normal fallback slot for a normal-less draw). Shared with the
    // reflection-probe capture, which builds into its own bake-owned buffers
    // against the probe eye; the instance tail is left untouched, so a bake
    // buffer must be zeroed first (a zero record is a disabled draw the cull
    // kernel skips, which is how the probe omits instanced geometry). Returns
    // the static draws' packing time.
    pub(in crate::vulkan) fn build_records_into(
        &self,
        objects: &super::allocator::PooledBuffer,
        args: &super::allocator::PooledBuffer,
        cam_pos: [f32; 3],
        history: HistoryMode,
    ) -> FanOutTiming {
        use concinnity_core::gfx::render_types::{
            GpuDrawArgs, GpuObjectData, albedo_pool_index, draw_args_bucket_bits, draw_args_flags,
            normal_pool_index, pack_skinned_record,
        };
        use concinnity_core::render::record_pack::{
            PackView, StaticPack, draw_bucket, pool_object_record,
        };
        let texture_count = u32::saturating_from(self.scene.textures.len());
        let record_stride = size_of::<GpuObjectData>();
        let args_stride = size_of::<GpuDrawArgs>();
        let mut model_history = self.state.model_history.borrow_mut();
        model_history.begin(history, self.state.draw.record_count());
        // Hand the dispatch the rebuild's prime request here, on the one thread
        // that owns the tracker; the encode runs on a worker.
        if model_history.take_prime()
            && let Some(mh) = self.cull.model_history.as_ref()
        {
            mh.prime.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        // A see-through glass mesh (Layer 2) is disabled in the opaque pass when
        // the RT path is live: it draws in the transparent pass instead.
        let mesh_glass_active = self.mesh_glass_active();
        let view = PackView {
            cam_pos,
            mesh_glass_active,
            horizon: self
                .state
                .horizon_occluder
                .and_then(|o| o.seen_from(cam_pos)),
            grass_route: self.grass_route(),
            tile_morph: self.state.tile_morph,
        };

        // The static draws, one record each at their own index, packed in
        // place across the job pool once that pays.
        let n_static = self.state.draw.n_objects.min(self.state.draw.objects.len());
        let static_draws = &self.state.draw.objects[..n_static];
        // SAFETY: `objects` and `args` are this frame's copies (its fence has
        // signaled) or bake buffers not yet submitted, so the GPU reads neither,
        // and these are the only views of either range until the pack returns.
        let packed = unsafe {
            (
                objects.records_mut::<GpuObjectData>(0, static_draws.len()),
                args.records_mut::<GpuDrawArgs>(0, static_draws.len()),
            )
        };
        let packing = if let (Some(records), Some(draw_args)) = packed {
            self.cull.packer.pack(
                StaticPack {
                    objects: static_draws,
                    records,
                    args: draw_args,
                    history: model_history.window(),
                },
                &view,
                &|obj| pool_object_record(obj, texture_count),
            )
        } else {
            tracing::error!(
                "vulkan: cull record buffers cannot hold {} static draws",
                static_draws.len()
            );
            FanOutTiming::default()
        };

        // Runtime objects -- streamed chunks and spawned clones -- one record each
        // in the reserved region at `[runtime_record_base() + k]`, packed like a
        // static object. Their geometry already lives in the shared VB/IB with
        // their own `base_vertex` and index slice, so they ride the static +
        // instance prefix indirect draw. Both kinds are non-cullable (NaN AABB,
        // because the init-time BVH cannot refit to admit them), so a resident
        // one draws unconditionally; a freed slot's `resident` clear disables
        // it, and the cull kernel skips a disabled record's stale object.
        let runtime_base = self.state.draw.runtime_record_base();
        let grass_route = self.grass_route();
        let n_resident_runtime = self.state.draw.for_each_runtime_record(|k, i, obj| {
            let mut record = pool_object_record(obj, texture_count);
            record.tile_morph = view.tile_morph_row(obj);
            objects.write_val((runtime_base + k) * record_stride, &record);
            // `camera_distance` falls back to the model translation for a
            // non-cullable object, so the LOD pick works off a NaN AABB. Chunks
            // carry no alternates and land on LOD0 either way; a clone inherits
            // its template's.
            let d = lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            // A clone copies its template's material, so a see-through one
            // leaves the opaque pass the way a static one does.
            let opaque_visible =
                obj.visible && !(mesh_glass_active && obj.material.see_through != 0);
            let rec = GpuDrawArgs {
                index_count: u32::saturating_from(index_count),
                index_offset: u32::saturating_from(index_offset),
                base_vertex: obj.base_vertex.cast_unsigned(),
                flags: draw_args_flags(opaque_visible, obj.resident, obj.cullable())
                    | draw_args_bucket_bits(draw_bucket(obj, cam_pos, grass_route))
                    | model_history.draw_flags(runtime_base + k, i),
            };
            args.write_val((runtime_base + k) * args_stride, &rec);
        });
        // Disable the unused reserve tail so vacated / never-used slots draw
        // nothing (the cull kernel skips `objects[i]` for an ENABLED-clear record).
        let disabled = GpuDrawArgs {
            index_count: 0,
            index_offset: 0,
            base_vertex: 0,
            flags: 0,
        };
        for k in n_resident_runtime..self.state.draw.n_runtime {
            args.write_val((runtime_base + k) * args_stride, &disabled);
        }

        // Skinned objects: one record each in the reserved tail at
        // `[skinned_record_base(), record_count())`. `model = obj.model` (applied
        // after the per-frame skin deform), flat-pool texture indices like a
        // static object, and a padded bind-pose AABB so the cull kernel can
        // frustum/Hi-Z test them, so they are cullable + resident. The main
        // pass's 2nd indirect draw binds the per-frame deformed VB + the skinned
        // IB, so `base_vertex = 0` (the deformed buffer mirrors global skinned
        // indexing) and the active-LOD slice is the element offset into the
        // skinned IB. `take(n_skinned)` no-ops when the fold is inactive.
        let skinned_base = self.state.draw.skinned_record_base();
        for (k, obj) in self
            .state
            .skinned
            .draw_objects
            .iter()
            .take(self.state.draw.n_skinned)
            .enumerate()
        {
            let albedo = albedo_pool_index(obj.texture_slot, texture_count);
            let normal = normal_pool_index(obj.normal_map_slot, texture_count);
            objects.write_val(
                (skinned_base + k) * record_stride,
                &pack_skinned_record(obj, albedo, normal),
            );
            let d = lod::skinned_camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let rec = GpuDrawArgs {
                index_count: u32::saturating_from(index_count),
                index_offset: u32::saturating_from(index_offset),
                base_vertex: 0,
                flags: draw_args_flags(obj.visible, true, true)
                    | model_history.skinned_flags(skinned_base + k, k),
            };
            args.write_val((skinned_base + k) * args_stride, &rec);
        }
        packing
    }

    // Record the frame's leading "start" command buffer: the timestamp block
    // reset and whole-frame start tick, then the TLAS refresh for moved props.
    fn record_frame_start(
        &mut self,
        frame_idx: usize,
        world_hidden: bool,
    ) -> RenderResult<vk::CommandBuffer> {
        let device = self.hw.device.clone();
        let device = &device;
        // Profiler-overlay timestamp pair. The pool slot for this frame is
        // reset (the matching `get_query_pool_results` already ran in
        // `read_gpu_timings`, after the fence wait that gated the previous trip's
        // writes), then the start tick is recorded as the first cmd-buffer
        // op. The matching end tick is written just before
        // `end_command_buffer` returns control. Mirrors the DirectX
        // EndQuery(TIMESTAMP) + ResolveQueryData pattern.
        // Outer "start" command buffer: the leading timestamp (the frame's
        // first GPU op), recorded into its own buffer so it can be submitted
        // before the per-pass buffers. The query-pool reset must precede every
        // pass, hence it lives here at the head of the batch.
        let start_cmd = self.commands.start_command_buffers[frame_idx];
        // SAFETY: `cmd` belongs to this frame slot, whose fence was already waited on, so it is not
        // in flight; reset then begin puts it in the recording state, which is what the subsequent
        // recording requires.
        unsafe {
            device
                .reset_command_buffer(start_cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| crate::vulkan::error::map_vk_result(e, "reset start cmd buf"))?;
            device
                .begin_command_buffer(
                    start_cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| crate::vulkan::error::map_vk_result(e, "begin start cmd buf"))?;
        }
        if let Some(pool) = self.hw.timestamp_query_pool {
            // Reset this frame's whole timestamp block (whole-frame pair + every
            // per-pass pair) here, before any per-pass buffer writes into it.
            // `start_cmd` is submitted first, so the reset precedes every write in
            // queue order. The whole-frame start goes in the block's first slot;
            // each pass writes its own pair (see graph_exec); the whole-frame end
            // is the block's second slot, written in the end buffer below.
            let block_base = pass_timing::frame_block_base(frame_idx);
            let (wf_start, _) = pass_timing::whole_frame_pair(frame_idx);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_reset_query_pool(
                    start_cmd,
                    pool,
                    block_base,
                    u32::saturating_from(pass_timing::SLOTS_PER_FRAME),
                );
                device.cmd_write_timestamp(
                    start_cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    pool,
                    wf_start,
                );
            }
        }
        // Hardware ray-traced reflections: rebuild the TLAS for moved props (when
        // the dynamic mode + dirty gate call for it) onto the start buffer, which
        // is submitted before every per-pass trace, then re-point this frame's RT
        // descriptor set at the live TLAS + geometry table. A no-op when RT is
        // off. Recorded here (not on a per-pass worker) because it needs
        // `&mut self`; the start buffer carries it within the frame's submit batch.
        // Skipped while the world is hidden, which traces nothing; a topology
        // change stays owed to the next frame that draws it.
        if !world_hidden {
            self.rt_dynamic_update(start_cmd, frame_idx);
        }
        // SAFETY: `cmd` is in the recording state, which is what `end_command_buffer` requires.
        unsafe {
            device
                .end_command_buffer(start_cmd)
                .map_err(|e| crate::vulkan::error::map_vk_result(e, "end start cmd buf"))?;
        }
        Ok(start_cmd)
    }

    // What this backend has built, for the frame's graph gates.
    fn resident_features(&self, render_size: [u32; 2]) -> ResidentFeatures {
        ResidentFeatures {
            hdr_width: render_size[0],
            hdr_height: render_size[1],
            hdr_sample_count: self.targets.msaa_samples.as_raw(),
            shadow: self.shadow.enabled(),
            shadow_map_size: self.shadow.map_size,
            cull: self.cull.cull_kernels.is_some(),
            two_pass: self.two_pass_occlusion_built(),
            hiz: self.cull.hiz.is_some(),
            bloom_chain: self.bloom.is_some(),
            upscaler: self.upscale.as_ref().map(|u| ResidentUpscaler {
                reads_reactive: u.reactive_reader().reads(),
            }),
            particles: self.particle.resources.is_some(),
            fog: self.fog.resources.is_some(),
            decals: self.decal.resources.is_some(),
            lines: self.lines.resources.is_some(),
            atmosphere: self.atmosphere.is_some(),
            rt_transparent: self
                .transparent
                .as_ref()
                .is_some_and(super::transparent::TransparentResources::rt_pipelines_ready),
            taa: self.taa.is_some(),
            model_history: self.gbuffer_targets().is_some() && self.cull.model_history.is_some(),
            reflection_composite: self.reflection_composite.is_some(),
        }
    }

    // What the world puts on screen this frame, for the frame's graph gates.
    fn frame_content(&self, lines_published: bool, world_hidden: bool) -> FrameContent {
        FrameContent {
            world_hidden,
            cull_records: self.state.draw.record_count(),
            shadowed_spot_count: self.spot_shadow.count(),
            bloom_intensity: self.post_process.bloom_intensity,
            bvh_live: self.rt.accel.is_some(),
            emitters: !self.particle.system.is_empty(),
            fog: self.fog.settings.is_some_and(|s| s.contributes()),
            decals: !self.decal.set.is_empty(),
            visible_surfaces: self
                .transparent
                .as_ref()
                .is_some_and(super::transparent::TransparentResources::any_visible),
            visible_see_through: self.see_through_mesh_visible(),
            planar_targets: self
                .planar_reflection
                .as_ref()
                .is_some_and(super::planar::PlanarReflectionSet::has_targets),
            water_planar_slot: self
                .transparent
                .as_ref()
                .is_some_and(super::transparent::TransparentResources::water_planar_slot_live),
            lines: lines_published,
            visible_volumes: self
                .raymarch
                .as_ref()
                .is_some_and(super::raymarch::RaymarchResources::any_visible),
            atmosphere: self.atmosphere.as_ref().is_some_and(|a| a.frame.is_some()),
            water_sim: self.water_sim_live(),
            grass: self.grass.is_some(),
            ssgi_motion: self.ssgi.as_ref().is_some_and(|s| s.settings.contributes()),
        }
    }

    // Plan this frame in core over what this backend holds, then put what the
    // encoders read from `&self` in place: the spot slices and the jitter the
    // upscaler is dispatched with.
    fn plan_frame(&mut self, args: PlanArgs) -> RenderResult<FramePlan> {
        let PlanArgs {
            camera,
            elapsed,
            render_size,
            lines_published,
            world_hidden,
        } = args;
        let resident = self.resident_features(render_size);
        let content = self.frame_content(lines_published, world_hidden);
        // The upscaler prescribes its jitter sequence (FSR queries its FFX-tuned
        // offsets; DLSS / XeSS use the shared Halton-2/3) at the plan's phase.
        let phase = self.state.planner.jitter_phase();
        let upscaler_jitter = self.upscale.as_ref().map(|up| up.jitter_offset(phase));
        let surface = SurfaceInputs {
            prefilter_mip_count: self.scene.prefilter_mip_count as f32,
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
                ssao: self.ssao.as_ref().map(|s| s.settings),
                ssr: self.ssr.as_ref().and_then(|s| s.settings),
                ssgi: self.ssgi.as_ref().map(|s| (s.settings, s.frame())),
                rt: self.rt_reflections.as_ref().map(|r| r.settings),
                history_scale: self.pre_exposure.history_scale(),
            },
        };
        let plan = self.state.planner.plan(&self.state.view, &inputs)?;
        self.spot_shadow.render_mask = self.state.planner.shadow.spot_mask;
        if let Some(up) = self.upscale.as_ref() {
            up.jitter().set(plan.camera.jitter_px);
        }
        Ok(plan)
    }
}

// The per-frame inputs `plan_frame` plans from.
struct PlanArgs {
    camera: CameraInputs,
    elapsed: f32,
    render_size: [u32; 2],
    lines_published: bool,
    world_hidden: bool,
}

// What `begin` acquired for a frame: its in-flight slot, the swapchain image,
// the outer "start" and "end" command buffers (both recording), and the
// render extent.
pub(crate) struct VkFrame {
    frame_idx: usize,
    image_index: u32,
    start_cmd: vk::CommandBuffer,
    cmd: vk::CommandBuffer,
    extent: vk::Extent2D,
}

impl FrameDevice for VkContext {
    type Frame = VkFrame;

    fn frame_scene(&mut self) -> &mut SceneState {
        &mut self.state
    }

    fn draws_atmosphere(&self) -> bool {
        self.atmosphere.is_some()
    }

    fn prepare(&mut self, params: &FrameParams<'_>) -> RenderResult<()> {
        self.frame_capture.arm(params.capture);
        if let Some(atmo) = self.atmosphere.as_mut() {
            atmo.frame = params.atmosphere;
        }
        self.apply_pending_rebuilds()?;
        // A moved world's decals and grass move with it whatever happens to
        // history.
        if let Some(rebase) = &params.rebase {
            self.decal.set.rebase(rebase);
            self.rebase_grass(rebase);
            self.rebase_transparent_water(rebase);
        }
        Ok(())
    }

    // The TAA and SSGI rings, and the upscaler's history on its next dispatch.
    fn reset_history(&mut self) {
        if let Some(taa) = &mut self.taa {
            taa.pass.reset_history();
        }
        if let Some(ssgi) = &mut self.ssgi {
            ssgi.reset_history();
        }
        if let Some(upscaler) = &self.upscale {
            upscaler.request_history_reset();
        }
    }

    fn begin(&mut self, params: &FrameParams<'_>) -> RenderResult<Option<VkFrame>> {
        // Minimized window: the client area is 0x0. Vulkan rejects every
        // zero-extent operation (swapchain, render area, viewport, image copy),
        // so park the whole frame (no acquire / record / submit / present) until
        // the window is restored. `window_closed` keeps pumping the message loop
        // each tick, so the restore is picked up.
        if self.frame_is_parked() {
            return Ok(None);
        }

        let frame_idx = self.current_frame;
        let mut gpu_wait = self.wait_frame_slot(frame_idx)?;
        self.retire_frame_captures(frame_idx);
        self.reserve_frame_capture();
        // Staged geometry writes go ahead of everything this frame submits.
        self.flush_geometry_uploads()?;
        self.service_background_work(narrow_f32(params.elapsed), frame_idx);
        let timings = self.read_gpu_timings(frame_idx);
        let gpu_done = self.read_gpu_done(frame_idx);
        self.frame_input_sampled_us[frame_idx] = params.input_sampled_us;
        self.begin_frame_stats(gpu_wait, timings, gpu_done);
        let Some(image_index) = self.acquire_frame(frame_idx, &mut gpu_wait)? else {
            return Ok(None);
        };
        // Only a frame that renders moves the history's pre-exposure on.
        self.pre_exposure.advance(self.post_process.exposure);

        // The outer "end" buffer: Composite + the post-graph work, submitted
        // after the "start" buffer and every per-pass buffer.
        let cmd = self.commands.command_buffers[frame_idx];
        // SAFETY: `cmd` belongs to this frame slot, whose fence was already waited on, so it is not
        // in flight; reset then begin puts it in the recording state, which is what the subsequent
        // recording requires.
        unsafe {
            self.hw
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| crate::vulkan::error::map_vk_result(e, "reset cmd buf"))?;
            self.hw
                .device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| crate::vulkan::error::map_vk_result(e, "begin cmd buf"))?;
        }

        // The scene rasterizes at render resolution (== swapchain extent unless
        // upscaling). Cascade / projection aspect, the HDR graph dims, and the
        // sub-pixel jitter all derive from this, not the display extent.
        let extent = self.targets.render_extent;
        self.size_grass([extent.width, extent.height], frame_idx)?;
        self.grass_lod = self.grass.as_ref().map(|g| {
            g.field
                .lod_view(concinnity_core::render::grass::lod::focal_length(
                    extent.height,
                    params.fov_y_radians,
                ))
        });

        let start_cmd = self.record_frame_start(frame_idx, params.world_hidden)?;

        // Push this frame's skinning matrices into the per-frame joint buffers
        // before the skin fold reads them. No-op when no SkinnedMesh is
        // declared.
        self.upload_joint_matrices(frame_idx);
        // Push this frame's morph weights into the per-frame weight buffers the
        // skin fold reads. No-op when no SkinnedMesh carries morph targets.
        self.upload_morph_weights(frame_idx);

        // Line resources: built on the first frame that publishes lines (and
        // this slot's vertex buffer grown to fit them), so the plan's graph gate
        // can see them live this same frame and a world that never draws a line
        // never compiles them. Safe here: the `wait_frame_slot` fence retired
        // everything that read this slot last trip.
        self.ensure_line_pipeline(frame_idx, params.lines);

        Ok(Some(VkFrame {
            frame_idx,
            image_index,
            start_cmd,
            cmd,
            extent,
        }))
    }

    fn plan(&mut self, frame: &VkFrame, params: &FrameParams<'_>) -> RenderResult<FramePlan> {
        self.plan_frame(PlanArgs {
            camera: camera_inputs(&self.state, params),
            elapsed: narrow_f32(params.elapsed),
            render_size: [frame.extent.width, frame.extent.height],
            lines_published: !params.lines.is_empty(),
            world_hidden: params.world_hidden,
        })
    }

    fn execute(
        &mut self,
        frame: VkFrame,
        plan: &FramePlan,
        frame_params: &FrameParams<'_>,
    ) -> RenderResult<()> {
        let FrameParams {
            elapsed,
            fov_y_radians,
            near,
            cam_pos,
            text_calls,
            lines,
            world_hidden,
            grass_benders,
            ..
        } = *frame_params;
        let VkFrame {
            frame_idx,
            image_index,
            start_cmd,
            cmd,
            ..
        } = frame;
        let device = self.hw.device.clone();
        let device = &device;

        // The plan's blocks into this frame's ring slots.
        if self.shadow.enabled() {
            upload_shadow_uniforms(
                &self.shadow.ubos[frame_idx],
                &self.state.planner.shadow.uniforms,
            );
        }
        self.prepare_atmosphere(frame_idx, plan.atmosphere);
        self.write_cluster_params(frame_idx, &plan.cluster);
        self.uniforms.view_ubo_buffers[frame_idx].write_val(0, &plan.view);

        // Light uniforms for this frame. Written only into slots a live edit
        // (directional set / ambient scale) has re-armed, so a steady world
        // writes nothing after the ring has caught up.
        if self.uniforms.lights.take_dirty(frame_idx) {
            upload_light_uniforms(
                &self.uniforms.light_ubo_buffers[frame_idx],
                self.uniforms.lights.uniforms(),
            );
        }
        self.uniforms.canopy = match (&self.grass, &self.grass_lod) {
            (Some(grass), Some(lod)) => self
                .state
                .canopy_params(&grass.field.canopy, (lod, grass.crossfade_phase())),
            _ => Default::default(),
        };
        self.uniforms.canopy_buffers[frame_idx].write_val(0, &self.uniforms.canopy);
        self.uniforms.canopy_tables.update(
            self.grass.as_ref().map_or(&[], |g| g.field.canopy.tables()),
            frame_idx,
        );

        // Reflection-probe count + records (global set 0 bindings 7 + 17): count
        // 0 (sky reflection) until a probe bakes, so the forward shader keeps the
        // sky path. Uploaded every frame so a later install is picked up
        // immediately.
        self.upload_probe_set(frame_idx)?;

        // Compute-cull host-side prep: rebuild this frame's `GpuObjectData` +
        // `GpuDrawArgs` storage buffers with the latest per-object state, on a
        // frame whose plan runs the cull, so the `PassId::Cull` dispatch the
        // graph executor records below sees fresh data.
        if let Some(history) = plan.records {
            let packing = self.build_record_buffers(frame_idx, cam_pos, history);
            let mut stats = self.frame_stats.get();
            stats.packing_fan_out = packing;
            self.frame_stats.set(stats);
        }

        let params = GraphFrameParams {
            cmd,
            image_index,
            frame_idx,
            text_calls,
            lines,
            world_hidden,
            frustum: &plan.camera.frustum,
            cam_pos,
            vp_mat: plan.camera.vp,
            cur_vp: plan.camera.cur_vp,
            fov_y_radians,
            elapsed,
            near,
            grass_benders,
            planar: self
                .planar_reflection
                .as_ref()
                .map(|set| set.frame_plan(plan.camera.vp))
                .unwrap_or_default(),
            cluster_params: plan.cluster,
            fog: plan.fog,
            gbuffer_view: plan.gbuffer_view,
            effects: plan.effects,
            reactive: plan.reactive,
        };
        // The per-frame submission order, excluding the outer "end" buffer
        // (`cmd`) the caller appends: the `start` buffer (leading timestamp)
        // followed by the per-pass buffers in graph order. Submission order is
        // GPU order on the single graphics queue, so every encoder's inline
        // barrier synchronizes against the prior pass across buffer boundaries.
        // Composite + the post-graph work below record into `cmd`.
        let mut submit = std::mem::take(&mut self.submit_scratch);
        submit.clear();
        submit.push(start_cmd);
        self.execute_graph(&plan.graph, &params, &mut submit)?;

        self.finish_frame_stats();

        // End-of-frame timestamp for the profiler overlay. Pairs with the
        // TOP_OF_PIPE write near the top of the function (the block's first pair).
        if let Some(pool) = self.hw.timestamp_query_pool {
            let (_, wf_end) = pass_timing::whole_frame_pair(frame_idx);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    wf_end,
                );
            }
        }

        self.finish_frame_stats();

        // End-of-frame timestamp for the profiler overlay. Pairs with the
        // TOP_OF_PIPE write in the start buffer (the block's first pair).
        if let Some(pool) = self.hw.timestamp_query_pool {
            let (_, wf_end) = pass_timing::whole_frame_pair(frame_idx);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    wf_end,
                );
            }
        }

        let capture = self.record_frame_capture(cmd, image_index);

        // SAFETY: `cmd` is in the recording state, which is what `end_command_buffer` requires.
        unsafe { device.end_command_buffer(cmd) }
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "end cmd buf"))?;
        // The outer "end" buffer (Composite + post-graph work + trailing
        // timestamp) submits last, after every per-pass buffer.
        submit.push(cmd);

        let started = std::time::Instant::now();
        let submitted = self.submit_and_present(frame_idx, image_index, &submit);
        self.submit_scratch = submit;
        let mut stats = self.frame_stats.get();
        stats.submit_present_us = concinnity_host::clock::micros_since(started);
        self.frame_stats.set(stats);
        // A submitted frame's capture waits for its slot's fence; a failed one
        // is reported as a dropped frame by the caller.
        if let Some(capture) = capture
            && submitted.is_ok()
        {
            self.frame_capture.park(frame_idx, capture);
        }
        submitted
    }

    // What the TAA and SSGI accumulations wrote is next frame's history.
    fn finish(&mut self, plan: &FramePlan) {
        if plan.motion_read
            && let Some(taa) = &mut self.taa
        {
            taa.pass.advance();
        }
        if let Some(ssgi) = &mut self.ssgi {
            ssgi.advance();
        }
    }
}

// Helper: write ShadowUniforms into one persistently-mapped slot of the
// per-frame-in-flight ring. The slot belongs to a frame whose fence the caller
// already waited.
pub(super) fn upload_shadow_uniforms(ubo: &super::allocator::PooledBuffer, su: &ShadowUniforms) {
    ubo.write_val(0, su);
}

// Helper: write LightUniforms into one persistently-mapped slot of the
// per-frame-in-flight ring. The slot belongs to a frame whose fence the caller
// already waited.
pub(super) fn upload_light_uniforms(
    light_ubo: &super::allocator::PooledBuffer,
    lu: &LightUniforms,
) {
    light_ubo.write_val(0, lu);
}

// Helper: one-shot upload of the per-scene local lights into the static SSBO.
// An empty scene leaves the 1-element placeholder buffer untouched
// (num_local_lights == 0 keeps the shader from reading it). Mirrors the single
// upload path of upload_light_uniforms; the SSBO is never rewritten per-frame.
pub(super) fn upload_static_records<T: Copy>(
    buffer: &super::allocator::PooledBuffer,
    records: &[T],
) {
    if records.is_empty() {
        return;
    }
    buffer.write_slice(0, records);
}
