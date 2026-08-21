// src/vulkan/probe.rs
//
// Scene-captured reflection probes on Vulkan. Each declared `ReflectionProbe`
// (or an auto-seeded grid when a world declares none) describes a cube to bake
// DISTINCT from `env_map`: the specular reflection term box-projects against the
// probe's influence box and samples its cube, so glossy surfaces reflect the
// actual surrounding geometry instead of the imported HDR sky, while the skybox +
// diffuse irradiance keep sampling `env_map` so the visible sky is never replaced.
//
// The cube math + the staggered-bake state machine are backend-agnostic
// (`crate::gfx::reflection_probe`); this module drives the placement intake + the
// GPU capture, mirroring `crate::directx::probe` / `crate::metal::probe`.
//
// `set_reflection_probes` converts the graphics-system placements (auto-seeding a
// grid from the scene bounds when a world declares none) into the stored placement
// list + an EMPTY `ProbeSet`, then enqueues them. `bake_pending_probes` (driven each
// frame from `draw_frame`) advances the shared `next_bake_action` transition table:
// it renders one cube face per frame into a bake-owned target on a per-face fence,
// reads the six faces back, runs the GGX prefilter convolution on a worker thread,
// and installs the prefiltered cube into the forward / SSR / RT cube array -- all
// without blocking the render loop (the sky reflection covers a probe until its
// cube installs). The forward / SSR / RT sampling lives in the main / resolve
// shaders (see the reflection_probes.md DX/VK port checklist).

use ash::vk;

use crate::vulkan::owned::{OwnedDescriptorPool, OwnedFramebuffer, VkDevice};

use super::allocator::PooledBuffer;
use super::context::{HDR_FORMAT, VkContext};
use super::cull::CullParams;
use super::descriptor_layout::{LOCAL_LIGHT_SSBO_BINDING, PROBE_CUBE_ARRAY_BINDING};
use super::draw::ViewUniforms;
use super::hiz::CullHizParams;
use super::resources::alloc_descriptor_sets;
use super::texture::{
    GpuImage, ImageSpec, create_image, create_image_view, upload_probe_prefilter_cube,
};
use crate::gfx::frustum::Frustum;
use crate::gfx::image_decode::f16_to_f32;
use crate::gfx::reflection_probe::{self, BakeAction, BakePhase, ProbePlacement};
use concinnity_render::uniforms::MAX_PROBES;
use concinnity_render::uniforms::ProbeSet;
use concinnity_render::uniforms::ProbeUniforms;

// Captured cube-face resolution (mip 0 of the prefilter chain). Matches the
// `EnvironmentMap` asset default + the DirectX / Metal `PROBE_FACE_SIZE`.
const PROBE_FACE_SIZE: u32 = 512;
// Irradiance cube resolution (diffuse is low frequency, so this stays small).
const PROBE_IRRADIANCE_FACE: u32 = 16;
// GGX prefilter samples per output texel (a runtime bake uses far fewer than the
// importer's 1024; the convolution is rayon-parallel).
const PROBE_PREFILTER_SAMPLES: u32 = 128;
// Firefly clamp during the prefilter convolution (matches the asset default).
const PROBE_PREFILTER_CLAMP: f32 = 12.0;
// Cube faces per probe.
const PROBE_FACE_COUNT: usize = 6;
// Depth format of the probe-face target (matches the main pass's DSV).
const PROBE_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
// Near / far for the 90-degree probe-face projection. A fixed wide range keeps
// the capture independent of the live camera; the cube is sampled by direction,
// so the exact far plane only affects depth precision during the bake.
const PROBE_NEAR: f32 = 0.05;
const PROBE_FAR: f32 = 2000.0;

// The cull push constant for an off-camera capture (a probe face or a planar
// mirror plane), which differs from the main camera's in one way: `bucket_count`
// is 1, so every record is routed into region 0 whatever shader bucket it belongs
// to. The capture callers allocate a single-region indirect buffer and draw it
// with the one default bindless pipeline, so a bucketed record must land in
// region 0 to appear at all -- with default shading, which is the documented
// trade the DirectX and Metal capture paths make too.
fn capture_cull_params(frustum: &Frustum, cam_pos: [f32; 3], n_cull: u32) -> CullParams {
    let mut params = CullParams {
        planes: [[0.0; 4]; 6],
        cam_pos,
        object_count: n_cull,
        bucket_count: 1,
        // Never indexed with `bucket_count == 1` (region 0 starts at 0), but it
        // names the region capacity the caller sized its buffer with.
        bucket_stride: n_cull,
    };
    for (i, p) in frustum.planes.iter().enumerate().take(6) {
        params.planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
    }
    params
}

impl VkContext {
    // Set the reflection-probe placements (declared `ReflectionProbe` assets,
    // converted to `ProbePlacement`s by the graphics system). An empty list
    // auto-seeds a grid from the scene bounds, so existing scenes still get local
    // reflections without authoring. Capped at the cube array's descriptor count,
    // so `probe_set.count` can never index past what the shader declares. Pushed
    // once after construction; the cube capture that fills the probe set runs
    // across later frames (next slice).
    pub(super) fn set_reflection_probes(&mut self, declared: &[ProbePlacement]) {
        let mut placements: Vec<ProbePlacement> = if declared.is_empty() {
            match self.scene_world_bounds() {
                Some((mn, mx)) => {
                    // Object AABBs as occupancy so a probe is not auto-captured from
                    // inside a wall; skip degenerate (non-finite) boxes.
                    let occupancy: Vec<([f32; 3], [f32; 3])> = self
                        .draw_objects
                        .iter()
                        .map(|o| (o.bb_min, o.bb_max))
                        .filter(|(mn, mx)| mn.iter().chain(mx).all(|c| c.is_finite()))
                        .collect();
                    reflection_probe::auto_seed_probes(mn, mx, &occupancy)
                }
                None => Vec::new(),
            }
        } else {
            declared.to_vec()
        };
        let bind_count = self.descriptors.probe_cube_count as usize;
        if placements.len() > bind_count {
            // Past the CPU ceiling means authored (or seeded) probes are dropped;
            // between the device's bind count and the ceiling is only what this
            // GPU's sampler headroom affords, which init already reported.
            if placements.len() > MAX_PROBES {
                tracing::warn!(
                    "reflection probes: {} placements, capping at {bind_count}",
                    placements.len()
                );
            } else {
                tracing::debug!(
                    "reflection probes: binding {bind_count} of {} placements",
                    placements.len()
                );
            }
            placements.truncate(bind_count);
        }
        // A re-placement (rare -- this is normally a one-time init call) abandons any
        // in-flight staggered bake and frees the previously baked cubes. Idle first
        // when a capture is in flight (its targets may still be on the GPU) or cubes
        // exist (the forward shader may sample them), reset every cube-array slot back
        // to the sky so none dangles, then drop the in-flight bake + the cubes. The
        // common first call has an empty queue + `probe_maps`, so it skips all of this.
        if self.probe_rendering.is_some() || !self.probe_maps.is_empty() {
            self.wait_idle();
        }
        let device = self.device.clone();
        if let Some(rendering) = self.probe_rendering.take() {
            rendering.destroy(&device, self.commands.command_pool);
        }
        self.probe_converting = None;
        if !self.probe_maps.is_empty() {
            self.reset_probe_cube_slots_to_sky();
            self.probe_maps.clear();
        }
        self.probe_placements = placements;
        self.probe_set = ProbeSet::EMPTY;
        // Enqueue the placements; `bake_pending_probes` (driven each frame from
        // `draw_frame`) renders + installs them staggered across later frames, so the
        // construction call no longer blocks on the capture.
        self.probe_bake_queue = reflection_probe::ProbeBakeQueue::new(self.probe_placements.len());
    }

    // World-space bounds over every static draw object, skipping degenerate
    // (non-finite) AABBs. `None` for an empty scene. Mirrors
    // `directx/probe.rs::scene_world_bounds`.
    pub(super) fn scene_world_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        reflection_probe::fold_world_bounds(self.draw_objects.iter().map(|o| (o.bb_min, o.bb_max)))
    }

    // Point every probe-cube-array slot (binding 8) of every frame's global set
    // back at the sky prefilter cube. The init path leaves them this way; this
    // restores it before a re-placement drops the old baked cubes, so no slot
    // dangles a freed view (Vulkan requires every descriptor in a bound set be
    // valid, even slots the shader's `i < count` loop never samples).
    fn reset_probe_cube_slots_to_sky(&self) {
        let sky: Vec<vk::DescriptorImageInfo> = (0..self.descriptors.probe_cube_count)
            .map(|_| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(self.env_map.prefilter.view)
                    .sampler(self.cube_sampler.handle())
            })
            .collect();
        for &set in &self.descriptors.global_sets {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(PROBE_CUBE_ARRAY_BINDING)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&sky);
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        }
    }

    // Advance the staggered asynchronous reflection-probe bake by one step. Called
    // every frame from `draw_frame` after this frame's slot fence wait; cheap once the
    // queue drains. Drives the shared `next_bake_action` transition table over two
    // pipelined slots (one Rendering, one Converting), so the capture spreads across
    // frames instead of blocking construction. Non-fatal: a failure abandons the
    // remaining bakes, keeping what already installed. Mirrors `directx::probe`.
    //
    // V1 simplifications (documented; shared with DirectX / Metal):
    //   * Static + streamed-chunk geometry only -- instanced + skinned draws are left
    //     disabled in the bake cull buffers (the kernel skips them). They still
    //     RECEIVE probe reflections.
    //   * Cold lighting -- shadows may be unpopulated on the first frames, exactly
    //     like the DX / Metal first-frame bake.
    pub(super) fn bake_pending_probes(&mut self) -> Result<(), String> {
        // Nothing queued and nothing in flight: cheap early-out once the bake drains.
        if !self.probe_bake_queue.pending()
            && self.probe_rendering.is_none()
            && self.probe_converting.is_none()
        {
            return Ok(());
        }
        // Permanent ineligibility: a probe only improves on a real captured
        // environment, and the capture renders through the bindless GPU cull. These
        // never become true after init, so abandon the queue rather than re-checking
        // forever (the forward specular keeps sampling the sky).
        if self.env_map.prefilter_mip_count <= 1
            || self.cull.cull_pipeline.is_none()
            || self.cull.bindless_pipeline.is_none()
        {
            if self.probe_rendering.is_some() {
                self.wait_idle();
            }
            let device = self.device.clone();
            if let Some(rendering) = self.probe_rendering.take() {
                rendering.destroy(&device, self.commands.command_pool);
            }
            self.probe_converting = None;
            self.probe_bake_queue.abort();
            return Ok(());
        }

        // Converting slot first: install the convolved cube once the worker finishes,
        // freeing the slot so the rendering slot can read its finished capture back
        // this same frame (keeps installs in queue order -> `probe_maps` aligned with
        // the placement list).
        let converting_occupied = self.probe_converting.is_some();
        let payload_ready = self
            .probe_converting
            .as_ref()
            .is_some_and(|c| c.payload.get().is_some());
        let install = reflection_probe::next_bake_action(
            if converting_occupied {
                BakePhase::Converting
            } else {
                BakePhase::Idle
            },
            false,
            payload_ready,
            false,
            false,
            false,
        ) == BakeAction::Install;
        if install && let Err(e) = self.probe_install() {
            self.fail_bake(e);
            return Ok(());
        }
        let converting_free = !converting_occupied || install;

        // Rendering slot: submit one face per frame; once all six retired on the GPU
        // (the last face's fence signalled) AND the converting slot is free, read the
        // faces back and hand them to the worker, or start the next placement.
        let rendering_occupied = self.probe_rendering.is_some();
        let more_faces = self
            .probe_rendering
            .as_ref()
            .is_some_and(|r| r.cursor < PROBE_FACE_COUNT);
        let done = self.probe_rendering.as_ref().is_some_and(|r| {
            r.cursor >= PROBE_FACE_COUNT
                // SAFETY: the fence was created from this device; the query only reads.
                && unsafe { self.device.get_fence_status(r.face_fences[r.last_fence()]) }
                    .unwrap_or(false)
        });
        // Transient ineligibility: geometry may still be streaming. A zero cull keeps
        // the queue cursor so a later frame retries rather than baking an empty cube.
        let eligible = self.cull_count() > 0;
        match reflection_probe::next_bake_action(
            if rendering_occupied {
                BakePhase::Rendering
            } else {
                BakePhase::Idle
            },
            done && converting_free,
            false,
            self.probe_bake_queue.pending(),
            eligible,
            more_faces,
        ) {
            BakeAction::RenderFace => {
                if let Err(e) = self.probe_render_next_face() {
                    self.fail_bake(e);
                }
            }
            BakeAction::Readback => {
                if let Err(e) = self.probe_readback_and_convolve() {
                    self.fail_bake(e);
                }
            }
            BakeAction::StartNext => {
                if let Err(e) = self.probe_start_next() {
                    self.fail_bake(e);
                }
            }
            BakeAction::Install | BakeAction::Idle => {}
        }
        Ok(())
    }

    // Abandon the rest of the bake after an unrecoverable error, keeping the cubes
    // already installed. The queue cursor advanced when the current probe started, so
    // aborting (cursor -> end) keeps `probe_maps` aligned with the placement list.
    fn fail_bake(&mut self, e: String) {
        tracing::warn!(
            "reflection probe bake failed, keeping {} baked: {e}",
            self.probe_maps.len()
        );
        // Idle before dropping the in-flight capture's GPU resources: its command
        // buffers may still be executing. A bake failure is rare (allocation / device
        // error), so the one-time stall is acceptable.
        if self.probe_rendering.is_some() {
            self.wait_idle();
        }
        let device = self.device.clone();
        if let Some(rendering) = self.probe_rendering.take() {
            rendering.destroy(&device, self.commands.command_pool);
        }
        self.probe_converting = None;
        self.probe_bake_queue.abort();
    }

    // Begin baking the next pending placement: build the bake-owned capture resources
    // (target + cull ring + per-face view UBOs + readback buffers) and fill the cull
    // buffers + the six per-face view uniforms ONCE (frustum-independent; each face
    // re-runs only the cull with its own frustum). No face is submitted here; the six
    // follow one per frame via `probe_render_next_face`.
    fn probe_start_next(&mut self) -> Result<(), String> {
        let Some(index) = self.probe_bake_queue.take_next() else {
            return Ok(());
        };
        let placement = self.probe_placements[index];
        let eye = placement.position;
        let bake = BakeResources::new(self)?;

        // Bake-owned cull buffers, zeroed first so the untouched instance tail reads
        // as disabled (a probe omits instanced geometry in V1), then filled with this
        // probe's static + chunk + skinned records (LOD by probe eye).
        let object_size =
            self.cull_count() * std::mem::size_of::<crate::gfx::render_types::GpuObjectData>();
        let args_size =
            self.cull_count() * std::mem::size_of::<crate::gfx::render_types::GpuDrawArgs>();
        bake.object_buf.zero_bytes(0, object_size);
        bake.draw_args_buf.zero_bytes(0, args_size);
        self.build_object_records_into(&bake.object_buf);
        self.build_draw_args_records_into(&bake.draw_args_buf, eye);

        // Per-face view uniforms (the only per-face binding), all six filled once.
        // reflections_enabled stays 0: no resolve runs over a probe face, so the bake
        // captures the full forward probe specular -- here the sky, since the bake
        // binds an EMPTY ProbeSet.
        let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;
        for face in 0..PROBE_FACE_COUNT {
            let vp = reflection_probe::face_view_projection(eye, face, PROBE_NEAR, PROBE_FAR);
            let view_mat = reflection_probe::face_view_matrix(eye, face);
            let view = ViewUniforms {
                vp,
                view: view_mat,
                elapsed: 0.0,
                reflections_enabled: 0.0,
                cam_pos: [eye[0], eye[1], eye[2]],
                prefilter_mip_count,
                // A probe capture is always lit, whatever the viewport shows.
                shade_mode: 0.0,
                _end_pad: 0.0,
            };
            bake.view_bufs[face].write_val(0, &view);
        }

        self.probe_rendering = Some(RenderingBake {
            index,
            placement,
            eye,
            cursor: 0,
            bake,
            face_cmds: Vec::with_capacity(PROBE_FACE_COUNT),
            face_fences: Vec::with_capacity(PROBE_FACE_COUNT),
        });
        Ok(())
    }

    // Write the whole live texture pool into a bake face's bindless set
    // (binding 1). Called right before the face records, so the face samples
    // the pool as it stands this frame.
    fn write_probe_face_pool(&self, set: vk::DescriptorSet) {
        let pool_infos: Vec<vk::DescriptorImageInfo> = self
            .textures
            .iter()
            .chain(self.normal_map_textures.iter())
            .map(|img| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(img.view)
                    .sampler(self.linear_sampler.handle())
            })
            .collect();
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&pool_infos);
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe {
            self.device
                .update_descriptor_sets(std::slice::from_ref(&write), &[])
        };
    }

    // Submit one cube face of the in-flight probe: a fresh command buffer that culls
    // for this face's frustum, draws the bindless main into the bake target, and
    // copies the resolved face into its readback buffer, on a per-face fence (polled,
    // never waited). The command buffer + fence are held in the `RenderingBake` until
    // readback, so the last face's fence retiring means the whole capture is done. One
    // face per frame spreads the capture so no frame pays the whole cost.
    fn probe_render_next_face(&mut self) -> Result<(), String> {
        let device = self.device.clone();
        let extent = vk::Extent2D {
            width: PROBE_FACE_SIZE,
            height: PROBE_FACE_SIZE,
        };
        // Copy the bake handles out (all Copy) so no borrow of `self.probe_rendering`
        // is held across the `&self` encode calls below.
        let (
            face,
            eye,
            cull_set,
            hiz_set,
            framebuffer,
            global_set,
            bindless_set,
            indirect,
            copy_src,
            readback,
        ) = {
            let r = self
                .probe_rendering
                .as_ref()
                .ok_or("probe: render face with no bake in flight")?;
            let b = &r.bake;
            (
                r.cursor,
                r.eye,
                b.cull_set,
                b.hiz_set,
                b.framebuffer.handle(),
                b.global_sets[r.cursor],
                b.bindless_sets[r.cursor],
                b.indirect_buf.buffer(),
                b.copy_source(),
                b.readback_bufs[r.cursor].buffer(),
            )
        };

        // Snapshot the live texture pool into this face's set. The set has
        // never been bound in a submitted command buffer (each face uses its
        // own), so the write is legal without a drain, and a texture streamed
        // in since the bake started is picked up here.
        self.write_probe_face_pool(bindless_set);

        // A fresh command buffer + fence for this face, from the one-shot pool.
        // Register both in the `RenderingBake` the instant they exist so a later
        // record / submit error still reclaims them via `fail_bake` ->
        // `RenderingBake::destroy` (which `wait_idle`s first); on the success path the
        // last-pushed fence is `face_fences[last_fence()]` after `cursor` advances.
        let cmd = {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.commands.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the create-info and every slice it borrows are live for the call, and each
            // handle it names belongs to this device.
            unsafe { device.allocate_command_buffers(&info) }
                .map_err(|e| format!("probe face cmd alloc: {e}"))?[0]
        };
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(f) => f,
            Err(e) => {
                // The command buffer is allocated but not yet tracked; free it before
                // bailing so it does not leak.
                // SAFETY: the handle was created from this device moments ago and never submitted,
                // so this cleanup is its only remaining use.
                unsafe {
                    device.free_command_buffers(
                        self.commands.command_pool,
                        std::slice::from_ref(&cmd),
                    );
                }
                return Err(format!("probe face fence: {e}"));
            }
        };
        {
            let r = self
                .probe_rendering
                .as_mut()
                .ok_or("probe: render face slot vanished")?;
            r.face_cmds.push(cmd);
            r.face_fences.push(fence);
        }

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cmd` was allocated from this device's pool and is not in flight (its face fence
        // was waited on), so it is in the initial state that `begin` requires.
        unsafe { device.begin_command_buffer(cmd, &begin) }
            .map_err(|e| format!("probe face begin: {e}"))?;
        // Order the previous face's readback copy + indirect-draw read (a prior
        // frame's submit) before this face's cull (rewrites the shared indirect
        // buffer) and resolve (rewrites the shared colour). Intra-queue, so the
        // queue's submission order preserves it across the separate submits.
        //
        // The attachment writes are here for a second reason: all six faces share
        // one framebuffer, and `main_render_pass` declares `initial_layout =
        // UNDEFINED`, so this face's `vkCmdBeginRenderPass` performs a layout
        // transition that write-after-writes the previous face's storeOp. The
        // render pass's own external dependency declares an empty src access mask,
        // an execution dependency with no availability operation, so nothing else
        // covers it.
        if face > 0 {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::INDIRECT_COMMAND_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .dst_access_mask(
                    vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                );
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER
                        | vk::PipelineStageFlags::DRAW_INDIRECT
                        | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&barrier),
                    &[],
                    &[],
                );
            }
        }
        let vp = reflection_probe::face_view_projection(eye, face, PROBE_NEAR, PROBE_FAR);
        let frustum = Frustum::from_view_projection(vp);
        self.encode_probe_cull(cmd, cull_set, hiz_set, &frustum, eye);
        self.encode_main_into_face(cmd, framebuffer, extent, global_set, bindless_set, indirect);
        // The face colour rests in SHADER_READ_ONLY_OPTIMAL after the render pass;
        // flip it to TRANSFER_SRC for the readback copy. This exact transition is the
        // one the shared layout-transition table omits.
        let to_src = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(copy_src)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: PROBE_FACE_SIZE,
                height: PROBE_FACE_SIZE,
                depth: 1,
            });
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_src),
            );
            device.cmd_copy_image_to_buffer(
                cmd,
                copy_src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback,
                std::slice::from_ref(&region),
            );
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("probe face end: {e}"))?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device
                .queue_submit(self.graphics_queue, std::slice::from_ref(&submit), fence)
                .map_err(|e| format!("probe face submit: {e}"))?;
        }

        // The command buffer + fence are already tracked (registered at allocation);
        // advance the cursor now that this face submitted, so `last_fence()` points at
        // it and `done` polls the right fence.
        let r = self
            .probe_rendering
            .as_mut()
            .ok_or("probe: render face slot vanished")?;
        r.cursor += 1;
        Ok(())
    }

    // The capture finished on the GPU (the last face's fence signalled): map the six
    // readback buffers, decode RGBA16F -> f32, free the capture's GPU resources, and
    // hand the faces to a worker thread that runs the GGX prefilter convolution off
    // the render thread. Moves the bake to the Converting slot.
    fn probe_readback_and_convolve(&mut self) -> Result<(), String> {
        let rendering = self
            .probe_rendering
            .take()
            .ok_or("probe: readback with no bake in flight")?;
        let device = self.device.clone();

        // Decode the six readbacks (tightly packed RGBA16F) to f32.
        let mut faces: [Vec<f32>; PROBE_FACE_COUNT] = std::array::from_fn(|_| Vec::new());
        let face_bytes = (PROBE_FACE_SIZE as u64) * (PROBE_FACE_SIZE as u64) * 8;
        for (slot, buf) in faces.iter_mut().zip(rendering.bake.readback_bufs.iter()) {
            // SAFETY: the buffer is HOST_COHERENT and `face_bytes` long; the last
            // face's fence is signalled, so on the single graphics queue all six copies
            // completed.
            let raw = unsafe { std::slice::from_raw_parts(buf.mapped_ptr(), face_bytes as usize) };
            *slot = decode_probe_face_rgba16f(raw, PROBE_FACE_SIZE);
        }
        let index = rendering.index;
        let placement = rendering.placement;
        // The capture's GPU resources (target + cull + per-face command buffers +
        // fences + readbacks) free here; the last face's fence signalled, so the GPU
        // is done with all of them.
        rendering.destroy(&device, self.commands.command_pool);

        // Convolve off the render thread: only the decoded CPU floats + the payload
        // slot cross the boundary (no vk handle), so it is Send-safe. A worker panic
        // yields an empty payload, which `probe_install` rejects -> `fail_bake`.
        let payload = std::sync::Arc::new(std::sync::OnceLock::new());
        let slot = std::sync::Arc::clone(&payload);
        std::thread::spawn(move || {
            let bytes = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reflection_probe::build_probe_payload(
                    &faces,
                    PROBE_FACE_SIZE,
                    PROBE_IRRADIANCE_FACE,
                    PROBE_PREFILTER_SAMPLES,
                    PROBE_PREFILTER_CLAMP,
                )
            }))
            .unwrap_or_else(|_| {
                tracing::error!("reflection probe convolution panicked; abandoning this probe");
                Vec::new()
            });
            let _ = slot.set(bytes);
        });

        self.probe_converting = Some(ConvertingBake {
            index,
            placement,
            payload,
        });
        Ok(())
    }

    // The off-thread convolution finished: deserialise the worker's payload, upload
    // the prefiltered radiance cube, and install it as probe `index` -- point this
    // probe's slot in every frame's cube array at the baked cube and record its
    // parallax box, bumping `probe_set.count` so the forward specular samples it.
    // Leaves `env_map` / the sky untouched. Mirrors `directx/probe.rs::probe_install`.
    fn probe_install(&mut self) -> Result<(), String> {
        let ConvertingBake {
            index,
            placement: p,
            payload,
        } = self
            .probe_converting
            .take()
            .ok_or("probe: install with no bake in flight")?;
        let bytes = payload.get().ok_or("probe: install before payload ready")?;
        let view = crate::build::environment_map::deserialise(bytes)
            .map_err(|e| format!("deserialise probe payload: {e}"))?;
        if view.prefilter_mip_bytes.is_empty() {
            return Err("probe payload has no prefilter mips".into());
        }
        let cube = upload_probe_prefilter_cube(
            &super::texture::GpuUploadContext {
                alloc: &self.alloc,
                device: &self.device,
                command_pool: self.commands.command_pool,
                queue: self.graphics_queue,
            },
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?;

        // Point this probe's slot in every frame's global set at the baked cube (it
        // held the sky prefilter until now). Safe to rewrite mid-frame-loop: the cube
        // upload's `one_shot_submit` just idled the graphics queue (no in-flight frame
        // is reading the global sets), and the shader's `i < count` loop never reaches
        // slot `index` until the count bump below, so no frame samples it mid-rewrite.
        let img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(cube.view)
            .sampler(self.cube_sampler.handle());
        for &set in &self.descriptors.global_sets {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(PROBE_CUBE_ARRAY_BINDING)
                .dst_array_element(index as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&img_info));
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        }

        // Installs run in queue order, so the cube array stays aligned with the
        // placement list.
        debug_assert_eq!(index, self.probe_maps.len());
        self.probe_maps.push(cube);
        self.probe_set.probes[index] = ProbeUniforms {
            box_min: [p.box_min[0], p.box_min[1], p.box_min[2], 1.0],
            box_max: [p.box_max[0], p.box_max[1], p.box_max[2], 0.0],
            probe_pos: [p.position[0], p.position[1], p.position[2], 0.0],
        };
        self.probe_set.count = self.probe_maps.len() as u32;
        if !self.probe_bake_queue.pending() && self.probe_rendering.is_none() {
            tracing::info!(
                "reflection probes: baked {}/{}",
                self.probe_maps.len(),
                self.probe_placements.len()
            );
        }
        Ok(())
    }

    // Dispatch the compute cull for one probe face (or one planar mirror plane)
    // into the caller's indirect buffer. A thin sibling of `encode_cull`: it binds
    // the given cull set (set 0) and -- when the world runs Hi-Z -- a Hi-Z set
    // (set 1, written with `hiz_enabled = 0` so the frustum-only cull never samples
    // the pyramid; the cull layout statically references set 1, so it must be
    // bound), pushes the face/plane frustum + eye, dispatches one invocation per
    // record, and orders the writes before the indirect draw's read. Shared by the
    // probe bake + the planar reflection's reflected-frustum cull.
    //
    // `bucket_count = 1` routes every record into region 0 whatever shader bucket
    // it belongs to, matching the single indirect region these callers allocate and
    // the one bindless pipeline `encode_main_into_face` draws it with: a bucketed
    // draw appears in the capture with default shading rather than not at all.
    pub(in crate::vulkan) fn encode_probe_cull(
        &self,
        cmd: vk::CommandBuffer,
        cull_set: vk::DescriptorSet,
        hiz_set: Option<vk::DescriptorSet>,
        frustum: &Frustum,
        cam_pos: [f32; 3],
    ) {
        let (Some(pipeline), Some(layout)) = (
            self.cull.cull_pipeline.as_ref(),
            self.cull.cull_pipeline_layout.as_ref(),
        ) else {
            return;
        };
        let device = &self.device;
        let params = capture_cull_params(frustum, cam_pos, self.cull_count() as u32);
        // SAFETY: `CullParams` is `repr(C)` and matches the push-constant block
        // cull.comp declares (pinned by the layout test in concinnity-render).
        let push = unsafe {
            std::slice::from_raw_parts(
                &params as *const CullParams as *const u8,
                std::mem::size_of::<CullParams>(),
            )
        };
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout.handle(),
                0,
                std::slice::from_ref(&cull_set),
                &[],
            );
            if let Some(hs) = hiz_set {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout.handle(),
                    1,
                    std::slice::from_ref(&hs),
                    &[],
                );
            }
            device.cmd_push_constants(cmd, layout.handle(), vk::ShaderStageFlags::COMPUTE, 0, push);
            device.cmd_dispatch(cmd, (self.cull_count() as u32).div_ceil(64), 1, 1);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    // Render the bindless static + instance + chunk prefix into a probe face (or a
    // planar mirror plane). A thin sibling of `encode_main_pass`'s bindless branch:
    // begins the render pass (reusing `main_render_pass`, render-pass-compatible
    // with the bindless pipeline), binds the caller's face/plane global set (set 0)
    // + bindless set (set 1), and issues one indirect draw of
    // `[0, skinned_record_base())` from the given indirect buffer. The skinned tail
    // is omitted (V1). Shared by the probe bake + the planar reflection render.
    pub(in crate::vulkan) fn encode_main_into_face(
        &self,
        cmd: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        extent: vk::Extent2D,
        global_set: vk::DescriptorSet,
        bindless_set: vk::DescriptorSet,
        indirect: vk::Buffer,
    ) {
        let (Some(pipeline), Some(layout)) = (
            self.cull.bindless_pipeline.as_ref(),
            self.cull.bindless_pipeline_layout.as_ref(),
        ) else {
            return;
        };
        let device = &self.device;
        let [r, g, b, a] = self.clear_color;
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [r, g, b, a],
            },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clears: &[vk::ClearValue] = if self.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            &[clear_color, clear_depth, vk::ClearValue::default()]
        } else {
            &[clear_color, clear_depth]
        };
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.main_render_pass.handle())
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D::default().extent(extent))
            .clear_values(clears);
        // Negative-height viewport (Y flip), matching the main pass so the captured
        // faces share the cube convention `face_view_projection` was built against.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.geometry.vertex_buffer.buffer()], &[0]);
            device.cmd_bind_index_buffer(
                cmd,
                self.geometry.index_buffer.buffer(),
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout.handle(),
                0,
                &[global_set, bindless_set],
                &[],
            );
            device.cmd_draw_indexed_indirect(
                cmd,
                indirect,
                0,
                self.skinned_record_base() as u32,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            device.cmd_end_render_pass(cmd);
        }
    }
}

// One in-flight probe's GPU capture state, held on `VkContext::probe_rendering`
// while its six faces submit one per frame. Reuses one `BakeResources` (built in
// `probe_start_next`, freed in `probe_readback_and_convolve`) across the faces; the
// per-face command buffers + fences accumulate until readback, when the last face's
// fence retiring guarantees the GPU is done with all of them. Mirrors
// `directx::probe::RenderingBake`.
pub(super) struct RenderingBake {
    index: usize,
    placement: ProbePlacement,
    eye: [f32; 3],
    // Next of `PROBE_FACE_COUNT` faces to submit; `more_faces = cursor < FACE_COUNT`.
    cursor: usize,
    bake: BakeResources,
    face_cmds: Vec<vk::CommandBuffer>,
    face_fences: Vec<vk::Fence>,
}

impl RenderingBake {
    // Index of the face whose fence completion means the whole capture retired (the
    // last submitted face; the single graphics queue retires the rest in order).
    fn last_fence(&self) -> usize {
        self.cursor.saturating_sub(1)
    }

    // Re-point this bake's Hi-Z set at a rebuilt pyramid view. Called by
    // `rebuild_swapchain` after `hiz.resize_to` retired the view this set
    // captured at bake start; `wait_idle` gated the in-flight faces, and
    // hiz_enabled = 0 keeps the binding unsampled, but it must not dangle.
    // Mirrors the planar cull set's treatment.
    pub(super) fn rewrite_hiz_view(
        &self,
        device: &VkDevice,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        let Some(set) = self.bake.hiz_set else { return };
        let img = img_info(view, sampler);
        let write = sampler_write(set, 0, &img);
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }

    // Free every owned GPU resource: the per-face command buffers (back to the
    // one-shot pool), the per-face fences, and the bake target / cull / sets. The
    // caller has ensured the GPU retired them (the last face's fence is signalled, or
    // the device is idle).
    pub(super) fn destroy(self, device: &VkDevice, command_pool: vk::CommandPool) {
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.
        unsafe {
            if !self.face_cmds.is_empty() {
                device.free_command_buffers(command_pool, &self.face_cmds);
            }
            for &fence in &self.face_fences {
                device.destroy_fence(fence, None);
            }
        }
        self.bake.destroy(device);
    }
}

// The prior probe whose read-back faces are convolving on a worker thread. Holds
// only the worker's payload slot (plain bytes), so it drops freely (no vk handle).
// Mirrors `directx::probe::ConvertingBake`.
pub(super) struct ConvertingBake {
    index: usize,
    placement: ProbePlacement,
    payload: std::sync::Arc<std::sync::OnceLock<Vec<u8>>>,
}

// The GPU resources for ONE reflection-probe bake: the 512x512 colour/depth
// (/resolve) target + framebuffer, a bake-owned cull ring + its descriptor sets,
// six per-face global sets carrying the face view + snapshot lighting, and six
// readback buffers. One per in-flight probe (held in `RenderingBake`); `destroy`
// frees it after the faces read back.
struct BakeResources {
    color: GpuImage,
    // Held for the bake's lifetime; the framebuffer and sets alias them.
    _depth: GpuImage,
    resolve: Option<GpuImage>,
    framebuffer: OwnedFramebuffer,
    object_buf: PooledBuffer,
    draw_args_buf: PooledBuffer,
    indirect_buf: PooledBuffer,
    _status_buf: PooledBuffer,
    _pool: OwnedDescriptorPool,
    cull_set: vk::DescriptorSet,
    // One texture-pool set per face, written from the live pool right before
    // that face records. A face's set is never touched after its submit, so a
    // streamed texture swap mid-bake needs no rewrite of pending sets (and no
    // device drain): the next face simply snapshots the current pool.
    bindless_sets: Vec<vk::DescriptorSet>,
    hiz_set: Option<vk::DescriptorSet>,
    _hiz_ubo: Option<PooledBuffer>,
    global_sets: Vec<vk::DescriptorSet>,
    view_bufs: Vec<PooledBuffer>,
    _light: PooledBuffer,
    _shadow: PooledBuffer,
    _probeset: PooledBuffer,
    readback_bufs: Vec<PooledBuffer>,
}

impl BakeResources {
    // The image the readback copy reads: the single-sample resolve when MSAA is on,
    // else the (single-sample) colour attachment. Both rest in SHADER_READ_ONLY
    // after the render pass.
    fn copy_source(&self) -> vk::Image {
        match &self.resolve {
            Some(r) => r.image,
            None => self.color.image,
        }
    }

    fn new(ctx: &VkContext) -> Result<BakeResources, String> {
        use crate::gfx::render_types::{GpuDrawArgs, GpuObjectData, LightUniforms, ShadowUniforms};
        let device = &ctx.device;
        let alloc = &ctx.alloc;
        let msaa = ctx.msaa_samples != vk::SampleCountFlags::TYPE_1;
        let size = PROBE_FACE_SIZE;

        // Colour + depth (+ single-sample resolve when MSAA), then a framebuffer
        // compatible with `main_render_pass`.
        let color_pooled = create_image(
            alloc,
            &ImageSpec {
                width: size,
                height: size,
                format: HDR_FORMAT,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: ctx.msaa_samples,
            },
        )?;
        let color_view = create_image_view(
            device,
            color_pooled.image(),
            HDR_FORMAT,
            vk::ImageAspectFlags::COLOR,
        )?;
        let color = GpuImage::from_pooled(color_pooled, color_view);
        let depth_pooled = create_image(
            alloc,
            &ImageSpec {
                width: size,
                height: size,
                format: PROBE_DEPTH_FORMAT,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: ctx.msaa_samples,
            },
        )?;
        let depth_view = create_image_view(
            device,
            depth_pooled.image(),
            PROBE_DEPTH_FORMAT,
            vk::ImageAspectFlags::DEPTH,
        )?;
        let depth = GpuImage::from_pooled(depth_pooled, depth_view);
        let resolve = if msaa {
            let resolve_pooled = create_image(
                alloc,
                &ImageSpec {
                    width: size,
                    height: size,
                    format: HDR_FORMAT,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::SAMPLED,
                    mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    samples: vk::SampleCountFlags::TYPE_1,
                },
            )?;
            let view = create_image_view(
                device,
                resolve_pooled.image(),
                HDR_FORMAT,
                vk::ImageAspectFlags::COLOR,
            )?;
            Some(GpuImage::from_pooled(resolve_pooled, view))
        } else {
            None
        };
        let fb_attachments: Vec<vk::ImageView> = if msaa {
            vec![
                color.view,
                depth.view,
                resolve
                    .as_ref()
                    .expect("a multisampled probe target has a resolve image")
                    .view,
            ]
        } else {
            vec![color.view, depth.view]
        };
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(ctx.main_render_pass.handle())
            .attachments(&fb_attachments)
            .width(size)
            .height(size)
            .layers(1);
        let framebuffer = device
            .create_framebuffer(&fb_info)
            .map_err(|e| format!("probe framebuffer: {e}"))?;

        // Bake-owned cull ring, sized like the per-frame rings.
        let n = ctx.cull_count();
        let object_size = (n * std::mem::size_of::<GpuObjectData>()) as u64;
        let args_size = (n * std::mem::size_of::<GpuDrawArgs>()) as u64;
        let indirect_size = (n * std::mem::size_of::<vk::DrawIndexedIndirectCommand>()) as u64;
        let status_size = (n * std::mem::size_of::<u32>()) as u64;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let object_buf =
            alloc.create_buffer(object_size, vk::BufferUsageFlags::STORAGE_BUFFER, host)?;
        let draw_args_buf =
            alloc.create_buffer(args_size, vk::BufferUsageFlags::STORAGE_BUFFER, host)?;
        let indirect_buf = alloc.create_buffer(
            indirect_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let status_buf = alloc.create_buffer(
            status_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Snapshot lighting (so all faces share one set), an EMPTY ProbeSet (count 0
        // so a probe face reflects only the sky), and six per-face view UBOs.
        let light = make_ubo_bytes(alloc, light_bytes(&ctx.uniforms.light_uniforms))?;
        let shadow = make_ubo_bytes(alloc, shadow_bytes(&ctx.shadow.uniforms))?;
        let probeset = make_ubo_bytes(alloc, probeset_bytes(&ProbeSet::EMPTY))?;
        let view_size = std::mem::size_of::<ViewUniforms>() as u64;
        let mut view_bufs = Vec::with_capacity(PROBE_FACE_COUNT);
        for _ in 0..PROBE_FACE_COUNT {
            view_bufs.push(alloc.create_buffer(
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                host,
            )?);
        }

        // A bake Hi-Z set (cull set 1) only when the world runs Hi-Z; written with
        // hiz_enabled = 0 so the pyramid is never sampled. The UBO is kept so it can
        // be freed in `destroy`.
        let mut hiz_ubo: Option<PooledBuffer> = None;

        // One dedicated descriptor pool for the bake's cull + per-face bindless +
        // global + Hi-Z sets.
        let tex_pool = (ctx.textures.len() + ctx.normal_map_textures.len()) as u32;
        let has_hiz = ctx.cull.hiz.is_some();
        // Per face: view + light + shadow + ProbeSet + ClusterParams UBOs.
        let uniform_count = PROBE_FACE_COUNT as u32 * 5 + u32::from(has_hiz);
        // 4 cull SSBOs + one bindless object SSBO per face + the binding-9
        // local-light, binding-11 cluster-list, binding-13 spot-shadow and
        // binding-14 area-light SSBOs, one of each per global set.
        let storage_count = 4 + PROBE_FACE_COUNT as u32 * 5;
        let sampler_count = PROBE_FACE_COUNT as u32
            * (tex_pool + 7 + ctx.descriptors.probe_cube_count)
            + u32::from(has_hiz);
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(uniform_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(storage_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(sampler_count.max(1)),
        ];
        let max_sets = 1 + 2 * PROBE_FACE_COUNT as u32 + u32::from(has_hiz);
        // The per-face bindless sets below come from `cull.bindless_set_layout`
        // and the per-face global sets from `descriptors.global_set_layout`, so
        // this pool has to declare update-after-bind whenever either layout does.
        let mut pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets);
        if ctx.cull.bindless_update_after_bind || ctx.descriptors.global_update_after_bind {
            pool_info = pool_info.flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        }
        let pool = device
            .create_descriptor_pool(&pool_info)
            .map_err(|e| format!("probe descriptor pool: {e}"))?;

        // Cull set (set 0): object / draw-args / indirect / status SSBOs.
        let cull_set = alloc_descriptor_sets(
            device,
            pool.handle(),
            std::slice::from_ref(
                &ctx.cull
                    .cull_set_layout
                    .as_ref()
                    .expect("cull descriptor set layout exists once culling is initialised")
                    .handle(),
            ),
        )?[0];
        write_storage(device, cull_set, 0, object_buf.buffer(), object_size);
        write_storage(device, cull_set, 1, draw_args_buf.buffer(), args_size);
        write_storage(device, cull_set, 2, indirect_buf.buffer(), indirect_size);
        write_storage(device, cull_set, 3, status_buf.buffer(), status_size);

        // Per-face bindless sets (set 1): object SSBO + the shared texture pool
        // array. Only the SSBO is written here; each face's pool array is
        // written from the live pool right before that face records
        // (`write_face_pool`), so a mid-bake streamed swap needs no rewrite of
        // a pending set.
        let bindless_layouts = vec![
            ctx.cull
                .bindless_set_layout
                .as_ref()
                .expect("bindless descriptor set layout exists once culling is initialised")
                .handle();
            PROBE_FACE_COUNT
        ];
        let bindless_sets = alloc_descriptor_sets(device, pool.handle(), &bindless_layouts)?;
        {
            let obj_info = vk::DescriptorBufferInfo::default()
                .buffer(object_buf.buffer())
                .offset(0)
                .range(object_size);
            let writes: Vec<vk::WriteDescriptorSet> = bindless_sets
                .iter()
                .map(|&set| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&obj_info))
                })
                .collect();
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        // Bake Hi-Z set (cull set 1), hiz_enabled = 0.
        let hiz_set = if let Some(hiz) = ctx.cull.hiz.as_ref() {
            let params = CullHizParams {
                prev_view_proj: [[0.0; 4]; 4],
                hiz_size: [1.0, 1.0],
                hiz_mip_count: 1,
                hiz_enabled: 0,
            };
            let ubo = make_ubo_bytes(alloc, hiz_params_bytes(&params))?;
            let (view, sampler) = hiz.read_set_sources();
            let layout = hiz.read_set_layout.handle();
            let set =
                alloc_descriptor_sets(device, pool.handle(), std::slice::from_ref(&layout))?[0];
            let img = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(sampler);
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(ubo.buffer())
                .offset(0)
                .range(std::mem::size_of::<CullHizParams>() as u64);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&img)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&ubo_info)),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            hiz_ubo = Some(ubo);
            Some(set)
        } else {
            None
        };

        // Six per-face global sets (set 0 of the bindless main pass): the face view
        // + shared snapshot lighting + env cubes + the SSAO white fallback + an
        // EMPTY ProbeSet + the sky-filled probe cube array. Mirrors init.rs.
        let layouts: Vec<_> = (0..PROBE_FACE_COUNT)
            .map(|_| ctx.descriptors.global_set_layout.handle())
            .collect();
        let global_sets = alloc_descriptor_sets(device, pool.handle(), &layouts)?;
        let probe_cube_sky: Vec<vk::DescriptorImageInfo> = (0..ctx.descriptors.probe_cube_count)
            .map(|_| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(ctx.env_map.prefilter.view)
                    .sampler(ctx.cube_sampler.handle())
            })
            .collect();
        for (face, &set) in global_sets.iter().enumerate() {
            let view_info = buf_info(view_bufs[face].buffer(), view_size);
            let light_info = buf_info(light.buffer(), std::mem::size_of::<LightUniforms>() as u64);
            let shadow_info = buf_info(
                shadow.buffer(),
                std::mem::size_of::<ShadowUniforms>() as u64,
            );
            let probeset_info = buf_info(probeset.buffer(), std::mem::size_of::<ProbeSet>() as u64);
            let shadow_img = img_info(ctx.shadow.map.view, ctx.shadow.sampler.handle());
            let irr_img = img_info(ctx.env_map.irradiance.view, ctx.cube_sampler.handle());
            let pre_img = img_info(ctx.env_map.prefilter.view, ctx.cube_sampler.handle());
            let ssao_img = img_info(ctx.ssao_white.view, ctx.linear_sampler.handle());
            let writes = [
                ubo_write(set, 0, &view_info),
                ubo_write(set, 1, &light_info),
                ubo_write(set, 2, &shadow_info),
                sampler_write(set, 3, &shadow_img),
                sampler_write(set, 4, &irr_img),
                sampler_write(set, 5, &pre_img),
                sampler_write(set, 6, &ssao_img),
                ubo_write(set, 7, &probeset_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(PROBE_CUBE_ARRAY_BINDING)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&probe_cube_sky),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            // Binding 9: the shared static per-scene local-light SSBO.
            write_storage(
                device,
                set,
                LOCAL_LIGHT_SSBO_BINDING,
                ctx.uniforms.local_light_buffer.buffer(),
                ctx.uniforms.local_light_size,
            );
            // Bindings 10 + 11: the `use_clusters = 0` ClusterParams (a cube face
            // does not match the main camera's grid) + the cluster lists, bound
            // because the forward shader references them unconditionally.
            let cluster_params_info = vk::DescriptorBufferInfo::default()
                .buffer(ctx.light_cull.unclustered_buffer.buffer())
                .offset(0)
                .range(std::mem::size_of::<crate::gfx::render_types::ClusterParams>() as u64);
            let cluster_write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(super::descriptor_layout::CLUSTER_PARAMS_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&cluster_params_info));
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&cluster_write), &[]) };
            write_storage(
                device,
                set,
                super::descriptor_layout::CLUSTER_LIGHT_LIST_SSBO_BINDING,
                ctx.light_cull.cluster_buffer.buffer(),
                super::light_cull::cluster_list_size(),
            );
            // Bindings 12 + 13: the spot shadow depth array + its per-slice
            // projections, bound exactly as the main camera binds them.
            let spot_img = img_info(ctx.spot_shadow.map.view, ctx.shadow.sampler.handle());
            let spot_write = sampler_write(
                set,
                super::descriptor_layout::SPOT_SHADOW_MAP_BINDING,
                &spot_img,
            );
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&spot_write), &[]) };
            write_storage(
                device,
                set,
                super::descriptor_layout::SPOT_SHADOW_DATA_SSBO_BINDING,
                ctx.spot_shadow.data_buffer.buffer(),
                vk::WHOLE_SIZE,
            );
            // Bindings 14..16: the area-light table and its two LTC lookups.
            write_storage(
                device,
                set,
                super::descriptor_layout::AREA_LIGHT_SSBO_BINDING,
                ctx.area_light.buffer.buffer(),
                vk::WHOLE_SIZE,
            );
            let ltc_m = img_info(
                ctx.area_light.ltc_matrix.view,
                ctx.area_light.sampler.handle(),
            );
            let ltc_g = img_info(
                ctx.area_light.ltc_magnitude.view,
                ctx.area_light.sampler.handle(),
            );
            let ltc_writes = [
                sampler_write(set, super::descriptor_layout::LTC_MATRIX_BINDING, &ltc_m),
                sampler_write(set, super::descriptor_layout::LTC_MAGNITUDE_BINDING, &ltc_g),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&ltc_writes, &[]) };
        }

        // Six readback buffers (one RGBA16F face each, tightly packed).
        let readback_size = (size as u64) * (size as u64) * 8;
        let mut readback_bufs = Vec::with_capacity(PROBE_FACE_COUNT);
        for _ in 0..PROBE_FACE_COUNT {
            readback_bufs.push(alloc.create_buffer(
                readback_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                host,
            )?);
        }

        Ok(BakeResources {
            color,
            _depth: depth,
            resolve,
            framebuffer,
            object_buf,
            draw_args_buf,
            indirect_buf,
            _status_buf: status_buf,
            _pool: pool,
            cull_set,
            bindless_sets,
            hiz_set,
            _hiz_ubo: hiz_ubo,
            global_sets,
            view_bufs,
            _light: light,
            _shadow: shadow,
            _probeset: probeset,
            readback_bufs,
        })
    }

    fn destroy(self, _device: &VkDevice) {
        // The images and pooled buffers retire through the allocator when this
        // drops; only the framebuffer and the descriptor pool are destroyed by
        // hand (the pool frees every set allocated from it).
    }
}

// Create a HOST_VISIBLE uniform buffer holding `bytes`, persistently mapped.
fn make_ubo_bytes(
    alloc: &super::allocator::DeviceAllocator,
    bytes: &[u8],
) -> Result<PooledBuffer, String> {
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let buf = alloc.create_buffer(
        bytes.len() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        host,
    )?;
    buf.write_bytes(0, bytes);
    Ok(buf)
}

fn light_bytes(u: &crate::gfx::render_types::LightUniforms) -> &[u8] {
    // SAFETY: `LightUniforms` is `#[repr(C)]` over 4-byte scalars and fixed-size arrays of them, so
    // it has no padding and every byte is initialised; the slice borrows it and does not outlive
    // it.
    unsafe {
        std::slice::from_raw_parts(
            u as *const _ as *const u8,
            std::mem::size_of::<crate::gfx::render_types::LightUniforms>(),
        )
    }
}

fn shadow_bytes(u: &crate::gfx::render_types::ShadowUniforms) -> &[u8] {
    // SAFETY: `ShadowUniforms` is `#[repr(C)]` over 4-byte scalars and fixed-size arrays of them,
    // so it has no padding and every byte is initialised; the slice borrows it and does not outlive
    // it.
    unsafe {
        std::slice::from_raw_parts(
            u as *const _ as *const u8,
            std::mem::size_of::<crate::gfx::render_types::ShadowUniforms>(),
        )
    }
}

fn probeset_bytes(p: &ProbeSet) -> &[u8] {
    bytemuck::bytes_of(p)
}

fn hiz_params_bytes(p: &CullHizParams) -> &[u8] {
    // SAFETY: `CullHizParams` is `#[repr(C)]` over 4-byte scalars and fixed-size arrays of them, so
    // it has no padding and every byte is initialised; the slice borrows it and does not outlive
    // it.
    unsafe {
        std::slice::from_raw_parts(
            p as *const _ as *const u8,
            std::mem::size_of::<CullHizParams>(),
        )
    }
}

fn buf_info(buffer: vk::Buffer, range: u64) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(range)
}

fn img_info(view: vk::ImageView, sampler: vk::Sampler) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view)
        .sampler(sampler)
}

fn ubo_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(info))
}

fn sampler_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorImageInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(info))
}

fn write_storage(
    device: &VkDevice,
    set: vk::DescriptorSet,
    binding: u32,
    buffer: vk::Buffer,
    range: u64,
) {
    let info = buf_info(buffer, range);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// Decode a tightly-packed `R16G16B16A16_SFLOAT` probe-cube face (the format the
// bake renders + copies back into a host buffer) to linear f32 RGBA, row major.
// The bake's `cmd_copy_image_to_buffer` uses `buffer_row_length(0)`, so the
// readback is tightly packed (8 bytes per texel, no row padding) -- unlike
// DirectX, whose `CopyTextureRegion` footprint is 256-byte-row-aligned and needs
// an explicit unpad. The six decoded faces feed
// `reflection_probe::build_probe_payload`, which wants each as
// `face_size * face_size` RGBA f32 in row-major order. Mirrors the decode half of
// `directx/probe.rs::read_face_rgba_f32`.
#[allow(dead_code)] // consumed by the probe capture-pass readback (next slice).
fn decode_probe_face_rgba16f(raw: &[u8], face_size: u32) -> Vec<f32> {
    let texels = (face_size as usize) * (face_size as usize);
    let mut out = vec![0.0f32; texels * 4];
    for (texel, px) in raw.chunks_exact(8).take(texels).enumerate() {
        let half = |o: usize| f16_to_f32(u16::from_le_bytes([px[o], px[o + 1]]));
        let base = texel * 4;
        out[base] = half(0);
        out[base + 1] = half(2);
        out[base + 2] = half(4);
        out[base + 3] = half(6);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // The readback decode unpacks tightly-packed RGBA16F (4 halfs per texel, no
    // row padding) into row-major f32, the layout `build_probe_payload` consumes.
    #[test]
    fn decode_probe_face_unpacks_tightly_packed_rgba16f() {
        // A 2x2 face = 4 texels. Each texel is four little-endian halfs.
        let texels: [[u16; 4]; 4] = [
            [0x3c00, 0x3800, 0x0000, 0x3c00], // (1.0, 0.5, 0.0, 1.0)
            [0x4000, 0xbc00, 0x0000, 0x3c00], // (2.0, -1.0, 0.0, 1.0)
            [0x0000, 0x0000, 0x0000, 0x0000], // (0.0, 0.0, 0.0, 0.0)
            [0x3800, 0x3800, 0x3800, 0x3c00], // (0.5, 0.5, 0.5, 1.0)
        ];
        let mut raw = Vec::new();
        for t in texels {
            for h in t {
                raw.extend_from_slice(&h.to_le_bytes());
            }
        }
        let out = decode_probe_face_rgba16f(&raw, 2);
        assert_eq!(out.len(), 16, "2x2 face decodes to 4 RGBA texels");
        assert_eq!(&out[0..4], &[1.0, 0.5, 0.0, 1.0]);
        assert_eq!(&out[4..8], &[2.0, -1.0, 0.0, 1.0]);
        assert_eq!(&out[8..12], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&out[12..16], &[0.5, 0.5, 0.5, 1.0]);
    }

    // A capture routes every record into region 0. Getting this wrong is
    // invisible in a screenshot of most worlds -- it only drops the records whose
    // material carries a world shader -- so it is pinned rather than eyeballed.
    #[test]
    fn a_capture_cull_routes_every_record_into_one_region() {
        let p = capture_cull_params(&Frustum::from_view_projection(IDENTITY), [0.0; 3], 12);
        assert_eq!(p.bucket_count, 1, "one region, whatever the world declares");
        assert_eq!(p.object_count, 12);
        assert_eq!(p.bucket_stride, 12, "stride names the region capacity");
    }

    // Every byte the shader reads must be written. `cmd_push_constants` takes a
    // slice, so a short one leaves the tail undefined -- and push constants do not
    // carry across command buffers, so the capture cull (which runs on a later
    // pass's buffer than the main cull) reads whatever the driver left there.
    // That is how the mirror render lost its draws.
    #[test]
    fn the_capture_push_covers_the_whole_shader_block() {
        let p = capture_cull_params(&Frustum::from_view_projection(IDENTITY), [1.0, 2.0, 3.0], 4);
        // SAFETY: `repr(C)`, read as its own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &p as *const CullParams as *const u8,
                std::mem::size_of::<CullParams>(),
            )
        };
        assert_eq!(bytes.len(), 120, "cull.comp's push_constant block is 120 B");
        // The two routing fields live in the last 8 bytes: the exact span a
        // 112-byte push left undefined.
        assert_eq!(
            &bytes[112..116],
            &1u32.to_le_bytes(),
            "bucket_count written"
        );
        assert_eq!(
            &bytes[116..120],
            &4u32.to_le_bytes(),
            "bucket_stride written"
        );
    }

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
}
