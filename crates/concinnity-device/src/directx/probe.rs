// src/directx/probe.rs
//
// Scene-captured reflection probes on DirectX. Each declared `ReflectionProbe`
// (or an auto-seeded grid when a world declares none) is baked into its own cube,
// DISTINCT from `env_map`: the specular reflection term box-projects against the
// probe's influence box and samples its cube, so glossy surfaces reflect the
// actual surrounding geometry instead of the imported HDR sky, while the skybox +
// diffuse irradiance keep sampling `env_map` so the visible sky is never replaced.
//
// The cube math + the staggered-bake state machine are backend-agnostic
// (`crate::gfx::reflection_probe`); this module drives the GPU capture, mirroring
// `crate::metal::probe`. The bake is STAGGERED + ASYNCHRONOUS across frames so the
// render thread never blocks: one probe is in flight at a time, its six cube faces
// submitted one per frame into a capture cube, then convolved into the probe cube
// by the compute kernels in `probe_prefilter.slang`. Nothing is read back and no
// convolution runs on the CPU.
//
// DirectX simplification vs Metal: a per-face fence VALUE gives ordered GPU
// completion for free (the queue is FIFO), so there is no completion handler / atomic
// -- a face is done when `frame_sync.fence` reaches the value signalled after it. The
// bake never calls `wait_idle` (that would reintroduce a multi-hundred-ms freeze);
// the convolution is deferred until the fence reaches the last face's value.
//
// Each probe passes through three phases (`gfx::reflection_probe::BakePhase`, driven by
// the pure `next_bake_action` transition table called once per pipeline slot per frame):
//   * Rendering    -- six cube faces submitted to the GPU (one per frame) into a RESERVED
//                     ring slot (`bake_ring_slot`) the frame never overwrites, each
//                     copied into its slice of the capture cube.
//   * Prefiltering -- the convolution runs as compute dispatches: the clamped mirror
//                     mip plus the capture's source pyramid in the first frame (all
//                     cheap), then ONE GGX mip per frame after it.
//   * (install)    -- the finished cube is installed into `probe.maps` + `probe.set`.
//                     No upload: the cube was written in place.
//
// Known V1 simplifications (documented intentionally; mirror Metal where noted):
//   * Static + instanced geometry only -- skinned meshes are not captured into the
//     probe (no per-bake deformed buffer yet). They still receive probe reflections.
//   * Single bounce + cold-first-frame lighting (the shadow map may be unpopulated when
//     a probe bakes on an early frame), exactly like Metal.

use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use super::context::{DxContext, FRAMES};
use super::probe_prefilter::PrefilterGpu;
use super::texture::{
    HDR_FORMAT, create_buffer, create_hdr_color_target, create_hdr_resolve_target,
    transition_barrier,
};
use crate::gfx::reflection_probe::{self, BakeAction, BakePhase, BakeSignals, PrefilterPlan};

// What a runtime capture bakes: face size, mip count, GGX sample count and firefly
// clamp, shared with the Metal and Vulkan backends (and with the build-time CPU
// convolution's roughness ramp) so a probe looks the same whichever backend
// captured it.
const PLAN: PrefilterPlan = PrefilterPlan::RUNTIME;
// Captured cube-face resolution (mip 0 of the prefilter chain).
const PROBE_FACE_SIZE: u32 = PLAN.face_size();
// Cube faces per probe, rendered one per frame.
const PROBE_FACE_COUNT: usize = 6;

// A baked prefilter cube, one per installed probe. Distinct from `env_map`;
// sampled only by the specular reflection term. The SRV into the probe cube array
// is written when the array is bound to the shaders.
pub(in crate::directx) struct ProbeCube {
    #[expect(
        dead_code,
        reason = "held to keep the prefilter cube resident; the array SRV is what the shaders bind"
    )]
    pub(in crate::directx) prefilter: ID3D12Resource,
}

// The GPU resources + state of one in-flight capture. The six faces share one
// (MSAA) colour + depth target reused across frames; each face has its own view
// CBV + command allocator/list (held until the convolution starts, so the fence
// guarantees their GPU work has retired before they drop).
pub(in crate::directx) struct RenderingBake {
    index: usize,
    placement: reflection_probe::ProbePlacement,
    // Next of `PROBE_FACE_COUNT` faces to submit (one per frame).
    cursor: usize,
    eye: [f32; 3],
    near: f32,
    far: f32,
    sample_count: u32,
    // Reused across the six faces.
    color: ID3D12Resource,
    _depth: ID3D12Resource,
    resolve: Option<ID3D12Resource>,
    _rtv_heap: ID3D12DescriptorHeap,
    _dsv_heap: ID3D12DescriptorHeap,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    // Per-face: a 160-byte ViewUniforms CBV (kept mapped) + its GVA.
    _view_cbvs: Vec<PooledBuffer>,
    view_gvas: Vec<u64>,
    // Per-capture light + shadow snapshots (so the six faces share one consistent
    // lighting set, decoupled from the frame's per-frame CBV writes).
    light_gva: u64,
    shadow_gva: u64,
    _light_cbv: PooledBuffer,
    _shadow_cbv: PooledBuffer,
    // The capture cube each face is copied into, and the probe cube the convolution
    // will write. Allocated with the capture because face 0 copies into it, and
    // handed to the prefiltering slot once every face has landed.
    prefilter: PrefilterGpu,
    // One fresh allocator + list per submitted face, held until the convolution
    // starts (the fence proves their GPU work retired before they drop).
    cmd_allocs: Vec<ID3D12CommandAllocator>,
    cmd_lists: Vec<ID3D12GraphicsCommandList>,
    // Fence value signalled after the LAST face; the convolution waits for the
    // shared `frame_sync.fence` to reach it.
    last_fence_value: u64,
}

// A finished capture convolving into its cube on the GPU, one destination mip per
// frame. Holds both cubes plus the allocator and list of every dispatch it has
// submitted, which install drops once the fence covers them. Nothing else bakes
// while this slot is full: the dispatches address their cubes through the one
// reserved descriptor block a starting capture would rewrite.
pub(in crate::directx) struct PrefilteringBake {
    index: usize,
    placement: reflection_probe::ProbePlacement,
    gpu: PrefilterGpu,
    // Next destination mip to convolve. Starts at 1: mip 0 is the clamped copy,
    // dispatched with the source pyramid when this slot is filled.
    cursor: u32,
    cmd_allocs: Vec<ID3D12CommandAllocator>,
    cmd_lists: Vec<ID3D12GraphicsCommandList>,
    // Fence value signalled after the LAST dispatch submitted so far.
    last_fence_value: u64,
}

// Colour + depth attachments for a probe-face / planar mirror capture.
#[derive(Clone, Copy)]
pub(in crate::directx) struct FaceTargets {
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
}

// GPU virtual addresses of the per-capture view / light / shadow constant buffers.
#[derive(Clone, Copy)]
pub(in crate::directx) struct FaceUniforms {
    pub view_gva: u64,
    pub light_gva: u64,
    pub shadow_ubo_gva: u64,
}

// The indirect draw for one capture region: the command buffer, its byte offset,
// and the per-object buffer address for bindless rendering.
#[derive(Clone, Copy)]
pub(in crate::directx) struct IndirectDraw<'a> {
    pub indirect: &'a ID3D12Resource,
    pub indirect_offset: u32,
    pub object_gva: u64,
}

// Render-target dimensions for the capture.
#[derive(Clone, Copy)]
pub(in crate::directx) struct FaceExtent {
    pub width: u32,
    pub height: u32,
}

impl DxContext {
    // Set the reflection-probe placements (declared `ReflectionProbe` assets,
    // converted to `ProbePlacement`s by the graphics system). An empty list
    // auto-seeds a grid from the scene bounds, so existing scenes still get local
    // reflections without authoring. Resets the staggered bake; capped at
    // `MAX_PROBES`.
    pub(super) fn set_reflection_probes(&mut self, declared: &[reflection_probe::ProbePlacement]) {
        use concinnity_core::render::uniforms::MAX_PROBES;
        use concinnity_core::render::uniforms::ProbeSet;
        let mut placements: Vec<reflection_probe::ProbePlacement> = if declared.is_empty() {
            match self.scene_world_bounds() {
                Some((mn, mx)) => {
                    // Object AABBs as occupancy so a probe is not auto-captured from
                    // inside a wall; skip degenerate (non-finite) boxes.
                    let occupancy: Vec<([f32; 3], [f32; 3])> = self
                        .draw
                        .objects
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
        if placements.len() > MAX_PROBES {
            tracing::warn!(
                "reflection probes: {} placements, capping at MAX_PROBES={}",
                placements.len(),
                MAX_PROBES
            );
            placements.truncate(MAX_PROBES);
        }
        // A re-placement mid-flight (rare -- this is normally an init-time call) would
        // free capture resources the GPU may still be reading. Idle the GPU first so
        // the dropped command lists + reserved-slot buffers are safe to release. The
        // first call has nothing in flight, so it never idles.
        self.abandon_in_flight_bakes();
        self.probe.placements = placements;
        self.probe.maps.clear();
        self.probe.set = ProbeSet::EMPTY;
        self.probe.bake_queue = reflection_probe::ProbeBakeQueue::new(self.probe.placements.len());
    }

    // The reserved transient-ring slot the asynchronous bake builds its bindless
    // buffers into: one past the frame's range `[0, FRAMES)`. The frame never writes
    // this slot, so the bake's CPU-written buffers stay valid across its capture.
    // The cull rings are sized `FRAMES + 1` in `init/pipelines.rs` to make room.
    fn bake_ring_slot(&self) -> usize {
        FRAMES
    }

    // GPU descriptor handle of the reflection-probe cube array table base (root param
    // [10] of the bindless main pass). The MAX_PROBES contiguous cube SRVs start here.
    pub(in crate::directx) fn probe_cube_table_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr
                + (self.descriptors.probe_cube_base_slot * self.descriptors.srv_descriptor_size)
                    as u64,
        }
    }

    // CPU descriptor handle of probe cube array slot `i` (for writing a baked cube's
    // SRV into the array at install time).
    fn probe_cube_slot_cpu(&self, i: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let base = unsafe {
            self.descriptors
                .srv_heap
                .GetCPUDescriptorHandleForHeapStart()
        };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr
                + (self.descriptors.probe_cube_base_slot + i)
                    * self.descriptors.srv_descriptor_size,
        }
    }

    // Whether the capture path can run: the bindless GPU-driven cull must be active
    // (the capture renders through the indirect command buffer) and the reserved ring
    // slot must exist.
    fn probe_capture_supported(&self) -> bool {
        self.cull.main_bindless_pso.is_some()
            && self.cull.cull_pso.is_some()
            && self.cull.object_buffer_resources.len() > FRAMES
            && self.cull.draw_args_buffer_resources.len() > FRAMES
            && self.cull.indirect_cmd_buffers.len() > FRAMES
    }

    // Advance the asynchronous reflection-probe bake by one step. Called every frame
    // from `draw_frame` after the frame-slot fence wait; cheap once the queue drains.
    // Drives the pure `next_bake_action` transition table over two pipelined slots.
    // Non-fatal: a failure abandons the remaining bakes, keeping what baked.
    pub(super) fn bake_pending_probes(
        &mut self,
        elapsed: f32,
        near: f32,
        far: f32,
    ) -> Result<(), String> {
        let _ = elapsed;
        if !self.probe.bake_queue.pending()
            && self.probe.rendering.is_none()
            && self.probe.prefiltering.is_none()
        {
            return Ok(());
        }
        // Permanent ineligibility: a probe only improves on a real environment, and
        // the capture renders through the bindless cull. Abandon the queue rather than
        // re-checking forever.
        if self.env_map.prefilter_mip_count <= 1
            || !self.probe_capture_supported()
            || self.probe.prefilter.is_none()
        {
            self.abandon_in_flight_bakes();
            self.probe.bake_queue.abort();
            return Ok(());
        }

        // Prefiltering slot first: convolve one mip, or install the finished cube,
        // freeing the slot so the rendering slot can hand its capture over this same
        // frame.
        let prefiltering_occupied = self.probe.prefiltering.is_some();
        let more_mips = self
            .probe
            .prefiltering
            .as_ref()
            .is_some_and(|p| p.cursor < PLAN.mips());
        // The install drops each dispatch's allocator and list, so unlike Metal it
        // must wait for the GPU to retire them, not just for them to be submitted.
        // SAFETY: the fence was created from this device; the query only reads.
        let completed = unsafe { self.frame_sync.fence.GetCompletedValue() };
        let mips_done = self
            .probe
            .prefiltering
            .as_ref()
            .is_some_and(|p| completed >= p.last_fence_value);
        match reflection_probe::next_bake_action(
            if prefiltering_occupied {
                BakePhase::Prefiltering
            } else {
                BakePhase::Idle
            },
            BakeSignals {
                more_mips,
                mips_done,
                ..Default::default()
            },
        ) {
            BakeAction::PrefilterMip => {
                if let Err(e) = self.probe_prefilter_next_mip() {
                    self.fail_bake(e);
                    return Ok(());
                }
            }
            BakeAction::Install => {
                if let Err(e) = self.probe_install() {
                    self.fail_bake(e);
                    return Ok(());
                }
            }
            _ => {}
        }
        let prefiltering_free = self.probe.prefiltering.is_none();

        // Rendering slot: submit one face per frame; once all six are done on the GPU
        // (the fence reached the last face's value) AND the prefiltering slot is free,
        // hand the capture over; or start the next placement.
        let rendering_occupied = self.probe.rendering.is_some();
        let more_faces = self
            .probe
            .rendering
            .as_ref()
            .is_some_and(|r| r.cursor < PROBE_FACE_COUNT);
        // SAFETY: the fence was created from this device; the query only reads.
        let completed = unsafe { self.frame_sync.fence.GetCompletedValue() };
        let done = self
            .probe
            .rendering
            .as_ref()
            .is_some_and(|r| r.cursor >= PROBE_FACE_COUNT && completed >= r.last_fence_value);
        // Transient ineligibility: geometry may still be streaming. A zero cull keeps
        // the queue cursor so a later frame retries rather than baking an empty cube.
        //
        // `prefiltering_free` is a DirectX-only term: both cubes of a bake are
        // addressed through ONE reserved SRV-heap block, written by `PrefilterGpu::new`
        // at the start of a capture, so starting a second bake while the first is
        // still convolving would rewrite the descriptors its remaining dispatches
        // bind. Metal and Vulkan give each bake its own resources and keep the
        // capture / convolution pipelined.
        let eligible = self.cull_count() > 0 && prefiltering_free;
        match reflection_probe::next_bake_action(
            if rendering_occupied {
                BakePhase::Rendering
            } else {
                BakePhase::Idle
            },
            BakeSignals {
                faces_done: done && prefiltering_free,
                queue_pending: self.probe.bake_queue.pending(),
                eligible,
                more_faces,
                ..Default::default()
            },
        ) {
            BakeAction::RenderFace => {
                if let Err(e) = self.probe_render_next_face() {
                    self.fail_bake(e);
                }
            }
            BakeAction::StartPrefilter => {
                if let Err(e) = self.probe_begin_prefilter() {
                    self.fail_bake(e);
                }
            }
            BakeAction::StartNext => {
                if let Err(e) = self.probe_start_next(near, far) {
                    self.fail_bake(e);
                }
            }
            BakeAction::PrefilterMip | BakeAction::Install | BakeAction::Idle => {}
        }
        Ok(())
    }

    // Drop whatever both bake slots hold, after idling the device: their command
    // lists may still be executing, and every payload owns resources a submission
    // could still name.
    fn abandon_in_flight_bakes(&mut self) {
        if self.probe.rendering.is_some() || self.probe.prefiltering.is_some() {
            self.wait_idle();
        }
        self.probe.rendering = None;
        self.probe.prefiltering = None;
    }

    // Abandon the rest of the bake after an unrecoverable error, keeping the cubes
    // already installed. The queue cursor advanced when the current probe started, so
    // aborting (cursor -> end) keeps `probe.maps` aligned with the placement list.
    fn fail_bake(&mut self, e: String) {
        tracing::warn!(
            "reflection probe bake failed, keeping {} baked: {e}",
            self.probe.maps.len()
        );
        // Idle before dropping either slot's GPU resources: their command lists may
        // still be executing. A bake failure is rare (allocation / device error), so
        // the one-time stall is acceptable.
        self.abandon_in_flight_bakes();
        self.probe.bake_queue.abort();
    }

    // Begin baking the next pending placement: build the reserved-slot bindless
    // buffers (object + draw-args, frustum-independent) ONCE, and allocate the capture
    // targets + per-face view CBVs + both cubes. No face is submitted here; the
    // faces follow one per frame via `probe_render_next_face`.
    fn probe_start_next(&mut self, near: f32, far: f32) -> Result<(), String> {
        let Some(index) = self.probe.bake_queue.take_next() else {
            return Ok(());
        };
        let placement = self.probe.placements[index];
        let eye = placement.position;
        let slot = self.bake_ring_slot();

        // Build the reserved-slot bindless buffers once: the per-object record buffer
        // and the draw-args buffer (LOD by distance from the probe eye). Both are
        // frustum-independent, reused by every face's cull.
        self.build_object_buffer(slot);
        self.build_draw_args_buffer(slot, eye);

        let alloc = &self.alloc;
        let device = &self.device;
        let sample_count = self.hdr.msaa_samples.max(1);
        let size = PROBE_FACE_SIZE;

        // One MSAA (or single-sample) colour + depth pair, reused across the six faces.
        let rtv_heap = create_rtv_heap(device)?;
        let dsv_heap = create_dsv_heap(device)?;
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let dsv = unsafe { dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        let color =
            create_hdr_color_target(device, size, size, sample_count, rtv, self.view.clear_color)?;
        let depth = create_bake_depth(device, size, sample_count, dsv)?;
        // A single-sample resolve target only when MSAA is on.
        let resolve = if sample_count > 1 {
            Some(create_hdr_resolve_target(device, size, size)?)
        } else {
            None
        };

        // Snapshot the frame's light + shadow uniforms into bake-owned CBVs so all six
        // faces share one temporally-consistent lighting set, and so the capture does
        // not read `light_ubo` / `shadow_ubo[frame]` while `record_frame` (which runs
        // after this) overwrites them on the same frame -- a CPU/GPU race on a mapped
        // buffer. The capture's lighting is the env live when it started.
        // SAFETY: `LightUniforms` is `#[repr(C)]` with explicit pad fields and no implicit padding
        // (pinned by the layout tests in `render_types.rs`), so all `size_of` bytes are
        // initialized, and the borrow keeps them live for the snapshot copy below.
        let light_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.uniforms.light_uniforms as *const crate::gfx::render_types::LightUniforms
                    as *const u8,
                std::mem::size_of::<crate::gfx::render_types::LightUniforms>(),
            )
        };
        let (light_cbv, light_gva) = make_snapshot_cbv(alloc, light_bytes)?;
        // SAFETY: `ShadowUniforms` is `#[repr(C)]` with an explicit trailing pad and no implicit
        // padding (pinned by the layout tests in `render_types.rs`), so all `size_of` bytes are
        // initialized, and the borrow keeps them live for the snapshot copy below.
        let shadow_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.shadow.uniforms as *const crate::gfx::render_types::ShadowUniforms
                    as *const u8,
                std::mem::size_of::<crate::gfx::render_types::ShadowUniforms>(),
            )
        };
        let (shadow_cbv, shadow_gva) = make_snapshot_cbv(alloc, shadow_bytes)?;

        // Per-face ViewUniforms CBVs, the only per-face binding.
        // The capture renders with the real env IBL (so the scene carries ambient
        // lighting), exactly like the main pass minus the SSR/RT resolve.
        let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;
        let mut view_cbvs = Vec::with_capacity(PROBE_FACE_COUNT);
        let mut view_gvas = Vec::with_capacity(PROBE_FACE_COUNT);
        for face in 0..PROBE_FACE_COUNT {
            let vp = reflection_probe::face_view_projection(eye, face, near, far);
            let view_mat = reflection_probe::face_view_matrix(eye, face);
            let view = super::draw::ViewUniforms {
                vp,
                view: view_mat,
                elapsed: 0.0,
                // No reflection resolve runs over the probe cube, so the forward
                // probe specular is the only reflection source here; keep it.
                reflections_enabled: 0.0,
                cam_pos: [eye[0], eye[1], eye[2]],
                prefilter_mip_count,
                // A probe capture is always lit, whatever the viewport shows.
                shade_mode: 0.0,
                _end_pad: 0.0,
            };
            let cbv = create_buffer(
                alloc,
                256,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { cbv.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("probe: map view cbv: {e}"))?;
            // SAFETY: the buffer is 256 bytes; ViewUniforms is 160.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &view as *const super::draw::ViewUniforms as *const u8,
                    ptr as *mut u8,
                    std::mem::size_of::<super::draw::ViewUniforms>(),
                );
            }
            view_gvas.push(com::gpu_va(&cbv));
            view_cbvs.push(cbv);
        }

        // The capture cube each face is copied into, and the probe cube the
        // convolution writes. Allocated with the capture rather than at the
        // convolution's start: face 0 copies into the cube, so it has to exist
        // before the first face records.
        let prefilter = PrefilterGpu::new(self, &PLAN)?;

        self.probe.rendering = Some(RenderingBake {
            index,
            placement,
            cursor: 0,
            eye,
            near,
            far,
            sample_count,
            color,
            _depth: depth,
            resolve,
            _rtv_heap: rtv_heap,
            _dsv_heap: dsv_heap,
            rtv,
            dsv,
            _view_cbvs: view_cbvs,
            view_gvas,
            light_gva,
            shadow_gva,
            _light_cbv: light_cbv,
            _shadow_cbv: shadow_cbv,
            prefilter,
            cmd_allocs: Vec::with_capacity(PROBE_FACE_COUNT),
            cmd_lists: Vec::with_capacity(PROBE_FACE_COUNT),
            last_fence_value: 0,
        });
        Ok(())
    }

    // Submit the in-flight capture's next cube face (one per frame): a fresh command
    // list that culls the face frustum into the reserved slot, renders the bindless
    // static + instance geometry into the face target, (resolves +) copies it into its
    // slice of the capture cube, then signals a fence value. The last face's value is
    // what the convolution waits for.
    fn probe_render_next_face(&mut self) -> Result<(), String> {
        let slot = self.bake_ring_slot();
        let (face, eye, near, far, sample_count, view_gva, light_gva, shadow_gva) = {
            let bake = self
                .probe
                .rendering
                .as_ref()
                .ok_or("probe: render face with no capture in flight")?;
            (
                bake.cursor,
                bake.eye,
                bake.near,
                bake.far,
                bake.sample_count,
                bake.view_gvas[bake.cursor],
                bake.light_gva,
                bake.shadow_gva,
            )
        };

        let vp = reflection_probe::face_view_projection(eye, face, near, far);
        let frustum = crate::gfx::frustum::Frustum::from_view_projection(vp);

        // A fresh allocator + list per face, held until the fence proves the face
        // retired, so no in-flight allocator is ever reset.
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        let alloc: ID3D12CommandAllocator = unsafe {
            self.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .map_err(|e| format!("probe: face allocator: {e}"))?;
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        let cmd: ID3D12GraphicsCommandList = unsafe {
            self.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
        }
        .map_err(|e| format!("probe: face cmd list: {e}"))?;
        // Register the recording on the bake before anything can fail: once it is
        // submitted, only `abandon_in_flight_bakes` idling the device makes it safe to
        // drop, and that reaches it only through the bake.
        if let Some(bake) = self.probe.rendering.as_mut() {
            bake.cmd_allocs.push(alloc);
            bake.cmd_lists.push(cmd.clone());
        }

        // Cull this face's frustum into the reserved indirect buffer, then render.
        self.encode_probe_cull(&cmd, slot, &frustum, eye);
        let (rtv, dsv) = {
            let bake = self
                .probe
                .rendering
                .as_ref()
                .expect("probe bake targets are live while a bake is recording");
            (bake.rtv, bake.dsv)
        };
        let indirect = &self.cull.indirect_cmd_buffers[slot];
        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[slot]);
        self.encode_main_into_face(
            &cmd,
            FaceTargets { rtv, dsv },
            FaceUniforms {
                view_gva,
                light_gva,
                shadow_ubo_gva: shadow_gva,
            },
            IndirectDraw {
                indirect,
                indirect_offset: 0,
                object_gva,
            },
            FaceExtent {
                width: PROBE_FACE_SIZE,
                height: PROBE_FACE_SIZE,
            },
        );

        // Resolve (MSAA) + copy the face into its slice of the capture cube.
        self.copy_face_to_capture(&cmd, face, sample_count)?;

        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { cmd.Close() }.map_err(|e| format!("probe: face close: {e}"))?;
        let list: ID3D12CommandList =
            windows::core::Interface::cast(&cmd).map_err(|e| format!("probe: face cast: {e}"))?;
        // SAFETY: every command list in the submission is live and closed, and the slice outlives
        // the call.
        unsafe { self.command_queue.ExecuteCommandLists(&[Some(list)]) };

        // Signal a unique fence value on the shared fence; the convolution waits for it.
        let fence_val = self.frame_sync.next_fence_value.get();
        self.frame_sync.next_fence_value.set(fence_val + 1);
        // SAFETY: the fence and the event were created from this device and are live for the call.
        unsafe { self.command_queue.Signal(&self.frame_sync.fence, fence_val) }
            .map_err(|e| format!("probe: face signal: {e}"))?;

        if let Some(bake) = self.probe.rendering.as_mut() {
            bake.last_fence_value = fence_val;
            bake.cursor += 1;
        }
        Ok(())
    }

    // Resolve (when MSAA) + copy the just-rendered face colour into slice `face` of
    // the capture cube. The colour rests in RENDER_TARGET and is restored to it for
    // the next face; the resolve target rests in PIXEL_SHADER_RESOURCE. Face order is
    // the hardware cube order (`gfx::cubemap`), so slice `face` is the face a sampler
    // finds looking that way.
    fn copy_face_to_capture(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        face: usize,
        sample_count: u32,
    ) -> Result<(), String> {
        let bake = self
            .probe
            .rendering
            .as_ref()
            .expect("probe bake targets are live while a bake is recording");
        // Subresource index of mip 0 of array slice `face`, which D3D12 orders
        // mip-major within a slice.
        let dst_subresource = face as u32 * bake.prefilter.mips();
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(bake.prefilter.capture()),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: dst_subresource,
            },
        };
        if sample_count > 1 {
            let resolve = bake
                .resolve
                .as_ref()
                .expect("a multisampled probe bake has a resolve image");
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.ResourceBarrier(&[
                    transition_barrier(
                        &bake.color,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                    ),
                    transition_barrier(
                        resolve,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_RESOLVE_DEST,
                    ),
                ]);
                cmd.ResolveSubresource(resolve, 0, &bake.color, 0, HDR_FORMAT);
                cmd.ResourceBarrier(&[
                    transition_barrier(
                        resolve,
                        D3D12_RESOURCE_STATE_RESOLVE_DEST,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                    transition_barrier(
                        &bake.color,
                        D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                    ),
                ]);
                let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: com::borrowed(resolve),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
                cmd.ResourceBarrier(&[transition_barrier(
                    resolve,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                )]);
            }
        } else {
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.ResourceBarrier(&[transition_barrier(
                    &bake.color,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: com::borrowed(&bake.color),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
                cmd.ResourceBarrier(&[transition_barrier(
                    &bake.color,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                )]);
            }
        }
        Ok(())
    }

    // The GPU has finished the capture (the fence reached the last face's value):
    // free the capture's draw resources (so the next probe can start rendering), take
    // ownership of the two cubes, and submit the cheap half of the convolution -- the
    // firefly-clamped mirror mip plus the capture's source pyramid. The bake moves to
    // the Prefiltering slot with the mip cursor at 1.
    fn probe_begin_prefilter(&mut self) -> Result<(), String> {
        let bake = self
            .probe
            .rendering
            .take()
            .ok_or("probe: convolve with no bake in flight")?;
        let RenderingBake {
            index,
            placement,
            prefilter,
            ..
        } = bake;
        // The capture's draw resources (targets + command lists) drop here; the fence
        // reached `last_fence_value`, so the GPU is done with all of them.

        let mut bake = PrefilteringBake {
            index,
            placement,
            gpu: prefilter,
            cursor: 1,
            cmd_allocs: Vec::with_capacity(PLAN.mips() as usize),
            cmd_lists: Vec::with_capacity(PLAN.mips() as usize),
            last_fence_value: 0,
        };
        // Store the bake whether or not the recording succeeded: it owns both cubes and
        // every list submitted for them, so a failure has to reach `fail_bake`'s idle
        // through the slot rather than dropping them here.
        let result = self.record_prefilter_step(&mut bake, |ctx, cmd, bake| {
            ctx.encode_probe_pyramid(cmd, &bake.gpu, &PLAN)
        });
        self.probe.prefiltering = Some(bake);
        result
    }

    // Convolve one destination mip of the in-flight probe cube (one per frame, so no
    // frame pays the whole convolution). Each dispatch reads the finished pyramid and
    // writes a mip nothing else touches, so consecutive mips need no barrier; the
    // queue's FIFO order puts every one of them after the pyramid build that produced
    // their source.
    fn probe_prefilter_next_mip(&mut self) -> Result<(), String> {
        let mut bake = self
            .probe
            .prefiltering
            .take()
            .ok_or("probe: convolve mip with no bake in flight")?;
        let cursor = bake.cursor;
        // The last mip's list also carries the cube into PIXEL_SHADER_RESOURCE. The
        // install has no list of its own to submit that transition on: it would have
        // to drop that list immediately, and D3D12 does not keep a command allocator
        // alive for the GPU.
        let last = cursor + 1 == PLAN.mips();
        let result = self.record_prefilter_step(&mut bake, |ctx, cmd, bake| {
            ctx.encode_probe_ggx_mip(cmd, &PLAN, cursor)?;
            if last {
                // SAFETY: the command list is in the recording state and the resource the
                // barrier names is live for the call.
                unsafe {
                    cmd.ResourceBarrier(&[transition_barrier(
                        bake.gpu.probe(),
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    )]);
                }
            }
            Ok(())
        });
        bake.cursor += 1;
        self.probe.prefiltering = Some(bake);
        result
    }

    // Record and submit one convolution step on a fresh allocator + list, registering
    // both on the bake so a later failure still reclaims them, and signalling a fence
    // value the install waits for. The shader-visible descriptor heaps are bound
    // first: every dispatch addresses its cubes through the SRV heap.
    fn record_prefilter_step(
        &self,
        bake: &mut PrefilteringBake,
        encode: impl FnOnce(&Self, &ID3D12GraphicsCommandList, &PrefilteringBake) -> Result<(), String>,
    ) -> Result<(), String> {
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        let alloc: ID3D12CommandAllocator = unsafe {
            self.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .map_err(|e| format!("probe: convolve allocator: {e}"))?;
        // SAFETY: as above.
        let cmd: ID3D12GraphicsCommandList = unsafe {
            self.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
        }
        .map_err(|e| format!("probe: convolve cmd list: {e}"))?;
        bake.cmd_allocs.push(alloc);
        bake.cmd_lists.push(cmd.clone());

        // SAFETY: the command list is in the recording state and the heaps it names are live.
        unsafe {
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
        }
        encode(self, &cmd, bake)?;
        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { cmd.Close() }.map_err(|e| format!("probe: convolve close: {e}"))?;
        let list: ID3D12CommandList = windows::core::Interface::cast(&cmd)
            .map_err(|e| format!("probe: convolve cast: {e}"))?;
        // SAFETY: every command list in the submission is live and closed, and the slice outlives
        // the call.
        unsafe { self.command_queue.ExecuteCommandLists(&[Some(list)]) };

        let fence_val = self.frame_sync.next_fence_value.get();
        self.frame_sync.next_fence_value.set(fence_val + 1);
        // SAFETY: the fence was created from this device and is live for the call.
        unsafe { self.command_queue.Signal(&self.frame_sync.fence, fence_val) }
            .map_err(|e| format!("probe: convolve signal: {e}"))?;
        bake.last_fence_value = fence_val;
        Ok(())
    }

    // Every mip is convolved and retired (the last one carried the cube into
    // PIXEL_SHADER_RESOURCE): point this probe's slot in the cube array at it and
    // bump `probe.set.count` so the forward specular samples it. Leaves `env_map` /
    // the sky untouched.
    //
    // Purely CPU work. Nothing is uploaded -- the cube was written in place -- and
    // the dispatch recordings free here, which the fence gate on this transition
    // proved the GPU had finished with.
    fn probe_install(&mut self) -> Result<(), String> {
        let bake = self
            .probe
            .prefiltering
            .take()
            .ok_or("probe: install with no bake in flight")?;
        let mips = bake.gpu.mips();
        let PrefilteringBake {
            index,
            placement: p,
            gpu,
            ..
        } = bake;
        let prefilter = gpu.into_probe_cube();

        // Point this probe's slot in the cube array at the baked cube (it held the sky
        // prefilter until now). The forward shader samples it once `probe.set.count`
        // covers this index.
        //
        // The slot is rewritten in place while up to FRAMES-1 submitted frames still
        // reference it. Both the sky prefilter and the probe cube are live, correctly
        // formatted cubes, so either one an in-flight frame reads shades; only the
        // frame the swap lands in is undefined about which it gets.
        super::texture::write_cube_srv_mips_format(
            &self.device,
            &prefilter,
            mips,
            super::probe_prefilter::PROBE_CUBE_FORMAT,
            self.probe_cube_slot_cpu(index),
        );

        debug_assert_eq!(index, self.probe.maps.len());
        self.probe.maps.push(ProbeCube { prefilter });
        self.probe.set.probes[index] = concinnity_core::render::uniforms::ProbeUniforms {
            box_min: [p.box_min[0], p.box_min[1], p.box_min[2], 1.0],
            box_max: [p.box_max[0], p.box_max[1], p.box_max[2], 0.0],
            probe_pos: [p.position[0], p.position[1], p.position[2], 0.0],
        };
        self.probe.set.count = self.probe.maps.len() as u32;
        tracing::info!(
            "reflection probes: baked {}/{}",
            index + 1,
            self.probe.placements.len()
        );
        Ok(())
    }

    // Render the bindless static + instance geometry into an off-screen target. A
    // thin sibling of `encode_main_pass`'s bindless branch: it clears + targets the
    // RTV/DSV, binds a per-view ViewUniforms CBV, and issues the static + instance
    // prefix `ExecuteIndirect` from `slot`'s indirect buffer. Skinned geometry is not
    // drawn (V1). No SSAO pre-pass, no HDR resolve -- the caller copies / resolves the
    // target out. Shared by the probe-face capture (square face, reserved bake
    // slot's indirect at offset 0) and the planar reflection mirror render
    // (render-resolution target, the planar indirect buffer at the plane's region
    // byte offset, drawn against the frame's object buffer). `indirect_offset` is a
    // byte offset into `indirect` to the region's first command.
    pub(in crate::directx) fn encode_main_into_face(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        targets: FaceTargets,
        uniforms: FaceUniforms,
        draw: IndirectDraw<'_>,
        extent: FaceExtent,
    ) {
        let FaceTargets { rtv, dsv } = targets;
        let FaceUniforms {
            view_gva,
            light_gva,
            shadow_ubo_gva,
        } = uniforms;
        let IndirectDraw {
            indirect,
            indirect_offset,
            object_gva,
        } = draw;
        let FaceExtent { width, height } = extent;
        let bindless_pso = self
            .cull
            .main_bindless_pso
            .as_ref()
            .expect("bindless PSO is live");
        let bindless_root = self
            .cull
            .main_bindless_root_sig
            .as_ref()
            .expect("bindless root signature is live alongside its PSO");
        let cull_sig = self
            .cull
            .cull_command_signature
            .as_ref()
            .expect("cull command signature is live alongside the bindless PSO");
        let local_lights_gva = com::gpu_va(&self.uniforms.local_light_buffer);

        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            cmd.ClearRenderTargetView(rtv, &self.view.clear_color, None);
            cmd.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);

            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);

            cmd.SetPipelineState(bindless_pso);
            cmd.SetGraphicsRootSignature(bindless_root);
            cmd.SetGraphicsRootConstantBufferView(1, view_gva);
            cmd.SetGraphicsRootConstantBufferView(2, light_gva);
            cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);
            cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(5, self.cull.bindless_pool_gpu[self.current_frame]);
            cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
            cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
            cmd.SetGraphicsRootShaderResourceView(8, object_gva);
            // [12] per-scene GpuLight storage buffer (t1). Probe + planar faces
            // reuse the bindless main PSO, which references it unconditionally.
            cmd.SetGraphicsRootShaderResourceView(12, local_lights_gva);
            // [13] ClusterParams + [14] the per-cluster light lists. These faces
            // bind the `use_clusters = 0` copy: their viewpoint differs from the
            // grid the main camera binned, so they iterate every local light.
            cmd.SetGraphicsRootConstantBufferView(
                13,
                self.cluster_params_gva(self.current_frame, false),
            );
            cmd.SetGraphicsRootShaderResourceView(14, self.cluster_list_gva());
            // [15]..[18] the spot shadow projections + depth array and the
            // area-light table + LTC lookups. Bound like any other main-pass
            // face: a shadowed spot occludes a probe capture, and an area light
            // lights it, exactly as they do for the main camera.
            self.bind_local_light_tables(cmd, super::draw::LocalLightParams::BINDLESS);
            cmd.SetGraphicsRootDescriptorTable(9, self.ssao_ao_srv_gpu());
            // [10] probe cube array (valid -- filled with the sky) + [11] the EMPTY
            // ProbeSet (count 0), so a probe face samples only the sky, not other
            // probes, and never reads the live ProbeSet ring while it is rewritten.
            cmd.SetGraphicsRootDescriptorTable(10, self.probe_cube_table_gpu());
            cmd.SetGraphicsRootConstantBufferView(11, com::gpu_va(&self.probe.set_empty_cbv));
            // Static + instance prefix `[0, skinned_record_base())`. Skinned tail
            // omitted (not captured into the probe in V1).
            cmd.ExecuteIndirect(
                cull_sig,
                self.skinned_record_base() as u32,
                indirect,
                indirect_offset as u64,
                None::<&ID3D12Resource>,
                0,
            );
        }
        self.inc_draw_calls(1);
    }

    // World-space bounds over every static draw object, skipping degenerate
    // (non-finite) AABBs. `None` for an empty scene. Mirrors
    // `metal/probe.rs::scene_world_bounds`.
    pub(super) fn scene_world_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        reflection_probe::fold_world_bounds(self.draw.objects.iter().map(|o| (o.bb_min, o.bb_max)))
    }
}

// Create a persistently-mapped UPLOAD constant buffer holding `bytes` (256-aligned)
// and return it with its GPU virtual address. Used for the bake's per-capture light
// + shadow snapshots, so the six faces share one lighting set decoupled from the
// frame's per-frame CBV writes.
fn make_snapshot_cbv(alloc: &DeviceAllocator, bytes: &[u8]) -> Result<(PooledBuffer, u64), String> {
    let size = (((bytes.len() as u64) + 255) & !255).max(256);
    let cbv = create_buffer(
        alloc,
        size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { cbv.Map(0, None, Some(&mut ptr)) }
        .map_err(|e| format!("probe: map snapshot cbv: {e}"))?;
    // SAFETY: the buffer is at least `bytes.len()` bytes (256-aligned).
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
    }
    let gva = com::gpu_va(&cbv);
    Ok((cbv, gva))
}

// A one-entry non-shader-visible RTV heap for a probe face colour target.
fn create_rtv_heap(device: &ID3D12Device) -> Result<ID3D12DescriptorHeap, String> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe { device.CreateDescriptorHeap(&desc) }.map_err(|e| format!("probe: rtv heap: {e}"))
}

// A one-entry non-shader-visible DSV heap for a probe face depth target.
fn create_dsv_heap(device: &ID3D12Device) -> Result<ID3D12DescriptorHeap, String> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe { device.CreateDescriptorHeap(&desc) }.map_err(|e| format!("probe: dsv heap: {e}"))
}

// Create a probe face depth target (D32_FLOAT, matching the main pass's DSV format
// + the face colour's sample count) and write its DSV. Created in DEPTH_WRITE and
// left there (only the bake uses it; it is cleared every face).
fn create_bake_depth(
    device: &ID3D12Device,
    size: u32,
    sample_count: u32,
    dsv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_D32_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 {
            DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                Depth: 1.0,
                Stencil: 0,
            },
        },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: size as u64,
        Height: size,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            Some(&clear_value),
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("probe: create face depth: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "probe: create face depth returned None".to_string())?;
    let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: DXGI_FORMAT_D32_FLOAT,
        ViewDimension: if sample_count > 1 {
            D3D12_DSV_DIMENSION_TEXTURE2DMS
        } else {
            D3D12_DSV_DIMENSION_TEXTURE2D
        },
        Flags: D3D12_DSV_FLAG_NONE,
        ..Default::default()
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateDepthStencilView(&texture, Some(&dsv_desc), dsv_cpu) };
    Ok(texture)
}
