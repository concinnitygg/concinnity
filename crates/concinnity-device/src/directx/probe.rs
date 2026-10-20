//! Scene-captured reflection probes on DirectX. Each declared `ReflectionProbe`
//! (or an auto-seeded grid when a world declares none) is baked into its own cube,
//! DISTINCT from `env_map`: the specular reflection term box-projects against the
//! probe's influence box and samples its cube, so glossy surfaces reflect the
//! actual surrounding geometry instead of the imported HDR sky, while the
//! background + diffuse irradiance keep sampling `env_map` so the visible sky is
//! never replaced.
//!
//! The cube math + the staggered-bake state machine are backend-agnostic
//! (`concinnity_engine::render::probe::capture`); this module drives the GPU capture, mirroring
//! `crate::metal::probe`. The bake is STAGGERED + ASYNCHRONOUS across frames so the
//! render thread never blocks: one probe is in flight at a time, its six cube faces
//! submitted one per frame into a capture cube, then convolved into the probe cube
//! by the compute kernels in `probe_prefilter.hlsl`. Nothing is read back and no
//! convolution runs on the CPU.
//!
//! DirectX simplification vs Metal: a per-face fence VALUE gives ordered GPU
//! completion for free (the queue is FIFO), so there is no completion handler / atomic
//! -- a face is done when `frame_sync.fence` reaches the value signaled after it. The
//! bake never calls `wait_idle` (that would reintroduce a multi-hundred-ms freeze);
//! the convolution is deferred until the fence reaches the last face's value.
//!
//! Each probe passes through three phases, sequenced by the shared `ProbeBake` this
//! module serves as a `ProbeBakeDevice`:
//!   * Rendering    -- six cube faces submitted to the GPU (one per frame) into a RESERVED
//!     ring slot (`bake_ring_slot`) the frame never overwrites, each
//!     copied into its slice of the capture cube.
//!   * Prefiltering -- the convolution runs as compute dispatches: the clamped mirror
//!     mip plus the capture's source pyramid in the first frame (all
//!     cheap), then ONE GGX mip per frame after it.
//!   * (install)    -- the probe's record joins the probe book, which makes the
//!     shaders read its cube of the array.
//!     No upload: the cube was written in place.
//!
//! Known V1 simplifications (documented intentionally; mirror Metal where noted):
//!   * Static + instanced geometry only -- skinned meshes are not captured into the
//!     probe (no per-bake deformed buffer yet). They still receive probe reflections.
//!   * Single bounce + cold-first-frame lighting (the shadow map may be unpopulated when
//!     a probe bakes on an early frame), exactly like Metal.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::depth::DEPTH_CLEAR;
use concinnity_engine::render::planar_reflection::PixelRect;
use concinnity_engine::render::probe::bake::{
    CAPTURE_FACES, ProbeBake, ProbeBakeDevice, capture_ring_slot,
};
use concinnity_engine::render::probe::book::ProbeBook;
use concinnity_engine::render::probe::capture::{self, PrefilterPlan, ProbePlacement};
use concinnity_engine::render::probe::radiance::FaceStorage;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use super::context::DxContext;
use super::error::map_hresult;
use super::light_cull::ClusterGrid;
use super::probe_prefilter::PrefilterGpu;
use super::texture::{
    HDR_FORMAT, create_hdr_color_target, create_hdr_resolve_target, transition_barrier,
};
use crate::directx::depth::optimized_clear;
use crate::directx::descriptor_slot::{DescriptorTables, SrvSlot};

// What a runtime capture bakes: face size, mip count, GGX sample count and firefly
// clamp, shared with the Metal and Vulkan backends (and with the build-time CPU
// convolution's roughness ramp) so a probe looks the same whichever backend
// captured it.
pub(super) const PLAN: PrefilterPlan = PrefilterPlan::RUNTIME;
// Captured cube-face resolution (mip 0 of the prefilter chain).
const PROBE_FACE_SIZE: u32 = PLAN.face_size();

// The GPU resources + state of one in-flight capture. The six faces share one
// (MSAA) color + depth target reused across frames; each face has its own view
// CBV + command allocator/list (held until the convolution starts, so the fence
// guarantees their GPU work has retired before they drop).
pub(crate) struct RenderingBake {
    eye: [f32; 3],
    capture_distance: Option<f32>,
    sample_count: u32,
    // Reused across the six faces.
    color: ID3D12Resource,
    _depth: ID3D12Resource,
    resolve: Option<ID3D12Resource>,
    _rtv_heap: ID3D12DescriptorHeap,
    _dsv_heap: ID3D12DescriptorHeap,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    // Per-face: a 208-byte ViewUniforms CBV (kept mapped) + its GVA.
    _view_cbvs: Vec<PooledBuffer>,
    view_gvas: Vec<u64>,
    // Per-capture light + shadow snapshots (so the six faces share one consistent
    // lighting set, decoupled from the frame's per-frame CBV writes).
    light_gva: u64,
    shadow_gva: u64,
    canopy_gva: u64,
    canopy_tables_gva: u64,
    _light_cbv: PooledBuffer,
    _shadow_cbv: PooledBuffer,
    _canopy_cbv: PooledBuffer,
    _canopy_tables: PooledBuffer,
    // The capture cube each face is copied into, and the probe cube the convolution
    // will write. Allocated with the capture because face 0 copies into it, and
    // handed to the prefiltering slot once every face has landed.
    prefilter: PrefilterGpu,
    // One fresh allocator + list per submitted face, held until the convolution
    // starts (the fence proves their GPU work retired before they drop).
    cmd_allocs: Vec<ID3D12CommandAllocator>,
    cmd_lists: Vec<ID3D12GraphicsCommandList>,
    // Fence value signaled after the LAST face; the convolution waits for the
    // shared `frame_sync.fence` to reach it.
    last_fence_value: u64,
}

// A finished capture convolving into its cube on the GPU, one destination mip per
// frame. Holds both cubes plus the allocator and list of every dispatch it has
// submitted, which install drops once the fence covers them. Nothing else bakes
// while this slot is full: the dispatches address their cubes through the one
// reserved descriptor block a starting capture would rewrite.
pub(crate) struct PrefilteringBake {
    gpu: PrefilterGpu,
    cmd_allocs: Vec<ID3D12CommandAllocator>,
    cmd_lists: Vec<ID3D12GraphicsCommandList>,
    // Fence value signaled after the LAST dispatch submitted so far.
    last_fence_value: u64,
}

// The bake's two slots: a capture rendering its faces and a capture convolving
// into its cube.
pub(super) type DxProbeBake = ProbeBake<RenderingBake, PrefilteringBake>;

// Color + depth attachments for a probe-face / planar mirror capture.
#[derive(Clone, Copy)]
pub(in crate::directx) struct FaceTargets {
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
}

// GPU virtual addresses of the per-capture view / light / shadow constant
// buffers, the cluster grid binned for the capture's viewpoint (`None`
// shades every local light and probe), and the scale the view block writes
// the face at, which its clear follows.
#[derive(Clone, Copy)]
pub(in crate::directx) struct FaceUniforms<'a> {
    pub view_gva: u64,
    pub light_gva: u64,
    pub shadow_ubo_gva: u64,
    pub canopy_gva: u64,
    pub canopy_tables_gva: u64,
    pub clusters: Option<ClusterGrid<'a>>,
    pub storage: FaceStorage,
}

// The indirect draw for one capture region: the command buffer, its byte offset,
// and the per-object buffer address for bindless rendering.
#[derive(Clone, Copy)]
pub(in crate::directx) struct IndirectDraw<'a> {
    pub indirect: &'a ID3D12Resource,
    pub indirect_offset: u32,
    pub object_gva: u64,
    pub material_params_gva: u64,
}

// Render-target dimensions for the capture, and the texel rectangle of them the
// render is cleared and drawn within (`None` for the whole target). Texels
// outside the rectangle keep whatever they held.
#[derive(Clone, Copy)]
pub(in crate::directx) struct FaceExtent {
    pub width: u32,
    pub height: u32,
    pub area: Option<PixelRect>,
}

impl DxContext {
    // Set the reflection-probe placements the graphics system resolved
    // (declared `ReflectionProbe` assets, or a grid auto-seeded over the
    // scene); an empty list bakes none. Resets the staggered bake, and grows the
    // cube array when the list outgrows it; a world whose array cannot grow keeps
    // the sky.
    pub(super) fn set_reflection_probes(&mut self, placements: &[ProbePlacement]) {
        let placements = placements.to_vec();
        let placed = self.with_probe_bake(|bake, ctx| bake.place(ctx, placements));
        crate::probe_report::report_probe_placement(placed);
    }

    // The reserved transient-ring slot the asynchronous bake builds its bindless
    // buffers into. The cull rings are sized one past the frames in
    // flight to make room.
    fn bake_ring_slot(&self) -> usize {
        capture_ring_slot(self.hw.frames())
    }

    // GPU descriptor handle of the reflection-probe cube array's SRV (root param
    // [10] of the bindless main pass).
    pub(in crate::directx) fn probe_cube_table_gpu(&self) -> SrvSlot {
        SrvSlot::at(
            &self.descriptors.srv_heap,
            self.descriptors.srv_descriptor_size,
            self.descriptors.layout.probe_cubes_srv_slot,
        )
    }

    // Advance the asynchronous reflection-probe bake one frame. Called every frame
    // from `draw_frame` after the frame-slot fence wait; cheap once the queue
    // drains. A failure abandons the remaining bakes, keeping what baked.
    pub(super) fn bake_pending_probes(&mut self, elapsed: f32) {
        let report = self.with_probe_bake(|bake, ctx| bake.advance(ctx, &elapsed));
        crate::probe_report::report_probe_bake(report);
    }

    // Run `f` over the bake with this context as its device. The slots are lent
    // to `f`, so the context reads them as empty for the call.
    fn with_probe_bake<R>(&mut self, f: impl FnOnce(&mut DxProbeBake, &mut Self) -> R) -> R {
        let mut bake = std::mem::take(&mut self.probe.bake);
        let out = f(&mut bake, self);
        self.probe.bake = bake;
        out
    }

    // Whether the GPU has finished everything submitted up to `fence_value`.
    fn fence_reached(&self, fence_value: u64) -> bool {
        // SAFETY: the fence was created from this device; the query only reads.
        let completed = unsafe { self.frame_sync.fence.GetCompletedValue() };
        completed >= fence_value
    }

    // Build the capture of the probe at `index`: the reserved-slot bindless
    // buffers (object + draw-args, frustum-independent) ONCE, and the capture
    // targets + per-face view CBVs + both cubes. No face is submitted here; the
    // faces follow one per frame via `record_probe_face`.
    fn start_probe_capture(
        &mut self,
        elapsed: f32,
        index: usize,
        placement: ProbePlacement,
    ) -> RenderResult<RenderingBake> {
        let eye = placement.position;
        let slot = self.bake_ring_slot();

        // Build the reserved-slot bindless buffers once: the per-object record buffer
        // and the draw-args buffer (LOD by distance from the probe eye). Both are
        // frustum-independent, reused by every face's cull.
        self.build_record_buffers(
            slot,
            eye,
            concinnity_engine::render::temporal::model_history::HistoryMode::Untracked,
        );
        if let Some(params) = self.cull.material_params.as_mut() {
            params.upload(slot);
        }

        let alloc = &self.hw.alloc;
        let device = &self.hw.device;
        let sample_count = self.targets.hdr.msaa_samples.max(1);
        let size = PROBE_FACE_SIZE;

        // One MSAA (or single-sample) color + depth pair, reused across the six faces.
        let rtv_heap = create_rtv_heap(device)?;
        let dsv_heap = create_dsv_heap(device)?;
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let dsv = unsafe { dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        let color = create_hdr_color_target(
            device,
            size,
            size,
            sample_count,
            rtv,
            FaceStorage::PROBE.clear_color(self.state.view.clear_color),
        )?;
        let depth = create_bake_depth(device, size, sample_count, dsv)?;
        // A single-sample resolve target only when MSAA is on.
        let resolve = if sample_count > 1 {
            Some(create_hdr_resolve_target(device, size, size)?)
        } else {
            None
        };

        // Snapshot the frame's light + shadow uniforms into bake-owned CBVs so all six
        // faces share one temporally-consistent lighting set, and so the capture does
        // not read `light_ubo[frame]` / `shadow_ubo[frame]` while `record_frame` (which runs
        // after this) overwrites them on the same frame -- a CPU/GPU race on a mapped
        // buffer. The capture's lighting is the env live when it started.
        // SAFETY: `LightUniforms` is `#[repr(C)]` with explicit pad fields and no implicit padding
        // (pinned by the layout tests in `render_types.rs`), so all `size_of` bytes are
        // initialized, and the borrow keeps them live for the snapshot copy below.
        let light_bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref::<render_types::LightUniforms>(self.uniforms.lights.uniforms())
                    .cast::<u8>(),
                size_of::<render_types::LightUniforms>(),
            )
        };
        let (light_cbv, light_gva) = make_snapshot_cbv(alloc, light_bytes)?;
        // SAFETY: `ShadowUniforms` is `#[repr(C)]` with an explicit trailing pad and no implicit
        // padding (pinned by the layout tests in `render_types.rs`), so all `size_of` bytes are
        // initialized, and the borrow keeps them live for the snapshot copy below.
        let shadow_bytes = unsafe {
            std::slice::from_raw_parts(
                (&raw const self.state.planner.shadow.uniforms).cast::<u8>(),
                size_of::<render_types::ShadowUniforms>(),
            )
        };
        let (shadow_cbv, shadow_gva) = make_snapshot_cbv(alloc, shadow_bytes)?;
        let (canopy_cbv, canopy_gva) =
            make_snapshot_cbv(alloc, bytemuck::bytes_of(&self.canopy.params.get()))?;
        let (canopy_tables, canopy_tables_gva) = make_snapshot_cbv(
            alloc,
            bytemuck::cast_slice(self.canopy.tables.borrow().as_slice()),
        )?;

        // Per-face ViewUniforms CBVs, the only per-face binding.
        // The capture renders with the real env IBL (so the scene carries ambient
        // lighting), exactly like the main pass minus the SSR/RT resolve.
        let prefilter_mip_count = self.scene.env_map.prefilter_mip_count as f32;
        let mut view_cbvs = Vec::with_capacity(CAPTURE_FACES);
        let mut view_gvas = Vec::with_capacity(CAPTURE_FACES);
        for face in 0..CAPTURE_FACES {
            let vp = capture::face_view_projection(eye, face);
            let view_mat = capture::face_view_matrix(eye, face);
            let view =
                super::draw::ViewUniforms::face(concinnity_engine::render::uniforms::FaceCamera {
                    vp,
                    view: view_mat,
                    cam_pos: eye,
                    elapsed,
                    prefilter_mip_count,
                    sky_rot: self.state.view.sky_rot,
                    irradiance: self.irradiance_lane(),
                    pre_exposure: FaceStorage::PROBE.scale(),
                });
            let cbv = alloc.alloc_buffer(
                256,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { cbv.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "probe: map view cbv"))?;
            // SAFETY: the buffer is 256 bytes; ViewUniforms is 208.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const view).cast::<u8>(),
                    ptr.cast::<u8>(),
                    size_of::<super::draw::ViewUniforms>(),
                );
            }
            view_gvas.push(com::gpu_va(&cbv));
            view_cbvs.push(cbv);
        }

        // The capture cube each face is copied into, and the descriptors naming
        // the cube of the array the convolution writes. Allocated with the capture
        // rather than at the convolution's start: face 0 copies into the capture,
        // so it has to exist before the first face records.
        let cubes = self
            .probe
            .gpu
            .cubes
            .as_ref()
            .ok_or_else(|| RenderError::Other("probe: no cube array for a placement".into()))?;
        let prefilter = PrefilterGpu::new(self, &PLAN, cubes, index)?;

        Ok(RenderingBake {
            eye,
            capture_distance: placement.capture_distance,
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
            canopy_gva,
            canopy_tables_gva,
            _light_cbv: light_cbv,
            _shadow_cbv: shadow_cbv,
            _canopy_cbv: canopy_cbv,
            _canopy_tables: canopy_tables,
            prefilter,
            cmd_allocs: Vec::with_capacity(CAPTURE_FACES),
            cmd_lists: Vec::with_capacity(CAPTURE_FACES),
            last_fence_value: 0,
        })
    }

    // Submit the in-flight capture's next cube face (one per frame): a fresh command
    // list that culls the face frustum into the reserved slot, renders the bindless
    // static + instance geometry into the face target, (resolves +) copies it into its
    // slice of the capture cube, then signals a fence value. The last face's value is
    // what the convolution waits for.
    fn record_probe_face(&self, bake: &mut RenderingBake, face: usize) -> RenderResult<()> {
        let slot = self.bake_ring_slot();
        let (eye, view_gva, light_gva, shadow_gva) = (
            bake.eye,
            bake.view_gvas[face],
            bake.light_gva,
            bake.shadow_gva,
        );

        let frustum = capture::face_frustum(eye, face, bake.capture_distance);

        // A fresh allocator + list per face, held until the fence proves the face
        // retired, so no in-flight allocator is ever reset.
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        let alloc: ID3D12CommandAllocator = unsafe {
            self.hw
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .map_err(|e| map_hresult(e.code(), "probe: face allocator"))?;
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        let cmd: ID3D12GraphicsCommandList = unsafe {
            self.hw
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
        }
        .map_err(|e| map_hresult(e.code(), "probe: face cmd list"))?;
        // Register the recording on the bake before anything can fail: once it is
        // submitted, only abandoning the bake (which idles the device) makes it
        // safe to drop, and that reaches it only through the bake.
        bake.cmd_allocs.push(alloc);
        bake.cmd_lists.push(cmd.clone());

        // Cull this face's frustum into the reserved indirect buffer, then render.
        self.encode_probe_cull(&cmd, slot, &frustum, eye);
        let (rtv, dsv) = (bake.rtv, bake.dsv);
        let indirect = &self.cull.indirect_cmd_buffers[slot];
        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[slot]);
        self.encode_main_into_face(
            &cmd,
            FaceTargets { rtv, dsv },
            FaceUniforms {
                view_gva,
                light_gva,
                shadow_ubo_gva: shadow_gva,
                canopy_gva: bake.canopy_gva,
                canopy_tables_gva: bake.canopy_tables_gva,
                clusters: None,
                storage: FaceStorage::PROBE,
            },
            IndirectDraw {
                indirect,
                indirect_offset: 0,
                object_gva,
                material_params_gva: self.material_params_gva(slot),
            },
            FaceExtent {
                width: PROBE_FACE_SIZE,
                height: PROBE_FACE_SIZE,
                area: None,
            },
        );

        // Resolve (MSAA) + copy the face into its slice of the capture cube.
        self.copy_face_to_capture(&cmd, bake, face)?;

        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { cmd.Close() }.map_err(|e| map_hresult(e.code(), "probe: face close"))?;
        let list: ID3D12CommandList = windows::core::Interface::cast(&cmd)
            .map_err(|e| map_hresult(e.code(), "probe: face cast"))?;
        // SAFETY: every command list in the submission is live and closed, and the slice outlives
        // the call.
        unsafe { self.hw.command_queue.ExecuteCommandLists(&[Some(list)]) };

        // Signal a unique fence value on the shared fence; the convolution waits for it.
        let fence_val = self.frame_sync.next_fence_value.get();
        self.frame_sync.next_fence_value.set(fence_val + 1);
        // SAFETY: the fence and the event were created from this device and are live for the call.
        unsafe {
            self.hw
                .command_queue
                .Signal(&self.frame_sync.fence, fence_val)
        }
        .map_err(|e| map_hresult(e.code(), "probe: face signal"))?;

        bake.last_fence_value = fence_val;
        Ok(())
    }

    // Resolve (when MSAA) + copy the just-rendered face color into slice `face` of
    // the capture cube. The color rests in RENDER_TARGET and is restored to it for
    // the next face; the resolve target rests in PIXEL_SHADER_RESOURCE. Face order is
    // the hardware cube order (`gfx::cubemap`), so slice `face` is the face a sampler
    // finds looking that way.
    fn copy_face_to_capture(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        bake: &RenderingBake,
        face: usize,
    ) -> RenderResult<()> {
        let sample_count = bake.sample_count;
        // Subresource index of mip 0 of array slice `face`, which D3D12 orders
        // mip-major within a slice.
        let dst_subresource = u32::saturating_from(face) * bake.prefilter.mips();
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
            let to_resolve = [
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
            ];
            // SAFETY: the list is recording, and the bake owns both the color and resolve images.
            unsafe { cmd.ResourceBarrier(&to_resolve) };
            // SAFETY: the barrier above put the color in RESOLVE_SOURCE and the resolve image in
            // RESOLVE_DEST, and the bake keeps both alive.
            unsafe { cmd.ResolveSubresource(resolve, 0, &bake.color, 0, HDR_FORMAT) };
            let to_copy = [
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
            ];
            // SAFETY: the list is recording, and both images are still owned by the bake.
            unsafe { cmd.ResourceBarrier(&to_copy) };
            let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: com::borrowed(resolve),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            // SAFETY: both locations borrow interfaces the bake keeps alive past the call, and the
            // resolve image is in COPY_SOURCE from the barrier above.
            unsafe { cmd.CopyTextureRegion(&raw const dst_loc, 0, 0, 0, &raw const src_loc, None) };
            let to_rest = transition_barrier(
                resolve,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            // SAFETY: the list is recording, and the resolve image is owned by the bake.
            unsafe { cmd.ResourceBarrier(&[to_rest]) };
        } else {
            let to_copy = transition_barrier(
                &bake.color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            // SAFETY: the list is recording, and the color image is owned by the bake.
            unsafe { cmd.ResourceBarrier(&[to_copy]) };
            let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: com::borrowed(&bake.color),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            // SAFETY: both locations borrow interfaces the bake keeps alive past the call, and the
            // color image is in COPY_SOURCE from the barrier above.
            unsafe { cmd.CopyTextureRegion(&raw const dst_loc, 0, 0, 0, &raw const src_loc, None) };
            let to_rt = transition_barrier(
                &bake.color,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            // SAFETY: the list is recording, and the color image is owned by the bake.
            unsafe { cmd.ResourceBarrier(&[to_rt]) };
        }
        Ok(())
    }

    // Record convolution step `mip` of a bake: mip 0 is the firefly-clamped mirror
    // mip plus the capture's source pyramid, each later mip one GGX convolution.
    // Each GGX dispatch reads the finished pyramid and writes a mip nothing else
    // touches, so consecutive mips need no barrier; the queue's FIFO order puts
    // every one of them after the pyramid build that produced their source.
    fn record_prefilter_mip(&self, bake: &mut PrefilteringBake, mip: u32) -> RenderResult<()> {
        if mip == 0 {
            return self.record_prefilter_step(bake, |ctx, cmd, bake| {
                ctx.encode_probe_pyramid(cmd, &bake.gpu, &PLAN)
            });
        }
        // The last mip's list also carries the cube back to PIXEL_SHADER_RESOURCE.
        // The install has no list of its own to submit that transition on: it would
        // have to drop that list immediately, and D3D12 does not keep a command
        // allocator alive for the GPU.
        let last = mip + 1 == PLAN.mips();
        self.record_prefilter_step(bake, |ctx, cmd, bake| {
            ctx.encode_probe_ggx_mip(cmd, &PLAN, mip)?;
            if last {
                let barriers = ctx.probe.gpu.bound_cubes().cube_barriers(
                    bake.gpu.cube(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    super::probe_set::PROBE_CUBES_STATE,
                );
                // SAFETY: the command list is in the recording state and the resource the
                // barriers name is live for the call.
                unsafe { cmd.ResourceBarrier(&barriers) };
            }
            Ok(())
        })
    }

    // Record and submit one convolution step on a fresh allocator + list, registering
    // both on the bake so a later failure still reclaims them, and signaling a fence
    // value the install waits for. The shader-visible descriptor heaps are bound
    // first: every dispatch addresses its cubes through the SRV heap.
    fn record_prefilter_step(
        &self,
        bake: &mut PrefilteringBake,
        encode: impl FnOnce(&Self, &ID3D12GraphicsCommandList, &PrefilteringBake) -> RenderResult<()>,
    ) -> RenderResult<()> {
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        let alloc: ID3D12CommandAllocator = unsafe {
            self.hw
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .map_err(|e| map_hresult(e.code(), "probe: convolve allocator"))?;
        // SAFETY: as above.
        let cmd: ID3D12GraphicsCommandList = unsafe {
            self.hw
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
        }
        .map_err(|e| map_hresult(e.code(), "probe: convolve cmd list"))?;
        bake.cmd_allocs.push(alloc);
        bake.cmd_lists.push(cmd.clone());

        // SAFETY: the command list is in the recording state and the heaps it names are live.
        unsafe {
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
        }
        encode(self, &cmd, bake)?;
        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { cmd.Close() }.map_err(|e| map_hresult(e.code(), "probe: convolve close"))?;
        let list: ID3D12CommandList = windows::core::Interface::cast(&cmd)
            .map_err(|e| map_hresult(e.code(), "probe: convolve cast"))?;
        // SAFETY: every command list in the submission is live and closed, and the slice outlives
        // the call.
        unsafe { self.hw.command_queue.ExecuteCommandLists(&[Some(list)]) };

        let fence_val = self.frame_sync.next_fence_value.get();
        self.frame_sync.next_fence_value.set(fence_val + 1);
        // SAFETY: the fence was created from this device and is live for the call.
        unsafe {
            self.hw
                .command_queue
                .Signal(&self.frame_sync.fence, fence_val)
        }
        .map_err(|e| map_hresult(e.code(), "probe: convolve signal"))?;
        bake.last_fence_value = fence_val;
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
        uniforms: FaceUniforms<'_>,
        draw: IndirectDraw<'_>,
        extent: FaceExtent,
    ) {
        let FaceTargets { rtv, dsv } = targets;
        let FaceUniforms {
            view_gva,
            light_gva,
            shadow_ubo_gva,
            canopy_gva,
            canopy_tables_gva,
            clusters,
            storage,
        } = uniforms;
        let (cluster_params_gva, cluster_list_gva) = match clusters {
            Some(grid) => (grid.params_gva, com::gpu_va(grid.lists)),
            None => (
                self.cluster_params_gva(self.current_frame, false),
                self.cluster_list_gva(),
            ),
        };
        let IndirectDraw {
            indirect,
            indirect_offset,
            object_gva,
            material_params_gva,
        } = draw;
        let FaceExtent {
            width,
            height,
            area,
        } = extent;
        let scissor = match area {
            Some(r) => windows::Win32::Foundation::RECT {
                left: i32::saturating_from(r.x),
                top: i32::saturating_from(r.y),
                right: i32::saturating_from(r.x + r.width),
                bottom: i32::saturating_from(r.y + r.height),
            },
            None => windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: i32::saturating_from(width),
                bottom: i32::saturating_from(height),
            },
        };
        let bindless_pso = self.face_bindless_pso().expect("bindless PSO is live");
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

        // SAFETY: the list is recording, and `rtv` and `dsv` address the face's live targets.
        unsafe { cmd.OMSetRenderTargets(1, Some(&raw const rtv), false, Some(&raw const dsv)) };
        let clear_color = storage.clear_color(self.state.view.clear_color);
        // SAFETY: the list is recording, `rtv` is bound above, and the color and rect are locals.
        unsafe { cmd.ClearRenderTargetView(rtv, &clear_color, Some(&[scissor])) };
        // SAFETY: the list is recording, `dsv` is bound above, and the rect is a live local.
        unsafe {
            cmd.ClearDepthStencilView(
                dsv,
                D3D12_CLEAR_FLAG_DEPTH,
                DEPTH_CLEAR,
                0,
                Some(&[scissor]),
            );
        }
        let vp = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        // SAFETY: the list is recording, and the viewport slice is a live local.
        unsafe { cmd.RSSetViewports(&[vp]) };
        // SAFETY: the list is recording, and the scissor slice is a live local.
        unsafe { cmd.RSSetScissorRects(&[scissor]) };

        // SAFETY: the list is recording; setting the topology names no resource.
        unsafe { cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) };
        // SAFETY: the view names the scene vertex buffer `self.scene.geometry` owns.
        unsafe { cmd.IASetVertexBuffers(0, Some(&[self.scene.geometry.vertex_buffer_view])) };
        // SAFETY: the view names the scene index buffer `self.scene.geometry` owns.
        unsafe { cmd.IASetIndexBuffer(Some(&raw const self.scene.geometry.index_buffer_view)) };
        let heaps = [
            Some(self.descriptors.srv_heap.clone()),
            Some(self.descriptors.sampler_heap.clone()),
        ];
        // SAFETY: the list is recording, and both heaps are owned by `self.descriptors`.
        unsafe { cmd.SetDescriptorHeaps(&heaps) };

        // SAFETY: the list is recording, and the bindless PSO is live while `self.cull` holds it.
        unsafe { cmd.SetPipelineState(bindless_pso) };
        // SAFETY: the list is recording, and the bindless root signature lives beside its PSO.
        unsafe { cmd.SetGraphicsRootSignature(bindless_root) };
        // SAFETY: the bindless signature declares a CBV at 1, and the face's view block is live.
        unsafe { cmd.SetGraphicsRootConstantBufferView(1, view_gva) };
        // SAFETY: the signature declares a CBV at 2, and the bake's light snapshot is live.
        unsafe { cmd.SetGraphicsRootConstantBufferView(2, light_gva) };
        // SAFETY: the signature declares a CBV at 3, and the bake's shadow snapshot is live.
        unsafe { cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva) };
        // SAFETY: the SRV heap is set above, and the shadow map's descriptor is live.
        unsafe { cmd.set_graphics_srv_table(4, self.shadow.srv_gpu) };
        let pool = self.cull.bindless_pool_gpu[self.current_frame];
        // SAFETY: the SRV heap is set above, and this frame's bindless pool table is live.
        unsafe { cmd.set_graphics_srv_table(5, pool) };
        // SAFETY: the sampler heap is set above, and the shadow sampler slot is live.
        unsafe { cmd.set_graphics_sampler_table(6, self.descriptors.shadow_sampler_gpu) };
        // SAFETY: the sampler heap is set above, and the linear sampler slot is live.
        unsafe { cmd.set_graphics_sampler_table(7, self.descriptors.linear_sampler_gpu) };
        // SAFETY: the list is recording, and the face's object buffer is live for the bake.
        unsafe { cmd.SetGraphicsRootShaderResourceView(8, object_gva) };
        let vertex_gva = com::gpu_va(&self.scene.geometry.vertex_buffer);
        // SAFETY: the list is recording, and the vertex buffer is owned by `self.scene.geometry`.
        unsafe {
            cmd.SetGraphicsRootShaderResourceView(
                crate::directx::draw::MAIN_MESH_VERTICES_PARAM,
                vertex_gva,
            );
        }
        // SAFETY: the list is recording, and the material params buffer is live for the frame.
        unsafe {
            cmd.SetGraphicsRootShaderResourceView(
                super::material_params::MATERIAL_PARAMS_ROOT_PARAM,
                material_params_gva,
            );
        }
        // SAFETY: the list is recording, and the canopy constant buffer is live for the frame.
        unsafe {
            cmd.SetGraphicsRootConstantBufferView(super::canopy::CANOPY_ROOT_PARAM, canopy_gva);
        }
        // SAFETY: the list is recording, and the canopy tables buffer is live for the frame.
        unsafe {
            cmd.SetGraphicsRootShaderResourceView(
                super::canopy::CANOPY_TABLES_ROOT_PARAM,
                canopy_tables_gva,
            );
        }
        // [12] per-scene GpuLight storage buffer (t1). Probe + planar faces
        // reuse the bindless main PSO, which references it unconditionally.
        // SAFETY: the list is recording, and the local light buffer is owned by `self.uniforms`.
        unsafe { cmd.SetGraphicsRootShaderResourceView(12, local_lights_gva) };
        // [13] ClusterParams + [14] the per-cluster light lists: the face's
        // own grid, or the `use_clusters = 0` copy that iterates every light.
        // SAFETY: the list is recording, and the chosen cluster params buffer is live.
        unsafe { cmd.SetGraphicsRootConstantBufferView(13, cluster_params_gva) };
        // SAFETY: the list is recording, and the chosen cluster light lists are live.
        unsafe { cmd.SetGraphicsRootShaderResourceView(14, cluster_list_gva) };
        // [15]..[18] the spot shadow projections + depth array and the
        // area-light table + LTC lookups. Bound like any other main-pass
        // face: a shadowed spot occludes a probe capture, and an area light
        // lights it, exactly as they do for the main camera.
        self.bind_local_light_tables(cmd, super::draw::LocalLightParams::BINDLESS);
        // SAFETY: the SRV heap is set above, and the SSAO output's descriptor is live.
        unsafe { cmd.set_graphics_srv_table(9, self.ssao_ao_srv_gpu()) };
        // [10] probe cube array + [11] the EMPTY ProbeSet (count 0) + [19] the
        // stand-in records, so a probe face samples only the sky, not other
        // probes, and never reads the live ring while it is rewritten.
        self.bind_main_probe_set(
            cmd,
            com::gpu_va(&self.uniforms.probe_set_empty_cbv),
            self.probe.gpu.stand_in_records.gpu_va(),
        );
        let max_commands = u32::saturating_from(self.state.draw.skinned_record_base());
        // Static + instance prefix `[0, skinned_record_base())`. Skinned tail
        // omitted (not captured into the probe in V1).
        // SAFETY: the command signature lives beside the bindless PSO, and `indirect` is the
        // cull's live argument buffer for this face.
        unsafe {
            cmd.ExecuteIndirect(
                cull_sig,
                max_commands,
                indirect,
                u64::from(indirect_offset),
                None::<&ID3D12Resource>,
                0,
            );
        }
        self.inc_draw_calls(1);
        // A face is always rendered lit, whatever the viewport shows.
        if self.draws_sky(concinnity_engine::gfx::view_modes::ViewMode::Lit) {
            self.encode_sky(cmd, view_gva, self.current_frame);
        }
    }
}

// Create a persistently-mapped UPLOAD constant buffer holding `bytes` (256-aligned)
// and return it with its GPU virtual address. Used for the bake's per-capture light
// + shadow snapshots, so the six faces share one lighting set decoupled from the
// frame's per-frame CBV writes.
impl ProbeBakeDevice for DxContext {
    type Capture = RenderingBake;
    type Prefilter = PrefilteringBake;
    // The frame's elapsed seconds, which a capture starting this frame
    // animates its faces at.
    type Frame<'f> = f32;

    fn book(&mut self) -> &mut ProbeBook {
        &mut self.probe.book
    }

    // The capture renders through the bindless GPU cull into the reserved ring
    // slot, neither of which comes or goes after init.
    fn capture_supported(&self) -> bool {
        let slot = self.bake_ring_slot();
        self.cull.main_bindless_pso.is_some()
            && self.cull.cull_kernels.is_some()
            && self.cull.object_buffer_resources.len() > slot
            && self.cull.draw_args_buffer_resources.len() > slot
            && self.cull.indirect_cmd_buffers.len() > slot
            && self.probe.prefilter.is_some()
    }

    // Geometry may still be streaming: a zero cull would bake an empty cube. Both
    // cubes of a bake are addressed through ONE reserved SRV-heap block, written
    // when a capture starts, so a capture waits for the previous convolution to
    // install rather than rewriting the descriptors its dispatches bind. An
    // atmosphere's sky and ambient light come from tables the frame writes, so
    // a capture also waits for the first of them.
    fn capture_ready(&self, prefilter_in_flight: bool) -> bool {
        self.state.draw.record_count() > 0 && !prefilter_in_flight && self.atmosphere_tables_ready()
    }

    fn reserve_cubes(&mut self, count: usize) -> RenderResult<()> {
        self.reserve_probe_cubes(&PLAN, count)
    }

    fn start_capture(
        &mut self,
        elapsed: &f32,
        index: usize,
        placement: ProbePlacement,
    ) -> RenderResult<RenderingBake> {
        self.start_probe_capture(*elapsed, index, placement)
    }

    fn render_face(
        &mut self,
        _frame: &f32,
        capture: &mut RenderingBake,
        face: usize,
    ) -> RenderResult<()> {
        self.record_probe_face(capture, face)
    }

    fn capture_retired(&self, capture: &RenderingBake) -> bool {
        self.fence_reached(capture.last_fence_value)
    }

    // The capture's targets + command lists drop here; the fence reached the
    // last face's value, so the GPU is done with all of them.
    fn begin_prefilter(
        &mut self,
        _index: usize,
        capture: RenderingBake,
    ) -> RenderResult<PrefilteringBake> {
        Ok(PrefilteringBake {
            gpu: capture.prefilter,
            cmd_allocs: Vec::with_capacity(PLAN.mips() as usize),
            cmd_lists: Vec::with_capacity(PLAN.mips() as usize),
            last_fence_value: 0,
        })
    }

    fn prefilter_mip(&mut self, prefilter: &mut PrefilteringBake, mip: u32) -> RenderResult<()> {
        self.record_prefilter_mip(prefilter, mip)
    }

    // The install drops each dispatch's allocator and list, so it waits for the
    // GPU to retire them, not just for them to be submitted.
    fn prefilter_retired(&self, prefilter: &PrefilteringBake) -> bool {
        self.fence_reached(prefilter.last_fence_value)
    }

    // Nothing is uploaded at install -- the cube was written in place -- and no
    // descriptor moves.
    fn finish_prefilter(&mut self, prefilter: PrefilteringBake) {
        drop(prefilter);
    }

    // Idle before dropping either slot: their command lists may still be
    // executing, and every payload owns resources a submission could still name.
    fn abandon(&mut self, capture: Option<RenderingBake>, prefilter: Option<PrefilteringBake>) {
        self.wait_idle();
        drop((capture, prefilter));
    }
}

fn make_snapshot_cbv(alloc: &DeviceAllocator, bytes: &[u8]) -> RenderResult<(PooledBuffer, u64)> {
    let size = (((bytes.len() as u64) + 255) & !255).max(256);
    let cbv = alloc.alloc_buffer(
        size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { cbv.Map(0, None, Some(&raw mut ptr)) }
        .map_err(|e| map_hresult(e.code(), "probe: map snapshot cbv"))?;
    // SAFETY: the buffer is at least `bytes.len()` bytes (256-aligned).
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
    }
    let gva = com::gpu_va(&cbv);
    Ok((cbv, gva))
}

// A one-entry non-shader-visible RTV heap for a probe face color target.
fn create_rtv_heap(device: &ID3D12Device) -> RenderResult<ID3D12DescriptorHeap> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe { device.CreateDescriptorHeap(&raw const desc) }
        .map_err(|e| map_hresult(e.code(), "probe: rtv heap"))
}

// A one-entry non-shader-visible DSV heap for a probe face depth target.
fn create_dsv_heap(device: &ID3D12Device) -> RenderResult<ID3D12DescriptorHeap> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe { device.CreateDescriptorHeap(&raw const desc) }
        .map_err(|e| map_hresult(e.code(), "probe: dsv heap"))
}

// Create a probe face depth target (D32_FLOAT, matching the main pass's DSV format
// + the face color's sample count) and write its DSV. Created in DEPTH_WRITE and
// left there (only the bake uses it; it is cleared every face).
fn create_bake_depth(
    device: &ID3D12Device,
    size: u32,
    sample_count: u32,
    dsv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> RenderResult<ID3D12Resource> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = optimized_clear();
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(size),
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
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            Some(&raw const clear_value),
            &raw mut tex_opt,
        )
    }
    .map_err(|e| map_hresult(e.code(), "probe: create face depth"))?;
    let texture = tex_opt
        .ok_or_else(|| RenderError::Other("probe: create face depth returned None".to_string()))?;
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
    unsafe { device.CreateDepthStencilView(&texture, Some(&raw const dsv_desc), dsv_cpu) };
    Ok(texture)
}
