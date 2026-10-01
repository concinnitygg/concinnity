//! Planar reflection for flat reflectors (glass panes + water surfaces) on the
//! D3D12 backend. The scene is rendered a second time from the camera reflected
//! across each reflector plane (mirror view + oblique near-plane clip so geometry
//! behind the plane never leaks in) into a mirror target; the reflector's
//! fragment shader then samples that target projectively for a sharp,
//! scene-correct reflection instead of the blurry box-projected probe cube.
//!
//! Mirrors src/metal/planar.rs. One mirror render per DISTINCT plane:
//! near-coplanar reflectors (one wall of windows) share a render, and reflectors
//! past the budget (`MAX_PLANAR_PLANES`) fall back to the probe cube. The layout,
//! the per-frame plan and the mirror matrices come from the pure, unit-tested
//! `planar_reflection`.
//!
//! A reflector reads its mirror only at its own screen pixels, so each frame's
//! `PlanarFramePlan` crops every mirror render to the rectangle its reflectors
//! cover and skips a plane whose reflectors are all off screen. Each rendered
//! plane gets a DEDICATED reflected-frustum mirror cull (`encode_planar_culls`,
//! mirroring `metal::cull::encode_mirror_cull`), narrowed to that rectangle: the
//! GPU cull re-runs into that plane's region of a per-frame indirect buffer,
//! reading the frame's camera-independent object + draw-args buffers. So geometry
//! visible only in the reflection (behind / beside the main camera, outside its
//! frustum) is captured; the reflected view-proj's oblique near-plane clip also
//! rejects geometry behind the reflector. The face render then executes that
//! region.
//!
//! V1 scope (documented, matches the probe capture's own simplification): static +
//! instanced + chunk geometry only -- skinned meshes are not drawn into the mirror
//! (the bindless face render omits the skinned tail), exactly like the probe capture.

use concinnity_core::gfx::frustum::Frustum;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::planar_reflection::{self, PlanarReflectors};
use concinnity_core::transform::mat4_inverse;
use concinnity_core::transform::mat4_mul;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use super::context::{DxContext, FRAMES, align256};
use super::cull::{INDIRECT_COMMAND_STRIDE, RegionCull};
use super::draw::ViewUniforms;
use super::error::map_hresult;
use super::graph_exec::GraphFrameParams;
use super::texture::{
    HDR_FORMAT, create_hdr_color_target, create_hdr_sampled_target, create_uav_buffer,
    transition_barrier, write_format_rtv, write_hdr_srv,
};
use crate::directx::descriptor_slot::SrvSlot;

// The engine capacity ceiling for distinct reflection planes: the count the
// reserved planar targets + resolve SRVs are sized to. Single-sourced from
// `gfx::planar_reflection` so the three backends stay in lockstep by construction.
// The per-frame budget passed to `assign_planar_slots` at init can be lower under a
// quality preset / GPU tier, never higher; panes past it fall back to the
// box-projected probe cube.
pub(in crate::directx) const MAX_PLANAR_PLANES: usize = planar_reflection::MAX_PLANAR_PLANES;

// Clip the reflection a hair toward the kept (camera) side of the plane so
// geometry exactly on the surface is not lost to near-plane precision. Matches
// `metal::planar::PLANAR_CLIP_BIAS`.
const PLANAR_CLIP_BIAS: f32 = 0.02;

// Texels a mirror's crop is grown by on every side, covering the bilinear
// footprint of the reflector's lookup.
const PLANAR_CROP_MARGIN: u32 = 2;

// Where a plane's mirror render lands. Multisampled planes share one MSAA color
// target and resolve out of it a plane at a time, the way the probe shares one
// face target across its six faces. Single-sampled planes have nothing to resolve,
// so each renders straight into its own resolve through a render-target view of
// it: no shared target, and no full-target copy per plane. Mirrors the Vulkan
// planar set, whose shared color is likewise `Some` only under MSAA.
enum PlanarColor {
    // The shared target every plane renders into, resolved into the plane's own
    // resolve before the next plane overwrites it.
    Multisampled(ID3D12Resource),
    // Nothing shared: each plane renders through an RTV of its own resolve.
    PerPlane,
}

// The set of distinct reflection planes for the world, each rendering its mirror
// into its own shader-readable resolve (directly, or through the shared MSAA
// color). A reflector samples the resolve of the slot it was assigned at init.
// The targets are rebuilt on resize alongside the HDR targets; the layout is
// fixed at init. `width` x `height` is the mirror target size, the render
// resolution scaled by the layout's mirror resolution.
pub(in crate::directx) struct PlanarReflectionSet {
    // The distinct reflector planes (one per resolve slot, re-oriented toward
    // the camera per frame), the reflector bounds a frame plans from, and the
    // mirror resolution.
    layout: PlanarReflectors,
    width: u32,
    height: u32,
    sample_count: u32,
    clear_color: [f32; 4],

    // The mirror render's color attachment(s) plus the depth shared across
    // planes. Own non-shader-visible RTV / DSV heaps: RTV slot 0 is the shared
    // MSAA target, or slot `i` is plane `i`'s resolve when single-sampled.
    color: PlanarColor,
    _depth: ID3D12Resource,
    _rtv_heap: ID3D12DescriptorHeap,
    _dsv_heap: ID3D12DescriptorHeap,
    rtv_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    rtv_stride: usize,
    depth_dsv: D3D12_CPU_DESCRIPTOR_HANDLE,

    // Per-plane shader-readable resolve + its SRV (CPU handle for the resize
    // rewrite, GPU handle for the glass pass to bind). The SRVs live in reserved
    // slots of the main shader-visible heap.
    resolves: Vec<ID3D12Resource>,
    resolve_srv_cpu: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    resolve_srv_gpu: Vec<SrvSlot>,

    // Per-(plane, frame) reflected `ViewUniforms` CBV ring, persistently mapped.
    // Indexed `plane * FRAMES + frame_idx`, so each frame writes its own slot and
    // never races the GPU reading a prior frame's reflected view.
    _view_cbvs: Vec<PooledBuffer>,
    view_ptrs: Vec<*mut u8>,
    view_gvas: Vec<u64>,

    // Per-frame reflected-frustum cull output: one indirect buffer per frame
    // holding `plane_count` regions of `n_cull` commands each, plus a per-frame
    // never-read cull-status scratch. Frame-indexed so frame N writes its own and
    // never races the GPU reading frame N-1's. The per-plane face render issues its
    // region (`region_offset`) of `planar_indirect[frame]`. Sized by the object
    // count (`n_cull`), which is fixed at init, so resize never touches them.
    planar_indirect: Vec<ID3D12Resource>,
    planar_status: Vec<ID3D12Resource>,
    n_cull: usize,
}

// SAFETY: the raw pointers `PlanarReflectionSet` holds are the mappings of upload buffers the
// struct also owns, so they stay valid for as long as it does and may move with it.
unsafe impl Send for PlanarReflectionSet {}
// SAFETY: the only writes through a shared reference are the mirror views, recorded by the one
// `PlanarReflection` pass a frame encodes (on whichever worker records it). Each write lands in
// its own `slot * FRAMES + frame` entry, which no other pass touches and which the GPU last read
// a frames-in-flight fence ago, so no two writers or a writer and the GPU share an entry.
unsafe impl Sync for PlanarReflectionSet {}

// Render-target build config for the planar set: MSAA sample count, the render
// resolution the mirror targets are scaled from, and the mirror-cull record count.
#[derive(Clone, Copy)]
pub(in crate::directx) struct PlanarConfig {
    // MSAA sample count matching the main pass.
    pub sample_count: u32,
    // Render width in pixels.
    pub width: u32,
    // Render height in pixels.
    pub height: u32,
    // Build-time draw-record count (`DxContext::cull_count`): sizes each plane's
    // region of the per-frame mirror-cull indirect buffer.
    pub n_cull: usize,
}

// Per-plane resolve descriptor handles (one entry per plane) plus the shared
// color clear value.
#[derive(Clone, Copy)]
pub(in crate::directx) struct PlanarTargets<'a> {
    pub resolve_srv_cpu: &'a [D3D12_CPU_DESCRIPTOR_HANDLE],
    pub resolve_srv_gpu: &'a [SrvSlot],
    pub clear_color: [f32; 4],
}

impl PlanarReflectionSet {
    // Build the planar set: shared color + depth at `width`x`height` (matching
    // the main pass's formats + sample count so the bindless face render binds the
    // standard pipeline), one resolve per plane with its SRV written into the
    // reserved heap slot, and the per-(plane, frame) reflected-view CBV ring.
    // `resolve_srv_cpu` / `resolve_srv_gpu` are the reserved heap descriptors, one
    // per plane in `layout`.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        config: PlanarConfig,
        layout: PlanarReflectors,
        targets: PlanarTargets,
    ) -> RenderResult<Self> {
        let planes = layout.planes();
        if planes.len() > MAX_PLANAR_PLANES {
            return Err(RenderError::Other(format!(
                "planar reflection: {} planes exceeds the {MAX_PLANAR_PLANES}-plane ceiling the \
                 reserved resolve descriptors are sized to",
                planes.len()
            )));
        }
        let device = alloc.device();
        let PlanarConfig {
            sample_count,
            width: render_w,
            height: render_h,
            n_cull,
        } = config;
        let (width, height) = layout.target_size(render_w, render_h);
        let PlanarTargets {
            resolve_srv_cpu,
            resolve_srv_gpu,
            clear_color,
        } = targets;
        let rtv_heap = create_rtv_heap(device, planes.len())?;
        let dsv_heap = create_dsv_heap(device)?;
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let rtv_base = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let depth_dsv = unsafe { dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: a property query on a live device; it only reads.
        let rtv_stride =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) }
                as usize;

        let depth =
            create_planar_depth(device, width.max(1), height.max(1), sample_count, depth_dsv)?;

        let mut resolves = Vec::with_capacity(planes.len());
        for (i, _) in planes.iter().enumerate() {
            let resolve =
                create_hdr_sampled_target(device, width.max(1), height.max(1), clear_color)?;
            write_hdr_srv(device, &resolve, resolve_srv_cpu[i]);
            resolves.push(resolve);
        }

        let color = create_planar_color(
            device,
            PlanarColorBuild {
                width: width.max(1),
                height: height.max(1),
                sample_count,
                clear_color,
                resolves: &resolves,
                rtv_base,
                rtv_stride,
            },
        )?;

        let mut view_cbvs = Vec::with_capacity(planes.len() * FRAMES);
        let mut view_ptrs = Vec::with_capacity(planes.len() * FRAMES);
        let mut view_gvas = Vec::with_capacity(planes.len() * FRAMES);
        for _ in 0..planes.len() * FRAMES {
            let cbv = alloc.alloc_buffer(
                256,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { cbv.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "planar: map view cbv"))?;
            view_gvas.push(com::gpu_va(&cbv));
            view_ptrs.push(ptr as *mut u8);
            view_cbvs.push(cbv);
        }

        // Per-frame mirror-cull output: `plane_count` regions of `n_cull` commands,
        // plus a never-read status scratch. Created in COMMON (D3D12 promotes buffers
        // from COMMON on first use), matching the shadow indirect buffers.
        let indirect_size =
            align256((planes.len() * n_cull * INDIRECT_COMMAND_STRIDE as usize) as u64);
        let status_size = align256((n_cull * std::mem::size_of::<u32>()) as u64).max(256);
        let mut planar_indirect = Vec::with_capacity(FRAMES);
        let mut planar_status = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            planar_indirect.push(create_uav_buffer(
                device,
                indirect_size.max(256),
                D3D12_RESOURCE_STATE_COMMON,
            )?);
            planar_status.push(create_uav_buffer(
                device,
                status_size,
                D3D12_RESOURCE_STATE_COMMON,
            )?);
        }

        Ok(Self {
            layout,
            width,
            height,
            sample_count,
            clear_color,
            color,
            _depth: depth,
            _rtv_heap: rtv_heap,
            _dsv_heap: dsv_heap,
            rtv_base,
            rtv_stride,
            depth_dsv,
            resolves,
            resolve_srv_cpu: resolve_srv_cpu.to_vec(),
            resolve_srv_gpu: resolve_srv_gpu.to_vec(),
            _view_cbvs: view_cbvs,
            view_ptrs,
            view_gvas,
            planar_indirect,
            planar_status,
            n_cull,
        })
    }

    // Recreate the depth + per-plane resolves + color attachment(s) for a new
    // render resolution and rewrite the RTV / DSV / resolve SRVs in place. The
    // descriptor slots do not move, so the transparent pass's GPU handles stay
    // valid. Mirrors the other `resize_to` resources.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        render_w: u32,
        render_h: u32,
    ) -> RenderResult<()> {
        let (w, h) = self.layout.target_size(render_w, render_h);
        self.width = w;
        self.height = h;
        self._depth = create_planar_depth(device, w, h, self.sample_count, self.depth_dsv)?;
        for i in 0..self.resolves.len() {
            let resolve = create_hdr_sampled_target(device, w, h, self.clear_color)?;
            write_hdr_srv(device, &resolve, self.resolve_srv_cpu[i]);
            self.resolves[i] = resolve;
        }
        self.color = create_planar_color(
            device,
            PlanarColorBuild {
                width: w,
                height: h,
                sample_count: self.sample_count,
                clear_color: self.clear_color,
                resolves: &self.resolves,
                rtv_base: self.rtv_base,
                rtv_stride: self.rtv_stride,
            },
        )?;
        Ok(())
    }

    // GPU descriptor handle of plane `slot`'s resolve SRV (what the glass pass
    // binds for a pane assigned to this slot).
    pub(in crate::directx) fn resolve_srv_gpu(&self, slot: usize) -> SrvSlot {
        self.resolve_srv_gpu[slot]
    }

    // Number of distinct reflector planes (at most one mirror render each per
    // frame).
    pub(in crate::directx) fn plane_count(&self) -> usize {
        self.layout.planes().len()
    }

    // This frame's mirror work under the (jittered) `view_proj` the reflectors
    // are rasterized with.
    pub(in crate::directx) fn frame_plan(
        &self,
        view_proj: [[f32; 4]; 4],
    ) -> planar_reflection::PlanarFramePlan {
        self.layout.frame_plan(view_proj)
    }

    // This frame's mirror-cull indirect buffer (the per-plane regions the face
    // render executes).
    fn indirect(&self, frame: usize) -> &ID3D12Resource {
        &self.planar_indirect[frame]
    }

    // GPU address of this frame's never-read mirror-cull status scratch.
    fn status_gva(&self, frame: usize) -> u64 {
        com::gpu_va(&self.planar_status[frame])
    }

    // Byte offset to plane `slot`'s region (of `n_cull` commands) in the indirect
    // buffer, for the face render's `ExecuteIndirect`.
    fn region_offset(&self, slot: usize) -> u32 {
        (slot * self.n_cull * INDIRECT_COMMAND_STRIDE as usize) as u32
    }

    // RTV plane `slot`'s mirror render draws through: the shared MSAA target, or
    // the plane's own resolve when single-sampled.
    fn face_rtv(&self, slot: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let index = rtv_slot_index(matches!(self.color, PlanarColor::Multisampled(_)), slot);
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.rtv_base.ptr + index * self.rtv_stride,
        }
    }

    // Open plane `slot`'s mirror render. Only the single-sampled path renders into
    // the resolve itself, so only it needs the state flip; under MSAA the render
    // targets the shared color, which never leaves RENDER_TARGET.
    fn begin_plane(&self, cmd: &ID3D12GraphicsCommandList, slot: usize) {
        if !matches!(self.color, PlanarColor::PerPlane) {
            return;
        }
        // SAFETY: the command list is in the recording state, and every resource, descriptor
        // and slice these commands name is live for the call.
        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(
                &self.resolves[slot],
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )]);
        }
    }

    // Close plane `slot`'s mirror render, leaving its resolve in
    // PIXEL_SHADER_RESOURCE for the glass sample. A multisampled render resolves
    // the shared color into it and puts the color back in RENDER_TARGET for the
    // next plane; a single-sampled one already wrote the resolve and only flips it
    // back.
    fn end_plane(&self, cmd: &ID3D12GraphicsCommandList, slot: usize) {
        let resolve = &self.resolves[slot];
        let PlanarColor::Multisampled(color) = &self.color else {
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.ResourceBarrier(&[transition_barrier(
                    resolve,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                )]);
            }
            return;
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor
        // and slice these commands name is live for the call.
        unsafe {
            cmd.ResourceBarrier(&[
                transition_barrier(
                    color,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                ),
                transition_barrier(
                    resolve,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_RESOLVE_DEST,
                ),
            ]);
            cmd.ResolveSubresource(resolve, 0, color, 0, HDR_FORMAT);
            cmd.ResourceBarrier(&[
                transition_barrier(
                    resolve,
                    D3D12_RESOURCE_STATE_RESOLVE_DEST,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ),
                transition_barrier(
                    color,
                    D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                ),
            ]);
        }
    }
}

impl DxContext {
    // Render the scene reflected across every plane the frame's plan keeps into
    // that plane's resolve, cropped to the plan's rectangle. A no-op (returns Ok)
    // when no set exists or no reflector is on screen. For each kept plane: a
    // dedicated reflected-frustum mirror cull (narrowed to the crop) fills the
    // plane's region of this frame's indirect buffer (reading the frame's
    // camera-independent object + draw-args), then the bindless face render draws
    // that region from the reflected view into the plane's color attachment,
    // limited to the crop, and leaves the plane's resolve shader-readable.
    // `Transparent` samples the resolves later in the same submission. Each plane
    // is oriented toward the camera so the oblique near-plane clip keeps the
    // camera's side.
    pub(in crate::directx) fn encode_planar_reflections(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        params: &GraphFrameParams<'_>,
    ) -> RenderResult<()> {
        let Some(set) = self.planar_reflection.as_ref() else {
            return Ok(());
        };
        let crops =
            params
                .planar
                .crops(set.plane_count(), set.width, set.height, PLANAR_CROP_MARGIN);
        let crops = crops.as_slice();
        if crops.is_empty() {
            return Ok(());
        }

        // Recover the (jittered) projection from this frame's view-projection so
        // the mirror render shares the main camera's projection + jitter, keeping
        // the reflection aligned with the reflective fragment's screen-space sample.
        let proj = mat4_mul(params.vp_mat, mat4_inverse(self.state.view.matrix));
        let prefilter_mip_count = self.scene.env_map.prefilter_mip_count as f32;

        // Per kept plane: compute the reflected matrices, write the reflected
        // view CBV, and collect the cropped reflected frustum + eye for the
        // mirror cull.
        let mut culls = [RegionCull::EMPTY; MAX_PLANAR_PLANES];
        let mut kept = 0;
        for &(slot, crop) in crops {
            let oriented =
                planar_reflection::orient_plane_toward(set.layout.planes()[slot], params.cam_pos);
            let m = planar_reflection::planar_matrices(
                self.state.view.matrix,
                proj,
                params.cam_pos,
                oriented,
                PLANAR_CLIP_BIAS,
            );
            let view = ViewUniforms {
                vp: m.view_proj,
                view: m.view,
                elapsed: params.elapsed,
                // No reflection resolve runs over the planar mirror render, so
                // the forward probe specular is its only reflection source.
                reflections_enabled: 0.0,
                cam_pos: [m.eye[0], m.eye[1], m.eye[2]],
                prefilter_mip_count,
                // A mirror render is always lit, whatever the viewport shows.
                shade_mode: 0.0,
                ambient_occlusion: 0.0,
                sky_rot: self.state.view.sky_rot,
            };
            let ring = slot * FRAMES + params.frame_idx;
            // SAFETY: `ring < planes.len() * FRAMES`; the CBV is 256 bytes and
            // `ViewUniforms` is 208. The slot is this frame's own, written before
            // the GPU reads it later on this cmd list.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &view as *const ViewUniforms as *const u8,
                    set.view_ptrs[ring],
                    std::mem::size_of::<ViewUniforms>(),
                );
            }
            if let Some(cull) = culls.get_mut(kept) {
                *cull = RegionCull {
                    region: slot,
                    frustum: Frustum::from_view_projection(crop.crop_view_projection(
                        m.view_proj,
                        set.width,
                        set.height,
                    )),
                    eye: m.eye,
                };
                kept += 1;
            }
        }

        // Reflected-frustum mirror cull into the kept planes' regions of this
        // frame's indirect buffer (one barrier flip around all of them).
        self.encode_planar_culls(
            cmd,
            params.frame_idx,
            &culls[..kept],
            set.indirect(params.frame_idx),
            set.status_gva(params.frame_idx),
            // Stride regions by the SAME fixed capacity `region_offset` reads with.
            set.n_cull,
        );

        // Per kept plane: render the culled region from the reflected view into
        // the plane's color attachment + the shared depth (against the frame's
        // object buffer), within the crop, then leave the plane's resolve
        // shader-readable.
        let frame_object_gva = com::gpu_va(&self.cull.object_buffer_resources[params.frame_idx]);
        let indirect = set.indirect(params.frame_idx);
        for &(slot, crop) in crops {
            let ring = slot * FRAMES + params.frame_idx;
            set.begin_plane(cmd, slot);
            self.encode_main_into_face(
                cmd,
                crate::directx::probe::FaceTargets {
                    rtv: set.face_rtv(slot),
                    dsv: set.depth_dsv,
                },
                crate::directx::probe::FaceUniforms {
                    view_gva: set.view_gvas[ring],
                    light_gva: params.light_gva,
                    shadow_ubo_gva: params.shadow_ubo_gva,
                },
                crate::directx::probe::IndirectDraw {
                    indirect,
                    indirect_offset: set.region_offset(slot),
                    object_gva: frame_object_gva,
                    material_params_gva: self.material_params_gva(params.frame_idx),
                },
                crate::directx::probe::FaceExtent {
                    width: set.width,
                    height: set.height,
                    area: Some(crop),
                },
            );
            set.end_plane(cmd, slot);
        }
        Ok(())
    }
}

// The non-shader-visible RTV heap for the planar color attachments: slot 0 is
// the shared MSAA target, or slot `i` is plane `i`'s resolve when single-sampled.
// Sized for the wider case so the sample count can pick either.
fn create_rtv_heap(
    device: &ID3D12Device,
    plane_count: usize,
) -> RenderResult<ID3D12DescriptorHeap> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: plane_count.max(1) as u32,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe { device.CreateDescriptorHeap(&desc) }
        .map_err(|e| map_hresult(e.code(), "planar: rtv heap"))
}

// Render dimensions, sample count and RTV heap slots for building the planar
// color attachment(s) over a set's per-plane resolves.
struct PlanarColorBuild<'a> {
    width: u32,
    height: u32,
    sample_count: u32,
    clear_color: [f32; 4],
    resolves: &'a [ID3D12Resource],
    rtv_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    rtv_stride: usize,
}

// Which RTV heap slot plane `slot` renders through. A multisampled set shares one
// color target in slot 0; a single-sampled one gives each plane the slot holding
// the view of its own resolve, so the slot IS the plane. Getting this wrong points
// every plane at one target, which reads as every mirror showing the first plane's
// reflection.
fn rtv_slot_index(multisampled: bool, slot: usize) -> usize {
    if multisampled { 0 } else { slot }
}

// Build the color attachment(s) and write their RTVs: one shared MSAA target in
// slot 0 when multisampled, else a view of each plane's resolve so the mirror
// render lands there directly. Called at init and again on resize, where the
// descriptor slots stay put and only the resources behind them change.
fn create_planar_color(
    device: &ID3D12Device,
    build: PlanarColorBuild<'_>,
) -> RenderResult<PlanarColor> {
    let PlanarColorBuild {
        width,
        height,
        sample_count,
        clear_color,
        resolves,
        rtv_base,
        rtv_stride,
    } = build;
    if sample_count > 1 {
        return Ok(PlanarColor::Multisampled(create_hdr_color_target(
            device,
            width,
            height,
            sample_count,
            rtv_base,
            clear_color,
        )?));
    }
    for (i, resolve) in resolves.iter().enumerate() {
        let rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr + i * rtv_stride,
        };
        write_format_rtv(device, resolve, rtv, HDR_FORMAT);
    }
    Ok(PlanarColor::PerPlane)
}

// A one-entry non-shader-visible DSV heap for the shared planar depth target.
fn create_dsv_heap(device: &ID3D12Device) -> RenderResult<ID3D12DescriptorHeap> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe { device.CreateDescriptorHeap(&desc) }
        .map_err(|e| map_hresult(e.code(), "planar: dsv heap"))
}

// Create the shared planar depth target (D32_FLOAT, matching the main pass's DSV
// format + the color's sample count) and write its DSV. Created in DEPTH_WRITE
// and left there (the face render clears it every plane). Mirrors
// `probe::create_bake_depth` at a rectangular render resolution.
fn create_planar_depth(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    sample_count: u32,
    dsv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> RenderResult<ID3D12Resource> {
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
        Width: width as u64,
        Height: height,
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
    .map_err(|e| map_hresult(e.code(), "planar: create depth"))?;
    let texture = tex_opt
        .ok_or_else(|| RenderError::Other("planar: create depth returned None".to_string()))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multisampled_planes_share_rtv_slot_zero() {
        // One shared color target, so every plane renders through the same view.
        for slot in 0..MAX_PLANAR_PLANES {
            assert_eq!(rtv_slot_index(true, slot), 0);
        }
    }

    #[test]
    fn single_sampled_planes_get_their_own_rtv_slot() {
        // Each plane renders straight into its own resolve, so the slot is the
        // plane. A collapse to 0 here would show plane 0's reflection everywhere.
        for slot in 0..MAX_PLANAR_PLANES {
            assert_eq!(rtv_slot_index(false, slot), slot);
        }
    }

    #[test]
    fn planar_capacity_is_four() {
        // The reserved planar-resolve heap block is sized off this. It now aliases
        // the single `gfx::planar_reflection` source, so this guards that the shared
        // capacity the heap layout assumes is still 4.
        assert_eq!(MAX_PLANAR_PLANES, 4);
        assert_eq!(MAX_PLANAR_PLANES, planar_reflection::MAX_PLANAR_PLANES);
    }
}
