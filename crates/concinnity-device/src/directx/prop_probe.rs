//! The prop reflection probe on DirectX: the six cube faces of the scene seen
//! from the prop, rendered every frame on the frame's command list from the
//! frame's own records like a planar mirror, and their convolution into the
//! probe's cube of the probe array. The record and the plan are the shared
//! `probe::prop`; this module owns the capture and encodes the `PropProbe`
//! graph node. Mirrors `metal::prop_probe`.
//!
//! Each face culls into its own region of the frame's indirect buffer, all six
//! before the first face draws, so no face's cull rewrites a list another face
//! still reads. The faces render into the layers of the capture's top mip
//! (through a shared multisampled target under MSAA), which the convolution
//! then reads.

use concinnity_core::math::{SaturatingFrom, narrow_f32, saturating_u32};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::probe::bake::CAPTURE_FACES;
use concinnity_engine::render::probe::capture::{self, PrefilterPlan};
use concinnity_engine::render::probe::prop::{
    FaceCycle, FaceSlice, PropProbeFrame, prop_probe_plan,
};
use concinnity_engine::render::probe::radiance::FaceStorage;
use concinnity_engine::render::uniforms::FaceCamera;
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::PooledBuffer;
use super::com;
use super::compact::DrawLists;
use super::context::{DxContext, align256};
use super::cull::{RegionCull, RegionPacking};
use super::draw::ViewUniforms;
use super::error::map_hresult;
use super::graph_exec::GraphFrameParams;
use super::probe::{
    FaceExtent, FaceTargets, FaceUniforms, IndirectDraw, create_bake_depth, create_dsv_heap,
    create_rtv_heap,
};
use super::probe_prefilter::{CaptureSetup, PROBE_MAX_MIPS, PrefilterGpu, PrefilterSlots};
use super::texture::{HDR_FORMAT, create_hdr_color_target, create_uav_buffer, transition_barrier};

/// Descriptors the prop probe's convolution reserves in the SRV heap.
pub(in crate::directx) const PROP_PROBE_SLOTS: usize =
    PrefilterSlots::span(saturating_u32(PROBE_MAX_MIPS));

// The state the capture rests in between frames: where the last GGX dispatch
// read it.
const CAPTURE_REST: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

// The world's prop probe: where it captures this frame, and the capture it
// renders into. `None` capture until the first frame that rides one.
#[derive(Default)]
pub(in crate::directx) struct PropProbe {
    // This frame's capture and the faces it renders.
    frame: Option<(PropProbeFrame, FaceSlice)>,
    cycle: FaceCycle,
    gpu: Option<PropProbeGpu>,
    // Set once building the capture failed, which is not retried.
    failed: bool,
}

// What the capture is built against; a change rebuilds it.
#[derive(Clone, Copy, PartialEq, Eq)]
struct CaptureKey {
    resolution: u32,
    cubes: usize,
    samples: u32,
}

// The capture and the convolution into the probe's cube, the face targets,
// and per frame in flight the face views and the six regions they cull into.
struct PropProbeGpu {
    key: CaptureKey,
    plan: PrefilterPlan,
    prefilter: PrefilterGpu,
    // The face targets: a view of each capture layer, or under MSAA the
    // shared multisampled color resolved into them, and the shared depth.
    _rtv_heap: ID3D12DescriptorHeap,
    _dsv_heap: ID3D12DescriptorHeap,
    face_rtvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    msaa: Option<(ID3D12Resource, D3D12_CPU_DESCRIPTOR_HANDLE)>,
    _depth: ID3D12Resource,
    dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    frames: usize,
    // Per (face, frame), at `face * frames + frame`: the face's mapped view
    // block.
    _view_cbvs: Vec<PooledBuffer>,
    view_ptrs: Vec<*mut u8>,
    view_gvas: Vec<u64>,
    // Per frame: six regions of `records` commands, their status and group
    // counts.
    records: usize,
    indirect: Vec<ID3D12Resource>,
    status: Vec<ID3D12Resource>,
    group_counts: Vec<ID3D12Resource>,
}

// SAFETY: the raw pointers are the mappings of upload buffers the struct also
// owns, so they stay valid for as long as it does and may move with it.
unsafe impl Send for PropProbeGpu {}
// SAFETY: the only writes through a shared reference are the face views,
// recorded by the one `PropProbe` pass a frame encodes. Each lands in its own
// `face * frames + frame` entry, which the GPU last read a frames-in-flight
// fence ago.
unsafe impl Sync for PropProbeGpu {}

impl PropProbeGpu {
    // Where frame `frame`'s regions pack each face's draws.
    fn packing(&self, frame: usize) -> RegionPacking<'_> {
        RegionPacking {
            group_counts: &self.group_counts[frame],
            group_stride: super::compact::group_count_bytes(self.records),
            status: &self.status[frame],
            status_stride: status_stride(self.records),
            lists_stride: super::compact::indirect_buffer_bytes(self.records),
        }
    }

    // Face `face`'s lists in frame `frame`'s indirect buffer.
    fn lists(&self, face: usize, frame: usize) -> DrawLists<'_> {
        DrawLists {
            buffer: &self.indirect[frame],
            base: face as u64 * super::compact::indirect_buffer_bytes(self.records),
            commands: self.records,
        }
    }
}

// Bytes of one face's cull status: a word per record.
fn status_stride(records: usize) -> u64 {
    align256((records * size_of::<u32>()) as u64).max(256)
}

impl DxContext {
    // Whether this backend can capture the world's prop probe: the bindless
    // cull it draws through, the convolution kernels, and a cube for it.
    fn prop_probe_supported(&self) -> bool {
        self.cull.main_bindless_pso.is_some()
            && self.cull.cull_kernels.is_some()
            && self.probe.prefilter.is_some()
            && self.probe.gpu.cubes.is_some()
            && self.probe.book.prop_cube().is_some()
            && !self.probe.prop.failed
    }

    // This frame's prop probe, ahead of the probe records and the graph: where
    // the `PropProbe` node captures, and the probe's record, which the shaders
    // read once a frame before has filled the cube. A frame whose graph does
    // not run the node withdraws the record.
    pub(in crate::directx) fn prepare_prop_probe(
        &mut self,
        frame: Option<PropProbeFrame>,
        enabled: bool,
    ) {
        let Some(frame) = frame.filter(|_| enabled && self.prop_probe_supported()) else {
            self.probe.prop.frame = None;
            self.probe.prop.cycle.reset();
            self.probe.book.set_prop_probe(None);
            return;
        };
        if let Err(e) = self.ensure_prop_capture(frame.resolution) {
            tracing::error!("reflection probes: the prop probe could not be built: {e}");
            self.probe.prop = PropProbe {
                failed: true,
                ..PropProbe::default()
            };
            self.probe.book.set_prop_probe(None);
            return;
        }
        let slice = self.probe.prop.cycle.advance(frame.faces_per_frame);
        self.probe.prop.frame = Some((frame, slice));
        self.probe
            .book
            .set_prop_probe(slice.filled.then_some(&frame));
    }

    // Build the capture for faces of `resolution` texels, unless the one held
    // already matches. A new capture has not filled the cube yet. Replacing
    // one idles the GPU first, since a command list does not keep the
    // resources it recorded alive.
    fn ensure_prop_capture(&mut self, resolution: u32) -> RenderResult<()> {
        let cubes =
            self.probe.gpu.cubes.as_ref().ok_or_else(|| {
                RenderError::Other("probe: no cube array for the prop probe".into())
            })?;
        let key = CaptureKey {
            resolution,
            cubes: windows::core::Interface::as_raw(cubes.resource()) as usize,
            samples: self.targets.hdr.msaa_samples.max(1),
        };
        if self.probe.prop.gpu.as_ref().is_some_and(|g| g.key == key) {
            return Ok(());
        }
        if self.probe.prop.gpu.is_some() {
            self.wait_idle();
        }
        self.probe.prop.gpu = None;
        self.probe.prop.frame = None;
        self.probe.prop.cycle.reset();
        self.probe.prop.gpu = Some(PropProbeGpu::new(self, key)?);
        Ok(())
    }

    // The `PropProbe` node: the culls of this frame's faces, each against its
    // face's frustum with the prop left out, into the face's region; then each
    // face drawn from its region into its layer of the capture; then, once the
    // cycle's last face is in, the convolution of the capture into the
    // probe's cube.
    pub(in crate::directx) fn encode_prop_probe(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        params: &GraphFrameParams<'_>,
    ) -> RenderResult<()> {
        let (Some((frame, slice)), Some(gpu)) =
            (self.probe.prop.frame, self.probe.prop.gpu.as_ref())
        else {
            return Ok(());
        };
        let f = params.frame_idx % gpu.frames;
        let eye = frame.eye();
        let size = gpu.plan.face_size();
        let camera = FaceCamera {
            vp: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            cam_pos: eye,
            elapsed: narrow_f32(params.elapsed),
            prefilter_mip_count: self.scene.env_map.prefilter_mip_count as f32,
            sky_rot: self.state.view.sky_rot,
            irradiance: self.irradiance_lane(),
            pre_exposure: FaceStorage::PROBE.scale(),
            daylight: self.state.view.daylight,
            tile_morph: self.state.tile_morph_view(),
        };
        let mut culls = [RegionCull::EMPTY; CAPTURE_FACES];
        let culls = &mut culls[..slice.count];
        for (face, cull) in slice.faces().zip(culls.iter_mut()) {
            let view = ViewUniforms::face(FaceCamera {
                vp: capture::face_view_projection(eye, face),
                view: capture::face_view_matrix(eye, face),
                ..camera
            })
            .with_stand_ins(frame.stand_ins());
            // SAFETY: the entry is this frame's own, sized for a `ViewUniforms`, and
            // written before the GPU reads it later on this list.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const view).cast::<u8>(),
                    gpu.view_ptrs[face * gpu.frames + f],
                    size_of::<ViewUniforms>(),
                );
            }
            *cull = RegionCull {
                region: face,
                frustum: capture::face_frustum(eye, face, frame.capture_distance),
                eye,
                exclude: frame.records,
                capture_tiles: self.state.placement.capture_tiles,
            };
        }
        self.encode_planar_culls(
            cmd,
            params.frame_idx,
            culls,
            &gpu.indirect[f],
            gpu.records,
            gpu.packing(f),
        );

        let capture = gpu.prefilter.capture();
        let arrival = if gpu.msaa.is_some() {
            D3D12_RESOURCE_STATE_RESOLVE_DEST
        } else {
            D3D12_RESOURCE_STATE_RENDER_TARGET
        };
        // SAFETY: the command list is recording; the capture rests in `CAPTURE_REST`.
        unsafe { cmd.ResourceBarrier(&[transition_barrier(capture, CAPTURE_REST, arrival)]) };
        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[params.frame_idx]);
        for face in slice.faces() {
            let rtv = gpu
                .msaa
                .as_ref()
                .map_or(gpu.face_rtvs[face], |(_, rtv)| *rtv);
            self.encode_main_into_face(
                cmd,
                FaceTargets { rtv, dsv: gpu.dsv },
                FaceUniforms {
                    view_gva: gpu.view_gvas[face * gpu.frames + f],
                    light_gva: params.light_gva,
                    shadow_ubo_gva: params.shadow_ubo_gva,
                    canopy_gva: self.canopy.gpu_va(params.frame_idx),
                    canopy_tables_gva: self.canopy.tables_gpu_va(params.frame_idx),
                    clusters: None,
                    storage: FaceStorage::PROBE,
                    probes: Some(params.frame_idx),
                    sky_from_cube: true,
                },
                IndirectDraw {
                    lists: gpu.lists(face, f),
                    object_gva,
                    material_params_gva: self.material_params_gva(params.frame_idx),
                },
                FaceExtent {
                    width: size,
                    height: size,
                    area: None,
                },
            );
            if let Some((color, _)) = gpu.msaa.as_ref() {
                resolve_face(cmd, color, capture, face, gpu.plan.mips());
            }
        }

        if !slice.convolve {
            // SAFETY: the command list is recording; the faces left the capture in its arrival
            // state.
            unsafe { cmd.ResourceBarrier(&[transition_barrier(capture, arrival, CAPTURE_REST)]) };
            return Ok(());
        }
        self.encode_probe_pyramid(cmd, &gpu.prefilter, &gpu.plan)?;
        for mip in 1..gpu.plan.mips() {
            self.encode_probe_ggx_mip(cmd, &gpu.prefilter, &gpu.plan, mip)?;
        }
        let barriers = self.probe.gpu.bound_cubes().cube_barriers(
            gpu.prefilter.cube(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            super::probe_set::PROBE_CUBES_STATE,
        );
        // SAFETY: the command list is recording; the pyramid build moved the cube to
        // UNORDERED_ACCESS and left the capture in `CAPTURE_REST`.
        unsafe { cmd.ResourceBarrier(&barriers) };
        Ok(())
    }
}

// Resolve the shared multisampled color into face `face` of the capture's top
// mip, and put the color back to render the next face.
fn resolve_face(
    cmd: &ID3D12GraphicsCommandList,
    color: &ID3D12Resource,
    capture: &ID3D12Resource,
    face: usize,
    mips: u32,
) {
    // SAFETY: the command list is recording; the color rests in RENDER_TARGET.
    unsafe {
        cmd.ResourceBarrier(&[transition_barrier(
            color,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
        )]);
    }
    let dst = u32::saturating_from(face) * mips;
    // SAFETY: the color is in RESOLVE_SOURCE and the capture in RESOLVE_DEST, both
    // `HDR_FORMAT`, and `dst` is face `face`'s top mip.
    unsafe { cmd.ResolveSubresource(capture, dst, color, 0, HDR_FORMAT) };
    // SAFETY: as above, the color is in RESOLVE_SOURCE.
    unsafe {
        cmd.ResourceBarrier(&[transition_barrier(
            color,
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        )]);
    }
}

// Write a render-target view of face `face` of a cube resource's top mip.
fn write_face_rtv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    face: usize,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: HDR_FORMAT,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_RTV {
                MipSlice: 0,
                FirstArraySlice: u32::saturating_from(face),
                ArraySize: 1,
                PlaneSlice: 0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and
    // the destination addresses a slot of a heap the caller owns.
    unsafe { device.CreateRenderTargetView(resource, Some(&raw const desc), rtv) };
}

impl PropProbeGpu {
    fn new(ctx: &DxContext, key: CaptureKey) -> RenderResult<Self> {
        let device = &ctx.hw.device;
        let alloc = &ctx.hw.alloc;
        let missing =
            |what: &str| RenderError::Other(format!("probe: no {what} for the prop probe"));
        let cubes = ctx
            .probe
            .gpu
            .cubes
            .as_ref()
            .ok_or_else(|| missing("cube array"))?;
        let cube = ctx.probe.book.prop_cube().ok_or_else(|| missing("cube"))?;
        let plan = prop_probe_plan(key.resolution);
        let size = plan.face_size();
        let multisampled = key.samples > 1;
        let prefilter = PrefilterGpu::new(
            ctx,
            &plan,
            cubes,
            cube,
            CaptureSetup {
                slots: PrefilterSlots::block(
                    ctx.descriptors.layout.prop_probe_base_slot,
                    plan.mips(),
                ),
                initial: CAPTURE_REST,
                arrival: if multisampled {
                    D3D12_RESOURCE_STATE_RESOLVE_DEST
                } else {
                    D3D12_RESOURCE_STATE_RENDER_TARGET
                },
                render_target: true,
            },
        )?;

        let rtv_count = CAPTURE_FACES + usize::from(multisampled);
        let rtv_heap = create_rtv_heap(device, u32::saturating_from(rtv_count))?;
        let dsv_heap = create_dsv_heap(device)?;
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let rtv_base = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let dsv = unsafe { dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: a property query on a live device; it only reads.
        let rtv_stride =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) }
                as usize;
        let rtv_at = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr + i * rtv_stride,
        };
        let face_rtvs: Vec<_> = (0..CAPTURE_FACES).map(rtv_at).collect();
        for (face, &rtv) in face_rtvs.iter().enumerate() {
            write_face_rtv(device, prefilter.capture(), face, rtv);
        }
        let clear = FaceStorage::PROBE.clear_color(ctx.state.view.clear_color);
        let msaa = multisampled
            .then(|| {
                let rtv = rtv_at(CAPTURE_FACES);
                create_hdr_color_target(device, size, size, key.samples, rtv, clear)
                    .map(|color| (color, rtv))
            })
            .transpose()?;
        let depth = create_bake_depth(device, size, key.samples, dsv)?;

        let frames = alloc.frames_in_flight();
        let ring = CAPTURE_FACES * frames;
        let mut view_cbvs = Vec::with_capacity(ring);
        let mut view_ptrs = Vec::with_capacity(ring);
        let mut view_gvas = Vec::with_capacity(ring);
        for _ in 0..ring {
            let cbv = alloc.alloc_buffer(
                align256(size_of::<ViewUniforms>() as u64),
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a
            // live local that receives the mapping.
            unsafe { cbv.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "prop probe: map view cbv"))?;
            view_gvas.push(com::gpu_va(&cbv));
            view_ptrs.push(ptr.cast::<u8>());
            view_cbvs.push(cbv);
        }

        // Six regions per frame, created in COMMON like the mirror culls'.
        let records = ctx.cull.bucket_stride;
        let faces = CAPTURE_FACES as u64;
        let mut indirect = Vec::with_capacity(frames);
        let mut status = Vec::with_capacity(frames);
        let mut group_counts = Vec::with_capacity(frames);
        for _ in 0..frames {
            indirect.push(create_uav_buffer(
                device,
                faces * super::compact::indirect_buffer_bytes(records),
            )?);
            status.push(create_uav_buffer(device, faces * status_stride(records))?);
            group_counts.push(create_uav_buffer(
                device,
                faces * super::compact::group_count_bytes(records),
            )?);
        }

        Ok(Self {
            key,
            plan,
            prefilter,
            _rtv_heap: rtv_heap,
            _dsv_heap: dsv_heap,
            face_rtvs,
            msaa,
            _depth: depth,
            dsv,
            frames,
            _view_cbvs: view_cbvs,
            view_ptrs,
            view_gvas,
            records,
            indirect,
            status,
            group_counts,
        })
    }
}
