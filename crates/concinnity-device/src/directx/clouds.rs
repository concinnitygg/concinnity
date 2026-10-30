//! The atmosphere's clouds for the D3D12 backend (see
//! `concinnity_engine::render::environment::clouds`): the noise volume the layers are
//! carved from, built by the first frame's atmosphere pass; the shadow map and
//! the panorama that pass writes for the shaded passes and the sky's cubes; the
//! half-resolution target the march writes; and the composite that blends it
//! over the scene behind the haze. Like every texture the GPU writes, they are
//! committed resources.
//!
//! Both rest readable by any shader stage; each kernel opens its output for
//! unordered access and closes it again inside the pass, so nothing crosses
//! the node boundary in a different state.
//!
//! Mirrors src/metal/clouds.rs and src/vulkan/clouds.rs.

use std::cell::Cell;

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::environment::atmosphere::{AtmosphereFrame, AtmosphereSettings};
use concinnity_engine::render::environment::clouds::{
    CloudParams, NOISE_SIZE, PANORAMA_SIZE, SHADOW_SIZE, cloud_target_size,
};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::builtin_shaders::{
    CLOUD_COMPOSITE_FRAG, CLOUD_MARCH, CLOUD_NOISE, CLOUD_PANORAMA, CLOUD_SHADOW, CompileProgram,
    FULLSCREEN_VERT,
};
use super::com;
use super::context::{DxContext, align256};
use super::descriptor_slot::{DescriptorTables, SrvSlot};
use super::error::map_hresult;
use super::pso::{Blend, GraphicsPso, Raster, compute_pso};
use super::root_sig::{RootSig, SamplerState, Visibility};
use super::texture::{HDR_FORMAT, transition_barrier};

const NOISE_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
const TARGET_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;
const RESTING: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
const NOISE_GROUP: u32 = 4;
const MARCH_GROUP: u32 = 8;

/// The heap slots the clouds own: the UAVs of the noise volume, the target,
/// the shadow map and the panorama, then their SRVs in the same order.
pub(in crate::directx) const CLOUD_SLOTS: usize = 8;
const NOISE_UAV: usize = 0;
const TARGET_UAV: usize = 1;
const SHADOW_UAV: usize = 2;
const PANORAMA_UAV: usize = 3;
const NOISE_SRV: usize = 4;
const TARGET_SRV: usize = 5;
const SHADOW_SRV: usize = 6;
const PANORAMA_SRV: usize = 7;

/// The CPU and GPU handles of the clouds' heap slots.
#[derive(Clone, Copy)]
pub(in crate::directx) struct CloudSlots {
    pub cpu: [D3D12_CPU_DESCRIPTOR_HANDLE; CLOUD_SLOTS],
    pub gpu: [SrvSlot; CLOUD_SLOTS],
}

pub(in crate::directx) struct DxClouds {
    // This frame's constants, `None` until a frame carries the clouds' inputs.
    params: Cell<Option<CloudParams>>,
    noise: ID3D12Resource,
    // Whether an atmosphere pass has built the noise volume yet.
    noise_built: Cell<bool>,
    shadow: ID3D12Resource,
    panorama: ID3D12Resource,
    target: ID3D12Resource,
    target_size: [u32; 2],
    slots: CloudSlots,
    params_buffers: Vec<PooledBuffer>,
    params_ptrs: Vec<*mut u8>,
    frames: Cell<u64>,
    noise_root_sig: ID3D12RootSignature,
    noise_pso: ID3D12PipelineState,
    march_root_sig: ID3D12RootSignature,
    march_pso: ID3D12PipelineState,
    shadow_pso: ID3D12PipelineState,
    panorama_pso: ID3D12PipelineState,
    composite_root_sig: ID3D12RootSignature,
    composite_pso: ID3D12PipelineState,
}

fn create_texture(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    label: &str,
) -> RenderResult<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut out: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
    // new COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &raw const heap,
            D3D12_HEAP_FLAG_NONE,
            desc,
            RESTING,
            None,
            &raw mut out,
        )
    }
    .map_err(|e| map_hresult(e.code(), &format!("create {label}")))?;
    out.ok_or_else(|| RenderError::Other(format!("create {label} returned None")))
}

fn texture_desc(extent: [u32; 3], format: DXGI_FORMAT) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: if extent[2] > 1 {
            D3D12_RESOURCE_DIMENSION_TEXTURE3D
        } else {
            D3D12_RESOURCE_DIMENSION_TEXTURE2D
        },
        Width: u64::from(extent[0]),
        Height: extent[1],
        DepthOrArraySize: u16::saturating_from(extent[2]),
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

// The SRV and UAV of a texture `depth` slices deep (1 for a 2D one).
fn write_views(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    format: DXGI_FORMAT,
    depth: u32,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let (srv, uav) = if depth > 1 {
        (
            D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            },
            D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_UAV {
                        MipSlice: 0,
                        FirstWSlice: 0,
                        WSize: depth,
                    },
                },
            },
        )
    } else {
        (
            D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            },
            D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV {
                        MipSlice: 0,
                        PlaneSlice: 0,
                    },
                },
            },
        )
    };
    // SAFETY: `srv` and `resource` are live for the call, and `srv_cpu` addresses a slot this
    // context reserved in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const srv), srv_cpu) };
    // SAFETY: `uav` and `resource` are live for the call, and `uav_cpu` is another reserved slot
    // in that same owned heap.
    unsafe { device.CreateUnorderedAccessView(resource, None, Some(&raw const uav), uav_cpu) };
}

// The target for a scene `size` pixels across, its views in `slots`.
fn create_target(
    device: &ID3D12Device,
    size: [u32; 2],
    slots: &CloudSlots,
) -> RenderResult<ID3D12Resource> {
    let target = create_texture(
        device,
        &texture_desc([size[0], size[1], 1], TARGET_FORMAT),
        "cloud target",
    )?;
    write_views(
        device,
        &target,
        TARGET_FORMAT,
        1,
        slots.cpu[TARGET_SRV],
        slots.cpu[TARGET_UAV],
    );
    Ok(target)
}

// The noise kernel's root signature:
//   [0] table  u0     the noise volume
fn noise_root_sig(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .uav_table(0, 1, Visibility::All)
        .build(device, "cloud noise root sig")
}

// The march's root signature:
//   [0] root CBV b0   AtmosphereParams
//   [1] root CBV b1   CloudParams
//   [2] table  t0     scene depth
//   [3] table  t1     aerial volume
//   [4] table  t2-t3  transmittance and multiple scattering
//   [5] table  t4     the noise volume
//   [6] table  u0     the target
// Static linear-clamp sampler s0, linear-wrap sampler s1.
fn march_root_sig(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .cbv(1, Visibility::All)
        .srv_table(0, 1, Visibility::All)
        .srv_table(1, 1, Visibility::All)
        .srv_table(2, 2, Visibility::All)
        .srv_table(4, 1, Visibility::All)
        .uav_table(0, 1, Visibility::All)
        .static_sampler(SamplerState::LinearClamp, 0, Visibility::All)
        .static_sampler(SamplerState::LinearWrap, 1, Visibility::All)
        .build(device, "cloud march root sig")
}

// The composite's root signature:
//   [0] root CBV b0   CloudParams
//   [1] table  t0     scene depth
//   [2] table  t1     the target
fn composite_root_sig(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::Pixel)
        .srv_table(0, 1, Visibility::Pixel)
        .srv_table(1, 1, Visibility::Pixel)
        .build(device, "cloud composite root sig")
}

fn map_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    program: &concinnity_engine::render::shader::programs::ShaderProgram,
    hot_reload: bool,
) -> RenderResult<ID3D12PipelineState> {
    compute_pso(
        device,
        root_sig,
        &program.compile(hot_reload)?,
        program.label,
    )
}

fn march_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
) -> RenderResult<ID3D12PipelineState> {
    compute_pso(
        device,
        root_sig,
        &CLOUD_MARCH.at(msaa_samples > 1).compile(hot_reload)?,
        CLOUD_MARCH.label,
    )
}

// `(light, 1 - T)` over the hazed scene -> `scene * T + light`.
fn composite_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
) -> RenderResult<ID3D12PipelineState> {
    let vs = FULLSCREEN_VERT.compile(hot_reload)?;
    let ps = CLOUD_COMPOSITE_FRAG
        .at(msaa_samples > 1)
        .compile(hot_reload)?;
    GraphicsPso::fullscreen(root_sig, &vs, &ps, HDR_FORMAT, Blend::PremultipliedOver)
        .raster(Raster {
            depth_clip: false,
            ..Raster::default()
        })
        .build(device, "cloud composite")
}

impl DxClouds {
    // The volume, a target for a scene `scene` pixels across, the pipelines
    // and the parameter ring, their views in `slots`.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        slots: CloudSlots,
        scene: [u32; 2],
        msaa_samples: u32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let noise = create_texture(
            device,
            &texture_desc([NOISE_SIZE; 3], NOISE_FORMAT),
            "cloud noise volume",
        )?;
        write_views(
            device,
            &noise,
            NOISE_FORMAT,
            NOISE_SIZE,
            slots.cpu[NOISE_SRV],
            slots.cpu[NOISE_UAV],
        );
        let target_size = cloud_target_size(scene[0], scene[1]);
        let target = create_target(device, target_size, &slots)?;
        let map = |size: [u32; 2], srv: usize, uav: usize, label: &str| {
            let map = create_texture(
                device,
                &texture_desc([size[0], size[1], 1], TARGET_FORMAT),
                label,
            )?;
            write_views(
                device,
                &map,
                TARGET_FORMAT,
                1,
                slots.cpu[srv],
                slots.cpu[uav],
            );
            Ok::<_, RenderError>(map)
        };
        let shadow = map([SHADOW_SIZE; 2], SHADOW_SRV, SHADOW_UAV, "cloud shadow map")?;
        let panorama = map(PANORAMA_SIZE, PANORAMA_SRV, PANORAMA_UAV, "cloud panorama")?;

        let size = align256(size_of::<CloudParams>() as u64);
        let mut params_buffers = Vec::new();
        let mut params_ptrs = Vec::new();
        for _ in 0..alloc.frames_in_flight() {
            let buf = alloc.alloc_buffer(
                size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map cloud params"))?;
            params_ptrs.push(ptr.cast::<u8>());
            params_buffers.push(buf);
        }

        let noise_root_sig = noise_root_sig(device)?;
        let noise_pso = compute_pso(
            device,
            &noise_root_sig,
            &CLOUD_NOISE.compile(hot_reload)?,
            CLOUD_NOISE.label,
        )?;
        let march_root_sig = march_root_sig(device)?;
        let march_pso = march_pso(device, &march_root_sig, msaa_samples, hot_reload)?;
        let shadow_pso = map_pso(device, &march_root_sig, &CLOUD_SHADOW, hot_reload)?;
        let panorama_pso = map_pso(device, &march_root_sig, &CLOUD_PANORAMA, hot_reload)?;
        let composite_root_sig = composite_root_sig(device)?;
        let composite_pso = composite_pso(device, &composite_root_sig, msaa_samples, hot_reload)?;
        Ok(Self {
            params: Cell::new(None),
            noise,
            noise_built: Cell::new(false),
            shadow,
            panorama,
            target,
            target_size,
            slots,
            params_buffers,
            params_ptrs,
            frames: Cell::new(0),
            noise_root_sig,
            noise_pso,
            march_root_sig,
            march_pso,
            shadow_pso,
            panorama_pso,
            composite_root_sig,
            composite_pso,
        })
    }

    // Size the target for a scene `scene` pixels across, its views rewritten
    // in place. The device is idle across a resize.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        scene: [u32; 2],
    ) -> RenderResult<()> {
        let size = cloud_target_size(scene[0], scene[1]);
        if size != self.target_size {
            self.target = create_target(device, size, &self.slots)?;
            self.target_size = size;
        }
        Ok(())
    }

    // Rebuild every pipeline from the shader sources (shader hot reload).
    pub(in crate::directx) fn rebuild_pipelines(
        &mut self,
        device: &ID3D12Device,
        msaa_samples: u32,
    ) -> RenderResult<()> {
        let noise_pso = compute_pso(
            device,
            &self.noise_root_sig,
            &CLOUD_NOISE.compile(true)?,
            CLOUD_NOISE.label,
        )?;
        let march_pso = march_pso(device, &self.march_root_sig, msaa_samples, true)?;
        let shadow_pso = map_pso(device, &self.march_root_sig, &CLOUD_SHADOW, true)?;
        let panorama_pso = map_pso(device, &self.march_root_sig, &CLOUD_PANORAMA, true)?;
        let composite_pso = composite_pso(device, &self.composite_root_sig, msaa_samples, true)?;
        self.noise_pso = noise_pso;
        self.march_pso = march_pso;
        self.shadow_pso = shadow_pso;
        self.panorama_pso = panorama_pso;
        self.composite_pso = composite_pso;
        self.noise_built.set(false);
        Ok(())
    }

    // This frame's constants for `settings` and `frame` over a scene `scene`
    // pixels across, written into frame slot `frame_idx`'s buffer.
    pub(in crate::directx) fn prepare(
        &self,
        settings: &AtmosphereSettings,
        frame: Option<&AtmosphereFrame>,
        scene: [u32; 2],
        frame_idx: usize,
    ) {
        let index = self.frames.get();
        let params = match (
            settings.clouds.as_ref(),
            frame.and_then(|f| f.clouds.as_ref()),
        ) {
            (Some(clouds), Some(cloud_frame)) => Some(CloudParams::new(
                clouds,
                settings.bottom_radius,
                cloud_frame,
                scene,
                index,
            )),
            _ => None,
        };
        self.params.set(params);
        self.frames.set(index.wrapping_add(1));
        if let (Some(params), Some(&ptr)) = (params, self.params_ptrs.get(frame_idx)) {
            // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer
            // sized for this block, and the source is a separate live value.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const params).cast::<u8>(),
                    ptr,
                    size_of::<CloudParams>(),
                );
            }
        }
    }

    // Whether this frame has the clouds' constants.
    pub(in crate::directx) fn live(&self) -> bool {
        self.params.get().is_some()
    }

    fn params_gva(&self, frame_idx: usize) -> Option<u64> {
        self.params_buffers.get(frame_idx).map(|b| com::gpu_va(b))
    }

    // Write the shadow map, and the panorama when `panorama` (the sky cube
    // redraws), after building the noise volume on the first frame. Inside the
    // atmosphere's pass, after its aerial volume; leaves the march root
    // signature bound.
    pub(in crate::directx) fn encode_maps(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        inputs: MarchInputs,
        frame_idx: usize,
        panorama: bool,
    ) {
        let Some(params_gva) = self.params_gva(frame_idx) else {
            return;
        };
        if !self.live() {
            return;
        }
        let open = |r: &ID3D12Resource| {
            transition_barrier(r, RESTING, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        let close = |r: &ID3D12Resource| {
            transition_barrier(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, RESTING)
        };
        if !self.noise_built.get() {
            // SAFETY: the command list is recording and the noise volume rests in `RESTING`.
            unsafe { cmd.ResourceBarrier(&[open(&self.noise)]) };
            // SAFETY: the noise root signature is owned by `clouds`.
            unsafe { cmd.SetComputeRootSignature(&self.noise_root_sig) };
            // SAFETY: the noise PSO is owned by `clouds` and matches the root signature above.
            unsafe { cmd.SetPipelineState(&self.noise_pso) };
            // SAFETY: table 0 of the noise root signature takes the volume's UAV, which lives in
            // the heap set above.
            unsafe { cmd.set_compute_srv_table(0, self.slots.gpu[NOISE_UAV]) };
            let groups = NOISE_SIZE.div_ceil(NOISE_GROUP);
            // SAFETY: the pipeline, root signature and table the dispatch writes are bound above.
            unsafe { cmd.Dispatch(groups, groups, groups) };
            // SAFETY: the volume was opened for unordered access above.
            unsafe { cmd.ResourceBarrier(&[close(&self.noise)]) };
            self.noise_built.set(true);
        }
        let shadow = [(&self.shadow, &self.shadow_pso, SHADOW_UAV, [SHADOW_SIZE; 2])];
        let both = [
            shadow[0],
            (
                &self.panorama,
                &self.panorama_pso,
                PANORAMA_UAV,
                PANORAMA_SIZE,
            ),
        ];
        let maps: &[(&ID3D12Resource, &ID3D12PipelineState, usize, [u32; 2])] =
            if panorama { &both } else { &shadow };
        bind_march(cmd, self, inputs, params_gva);
        for &(map, pso, uav, size) in maps {
            // SAFETY: the command list is recording and the map rests in `RESTING`.
            unsafe { cmd.ResourceBarrier(&[open(map)]) };
            // SAFETY: the kernel is owned by `clouds` and matches the march root signature bound
            // above.
            unsafe { cmd.SetPipelineState(pso) };
            // SAFETY: table 6 of the march root signature takes the map's UAV in the bound heap.
            unsafe { cmd.set_compute_srv_table(6, self.slots.gpu[uav]) };
            let groups = size.map(|s| s.div_ceil(MARCH_GROUP));
            // SAFETY: the pipeline, root signature, constants and tables are bound.
            unsafe { cmd.Dispatch(groups[0], groups[1], 1) };
            // SAFETY: the map was opened for unordered access above.
            unsafe { cmd.ResourceBarrier(&[close(map)]) };
        }
    }

    // The shadow map's SRV, which the shaded passes read.
    pub(in crate::directx) fn shadow_srv(&self) -> SrvSlot {
        self.slots.gpu[SHADOW_SRV]
    }

    // The panorama's SRV, which the sky cube and irradiance kernels read.
    pub(in crate::directx) fn panorama_srv(&self) -> SrvSlot {
        self.slots.gpu[PANORAMA_SRV]
    }
}

/// What the march borrows from the atmosphere: this frame's constants, its
/// tables' SRVs and the scene depth's.
#[derive(Clone, Copy)]
pub(in crate::directx) struct MarchInputs {
    pub atmo_params_gva: u64,
    pub depth_srv: SrvSlot,
    pub aerial_srv: SrvSlot,
    // The transmittance table's SRV, with the multiple-scattering table's
    // beside it.
    pub tables_srv: SrvSlot,
}

impl DxContext {
    // March the clouds into their target.
    pub(in crate::directx) fn encode_clouds(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        clouds: &DxClouds,
        inputs: MarchInputs,
        frame_idx: usize,
    ) {
        let Some(params_gva) = clouds.params_gva(frame_idx) else {
            return;
        };
        if !clouds.live() {
            return;
        }
        let open = |r: &ID3D12Resource| {
            transition_barrier(r, RESTING, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        let close = |r: &ID3D12Resource| {
            transition_barrier(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, RESTING)
        };
        // SAFETY: the descriptor heap is owned by `self.descriptors` and outlives the frame.
        unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
        // SAFETY: the command list is recording and the target rests in `RESTING`.
        unsafe { cmd.ResourceBarrier(&[open(&clouds.target)]) };
        bind_march(cmd, clouds, inputs, params_gva);
        // SAFETY: the march PSO is owned by `clouds` and matches the root signature bound above.
        unsafe { cmd.SetPipelineState(&clouds.march_pso) };
        // SAFETY: table 6 takes the target's UAV in the heap set above.
        unsafe { cmd.set_compute_srv_table(6, clouds.slots.gpu[TARGET_UAV]) };
        let [w, h] = clouds.target_size;
        // SAFETY: the pipeline, root signature, constants and tables are bound above.
        unsafe { cmd.Dispatch(w.div_ceil(MARCH_GROUP), h.div_ceil(MARCH_GROUP), 1) };
        // SAFETY: the target was opened for unordered access above.
        unsafe { cmd.ResourceBarrier(&[close(&clouds.target)]) };
    }

    // Blend the clouds over the hazed scene.
    pub(in crate::directx) fn encode_cloud_composite(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        clouds: &DxClouds,
        depth_srv: SrvSlot,
        frame_idx: usize,
    ) {
        let Some(params_gva) = clouds.params_gva(frame_idx) else {
            return;
        };
        if !clouds.live() {
            return;
        }
        // The pre-TAA scene and the main depth are graph resources: the
        // executor has put the scene in RENDER_TARGET and the depth in a
        // shader-resource state for this pass.
        let scene_rtv = self.post_scene_rtv();
        let w = self.targets.extent.render_width;
        let h = self.targets.extent.render_height;
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: w as f32,
            Height: h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: i32::saturating_from(w),
            bottom: i32::saturating_from(h),
        };
        // SAFETY: the command list is recording, and `scene_rtv` is a live descriptor the
        // executor has put in RENDER_TARGET for this pass.
        unsafe { cmd.OMSetRenderTargets(1, Some(&raw const scene_rtv), false, None) };
        // SAFETY: the viewport slice is a local that outlives the call.
        unsafe { cmd.RSSetViewports(&[viewport]) };
        // SAFETY: the scissor slice is a local that outlives the call.
        unsafe { cmd.RSSetScissorRects(&[scissor]) };
        // SAFETY: setting the topology only needs a recording command list.
        unsafe { cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) };
        // SAFETY: the composite PSO is owned by `clouds` and outlives the recording.
        unsafe { cmd.SetPipelineState(&clouds.composite_pso) };
        // SAFETY: the composite root signature is owned by `clouds` and matches the PSO above.
        unsafe { cmd.SetGraphicsRootSignature(&clouds.composite_root_sig) };
        // SAFETY: the descriptor heap is owned by `self.descriptors` and outlives the frame.
        unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
        // SAFETY: `params_gva` addresses this frame's live constant buffer, and root parameter 0
        // of the composite root signature is a CBV.
        unsafe { cmd.SetGraphicsRootConstantBufferView(0, params_gva) };
        // SAFETY: table 1 takes the depth's slot in the heap set above; the executor has put the
        // depth in a shader-resource state.
        unsafe { cmd.set_graphics_srv_table(1, depth_srv) };
        // SAFETY: table 2 takes the target's SRV in the heap set above; the march left the target
        // readable.
        unsafe { cmd.set_graphics_srv_table(2, clouds.slots.gpu[TARGET_SRV]) };
        // SAFETY: the target, pipeline, root signature and every table the draw reads are bound
        // above.
        unsafe { cmd.DrawInstanced(3, 1, 0, 0) };
    }
}

impl DxContext {
    // The clouds' shadow map's SRV the shaded passes read. Without clouds the
    // shadow lanes are off and the table is never read, so the LTC matrix's
    // SRV holds it.
    pub(in crate::directx) fn cloud_shadow_srv(&self) -> SrvSlot {
        self.atmosphere
            .as_ref()
            .and_then(|a| a.clouds.as_ref())
            .map_or(self.scene.area_light.ltc_table_gpu, DxClouds::shadow_srv)
    }
}

// The march root signature with this frame's constants and the tables
// every cloud kernel reads.
fn bind_march(
    cmd: &ID3D12GraphicsCommandList,
    clouds: &DxClouds,
    inputs: MarchInputs,
    params_gva: u64,
) {
    // SAFETY: the march root signature is owned by `clouds`.
    unsafe { cmd.SetComputeRootSignature(&clouds.march_root_sig) };
    // SAFETY: the atmosphere's constants for this frame are live, and parameter 0 of the
    // march root signature is a CBV.
    unsafe { cmd.SetComputeRootConstantBufferView(0, inputs.atmo_params_gva) };
    // SAFETY: the clouds' constants for this frame are live, and parameter 1 is a CBV.
    unsafe { cmd.SetComputeRootConstantBufferView(1, params_gva) };
    // SAFETY: table 2 takes the depth's slot in the bound heap; a kernel that never reads
    // the depth leaves it unread.
    unsafe { cmd.set_compute_srv_table(2, inputs.depth_srv) };
    // SAFETY: table 3 takes the aerial volume's slot in the bound heap.
    unsafe { cmd.set_compute_srv_table(3, inputs.aerial_srv) };
    // SAFETY: table 4 takes two slots, the transmittance and multiple-scattering tables',
    // which sit side by side in the bound heap.
    unsafe { cmd.set_compute_srv_table(4, inputs.tables_srv) };
    // SAFETY: table 5 takes the noise volume's SRV in the bound heap.
    unsafe { cmd.set_compute_srv_table(5, clouds.slots.gpu[NOISE_SRV]) };
}
