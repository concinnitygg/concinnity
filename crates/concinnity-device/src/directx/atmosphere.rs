//! The atmosphere for the D3D12 backend (see `concinnity_core::render::atmosphere`):
//! the tables, the compute pass that builds them, the haze pass that blends
//! the aerial volume over the opaque scene, and the sky pipeline that draws
//! the sky-view table behind it. The irradiance cube the last kernel writes
//! takes the environment's irradiance SRV slot, and the sky cube the
//! reflection prefilter convolves takes its prefilter slot, so every table
//! that reaches the environment's light reads the atmosphere's. Like every
//! texture the GPU writes, the tables and the cubes are committed resources.
//!
//! Every table and the cube rest readable by any shader stage; each kernel
//! opens its output for unordered access and closes it again inside the pass,
//! so nothing crosses the node boundary in a different state.
//!
//! Mirrors src/metal/atmosphere.rs and src/vulkan/atmosphere.rs.

use concinnity_core::math::SaturatingFrom;
use std::cell::Cell;

use concinnity_core::render::atmosphere::{
    AERIAL_SIZE, AirTables, AtmosphereFrame, AtmosphereParams, AtmosphereSettings, AtmosphereView,
    IRRADIANCE_FACE, MULTISCATTER_SIZE, SKY_CUBE_FACE, SKY_CUBE_PLAN, SKY_VIEW_HEIGHT,
    SKY_VIEW_WIDTH, SkyCubeKey, SkyCubeSchedule, SkyCubeWork, TRANSMITTANCE_HEIGHT,
    TRANSMITTANCE_WIDTH, TableReadiness,
};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shader_programs::ShaderProgram;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::builtin_shaders::{
    AERIAL_PERSPECTIVE_FRAG, ATMOSPHERE_AERIAL, ATMOSPHERE_IRRADIANCE, ATMOSPHERE_MULTISCATTER,
    ATMOSPHERE_SKY_CUBE, ATMOSPHERE_SKY_VIEW, ATMOSPHERE_TRANSMITTANCE, CompileProgram,
    FULLSCREEN_VERT, SKY_ATMOSPHERE_FRAG,
};
use super::com;
use super::context::{DxContext, align256};
use super::descriptor_slot::{DescriptorTables, SrvSlot};
use super::error::map_hresult;
use super::probe_prefilter::{
    PrefilterSlots, ProbePrefilterPipelines, typed_uav_load_supported, uav_barrier,
    write_cube_srv as write_mips_cube_srv, write_prefilter_views,
};
use super::probe_set::{CubeResource, create_cube_resource, write_cube_mip_uav};
use super::pso::{Blend, GraphicsPso, Raster, compute_pso};
use super::root_sig::{RootSig, SamplerState, Visibility};
use super::texture::{HDR_FORMAT, transition_barrier};

const TABLE_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;
// The raymarch pass views the environment's irradiance cube as RGBA32F, so the
// atmosphere's matches it.
const CUBE_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R32G32B32A32_FLOAT;
const RESTING: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
const GROUP: u32 = 8;

/// The heap slots the atmosphere owns: the four tables' SRVs (transmittance,
/// multiple scattering, sky view, aerial volume, contiguous so one table of
/// three covers the kernels' inputs) and five UAVs (the four tables and the
/// irradiance cube's faces).
pub(in crate::directx) const ATMOSPHERE_SRV_SLOTS: usize = 4;
/// See [`ATMOSPHERE_SRV_SLOTS`].
pub(in crate::directx) const ATMOSPHERE_UAV_SLOTS: usize = 5;
/// The heap slots the sky cube's convolution takes, laid out by
/// `PrefilterSlots::block`.
pub(in crate::directx) const SKY_CUBE_SLOTS: usize = PrefilterSlots::span(SKY_CUBE_PLAN.mips());

// The CPU and GPU handles of the atmosphere's heap slots, the environment's
// irradiance slot, and the main depth SRV the haze pass reads.
#[derive(Clone, Copy)]
pub(in crate::directx) struct AtmosphereSlots {
    pub srv_cpu: [D3D12_CPU_DESCRIPTOR_HANDLE; ATMOSPHERE_SRV_SLOTS],
    pub srv_gpu: [SrvSlot; ATMOSPHERE_SRV_SLOTS],
    pub uav_cpu: [D3D12_CPU_DESCRIPTOR_HANDLE; ATMOSPHERE_UAV_SLOTS],
    pub uav_gpu: [SrvSlot; ATMOSPHERE_UAV_SLOTS],
    pub irradiance_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub depth_srv_gpu: SrvSlot,
}

pub(in crate::directx) struct DxAtmosphere {
    settings: AtmosphereSettings,
    // This frame's inputs; `None` leaves the passes out of the graph.
    pub(in crate::directx) frame: Option<AtmosphereFrame>,
    // The constants built from them for this frame's view, set while the frame
    // records.
    pub(in crate::directx) params: Cell<Option<AtmosphereParams>>,
    // Transmittance, multiple scattering, sky view, aerial volume.
    tables: [ID3D12Resource; 4],
    irradiance: ID3D12Resource,
    irradiance_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    params_buffers: Vec<PooledBuffer>,
    params_ptrs: Vec<*mut u8>,
    srv_gpu: [SrvSlot; ATMOSPHERE_SRV_SLOTS],
    uav_gpu: [SrvSlot; ATMOSPHERE_UAV_SLOTS],
    depth_srv_gpu: SrvSlot,
    kernel_root_sig: ID3D12RootSignature,
    kernels: [ID3D12PipelineState; 5],
    sky_root_sig: ID3D12RootSignature,
    sky_pso: ID3D12PipelineState,
    aerial_root_sig: ID3D12RootSignature,
    aerial_pso: ID3D12PipelineState,
    // `None` on a device that cannot load the capture's format through a
    // UAV, which the prefilter needs; the surfaces then reflect no sky.
    sky_cube: Option<DxSkyCube>,
    // The ground the tables that depend on the air alone were built over.
    air_tables: AirTables,
    // The frames that recorded the tables, for work recorded outside the frame.
    readiness: TableReadiness,
}

/// The sky drawn into a capture cube and convolved by the reflection-probe
/// kernels into the cube whose SRV takes the environment's prefilter slot, a
/// few steps a frame. Both cubes rest readable by any shader stage.
pub(in crate::directx) struct DxSkyCube {
    kernel: ID3D12PipelineState,
    prefilter: ProbePrefilterPipelines,
    capture: ID3D12Resource,
    cube: ID3D12Resource,
    slots: PrefilterSlots,
    prefilter_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    schedule: Cell<SkyCubeSchedule>,
    // This frame's steps, planned when its constants are.
    work: Cell<SkyCubeWork>,
}

impl DxSkyCube {
    /// The two cubes and their descriptors in `slots`, the cube's SRV in the
    /// environment's prefilter slot `prefilter_srv_cpu`. `None` when the
    /// device cannot run the prefilter.
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        slots: PrefilterSlots,
        cpu: impl Fn(usize) -> D3D12_CPU_DESCRIPTOR_HANDLE,
        prefilter_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        hot_reload: bool,
    ) -> RenderResult<Option<Self>> {
        if !typed_uav_load_supported(device) {
            return Ok(None);
        }
        let cube_resource = |label: &'static str| {
            create_cube_resource(
                device,
                CubeResource {
                    face_size: SKY_CUBE_PLAN.face_size(),
                    mips: SKY_CUBE_PLAN.mips(),
                    cubes: 1,
                    state: RESTING,
                    label,
                },
            )
        };
        let capture = cube_resource("atmosphere sky capture")?;
        let cube = cube_resource("atmosphere sky cube")?;
        let mips = SKY_CUBE_PLAN.mips();
        write_prefilter_views(device, slots, cpu, &capture, mips, |mip, handle| {
            write_cube_mip_uav(device, &cube, 0, mip, handle);
        });
        write_mips_cube_srv(device, &cube, mips, prefilter_srv_cpu);
        let kernel_root_sig = kernel_root_sig(device)?;
        Ok(Some(Self {
            kernel: kernel(device, &kernel_root_sig, &ATMOSPHERE_SKY_CUBE, hot_reload)?,
            prefilter: ProbePrefilterPipelines::new(device, hot_reload)?,
            capture,
            cube,
            slots,
            prefilter_srv_cpu,
            schedule: Cell::new(SkyCubeSchedule::new(mips)),
            work: Cell::new(SkyCubeWork::default()),
        }))
    }
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
    .map_err(|e| map_hresult(e.code(), &format!("create atmosphere {label}")))?;
    out.ok_or_else(|| RenderError::Other(format!("create atmosphere {label} returned None")))
}

fn table_desc(extent: [u32; 3]) -> D3D12_RESOURCE_DESC {
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
        Format: TABLE_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

// SRV + UAV of one table.
fn write_table_views(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    volume: bool,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv = if volume {
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: TABLE_FORMAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    ResourceMinLODClamp: 0.0,
                },
            },
        }
    } else {
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: TABLE_FORMAT,
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
        }
    };
    let uav = if volume {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: TABLE_FORMAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_UAV {
                    MipSlice: 0,
                    FirstWSlice: 0,
                    WSize: AERIAL_SIZE,
                },
            },
        }
    } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: TABLE_FORMAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        }
    };
    // SAFETY: the view descriptors and the resource they name are live for the calls, and the
    // destination handles address slots this context reserved in a heap it owns.
    unsafe {
        device.CreateShaderResourceView(resource, Some(&raw const srv), srv_cpu);
        device.CreateUnorderedAccessView(resource, None, Some(&raw const uav), uav_cpu);
    }
}

// The irradiance cube's SRV, at `srv_cpu`.
fn write_cube_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: CUBE_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCube: D3D12_TEXCUBE_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const srv), srv_cpu) };
}

// The irradiance cube: its cube SRV in the environment's slot and a UAV over
// its six faces.
fn write_cube_views(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    write_cube_srv(device, resource, srv_cpu);
    let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: CUBE_FORMAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                MipSlice: 0,
                FirstArraySlice: 0,
                ArraySize: 6,
                PlaneSlice: 0,
            },
        },
    };
    // SAFETY: as `write_cube_srv`.
    unsafe { device.CreateUnorderedAccessView(resource, None, Some(&raw const uav), uav_cpu) };
}

// The kernels' root signature:
//   [0] root CBV b0   AtmosphereParams
//   [1] table  t0..t2 transmittance, multiple scattering, sky view
//   [2] table  u0     the table the kernel writes
// Static linear-clamp sampler s0.
fn kernel_root_sig(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .srv_table(0, 3, Visibility::All)
        .uav_table(0, 1, Visibility::All)
        .static_sampler(SamplerState::LinearClamp, 0, Visibility::All)
        .build(device, "atmosphere kernel root sig")
}

// The atmosphere sky's root signature:
//   [0] root CBV b0   ViewUniforms of the pass it completes
//   [1] root CBV b1   AtmosphereParams
//   [2] table  t0     the sky-view table
// Static linear-clamp sampler s0.
fn sky_root_sig(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .cbv(1, Visibility::Pixel)
        .srv_table(0, 1, Visibility::Pixel)
        .static_sampler(SamplerState::LinearClamp, 0, Visibility::Pixel)
        .build(device, "atmosphere sky root sig")
}

// The haze pass's root signature:
//   [0] root CBV b0   AtmosphereParams
//   [1] table  t0     scene depth
//   [2] table  t1     aerial volume
// Static linear-clamp sampler s0.
fn aerial_root_sig(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::Pixel)
        .srv_table(0, 1, Visibility::Pixel)
        .srv_table(1, 1, Visibility::Pixel)
        .static_sampler(SamplerState::LinearClamp, 0, Visibility::Pixel)
        .build(device, "aerial perspective root sig")
}

fn kernel(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    program: &ShaderProgram,
    hot_reload: bool,
) -> RenderResult<ID3D12PipelineState> {
    compute_pso(
        device,
        root_sig,
        &program.compile(hot_reload)?,
        program.label,
    )
}

fn kernels(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    hot_reload: bool,
) -> RenderResult<[ID3D12PipelineState; 5]> {
    Ok([
        kernel(device, root_sig, &ATMOSPHERE_TRANSMITTANCE, hot_reload)?,
        kernel(device, root_sig, &ATMOSPHERE_MULTISCATTER, hot_reload)?,
        kernel(device, root_sig, &ATMOSPHERE_SKY_VIEW, hot_reload)?,
        kernel(device, root_sig, &ATMOSPHERE_AERIAL, hot_reload)?,
        kernel(device, root_sig, &ATMOSPHERE_IRRADIANCE, hot_reload)?,
    ])
}

// `(in-scatter, 1 - T)` over the resolved scene -> `scene * T + in-scatter`.
fn aerial_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
) -> RenderResult<ID3D12PipelineState> {
    let vs = FULLSCREEN_VERT.compile(hot_reload)?;
    let ps = AERIAL_PERSPECTIVE_FRAG
        .at(msaa_samples > 1)
        .compile(hot_reload)?;
    GraphicsPso::fullscreen(root_sig, &vs, &ps, HDR_FORMAT, Blend::PremultipliedOver)
        .raster(Raster {
            depth_clip: false,
            ..Raster::default()
        })
        .build(device, "aerial perspective")
}

impl DxAtmosphere {
    // The tables, pipelines and parameter ring, and the irradiance cube,
    // whose SRV replaces the environment's in its slot.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        settings: AtmosphereSettings,
        slots: AtmosphereSlots,
        sky_cube: Option<DxSkyCube>,
        msaa_samples: u32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let extents = [
            [TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT, 1],
            [MULTISCATTER_SIZE, MULTISCATTER_SIZE, 1],
            [SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT, 1],
            [AERIAL_SIZE, AERIAL_SIZE, AERIAL_SIZE],
        ];
        let labels = [
            "transmittance",
            "multiple scattering",
            "sky view",
            "aerial volume",
        ];
        let mut tables = Vec::with_capacity(4);
        for (i, extent) in extents.iter().enumerate() {
            let table = create_texture(device, &table_desc(*extent), labels[i])?;
            write_table_views(
                device,
                &table,
                extent[2] > 1,
                slots.srv_cpu[i],
                slots.uav_cpu[i],
            );
            tables.push(table);
        }
        let tables: [ID3D12Resource; 4] = tables
            .try_into()
            .map_err(|_| RenderError::Other("atmosphere: four tables".into()))?;
        let irradiance = create_texture(
            device,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(IRRADIANCE_FACE),
                Height: IRRADIANCE_FACE,
                DepthOrArraySize: 6,
                MipLevels: 1,
                Format: CUBE_FORMAT,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            "irradiance cube",
        )?;
        write_cube_views(
            device,
            &irradiance,
            slots.irradiance_srv_cpu,
            slots.uav_cpu[4],
        );

        let size = align256(size_of::<AtmosphereParams>() as u64);
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
                .map_err(|e| map_hresult(e.code(), "map atmosphere params"))?;
            params_ptrs.push(ptr.cast::<u8>());
            params_buffers.push(buf);
        }

        let kernel_root_sig = kernel_root_sig(device)?;
        let kernels = kernels(device, &kernel_root_sig, hot_reload)?;
        let sky_root_sig = sky_root_sig(device)?;
        let sky_pso = super::sky::build_sky_pso(
            device,
            &SKY_ATMOSPHERE_FRAG,
            &sky_root_sig,
            msaa_samples,
            hot_reload,
        )?;
        let aerial_root_sig = aerial_root_sig(device)?;
        let aerial_pso = aerial_pso(device, &aerial_root_sig, msaa_samples, hot_reload)?;
        Ok(Self {
            settings,
            frame: None,
            params: Cell::new(None),
            tables,
            irradiance,
            irradiance_srv_cpu: slots.irradiance_srv_cpu,
            params_buffers,
            params_ptrs,
            srv_gpu: slots.srv_gpu,
            uav_gpu: slots.uav_gpu,
            depth_srv_gpu: slots.depth_srv_gpu,
            kernel_root_sig,
            kernels,
            sky_root_sig,
            sky_pso,
            aerial_root_sig,
            aerial_pso,
            sky_cube,
            air_tables: AirTables::default(),
            readiness: TableReadiness::default(),
        })
    }

    // The irradiance cube, for a pass that views it in its own slots.
    pub(in crate::directx) fn irradiance(&self) -> &ID3D12Resource {
        &self.irradiance
    }

    // The prefiltered sky cube and its mip count, for a pass that views it in
    // its own slots; `None` when the device has none.
    pub(in crate::directx) fn prefilter(&self) -> Option<(&ID3D12Resource, u32)> {
        self.sky_cube
            .as_ref()
            .map(|c| (&c.cube, SKY_CUBE_PLAN.mips()))
    }

    // Write the cubes' SRVs back into the environment's slots, after an
    // environment reload wrote the map's there.
    pub(in crate::directx) fn reclaim_environment_slots(&self, device: &ID3D12Device) {
        write_cube_srv(device, &self.irradiance, self.irradiance_srv_cpu);
        if let Some(c) = &self.sky_cube {
            write_mips_cube_srv(device, &c.cube, SKY_CUBE_PLAN.mips(), c.prefilter_srv_cpu);
        }
    }

    // Rebuild every pipeline from the shader sources (shader hot reload).
    pub(in crate::directx) fn rebuild_pipelines(
        &mut self,
        device: &ID3D12Device,
        msaa_samples: u32,
    ) -> RenderResult<()> {
        let kernels = kernels(device, &self.kernel_root_sig, true)?;
        if let Some(c) = self.sky_cube.as_mut() {
            c.kernel = kernel(device, &self.kernel_root_sig, &ATMOSPHERE_SKY_CUBE, true)?;
            c.schedule.set(SkyCubeSchedule::new(SKY_CUBE_PLAN.mips()));
        }
        let sky_pso = super::sky::build_sky_pso(
            device,
            &SKY_ATMOSPHERE_FRAG,
            &self.sky_root_sig,
            msaa_samples,
            true,
        )?;
        let aerial_pso = aerial_pso(device, &self.aerial_root_sig, msaa_samples, true)?;
        self.kernels = kernels;
        self.sky_pso = sky_pso;
        self.aerial_pso = aerial_pso;
        self.air_tables.invalidate();
        Ok(())
    }

    fn params_gva(&self, frame_idx: usize) -> Option<u64> {
        self.params_buffers.get(frame_idx).map(|b| com::gpu_va(b))
    }
}

impl DxContext {
    // This frame's atmosphere constants for `view`, written into the frame
    // slot's ring.
    pub(in crate::directx) fn prepare_atmosphere(&self, frame_idx: usize, view: AtmosphereView) {
        let Some(atmo) = self.atmosphere.as_ref() else {
            return;
        };
        let params = atmo
            .frame
            .map(|f| AtmosphereParams::new(&atmo.settings, &f, &view));
        atmo.params.set(params);
        if let Some(cube) = &atmo.sky_cube {
            let mut schedule = cube.schedule.get();
            cube.work.set(match atmo.frame {
                Some(f) => schedule.plan(SkyCubeKey::new(&atmo.settings, &f, &view.sky_rot)),
                None => SkyCubeWork::default(),
            });
            cube.schedule.set(schedule);
        }
        if let (Some(params), Some(&ptr)) = (params, atmo.params_ptrs.get(frame_idx)) {
            // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer
            // init sized for this block, and the source is a separate live value.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const params).cast::<u8>(),
                    ptr,
                    size_of::<AtmosphereParams>(),
                );
            }
        }
    }

    // Build this frame's tables: the sky view, the aerial volume and the
    // irradiance cube, after the transmittance and multiple-scattering tables
    // when the ground they are built over has changed; then this frame's steps
    // of the sky cube.
    pub(in crate::directx) fn encode_atmosphere(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) {
        let Some(atmo) = self.atmosphere.as_ref() else {
            return;
        };
        let Some(params_gva) = atmo.params_gva(frame_idx) else {
            return;
        };
        let ground = atmo.params.get().map_or(0.0, |p| p.radii[0]);
        let open = |r: &ID3D12Resource| {
            transition_barrier(r, RESTING, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        let close = |r: &ID3D12Resource| {
            transition_barrier(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, RESTING)
        };
        let dispatch = |k: usize, groups: [u32; 3]| {
            // SAFETY: the command list is in the recording state with the kernel root signature
            // bound, and every descriptor these commands name is live for the call.
            unsafe {
                cmd.SetPipelineState(&atmo.kernels[k]);
                cmd.set_compute_srv_table(2, atmo.uav_gpu[k]);
                cmd.Dispatch(groups[0], groups[1], groups[2]);
            }
        };
        let [transmittance, multiscatter, sky_view, aerial] = &atmo.tables;
        atmo.readiness.note_recorded();
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call; each barrier's before-state is the state
        // the resource rests in or was last moved to above.
        unsafe {
            cmd.SetComputeRootSignature(&atmo.kernel_root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetComputeRootConstantBufferView(0, params_gva);
            cmd.set_compute_srv_table(1, atmo.srv_gpu[0]);
            if atmo.air_tables.take_rebuild(ground) {
                cmd.ResourceBarrier(&[open(transmittance)]);
                dispatch(
                    0,
                    [
                        TRANSMITTANCE_WIDTH.div_ceil(GROUP),
                        TRANSMITTANCE_HEIGHT.div_ceil(GROUP),
                        1,
                    ],
                );
                cmd.ResourceBarrier(&[close(transmittance), open(multiscatter)]);
                dispatch(1, [MULTISCATTER_SIZE, MULTISCATTER_SIZE, 1]);
                cmd.ResourceBarrier(&[close(multiscatter)]);
            }
            cmd.ResourceBarrier(&[open(sky_view), open(aerial)]);
            dispatch(
                2,
                [
                    SKY_VIEW_WIDTH.div_ceil(GROUP),
                    SKY_VIEW_HEIGHT.div_ceil(GROUP),
                    1,
                ],
            );
            dispatch(
                3,
                [AERIAL_SIZE.div_ceil(GROUP), AERIAL_SIZE.div_ceil(GROUP), 1],
            );
            cmd.ResourceBarrier(&[close(sky_view), close(aerial), open(&atmo.irradiance)]);
            dispatch(4, [1, 1, 1]);
            cmd.ResourceBarrier(&[close(&atmo.irradiance)]);
        }
        if let Some(cube) = &atmo.sky_cube {
            self.encode_sky_cube(cmd, cube);
        }
    }

    // This frame's steps of the sky cube: the sky drawn into the capture's top
    // mip, the mirror copy and the pyramid, then the GGX mips. The capture is
    // open for unordered access only while the sky and the pyramid are
    // written, the cube while any of its mips are.
    fn encode_sky_cube(&self, cmd: &ID3D12GraphicsCommandList, cube: &DxSkyCube) {
        let work = cube.work.get();
        if work.is_empty() {
            return;
        }
        let open = |r: &ID3D12Resource| {
            transition_barrier(r, RESTING, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        let close = |r: &ID3D12Resource| {
            transition_barrier(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, RESTING)
        };
        if work.capture {
            let groups = SKY_CUBE_FACE.div_ceil(GROUP);
            // SAFETY: the command list is in the recording state with the kernel root signature,
            // the constants and the table inputs still bound from the atmosphere's kernels, and
            // every resource and descriptor these commands name is live for the call.
            unsafe {
                cmd.ResourceBarrier(&[open(&cube.capture), open(&cube.cube)]);
                cmd.SetPipelineState(&cube.kernel);
                cmd.set_compute_srv_table(
                    2,
                    self.descriptors.slot_gpu(cube.slots.capture_uav_base),
                );
                cmd.Dispatch(groups, groups, 6);
            }
            uav_barrier(cmd, &cube.capture);
            self.record_prefilter_pyramid(
                cmd,
                &cube.prefilter,
                cube.slots,
                &cube.capture,
                &SKY_CUBE_PLAN,
            );
            // SAFETY: as above.
            unsafe { cmd.ResourceBarrier(&[close(&cube.capture)]) };
        } else {
            // SAFETY: as above.
            unsafe { cmd.ResourceBarrier(&[open(&cube.cube)]) };
        }
        for mip in work.ggx_mips() {
            self.record_prefilter_ggx(cmd, &cube.prefilter, cube.slots, &SKY_CUBE_PLAN, mip);
        }
        // SAFETY: as above.
        unsafe { cmd.ResourceBarrier(&[close(&cube.cube)]) };
    }

    // Blend the air in front of each opaque surface over the resolved scene.
    pub(in crate::directx) fn encode_aerial_perspective(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) {
        let Some(atmo) = self.atmosphere.as_ref() else {
            return;
        };
        let Some(params_gva) = atmo.params_gva(frame_idx) else {
            return;
        };
        // The scene spine and the main depth are graph resources: the executor
        // has put the scene in RENDER_TARGET and the depth in a shader-resource
        // state for this pass.
        let scene_rtv = self.hdr_scene_rtv();
        let w = self.targets.extent.render_width;
        let h = self.targets.extent.render_height;
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&raw const scene_rtv), false, None);
            cmd.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            cmd.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: i32::saturating_from(w),
                bottom: i32::saturating_from(h),
            }]);
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.SetPipelineState(&atmo.aerial_pso);
            cmd.SetGraphicsRootSignature(&atmo.aerial_root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, params_gva);
            cmd.set_graphics_srv_table(1, atmo.depth_srv_gpu);
            cmd.set_graphics_srv_table(2, atmo.srv_gpu[3]);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
    }

    // Draw the atmosphere's sky in place of the environment map's, when a
    // frame has carried its inputs. `false` leaves the caller to draw the map.
    pub(in crate::directx) fn encode_atmosphere_sky(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        view_gva: u64,
        frame_idx: usize,
    ) -> bool {
        let Some(atmo) = self.atmosphere.as_ref() else {
            return false;
        };
        let Some(params_gva) = atmo.params.get().and_then(|_| atmo.params_gva(frame_idx)) else {
            return false;
        };
        // SAFETY: the command list is in the recording state with the CBV/SRV/UAV heap set, and
        // the root signature bound here declares the two CBVs and the table these commands set.
        unsafe {
            cmd.SetPipelineState(&atmo.sky_pso);
            cmd.SetGraphicsRootSignature(&atmo.sky_root_sig);
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            cmd.SetGraphicsRootConstantBufferView(1, params_gva);
            cmd.set_graphics_srv_table(2, atmo.srv_gpu[2]);
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
        self.inc_draw_calls(1);
        true
    }

    // The view block's `irradiance` lane.
    pub(in crate::directx) fn irradiance_lane(&self) -> f32 {
        concinnity_core::render::sky::irradiance_lane(
            self.scene.env_map.prefilter_mip_count > 0,
            self.atmosphere.is_some(),
        )
    }

    // Whether work recorded outside the frame graph, such as a probe face,
    // reads tables this atmosphere has written. Always true without one.
    pub(in crate::directx) fn atmosphere_tables_ready(&self) -> bool {
        self.atmosphere
            .as_ref()
            .is_none_or(|atmo| atmo.readiness.ready(self.hw.frames()))
    }
}
