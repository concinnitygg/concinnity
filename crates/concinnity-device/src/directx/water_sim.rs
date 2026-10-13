//! The simulated ocean on D3D12 (see `concinnity_core::render::water::ocean`):
//! the cascade arrays the water draw samples, and the `WaterSim` node that
//! fills them each frame from the shared `water_fft.hlsl` kernels.
//!
//! Both arrays rest readable by any shader stage; the node opens them for
//! unordered access at its head and hands each mip back as soon as it is
//! written, so the next reduction reads it as a shader resource and nothing
//! crosses the node boundary in a different state. Their descriptors take the
//! heap's ocean block: the two array SRVs the water draw reads as one table,
//! then, per mip level, the table one reduction binds: an SRV of each array's
//! source mip, then a UAV of each array's destination mip.

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::uniforms::ocean::{
    OceanFftParams, ocean_mip_groups, ocean_mip_levels,
};
use concinnity_core::render::water::ocean::initial::initial_spectrum;
use concinnity_core::render::water::ocean::{OceanPlan, OceanQuality};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::builtin_shaders::{CompileProgram, WATER_FFT_COLUMNS, WATER_FFT_ROWS, WATER_MIP};
use super::com;
use super::context::DxContext;
use super::descriptor_slot::{DescriptorTables, SrvSlot};
use super::error::map_hresult;
use super::probe_prefilter::uav_barrier;
use super::pso::compute_pso;
use super::root_constants::RootConstants;
use super::root_sig::{Range, RootSig, Visibility};
use super::texture::{
    create_uav_buffer, subresource_transition_barrier, transition_barrier, upload_buffer,
};

const CASCADE_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;
const RESTING: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;

// The most mip levels a cascade array has.
const MAX_MIPS: usize = OceanQuality::MAX_SIZE.trailing_zeros() as usize + 1;

/// The heap slots the ocean owns: both arrays' SRVs, then four views per mip
/// level (the source SRV and destination UAV of each array). Always reserved.
pub(in crate::directx) const OCEAN_SLOTS: usize = 2 + 4 * MAX_MIPS;

// Root parameters of the kernels' signature.
const PARAMS_ROOT: u32 = 0;
const SPECTRUM_ROOT: u32 = 1;
const ROWS_ROOT: u32 = 2;
const MIPS_ROOT: u32 = 3;

// Where the ocean block lies in the shader-visible heap.
pub(in crate::directx) struct OceanSlots<F: Fn(usize) -> D3D12_CPU_DESCRIPTOR_HANDLE> {
    // The CPU handle of block slot `i`.
    pub cpu: F,
    // The block's first slot, and the heap's descriptor stride.
    pub gpu: SrvSlot,
    pub stride: usize,
}

// The world's simulated oceans, or a stand-in pair of arrays for a world
// whose water is all summed, which the water draw binds and never samples.
pub(in crate::directx) struct DxOceanSim {
    displacement: ID3D12Resource,
    slopes: ID3D12Resource,
    // The two array SRVs, one table.
    srv_gpu: SrvSlot,
    kernels: Option<OceanKernels>,
}

struct OceanKernels {
    plan: OceanPlan,
    root_sig: ID3D12RootSignature,
    rows: ID3D12PipelineState,
    columns: ID3D12PipelineState,
    mip: ID3D12PipelineState,
    spectrum: PooledBuffer,
    row_data: ID3D12Resource,
    // Each mip level's table: two source SRVs, two destination UAVs.
    mip_tables: Vec<SrvSlot>,
    // Mip levels per array.
    mips: u32,
}

impl DxOceanSim {
    pub(in crate::directx) fn new<F: Fn(usize) -> D3D12_CPU_DESCRIPTOR_HANDLE>(
        alloc: &DeviceAllocator,
        plan: &OceanPlan,
        slots: OceanSlots<F>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let (size, layers, mips) = if plan.is_empty() {
            (1, 1, 1)
        } else {
            let size = plan.quality.size;
            (size, plan.layer_count(), ocean_mip_levels(size))
        };
        let displacement = cascade_array(device, size, layers, mips, "ocean displacement")?;
        let slopes = cascade_array(device, size, layers, mips, "ocean slopes")?;
        for (i, array) in [&displacement, &slopes].into_iter().enumerate() {
            write_array_srv(device, array, layers, mips, (slots.cpu)(i));
        }
        let mip_tables: Vec<SrvSlot> = (0..mips as usize)
            .map(|level| {
                let base = 2 + 4 * level;
                let src = level.saturating_sub(1) as u32;
                for (j, array) in [&displacement, &slopes].into_iter().enumerate() {
                    write_mip_srv(device, array, layers, src, (slots.cpu)(base + j));
                    let uav = (slots.cpu)(base + 2 + j);
                    write_mip_uav(device, array, layers, level as u32, uav);
                }
                slots.gpu.offset(base, slots.stride)
            })
            .collect();
        let kernels = if plan.is_empty() {
            None
        } else {
            Some(OceanKernels::new(
                alloc, plan, mip_tables, mips, hot_reload,
            )?)
        };
        Ok(Self {
            displacement,
            slopes,
            srv_gpu: slots.gpu,
            kernels,
        })
    }

    // The table of both array SRVs the water draw binds.
    pub(in crate::directx) fn srv_table(&self) -> SrvSlot {
        self.srv_gpu
    }

    // Whether any body simulates an ocean, which puts `WaterSim` in the graph.
    pub(in crate::directx) fn simulates(&self) -> bool {
        self.kernels.is_some()
    }
}

impl OceanKernels {
    fn new(
        alloc: &DeviceAllocator,
        plan: &OceanPlan,
        mip_tables: Vec<SrvSlot>,
        mips: u32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let texels: Vec<[f32; 4]> = plan
            .simulations
            .iter()
            .flat_map(|sim| {
                (0..sim.cascades.count())
                    .flat_map(move |c| initial_spectrum(&sim.sea, &sim.cascades, c))
            })
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&texels);
        let spectrum = upload_buffer(alloc, bytes, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)?;
        let row_data = create_uav_buffer(
            device,
            2 * bytes.len() as u64,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        let sig = RootSig::new()
            .constants::<OceanFftParams>(0, Visibility::All)
            .srv(5, Visibility::All)
            .uav(6, Visibility::All)
            .table(&[Range::srv(0, 2), Range::uav(2, 2)], Visibility::All);
        debug_assert_eq!(sig.len(), MIPS_ROOT + 1);
        let root_sig = sig.build(device, "ocean root sig")?;
        let kernel = |program: &concinnity_core::render::shader_programs::ShaderProgram,
                      label: &str| {
            compute_pso(device, &root_sig, &program.compile(hot_reload)?, label)
        };
        Ok(Self {
            plan: plan.clone(),
            rows: kernel(&WATER_FFT_ROWS, "ocean rows")?,
            columns: kernel(&WATER_FFT_COLUMNS, "ocean columns")?,
            mip: kernel(&WATER_MIP, "ocean mips")?,
            root_sig,
            spectrum,
            row_data,
            mip_tables,
            mips,
        })
    }
}

// A `layers`-deep array of `size`-square cascades with `mips` levels, resting
// readable by any shader stage.
fn cascade_array(
    device: &ID3D12Device,
    size: u32,
    layers: u32,
    mips: u32,
    label: &str,
) -> RenderResult<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: size as u64,
        Height: size,
        DepthOrArraySize: layers as u16,
        MipLevels: mips as u16,
        Format: CASCADE_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
    // new COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            RESTING,
            None,
            &mut resource,
        )
    }
    .map_err(|e| map_hresult(e.code(), &format!("create {label}")))?;
    resource.ok_or_else(|| RenderError::Other(format!("create {label} returned None")))
}

fn write_array_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    layers: u32,
    mips: u32,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: CASCADE_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: mips,
                FirstArraySlice: 0,
                ArraySize: layers,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&desc), srv_cpu) };
}

// A read-only view of mip `mip` of every layer.
fn write_mip_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    layers: u32,
    mip: u32,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: CASCADE_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                MostDetailedMip: mip,
                MipLevels: 1,
                FirstArraySlice: 0,
                ArraySize: layers,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&desc), srv_cpu) };
}

fn write_mip_uav(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    layers: u32,
    mip: u32,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: CASCADE_FORMAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                MipSlice: mip,
                FirstArraySlice: 0,
                ArraySize: layers,
                PlaneSlice: 0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateUnorderedAccessView(resource, None, Some(&desc), uav_cpu) };
}

impl DxContext {
    // Encode the `WaterSim` node: for every simulation, evolve and transform
    // its rows, then its columns into mip 0, then reduce down the mips, each
    // mip of its layers handed back to the resting state once written.
    pub(in crate::directx) fn encode_water_sim(&self, cmd: &ID3D12GraphicsCommandList, time: f64) {
        let Some(ocean) = self.water_sim.as_ref() else {
            return;
        };
        let Some(k) = ocean.kernels.as_ref() else {
            return;
        };
        let arrays = [&ocean.displacement, &ocean.slopes];
        // Mip `level` of `cascades` layers from `first`, written, made readable.
        let written = |level: u32, first: u32, cascades: u32| -> Vec<D3D12_RESOURCE_BARRIER> {
            arrays
                .iter()
                .flat_map(|a| {
                    (first..first + cascades).map(move |layer| {
                        subresource_transition_barrier(
                            a,
                            level + layer * k.mips,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            RESTING,
                        )
                    })
                })
                .collect()
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call; each barrier's before-state is the state
        // the arrays rest in or were last moved to above. Every layer belongs to one simulation,
        // so once each has handed back all of its mips the arrays rest again.
        unsafe {
            cmd.ResourceBarrier(
                &arrays
                    .map(|a| transition_barrier(a, RESTING, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)),
            );
            cmd.SetComputeRootSignature(&k.root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetComputeRootShaderResourceView(SPECTRUM_ROOT, com::gpu_va(&k.spectrum));
            cmd.SetComputeRootUnorderedAccessView(ROWS_ROOT, com::gpu_va(&k.row_data));
            for sim in &k.plan.simulations {
                let params = OceanFftParams::simulate(sim, time);
                let [size, _, first, cascades] = params.grid;
                cmd.set_compute_root_constants(PARAMS_ROOT, &params);
                cmd.set_compute_srv_table(MIPS_ROOT, k.mip_tables[0]);
                cmd.SetPipelineState(&k.rows);
                cmd.Dispatch(size, cascades, 1);
                uav_barrier(cmd, &k.row_data);
                cmd.SetPipelineState(&k.columns);
                cmd.Dispatch(size, cascades, 1);
                cmd.SetPipelineState(&k.mip);
                for level in 1..k.mips {
                    cmd.ResourceBarrier(&written(level - 1, first, cascades));
                    let reduce = params.reduce(level);
                    let groups = ocean_mip_groups(reduce.mip_size);
                    cmd.set_compute_root_constants(PARAMS_ROOT, &reduce);
                    cmd.set_compute_srv_table(MIPS_ROOT, k.mip_tables[level as usize]);
                    cmd.Dispatch(groups, groups, cascades);
                }
                cmd.ResourceBarrier(&written(k.mips - 1, first, cascades));
                uav_barrier(cmd, &k.row_data);
            }
        }
    }
}
