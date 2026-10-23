//! The water's wake on D3D12 (see
//! `concinnity_engine::render::environment::water::wake`): the state the
//! shared `water_wake.hlsl` kernels run forward each frame, the field a planet
//! sea's draw samples, and their dispatch inside the `WaterSim` node, after
//! the ocean's cascades.
//!
//! The field rests readable by any shader stage, like the cascade arrays: the
//! node opens it for unordered access and hands each mip back as soon as it is
//! written, so the next reduction reads it as a shader resource. Its
//! descriptors take the tail of the ocean block: per mip level, the table one
//! kernel binds, a UAV of the level then an SRV of the one above it. The state
//! and the transform stay in unordered access and are bound as root views; the
//! block is a root CBV in an upload-heap block per frame in flight.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::environment::water::wake::{
    WAKE_COLUMN_GROUPS, WAKE_MIP_LEVELS, WAKE_RESOLUTION, WAKE_ROW_GROUPS, wake_groups,
};
use concinnity_engine::render::uniforms::wake::WaterWakeParams;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::DeviceAllocator;
use super::builtin_shaders::{
    CompileProgram, WATER_WAKE_COLUMNS, WATER_WAKE_INVERSE, WATER_WAKE_MIP, WATER_WAKE_ROWS,
    WATER_WAKE_SHADE,
};
use super::com;
use super::descriptor_slot::{DescriptorTables, SrvSlot};
use super::error::map_hresult;
use super::mapped_blocks::MappedBlocks;
use super::probe_prefilter::uav_barrier;
use super::pso::compute_pso;
use super::root_sig::{Range, RootSig, Visibility};
use super::texture::{create_uav_buffer, subresource_transition_barrier, transition_barrier};

const FIELD_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;
const RESTING: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;

/// The heap slots the wake's mip tables take: a UAV and an SRV per level.
pub(in crate::directx) const WAKE_SLOTS: usize = 2 * WAKE_MIP_LEVELS as usize;

// Root parameters of the kernels' signature.
const PARAMS_ROOT: u32 = 0;
const STATE_ROOT: u32 = 1;
const WORK_ROOT: u32 = 2;
const MIPS_ROOT: u32 = 3;

// The wake's kernels, state and field.
pub(in crate::directx) struct WakeSim {
    root_sig: ID3D12RootSignature,
    rows: ID3D12PipelineState,
    columns: ID3D12PipelineState,
    inverse: ID3D12PipelineState,
    shade: ID3D12PipelineState,
    mip: ID3D12PipelineState,
    params: MappedBlocks,
    // Two halves of (height, rise, foam, _) per cell, kept across frames.
    state: ID3D12Resource,
    // The transform between the passes, one complex value per cell.
    work: ID3D12Resource,
    field: ID3D12Resource,
    // Each mip level's table: the level's UAV, the level above's SRV.
    mip_tables: Vec<SrvSlot>,
}

impl WakeSim {
    // The wake, its descriptors written where `slots` says.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        slots: &WakeSlots<impl Fn(usize) -> D3D12_CPU_DESCRIPTOR_HANDLE>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let cells = u64::from(WAKE_RESOLUTION * WAKE_RESOLUTION);
        let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let state = create_uav_buffer(device, 2 * cells * size_of::<[f32; 4]>() as u64, uav)?;
        let work = create_uav_buffer(device, cells * size_of::<[f32; 2]>() as u64, uav)?;
        let field = field_texture(device, WAKE_RESOLUTION, WAKE_MIP_LEVELS)?;
        write_field_srv(device, Some(&field), 0, WAKE_MIP_LEVELS, slots.field_cpu);
        let mip_tables = (0..WAKE_MIP_LEVELS)
            .map(|level| {
                let base = slots.first + 2 * level as usize;
                write_mip_uav(device, &field, level, (slots.cpu)(base));
                let src = level.saturating_sub(1);
                write_field_srv(device, Some(&field), src, 1, (slots.cpu)(base + 1));
                slots.gpu.offset(base, slots.stride)
            })
            .collect();
        let sig = RootSig::new()
            .cbv(0, Visibility::All)
            .uav(1, Visibility::All)
            .uav(2, Visibility::All)
            .table(&[Range::uav(3, 1), Range::srv(4, 1)], Visibility::All);
        debug_assert_eq!(sig.len(), MIPS_ROOT + 1);
        let root_sig = sig.build(device, "wake root sig")?;
        let kernel = |program: &concinnity_engine::render::shader::programs::ShaderProgram,
                      label: &str| {
            compute_pso(device, &root_sig, &program.compile(hot_reload)?, label)
        };
        Ok(Self {
            rows: kernel(&WATER_WAKE_ROWS, "wake rows")?,
            columns: kernel(&WATER_WAKE_COLUMNS, "wake columns")?,
            inverse: kernel(&WATER_WAKE_INVERSE, "wake inverse")?,
            shade: kernel(&WATER_WAKE_SHADE, "wake shading")?,
            mip: kernel(&WATER_WAKE_MIP, "wake mips")?,
            root_sig,
            params: MappedBlocks::new(alloc, size_of::<WaterWakeParams>(), "map wake params")?,
            state,
            work,
            field,
            mip_tables,
        })
    }

    // Run the wake one frame forward under `params` in frame slot
    // `frame_idx`, and reduce its field, each mip handed back to the resting
    // state once written. The caller has set the SRV heap. Returns whether it
    // encoded the run.
    pub(in crate::directx) fn encode(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        params: &WaterWakeParams,
    ) -> bool {
        let (Some(&first), Some(params_gva)) = (
            self.mip_tables.first(),
            self.params.write(frame_idx, params),
        ) else {
            return false;
        };
        let to_uav = [transition_barrier(
            &self.field,
            RESTING,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )];
        // SAFETY: the list is recording, and the field rests in `RESTING`, the barrier's
        // before-state; every mip is handed back below.
        unsafe { cmd.ResourceBarrier(&to_uav) };
        // SAFETY: the list is recording, and the wake's root signature is owned by `self`.
        unsafe { cmd.SetComputeRootSignature(&self.root_sig) };
        // SAFETY: the signature declares the block's CBV here, and this frame's block is live.
        unsafe { cmd.SetComputeRootConstantBufferView(PARAMS_ROOT, params_gva) };
        // SAFETY: the signature declares the state UAV here, and `self` owns the buffer.
        unsafe { cmd.SetComputeRootUnorderedAccessView(STATE_ROOT, com::gpu_va(&self.state)) };
        // SAFETY: the signature declares the transform UAV here, and `self` owns the buffer.
        unsafe { cmd.SetComputeRootUnorderedAccessView(WORK_ROOT, com::gpu_va(&self.work)) };
        // SAFETY: the caller set the SRV heap, and the mip 0 table `self` reserved is live.
        unsafe { cmd.set_compute_srv_table(MIPS_ROOT, first) };
        for (pso, groups) in [
            (&self.rows, WAKE_ROW_GROUPS),
            (&self.columns, WAKE_COLUMN_GROUPS),
            (&self.inverse, WAKE_ROW_GROUPS),
        ] {
            // SAFETY: the list is recording, and the PSO is owned by `self`.
            unsafe { cmd.SetPipelineState(pso) };
            // SAFETY: the PSO and every binding it reads are set above.
            unsafe { cmd.Dispatch(groups, 1, 1) };
            uav_barrier(cmd, &self.state);
            uav_barrier(cmd, &self.work);
        }
        // SAFETY: the list is recording, and the shading PSO is owned by `self`.
        unsafe { cmd.SetPipelineState(&self.shade) };
        // SAFETY: the shading reads the state the UAV barriers above made visible.
        unsafe { cmd.Dispatch(wake_groups(0), wake_groups(0), 1) };
        // SAFETY: the list is recording, and the mip reduction PSO is owned by `self`.
        unsafe { cmd.SetPipelineState(&self.mip) };
        for (level, &table) in (1..WAKE_MIP_LEVELS).zip(self.mip_tables.iter().skip(1)) {
            self.hand_back(cmd, level - 1);
            // SAFETY: the SRV heap is set, and this level's table is live.
            unsafe { cmd.set_compute_srv_table(MIPS_ROOT, table) };
            let g = wake_groups(level);
            // SAFETY: the reduction PSO and this level's bindings are set above.
            unsafe { cmd.Dispatch(g, g, 1) };
        }
        self.hand_back(cmd, WAKE_MIP_LEVELS - 1);
        uav_barrier(cmd, &self.state);
        true
    }

    // Mip `level`, written, made readable.
    fn hand_back(&self, cmd: &ID3D12GraphicsCommandList, level: u32) {
        let barrier = [subresource_transition_barrier(
            &self.field,
            level,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            RESTING,
        )];
        // SAFETY: the level is still in the UNORDERED_ACCESS the field was moved to at the
        // node's head, and the field is owned by `self`.
        unsafe { cmd.ResourceBarrier(&barrier) };
    }
}

// Where the wake's descriptors go in the ocean block.
pub(in crate::directx) struct WakeSlots<F: Fn(usize) -> D3D12_CPU_DESCRIPTOR_HANDLE> {
    // The CPU handle of block slot `i`.
    pub cpu: F,
    // The block's first slot, the heap's descriptor stride, and the slot the
    // wake's tables start at.
    pub gpu: SrvSlot,
    pub stride: usize,
    pub first: usize,
    // Where the whole field's SRV goes.
    pub field_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
}

// The SRV a sea reads where no wake runs: a null view, which reads zero.
pub(in crate::directx) fn write_stand_in_srv(
    device: &ID3D12Device,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    write_field_srv(device, None, 0, 1, srv_cpu);
}

// A `size`-square field with `mips` levels, resting readable by any shader
// stage.
fn field_texture(device: &ID3D12Device, size: u32, mips: u32) -> RenderResult<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(size),
        Height: size,
        DepthOrArraySize: 1,
        MipLevels: u16::saturating_from(mips),
        Format: FIELD_FORMAT,
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
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            RESTING,
            None,
            &raw mut resource,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create the wake field"))?;
    resource.ok_or_else(|| RenderError::Other("create the wake field returned None".into()))
}

// A read-only view of `mips` levels of `resource` from `first`, or a null
// view without one.
fn write_field_srv(
    device: &ID3D12Device,
    resource: Option<&ID3D12Resource>,
    first: u32,
    mips: u32,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: FIELD_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: first,
                MipLevels: mips,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names, if any, are live for the call, and
    // the destination handle addresses a slot this context reserved for the view in a heap it
    // owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const desc), srv_cpu) };
}

fn write_mip_uav(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    mip: u32,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: FIELD_FORMAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV {
                MipSlice: mip,
                PlaneSlice: 0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateUnorderedAccessView(resource, None, Some(&raw const desc), uav_cpu) };
}
