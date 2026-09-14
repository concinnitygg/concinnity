//! The DirectX-only descriptor heaps: the RTV heap holding the swapchain's
//! back-buffer views, the DSV heap, and the sampler heap with its static
//! samplers, plus the handle arithmetic every stage addresses a slot with.

use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::InitGpu;
use super::bootstrap::DxgiSwapchain;
use super::heap_layout::{DSV_SLOTS, RtvHeapLayout};
use crate::directx::context::{FRAMES, SwapchainState};

// Sampler heap slots: [0] shadow comparison, [1] linear repeat, [2] cube
// linear-clamp + mip, [3] linear clamp (text). Linear and cube are placed
// contiguously so the main pass binds them via a single 2-descriptor table
// range. Slots [4..7] are the raymarch pass's contiguous descriptor table:
// shadow comparison, cube linear-clamp, and a linear-clamp scene sampler. These
// duplicate samplers at slots 0 / 2 so the raymarch root sig can bind a single
// 3-slot range; the cost is three extra descriptors (a few bytes). Reserved
// unconditionally so the heap layout stays anchored.
pub(super) const SHADOW_SAMPLER_SLOT: usize = 0;
pub(super) const LINEAR_SAMPLER_SLOT: usize = 1;
pub(super) const TEXT_SAMPLER_SLOT: usize = 3;
pub(super) const RAYMARCH_SAMPLER_BASE_SLOT: usize = 4;
const SAMPLER_SLOTS: u32 = 7;

// CPU handle of `slot` in a heap whose descriptors sit `stride` apart.
pub(super) fn cpu_handle(
    heap: &ID3D12DescriptorHeap,
    stride: usize,
    slot: usize,
) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    // SAFETY: a property query on a live descriptor heap; it only reads.
    let base = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: base.ptr + slot * stride,
    }
}

// GPU handle of `slot` in a shader-visible heap whose descriptors sit `stride`
// apart.
pub(super) fn gpu_handle(
    heap: &ID3D12DescriptorHeap,
    stride: usize,
    slot: usize,
) -> D3D12_GPU_DESCRIPTOR_HANDLE {
    // SAFETY: a property query on a live descriptor heap; it only reads.
    let base = unsafe { heap.GetGPUDescriptorHandleForHeapStart() };
    D3D12_GPU_DESCRIPTOR_HANDLE {
        ptr: base.ptr + (slot * stride) as u64,
    }
}

pub(super) fn descriptor_size(device: &ID3D12Device, kind: D3D12_DESCRIPTOR_HEAP_TYPE) -> usize {
    // SAFETY: a property query on a live device; it only reads.
    unsafe { device.GetDescriptorHandleIncrementSize(kind) as usize }
}

impl SwapchainState {
    // CPU handle of RTV heap `slot`.
    pub(super) fn rtv(&self, slot: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        cpu_handle(&self.rtv_heap, self.rtv_descriptor_size, slot)
    }
}

// The swapchain with its RTV heap sized to `rtv`: a view per back buffer in
// `[0, FRAMES)`, every later slot written by the stage that renders into it.
pub(super) fn build_swapchain(
    gpu: &InitGpu<'_>,
    swapchain: DxgiSwapchain,
    rtv: &RtvHeapLayout,
    vsync: bool,
) -> RenderResult<SwapchainState> {
    let device = &gpu.hw.device;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
    // new COM object lands in a binding that owns it.
    let rtv_heap: ID3D12DescriptorHeap = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: rtv.rtv_slots as u32,
            ..Default::default()
        })
    }
    .map_err(|e| format!("RTV heap: {e}"))?;
    let rtv_descriptor_size = descriptor_size(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    let mut back_buffers = Vec::with_capacity(FRAMES);
    for i in 0..FRAMES {
        // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters
        // it fills are live locals that outlive the call.
        let buf: ID3D12Resource = unsafe { swapchain.handle.GetBuffer(i as u32) }
            .map_err(|e| format!("GetBuffer[{i}]: {e}"))?;
        let rtv_handle = cpu_handle(&rtv_heap, rtv_descriptor_size, i);
        // SAFETY: the view descriptor and the resource it names are live for the call, and the
        // destination handle addresses a slot this context reserved for the view in a heap it
        // owns.
        unsafe {
            device.CreateRenderTargetView(&buf, None, rtv_handle);
        }
        back_buffers.push(buf);
    }

    // Presentation pacing derived from the vsync request + tearing support.
    // vsync on -> sync interval 1 (lock to refresh). vsync off + tearing ->
    // sync interval 0 with the tearing present flag (true uncapped). vsync
    // off without tearing -> sync interval 0, no flag (flip-model refresh
    // pacing, the best available fallback).
    let present_sync_interval: u32 = if vsync { 1 } else { 0 };
    Ok(SwapchainState {
        handle: swapchain.handle,
        back_buffers,
        rtv_heap,
        rtv_descriptor_size,
        format: swapchain.format,
        present_sync_interval,
        allow_tearing: swapchain.allow_tearing,
        last_present_index: None,
    })
}

// The DSV heap, sized to `heap_layout`'s DSV slots.
pub(super) fn create_dsv_heap(device: &ID3D12Device) -> RenderResult<ID3D12DescriptorHeap> {
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
    // new COM object lands in a binding that owns it.
    let dsv_heap: ID3D12DescriptorHeap = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            NumDescriptors: DSV_SLOTS as u32,
            ..Default::default()
        })
    }
    .map_err(|e| format!("DSV heap: {e}"))?;
    Ok(dsv_heap)
}

// The shader-visible sampler heap with its static samplers written.
pub(super) fn create_sampler_heap(
    device: &ID3D12Device,
    anisotropy: u32,
) -> RenderResult<ID3D12DescriptorHeap> {
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
    // new COM object lands in a binding that owns it.
    let sampler_heap: ID3D12DescriptorHeap = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: SAMPLER_SLOTS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        })
    }
    .map_err(|e| format!("sampler heap: {e}"))?;
    let sampler_descriptor_size = descriptor_size(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    create_samplers(
        device,
        cpu_handle(&sampler_heap, sampler_descriptor_size, 0),
        sampler_descriptor_size,
        anisotropy,
    );
    Ok(sampler_heap)
}

fn create_samplers(
    device: &ID3D12Device,
    base_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    stride: usize,
    // Scene-sampler max anisotropy from GraphicsConfig.anisotropy, clamped to the
    // D3D12 1..16 range below.
    anisotropy: u32,
) {
    // [0] Shadow comparison sampler (LESS_EQUAL).
    let shadow_samp = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe {
        device.CreateSampler(
            &shadow_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base_cpu.ptr },
        )
    };

    // [1] Anisotropic repeat (albedo + normal map). Anisotropic filtering plus
    // the unclamped MaxLOD lets minified scene textures trilinear-select down
    // their mip chain instead of aliasing from mip 0. The degree comes from
    // GraphicsConfig.anisotropy (default 8), clamped to the D3D12 feature-level-11
    // guaranteed 1..16 range.
    let linear_samp = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_ANISOTROPIC,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        MaxAnisotropy: anisotropy.clamp(1, 16),
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe {
        device.CreateSampler(
            &linear_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base_cpu.ptr + stride,
            },
        )
    };

    // [2] Cube linear-clamp + mip linear (IBL irradiance / prefilter).
    let cube_samp = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe {
        device.CreateSampler(
            &cube_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base_cpu.ptr + stride * 2,
            },
        )
    };

    // [3] Linear clamp, mip 0 only (text atlas). The text atlas is a tightly
    // packed glyph SDF: its coarse mips bleed adjacent glyphs together, so
    // trilinear minification samples that garbage and the text reads choppy.
    // Clamp MaxLOD to 0 so only the full-resolution (supersampled) mip 0 is
    // sampled; the SDF stays crisp under bilinear minification on its own.
    // Mirrors the Vulkan text sampler (`create_sampler_linear_clamp`, whose
    // max_lod defaults to 0).
    let clamp_samp = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MinLOD: 0.0,
        MaxLOD: 0.0,
        ..Default::default()
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe {
        device.CreateSampler(
            &clamp_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base_cpu.ptr + stride * 3,
            },
        )
    };
}
