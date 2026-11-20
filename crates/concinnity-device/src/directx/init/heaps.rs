//! The DirectX-only descriptor heaps: the RTV heap holding the swapchain's
//! back-buffer views, the DSV heap, and the sampler heap with its static
//! samplers, plus the handle arithmetic every stage addresses a slot with.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::render::backend::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::InitGpu;
use super::bootstrap::DxgiSwapchain;
use super::heap_layout::{DSV_SLOTS, RtvHeapLayout};
use crate::directx::context::{BACK_BUFFERS, SwapchainState};
use crate::directx::depth::shadow_sample_compare;
use crate::directx::error::map_hresult;

// Sampler heap slots: [0] shadow compare, [1] linear repeat, [2] cube
// linear-clamp (contiguous with [1] for one table), [3] text linear-clamp, and
// [4..7] the raymarch table duplicating shadow, cube and a scene clamp.
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
// `[0, BACK_BUFFERS)`, every later slot written by the stage that renders into it.
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
            NumDescriptors: u32::saturating_from(rtv.rtv_slots),
            ..Default::default()
        })
    }
    .map_err(|e| map_hresult(e.code(), "RTV heap"))?;
    let rtv_descriptor_size = descriptor_size(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    let mut back_buffers = Vec::with_capacity(BACK_BUFFERS);
    for i in 0..BACK_BUFFERS {
        // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters
        // it fills are live locals that outlive the call.
        let buf: ID3D12Resource = unsafe { swapchain.handle.GetBuffer(u32::saturating_from(i)) }
            .map_err(|e| map_hresult(e.code(), &format!("GetBuffer[{i}]")))?;
        let rtv_handle = cpu_handle(&rtv_heap, rtv_descriptor_size, i);
        // SAFETY: the view descriptor and the resource it names are live for the call, and the
        // destination handle addresses a slot this context reserved for the view in a heap it
        // owns.
        unsafe {
            device.CreateRenderTargetView(&buf, None, rtv_handle);
        }
        back_buffers.push(buf);
    }

    // Vsync uses interval 1; uncapped uses interval 0, with the tearing flag when
    // supported.
    let present_sync_interval: u32 = u32::from(vsync);
    // Hold the presents DXGI queues ahead of the display to the frames in
    // flight (its default is three), so a refresh-paced run does not add a
    // frame of input latency on top of the CPU's own lead.
    // DXGI accepts 1 to 16.
    let frames = u32::saturating_from(gpu.hw.swapchain_config.frames_in_flight.clamp(1, 16));
    // SAFETY: a setter on the live swapchain, which was created with the waitable-object flag the
    // call requires; it takes only a scalar.
    unsafe { swapchain.handle.SetMaximumFrameLatency(frames) }
        .map_err(|e| map_hresult(e.code(), "SetMaximumFrameLatency"))?;
    // SAFETY: a query on the live swapchain, created with the waitable-object flag. Each call
    // returns a new handle, which `SwapchainState::frame_latency` owns from here.
    let frame_latency = unsafe { swapchain.handle.GetFrameLatencyWaitableObject() };
    Ok(SwapchainState {
        handle: swapchain.handle,
        back_buffers,
        rtv_heap,
        rtv_descriptor_size,
        format: swapchain.format,
        present_sync_interval,
        allow_tearing: swapchain.allow_tearing,
        frame_latency,
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
            NumDescriptors: u32::saturating_from(DSV_SLOTS),
            ..Default::default()
        })
    }
    .map_err(|e| map_hresult(e.code(), "DSV heap"))?;
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
    .map_err(|e| map_hresult(e.code(), "sampler heap"))?;
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
    // Scene-sampler max anisotropy from TextureFiltering.anisotropy, clamped to the
    // D3D12 1..16 range below.
    anisotropy: u32,
) {
    // [0] Shadow comparison sampler.
    let shadow_samp = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: shadow_sample_compare(),
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe {
        device.CreateSampler(
            &raw const shadow_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base_cpu.ptr },
        );
    };

    // [1] Anisotropic repeat for scene textures, unclamped LOD, degree from
    // `TextureFiltering.anisotropy` clamped to 1..16.
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
            &raw const linear_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base_cpu.ptr + stride,
            },
        );
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
            &raw const cube_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base_cpu.ptr + stride * 2,
            },
        );
    };

    // [3] Linear clamp at mip 0 only, for the text atlas, whose coarse mips bleed
    // glyphs together.
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
            &raw const clamp_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base_cpu.ptr + stride * 3,
            },
        );
    };
}
