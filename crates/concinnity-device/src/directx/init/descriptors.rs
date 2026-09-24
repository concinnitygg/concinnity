//! Shader-visible descriptors: the CBV/SRV/UAV heap laid out by
//! `heap_layout.rs`, the sampler heap's GPU handles, and the shared post
//! passes' descriptor block.

use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::heap_layout::{RtvHeapLayout, SrvHeapLayout, SrvHeapParams};
use super::{InitGpu, heaps};
use crate::directx::context::{DxDescriptors, SwapchainState};
use crate::directx::descriptor_slot::{SamplerSlot, SrvSlot};
use crate::directx::error::map_hresult;
use crate::directx::post::descriptors::PostDescriptors;

pub(super) fn build_descriptors(
    gpu: &InitGpu<'_>,
    params: &SrvHeapParams,
    anisotropy: u32,
) -> RenderResult<DxDescriptors> {
    let device = &gpu.hw.device;
    // CBV/SRV/UAV heap slot layout. The full per-block map + the
    // positional cascade live in `heap_layout.rs`, which a unit test
    // anchors so a stray offset edit fails a test instead of silently
    // misbinding a descriptor at shader time.
    let layout = SrvHeapLayout::compute(params);
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
    // new COM object lands in a binding that owns it.
    let srv_heap: ID3D12DescriptorHeap = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            // `srv_slots` is the running total of every block in the
            // heap_layout cascade, so it sizes the heap to exactly cover
            // the highest slot any descriptor write addresses.
            NumDescriptors: layout.srv_slots as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        })
    }
    .map_err(|e| map_hresult(e.code(), "SRV heap"))?;
    let srv_descriptor_size =
        heaps::descriptor_size(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    let sampler_heap = heaps::create_sampler_heap(device, anisotropy)?;
    let sampler_descriptor_size =
        heaps::descriptor_size(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    let sampler_gpu = |slot| SamplerSlot::at(&sampler_heap, sampler_descriptor_size, slot);
    Ok(DxDescriptors {
        shadow_sampler_gpu: sampler_gpu(heaps::SHADOW_SAMPLER_SLOT),
        linear_sampler_gpu: sampler_gpu(heaps::LINEAR_SAMPLER_SLOT),
        text_sampler_gpu: sampler_gpu(heaps::TEXT_SAMPLER_SLOT),
        srv_heap,
        srv_descriptor_size,
        flat_pool_len: params.albedo_count + params.normal_count,
        layout,
        sampler_heap,
    })
}

impl DxDescriptors {
    // CPU handle of SRV heap `slot`.
    pub(in crate::directx) fn slot_cpu(&self, slot: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        heaps::cpu_handle(&self.srv_heap, self.srv_descriptor_size, slot)
    }

    // GPU handle of SRV heap `slot`.
    pub(super) fn slot_gpu(&self, slot: usize) -> SrvSlot {
        SrvSlot::at(&self.srv_heap, self.srv_descriptor_size, slot)
    }
}

// The shared post passes' descriptor block: SRVs after the color LUT, RTVs
// after the bloom mips.
pub(super) fn build_post_descriptors(
    descriptors: &DxDescriptors,
    swapchain: &SwapchainState,
    rtv: &RtvHeapLayout,
) -> PostDescriptors {
    let post_srv_base_slot = descriptors.layout.post_srv_base_slot;
    PostDescriptors::new(
        descriptors.slot_cpu(post_srv_base_slot),
        descriptors.slot_gpu(post_srv_base_slot),
        descriptors.srv_descriptor_size,
        swapchain.rtv(rtv.post_base_slot),
        swapchain.rtv_descriptor_size,
    )
}
