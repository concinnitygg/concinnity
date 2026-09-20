//! GPU descriptor handles, typed by the heap they came from.
//!
//! A shader-visible bind names a heap slot, and D3D12 has two heaps that can be
//! bound at once: CBV/SRV/UAV and sampler. Both hand out
//! `D3D12_GPU_DESCRIPTOR_HANDLE`, a bare `u64`, so nothing but care keeps a
//! sampler handle out of a table whose range declares SRVs -- a mistake the
//! compiler cannot see and the GPU reads as garbage. These two types carry the
//! heap in the handle's own type instead, and the bind methods below take the
//! one their root-parameter range was declared for.

use windows::Win32::Graphics::Direct3D12::*;

/// A slot in the CBV/SRV/UAV heap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SrvSlot(D3D12_GPU_DESCRIPTOR_HANDLE);

/// A slot in the sampler heap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::directx) struct SamplerSlot(D3D12_GPU_DESCRIPTOR_HANDLE);

impl SrvSlot {
    /// Slot `index` of `heap`, whose descriptors are `stride` bytes apart.
    pub(in crate::directx) fn at(heap: &ID3D12DescriptorHeap, stride: usize, index: usize) -> Self {
        Self(gpu_handle(heap, stride, index))
    }

    /// The slot `count` descriptors past this one.
    pub(in crate::directx) fn offset(self, count: usize, stride: usize) -> Self {
        Self(D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.0.ptr + (count * stride) as u64,
        })
    }

    /// This slot's CPU handle, for the writes a shader-visible heap still takes
    /// through its CPU side. Both bases come from the same heap, so the
    /// shader-visible offset carries over unchanged.
    pub(in crate::directx) fn cpu_in(
        self,
        cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
        gpu_base: Self,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu_base.ptr + (self.0.ptr - gpu_base.0.ptr) as usize,
        }
    }

    /// A slot at a made-up address, for tests over the arithmetic alone.
    #[cfg(test)]
    pub(in crate::directx) fn for_test(ptr: u64) -> Self {
        Self(D3D12_GPU_DESCRIPTOR_HANDLE { ptr })
    }
}

impl SamplerSlot {
    /// Slot `index` of `heap`, whose descriptors are `stride` bytes apart.
    pub(in crate::directx) fn at(heap: &ID3D12DescriptorHeap, stride: usize, index: usize) -> Self {
        Self(gpu_handle(heap, stride, index))
    }
}

fn gpu_handle(
    heap: &ID3D12DescriptorHeap,
    stride: usize,
    index: usize,
) -> D3D12_GPU_DESCRIPTOR_HANDLE {
    // SAFETY: a property query on a live descriptor heap; it only reads.
    let base = unsafe { heap.GetGPUDescriptorHandleForHeapStart() };
    D3D12_GPU_DESCRIPTOR_HANDLE {
        ptr: base.ptr + (index * stride) as u64,
    }
}

pub(in crate::directx) trait DescriptorTables {
    // SAFETY: the caller holds the list in the recording state, with a graphics
    // root signature bound whose parameter `param` is a table of SRV / CBV / UAV
    // ranges, and the CBV/SRV/UAV heap set on the list.
    unsafe fn set_graphics_srv_table(&self, param: u32, slot: SrvSlot);

    // SAFETY: as `set_graphics_srv_table`, for a table of sampler ranges with
    // the sampler heap set on the list.
    unsafe fn set_graphics_sampler_table(&self, param: u32, slot: SamplerSlot);

    // SAFETY: as `set_graphics_srv_table`, for the compute root signature.
    unsafe fn set_compute_srv_table(&self, param: u32, slot: SrvSlot);
}

impl DescriptorTables for ID3D12GraphicsCommandList {
    unsafe fn set_graphics_srv_table(&self, param: u32, slot: SrvSlot) {
        // SAFETY: forwarded from the caller; the slot names a live descriptor in
        // the heap its type stands for.
        unsafe { self.SetGraphicsRootDescriptorTable(param, slot.0) }
    }

    unsafe fn set_graphics_sampler_table(&self, param: u32, slot: SamplerSlot) {
        // SAFETY: forwarded from the caller; the slot names a live descriptor in
        // the heap its type stands for.
        unsafe { self.SetGraphicsRootDescriptorTable(param, slot.0) }
    }

    unsafe fn set_compute_srv_table(&self, param: u32, slot: SrvSlot) {
        // SAFETY: forwarded from the caller; the slot names a live descriptor in
        // the heap its type stands for.
        unsafe { self.SetComputeRootDescriptorTable(param, slot.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(ptr: u64) -> SrvSlot {
        SrvSlot(D3D12_GPU_DESCRIPTOR_HANDLE { ptr })
    }

    #[test]
    fn offset_steps_by_whole_descriptors() {
        assert_eq!(slot(1024).offset(3, 32), slot(1024 + 96));
        assert_eq!(slot(1024).offset(0, 32), slot(1024));
    }

    #[test]
    fn cpu_in_keeps_the_slot_offset() {
        let base = slot(4096);
        let cpu_base = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 900 };
        assert_eq!(base.offset(2, 32).cpu_in(cpu_base, base).ptr, 900 + 64);
        assert_eq!(base.cpu_in(cpu_base, base).ptr, 900);
    }
}
