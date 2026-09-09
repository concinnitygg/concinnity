// src/directx/post/descriptors.rs
//
// The descriptor slots the shared fullscreen post passes allocate from, in
// place of the per-pass reservations `init/heap_layout.rs` used to carve for
// each effect by name.
//
// A block of the shader-visible CBV/SRV/UAV heap and a block of the RTV heap,
// each sub-allocated by a bump cursor and rewound as a unit when the targets
// they describe are recreated. A post target is persistent -- a temporal pass
// accumulates across frames, and the bloom prefilter and the composite bind its
// output by a handle they hold -- so the descriptors describing one are
// persistent too, and a per-frame ring would only force every consumer to
// re-read a handle each frame. What the block removes is the per-pass naming:
// a new post pass takes what it needs from here instead of adding a
// `<effect>_srv_extra` to the heap cascade.

use std::cell::Cell;

use windows::Win32::Graphics::Direct3D12::*;

// Post targets the shared passes may hold at once. The temporal resolve is two;
// this leaves room for the remaining fullscreen passes to follow without the
// block being resized.
pub(in crate::directx) const POST_TARGET_SLOTS: usize = 16;

// One post target's descriptors: the shader-visible SRV a consumer samples it
// through, and the RTV the pass writes it through.
#[derive(Clone, Copy)]
pub(in crate::directx) struct PostTargetDescriptors {
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
}

// The reserved blocks and where the next target's descriptors come from.
//
// `Cell` because allocation happens while building or resizing resources, which
// is a single-threaded `&self` path; the parallel graph executor only *reads*
// handles a target already holds.
pub(in crate::directx) struct PostDescriptors {
    srv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE,
    srv_size: usize,
    rtv_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    rtv_size: usize,
    next: Cell<usize>,
}

impl PostDescriptors {
    // The block starting at the given heap slots.
    pub(in crate::directx) fn new(
        srv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
        srv_gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE,
        srv_size: usize,
        rtv_base: D3D12_CPU_DESCRIPTOR_HANDLE,
        rtv_size: usize,
    ) -> Self {
        Self {
            srv_cpu_base,
            srv_gpu_base,
            srv_size,
            rtv_base,
            rtv_size,
            next: Cell::new(0),
        }
    }

    // The next free target's descriptors.
    pub(in crate::directx) fn allocate(&self) -> Result<PostTargetDescriptors, String> {
        let i = self.next.get();
        if i >= POST_TARGET_SLOTS {
            return Err(format!(
                "the shared post passes asked for more than {POST_TARGET_SLOTS} targets"
            ));
        }
        self.next.set(i + 1);
        Ok(PostTargetDescriptors {
            srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.srv_cpu_base.ptr + i * self.srv_size,
            },
            srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: self.srv_gpu_base.ptr + (i * self.srv_size) as u64,
            },
            rtv: D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.rtv_base.ptr + i * self.rtv_size,
            },
        })
    }

    // Rewind the block. Called when every post target is about to be recreated
    // (a resize, or a quality toggle that rebuilds the passes), so the new set
    // lands on the same slots the old one held and any handle a consumer still
    // caches keeps describing the right target.
    pub(in crate::directx) fn rewind(&self) {
        self.next.set(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block() -> PostDescriptors {
        PostDescriptors::new(
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 4096 },
            D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 8192 },
            32,
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 512 },
            16,
        )
    }

    #[test]
    fn slots_are_handed_out_in_order_at_the_heap_stride() {
        let b = block();
        let a = b.allocate().expect("first slot");
        let c = b.allocate().expect("second slot");
        assert_eq!(a.srv_cpu.ptr, 4096);
        assert_eq!(a.srv_gpu.ptr, 8192);
        assert_eq!(a.rtv.ptr, 512);
        assert_eq!(c.srv_cpu.ptr, 4096 + 32);
        assert_eq!(c.srv_gpu.ptr, 8192 + 32);
        assert_eq!(c.rtv.ptr, 512 + 16);
    }

    #[test]
    fn a_rewind_reissues_the_same_slots() {
        // A resize recreates every target; a consumer holding a handle must
        // still be describing the target that replaced the one it named.
        let b = block();
        let first: Vec<_> = (0..3).map(|_| b.allocate().expect("slot")).collect();
        b.rewind();
        let again: Vec<_> = (0..3).map(|_| b.allocate().expect("slot")).collect();
        for (a, c) in first.iter().zip(&again) {
            assert_eq!(a.srv_cpu.ptr, c.srv_cpu.ptr);
            assert_eq!(a.srv_gpu.ptr, c.srv_gpu.ptr);
            assert_eq!(a.rtv.ptr, c.rtv.ptr);
        }
    }

    #[test]
    fn running_out_of_slots_is_an_error_not_an_overrun() {
        // Past the reservation the next handle would address another feature's
        // descriptors, so the allocation fails instead.
        let b = block();
        for _ in 0..POST_TARGET_SLOTS {
            b.allocate().expect("reserved slot");
        }
        assert!(b.allocate().is_err());
    }
}
