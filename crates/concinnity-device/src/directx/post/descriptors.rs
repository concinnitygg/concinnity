// src/directx/post/descriptors.rs
//
// The descriptor slots the shared fullscreen post passes allocate from, in
// place of the per-pass reservations `init/heap_layout.rs` used to carve for
// each effect by name.
//
// A block of the shader-visible CBV/SRV/UAV heap and a block of the RTV heap,
// handed out a slot at a time and returned when the target holding the slot is
// dropped. A post target is persistent, so the descriptors describing one are
// too; a pass that recreates its targets drops the old ones first and gets the
// same slots back. What the block removes is the per-pass naming: a new post
// pass takes what it needs from here instead of adding a `<effect>_srv_extra`
// to the heap cascade.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use windows::Win32::Graphics::Direct3D12::*;

// Post targets the shared passes may hold at once: the temporal resolve's two,
// the reflection target, the indirect-light gather target, and room for the
// passes still to follow.
pub(in crate::directx) const POST_TARGET_SLOTS: usize = 16;

// The occupancy mask is one bit per slot.
const _: () = assert!(POST_TARGET_SLOTS <= u32::BITS as usize);

// One post target's descriptors: the shader-visible SRV a consumer samples it
// through, and the RTV the pass writes it through. The slot returns to the
// block when this is dropped.
pub(in crate::directx) struct PostTargetDescriptors {
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    _lease: SlotLease,
}

// A held slot, released on drop.
struct SlotLease {
    used: Arc<AtomicU32>,
    index: usize,
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        self.used.fetch_and(!(1 << self.index), Ordering::Relaxed);
    }
}

// The reserved blocks and which of their slots are held.
pub(in crate::directx) struct PostDescriptors {
    srv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE,
    srv_size: usize,
    rtv_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    rtv_size: usize,
    used: Arc<AtomicU32>,
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
            used: Arc::new(AtomicU32::new(0)),
        }
    }

    // The lowest free slot's descriptors.
    pub(in crate::directx) fn allocate(&self) -> Result<PostTargetDescriptors, String> {
        let mut current = self.used.load(Ordering::Relaxed);
        let i = loop {
            let i = (!current).trailing_zeros() as usize;
            if i >= POST_TARGET_SLOTS {
                return Err(format!(
                    "the shared post passes asked for more than {POST_TARGET_SLOTS} targets"
                ));
            }
            match self.used.compare_exchange_weak(
                current,
                current | (1 << i),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break i,
                Err(actual) => current = actual,
            }
        };
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
            _lease: SlotLease {
                used: Arc::clone(&self.used),
                index: i,
            },
        })
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
    fn a_dropped_set_is_reissued_on_the_same_slots() {
        // A resize drops a pass's targets and recreates them: the new set lands
        // where the old one was.
        let b = block();
        let first: Vec<_> = (0..3).map(|_| b.allocate().expect("slot")).collect();
        let ptrs: Vec<_> = first.iter().map(|d| d.srv_gpu.ptr).collect();
        drop(first);
        let again: Vec<_> = (0..3).map(|_| b.allocate().expect("slot")).collect();
        assert_eq!(
            ptrs,
            again.iter().map(|d| d.srv_gpu.ptr).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_held_slot_is_never_reissued() {
        // Two passes share the block: recreating one must not hand its slot to
        // a target of the other while that target is still live.
        let b = block();
        let taa = [b.allocate().expect("slot"), b.allocate().expect("slot")];
        let ssr = b.allocate().expect("slot");
        let ssr_slot = ssr.srv_gpu.ptr;
        drop(taa);
        let rebuilt = [b.allocate().expect("slot"), b.allocate().expect("slot")];
        assert!(rebuilt.iter().all(|d| d.srv_gpu.ptr != ssr_slot));
        assert_eq!(ssr.srv_gpu.ptr, ssr_slot);
    }

    #[test]
    fn a_freed_middle_slot_is_the_next_one_handed_out() {
        let b = block();
        let _first = b.allocate().expect("slot");
        let middle = b.allocate().expect("slot");
        let _last = b.allocate().expect("slot");
        let middle_ptr = middle.rtv.ptr;
        drop(middle);
        assert_eq!(b.allocate().expect("slot").rtv.ptr, middle_ptr);
    }

    #[test]
    fn running_out_of_slots_is_an_error_not_an_overrun() {
        // Past the reservation the next handle would address another feature's
        // descriptors, so the allocation fails instead.
        let b = block();
        let held: Vec<_> = (0..POST_TARGET_SLOTS)
            .map(|_| b.allocate().expect("reserved slot"))
            .collect();
        assert!(b.allocate().is_err());
        drop(held);
        assert!(b.allocate().is_ok());
    }
}
