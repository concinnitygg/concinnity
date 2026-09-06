// src/metal/transient.rs
//
// Ring-buffered per-frame upload buffers. The bindless object / draw-args /
// texture-argument buffers (and skinned joint palettes) used to be freshly
// `newBufferWith*`'d every frame: one driver allocation per buffer per frame,
// retained by the committed command buffer until the GPU retired it. With the
// frames-in-flight fence (`metal/frame_pacing.rs`) bounding the CPU to at most
// `frames_in_flight` frames ahead of the GPU, those allocations collapse into a
// small ring of persistent `StorageModeShared` buffers: frame `R` writes ring
// slot `R % depth` and binds it, and because the fence guarantees frame `R −
// depth` has already retired before frame `R` can acquire a slot, the slot the
// CPU is about to overwrite is provably no longer being read by the GPU.
//
// Each slot grows power-of-two on demand (like `ensure_icb_capacity`) and is
// never shrunk, so steady state does zero allocation.

use std::collections::VecDeque;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::context::{bytes_of_slice, write_buffer_region};

// The capacity a ring slot must be (re)allocated to in order to hold `needed`
// bytes, or `None` when its current `have` bytes already do. Rounding up to a
// power of two keeps a slowly-growing list from reallocating every frame; the
// 256-byte floor keeps tiny buffers off the page-size floor. Pure so the growth
// policy is unit-testable.
pub(super) fn grow_to(have: usize, needed: usize) -> Option<usize> {
    let needed = needed.max(1);
    if have >= needed {
        return None;
    }
    Some(needed.next_power_of_two().max(256))
}

// A frame-tagged deferred-free pool. A GPU resource an in-flight frame may
// still read cannot be freed or overwritten the instant the CPU replaces it;
// instead the old handle is parked here, tagged with the frame that retired it,
// and dropped only once the frames-in-flight fence guarantees every frame that
// could still reference it has retired on the GPU.
//
// This deliberately does NOT key storage by `frame_index % depth` the way the
// rings above do. A modulo ring is safe only when every slot is rewritten every
// `depth` frames, so the fence's "slot `R − depth` has retired" guarantee
// covers the slot about to be reused. A resource the reflection trace reads
// breaks that assumption: a static / sparsely-moving scene keeps tracing the
// same last-built acceleration structure for many frames without rebuilding, so
// that resource is read by frames the fence does not pair with its writer.
// Deferring the free by retirement frame is the correct general mechanism.
//
// Generic over the payload so the retirement timing is unit-testable without a
// GPU; the real pool stores `Retained` Metal handles.
pub(super) struct RetirePool<T> {
    // (retiring_frame, payload), pushed in nondecreasing frame order.
    pending: VecDeque<(u64, T)>,
}

impl<T> RetirePool<T> {
    pub(super) fn new() -> Self {
        Self {
            pending: VecDeque::new(),
        }
    }

    // Park `payload`, keeping it alive until `collect` is called for a frame at
    // least `depth` ahead of `retired_at`. Call when the CPU replaces a GPU
    // resource an in-flight frame may still be reading.
    pub(super) fn push(&mut self, retired_at: u64, payload: T) {
        self.pending.push_back((retired_at, payload));
    }

    // Drop every payload whose retiring frame is at least `depth` frames behind
    // `frame_id` (`retired_at + depth <= frame_id`). The frames-in-flight fence
    // guarantees frame `frame_id − depth`, and every earlier frame, has retired
    // on the GPU, so any frame that could still reference such a payload is
    // done. Entries are pushed in nondecreasing frame order, so draining
    // front-to-back can stop at the first still-live entry.
    pub(super) fn collect(&mut self, frame_id: u64, depth: u64) {
        while let Some(&(retired_at, _)) = self.pending.front() {
            if retired_at.saturating_add(depth) <= frame_id {
                self.pending.pop_front();
            } else {
                break;
            }
        }
    }

    // Number of payloads still held alive. Diagnostics / tests only.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.pending.len()
    }
}

// A small ring of persistent shared-storage buffers, one usable slot per
// frame-in-flight. Hand out a slot's buffer for the current frame via
// [`Self::slot`] (capacity only) or [`Self::write`] (capacity + memcpy).
pub(super) struct TransientRing {
    slots: Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
}

impl TransientRing {
    // `depth` is the frames-in-flight count; clamped to ≥1. Buffers are
    // allocated lazily on first use of each slot.
    pub(super) fn new(depth: usize) -> Self {
        Self {
            slots: (0..depth.max(1)).map(|_| None).collect(),
        }
    }

    // Return a cloned handle to `slot`'s buffer, (re)allocating it shared and
    // power-of-two-grown to hold at least `min_len` bytes. Contents are left
    // as-is: use this for buffers an argument encoder fills in place.
    pub(super) fn slot(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        min_len: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let idx = slot % self.slots.len();
        let have = self.slots[idx].as_ref().map_or(0, |buf| buf.length());
        if let Some(cap) = grow_to(have, min_len) {
            let buf = device
                .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
                .ok_or("failed to allocate transient ring buffer")?;
            self.slots[idx] = Some(buf);
        }
        Ok(self.slots[idx]
            .as_ref()
            .expect("ring slot was just ensured")
            .clone())
    }

    // Copy `bytes` into `slot`'s buffer (growing it first) and return a cloned
    // handle to bind. The handle is a cheap refcount bump on a buffer the ring
    // owns; the committed command buffer keeps it resident until the GPU is
    // done, and the fence prevents the next writer from racing that read.
    pub(super) fn write(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        bytes: &[u8],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let buf = self.slot(device, slot, bytes.len().max(1))?;
        write_buffer_region(&buf, 0, bytes)?;
        Ok(buf)
    }
}

// Ring of per-skinned-object upload buffers for one pose stream (the current
// pose, or the previous pose the velocity pre-pass reprojects from). Each ring
// slot holds one buffer per skinned object per column; a `write_*` fills this
// frame's slot and returns cloned handles in object order, matching the shape
// the per-pass encoders bind so they need no change. Current and previous poses
// use separate `JointRing`s because the velocity pass reads both in the same
// frame and they must not alias the same slot.
//
// The morph weights ride the same slot table as the joint palettes rather than a
// ring of their own: both are per-skinned-object, per-frame, and written from
// the same place, so one growth policy and one slot walk cover them. They stay
// separate buffers because the kernel binds them at separate indices. Only the
// current-pose ring writes the weights column; the previous-pose ring leaves it
// unallocated.
//
// One skinned object's buffers for a given ring slot, each absent until that
// column is first written.
type ObjectBuffer = Option<Retained<ProtocolObject<dyn MTLBuffer>>>;

#[derive(Default)]
struct SkinSlot {
    palette: ObjectBuffer,
    weights: ObjectBuffer,
}

pub(super) struct JointRing {
    // slots[ring_slot][object] -> that object's buffers
    slots: Vec<Vec<SkinSlot>>,
}

impl JointRing {
    pub(super) fn new(depth: usize) -> Self {
        Self {
            slots: (0..depth.max(1)).map(|_| Vec::new()).collect(),
        }
    }

    // Ensure this frame's ring slot has a palette buffer per `palettes` entry
    // (each grown to fit its matrices), copy each palette in, and return cloned
    // handles in order. Empty when `palettes` is empty (no skinned meshes).
    pub(super) fn write_all(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        palettes: &[Vec<[[f32; 4]; 4]>],
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, String> {
        self.objects(slot, palettes.len())
            .iter_mut()
            .zip(palettes)
            .map(|(o, mats)| fill(device, &mut o.palette, bytes_of_slice(mats), "joint"))
            .collect()
    }

    // The same, for the per-object morph weights the skin kernel indexes by
    // morph target. An object without targets still gets a buffer (the ring's
    // size floor), because the kernel binds the slot unconditionally and reads
    // it only when that object's `target_count` is non-zero.
    pub(super) fn write_weights(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        weights: &[Vec<f32>],
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, String> {
        self.objects(slot, weights.len())
            .iter_mut()
            .zip(weights)
            .map(|(o, w)| fill(device, &mut o.weights, bytes_of_slice(w), "morph weight"))
            .collect()
    }

    // This frame's slot, grown to `count` objects.
    fn objects(&mut self, slot: usize, count: usize) -> &mut [SkinSlot] {
        let idx = slot % self.slots.len();
        let slots = &mut self.slots[idx];
        if slots.len() < count {
            slots.resize_with(count, SkinSlot::default);
        }
        &mut slots[..count]
    }
}

// Grow one ring buffer to hold `bytes`, copy them in, and return a cloned
// handle. `what` names the column for the allocation-failure message.
fn fill(
    device: &ProtocolObject<dyn MTLDevice>,
    cell: &mut ObjectBuffer,
    bytes: &[u8],
    what: &str,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
    let have = cell.as_ref().map_or(0, |buf| buf.length());
    if let Some(cap) = grow_to(have, bytes.len()) {
        let buf = device
            .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| format!("failed to allocate {what} ring buffer"))?;
        *cell = Some(buf);
    }
    let buf = cell.as_ref().expect("ring slot was just ensured");
    write_buffer_region(buf, 0, bytes)?;
    Ok(buf.clone())
}

#[cfg(test)]
mod tests {
    use super::{RetirePool, grow_to};

    #[test]
    fn grow_to_keeps_a_slot_that_already_fits() {
        assert_eq!(grow_to(256, 200), None);
        assert_eq!(grow_to(256, 256), None);
        // A zero-byte request still needs one byte of storage, which any
        // existing slot has.
        assert_eq!(grow_to(256, 0), None);
    }

    #[test]
    fn grow_to_rounds_up_past_the_floor() {
        // An empty slot always allocates, never below the 256-byte floor.
        assert_eq!(grow_to(0, 1), Some(256));
        assert_eq!(grow_to(0, 0), Some(256));
        assert_eq!(grow_to(0, 300), Some(512));
        // Growth is power-of-two so a slowly-growing list stops reallocating.
        assert_eq!(grow_to(512, 513), Some(1024));
        assert_eq!(grow_to(1024, 4096), Some(4096));
    }

    #[test]
    fn retire_pool_holds_payloads_for_depth_frames() {
        // depth = 2: a payload retired at frame N must survive until the frame
        // that retires N (N + depth) so any frame that still reads it (≤ N − 1,
        // all retired by N + depth − 1) is provably done.
        let mut pool: RetirePool<u32> = RetirePool::new();
        pool.push(0, 100);
        pool.push(1, 101);
        pool.push(2, 102);

        // Frame 1 with depth 2: 0 + 2 = 2 > 1, nothing freed yet.
        pool.collect(1, 2);
        assert_eq!(pool.len(), 3);

        // Frame 2: 0 + 2 = 2 ≤ 2 frees the frame-0 payload; 1 + 2 = 3 > 2 stays.
        pool.collect(2, 2);
        assert_eq!(pool.len(), 2);

        // Frame 3 frees the frame-1 payload.
        pool.collect(3, 2);
        assert_eq!(pool.len(), 1);

        // A jump well past the last tag drains the rest.
        pool.collect(100, 2);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn retire_pool_depth_one_frees_one_frame_later() {
        let mut pool: RetirePool<u32> = RetirePool::new();
        pool.push(5, 7);
        // Same frame: 5 + 1 = 6 > 5, still live.
        pool.collect(5, 1);
        assert_eq!(pool.len(), 1);
        // Next frame: 5 + 1 = 6 ≤ 6, freed.
        pool.collect(6, 1);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn retire_pool_drains_multiple_same_frame_pushes() {
        // More than one payload can retire in the same frame (e.g. a TLAS plus
        // its geometry table). They share a tag and free together.
        let mut pool: RetirePool<u32> = RetirePool::new();
        pool.push(4, 1);
        pool.push(4, 2);
        pool.push(4, 3);
        pool.collect(5, 2); // 4 + 2 = 6 > 5, all live
        assert_eq!(pool.len(), 3);
        pool.collect(6, 2); // 4 + 2 = 6 ≤ 6, all freed
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn retire_pool_collect_is_idempotent_and_empty_safe() {
        let mut pool: RetirePool<u32> = RetirePool::new();
        pool.collect(10, 2); // empty pool, no panic
        assert_eq!(pool.len(), 0);
        pool.push(0, 1);
        pool.collect(10, 2);
        pool.collect(10, 2); // second call is a no-op
        assert_eq!(pool.len(), 0);
    }
}
