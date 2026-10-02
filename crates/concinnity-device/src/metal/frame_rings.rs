//! Ring-buffered per-frame upload buffers for the bindless object / draw-args /
//! texture-argument buffers and skinned joint palettes. The frames-in-flight
//! fence (`metal/frame_pacing.rs`) bounds the CPU to at most `frames_in_flight`
//! frames ahead of the GPU, so each is a small ring of persistent
//! `StorageModeShared` buffers: frame `R` writes ring slot `R % depth` and binds
//! it, and because the fence guarantees frame `R - depth` has already retired
//! before frame `R` can acquire a slot, the slot the CPU is about to overwrite is
//! provably no longer being read by the GPU.
//!
//! Each slot grows power-of-two on demand (like `ensure_icb_capacity`) and is
//! never shrunk, so steady state does zero allocation.

use super::error::allocation_failed;
use concinnity_core::render::buffer_growth::grow_capacity;
use concinnity_core::render::error::RenderResult;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::context::{bytes_of_slice, write_buffer_region};

// The capacity a ring slot must be (re)allocated to in order to hold `needed`
// bytes, or `None` when its current `have` bytes already do. An empty slot always
// allocates, so a zero-byte request still gets a buffer to bind.
pub(super) fn grow_to(have: usize, needed: usize) -> Option<usize> {
    grow_capacity(have as u64, needed.max(1) as u64, 256).map(|c| c as usize)
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
    ) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
        Ok(self.slot_fresh(device, slot, min_len)?.0)
    }

    // [`Self::slot`], also reporting whether the slot was (re)allocated by this
    // call. A producer that skips re-encoding an unchanged slot needs to know:
    // a fresh buffer holds nothing it wrote.
    pub(super) fn slot_fresh(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        min_len: usize,
    ) -> RenderResult<(Retained<ProtocolObject<dyn MTLBuffer>>, bool)> {
        let idx = slot % self.slots.len();
        let have = self.slots[idx].as_ref().map_or(0, |buf| buf.length());
        let grown = grow_to(have, min_len);
        if let Some(cap) = grown {
            let buf = device
                .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| allocation_failed("transient ring buffer"))?;
            self.slots[idx] = Some(buf);
        }
        Ok((
            self.slots[idx]
                .as_ref()
                .expect("ring slot was just ensured")
                .clone(),
            grown.is_some(),
        ))
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
    ) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
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
    ) -> RenderResult<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>> {
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
    ) -> RenderResult<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>> {
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
) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let have = cell.as_ref().map_or(0, |buf| buf.length());
    if let Some(cap) = grow_to(have, bytes.len()) {
        let buf = device
            .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| allocation_failed(format_args!("{what} ring buffer")))?;
        *cell = Some(buf);
    }
    let buf = cell.as_ref().expect("ring slot was just ensured");
    write_buffer_region(buf, 0, bytes)?;
    Ok(buf.clone())
}

#[cfg(test)]
mod tests {
    use super::{TransientRing, grow_to};

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLBuffer, MTLDevice};

    // `None` on a machine with no Metal device, which skips the device-backed
    // tests rather than failing them.
    fn device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
        objc2_metal::MTLCreateSystemDefaultDevice()
    }

    // The bytes a shared-storage buffer currently holds, truncated to `len`.
    fn read_back(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<u8> {
        // SAFETY: the ring allocates every slot `StorageModeShared` and at least
        // `len` bytes long (the caller passes the length it just wrote), so
        // `contents()` is a live CPU mapping of at least that many bytes.
        unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len).to_vec() }
    }

    #[test]
    fn an_empty_slot_always_gets_a_buffer() {
        assert_eq!(grow_to(0, 0), Some(256));
        assert_eq!(grow_to(256, 0), None);
    }

    #[test]
    fn steady_state_writes_reuse_one_buffer_per_slot() {
        let Some(device) = device() else {
            return;
        };
        let mut ring = TransientRing::new(2);
        let first = ring.write(&device, 0, &[7u8; 64]).expect("first write");
        for _ in 0..8 {
            let again = ring.write(&device, 0, &[7u8; 64]).expect("repeat write");
            assert!(
                std::ptr::eq(&*first, &*again),
                "a slot that already fits must not reallocate"
            );
        }
    }

    #[test]
    fn write_copies_the_bytes_into_the_slot() {
        let Some(device) = device() else {
            return;
        };
        let mut ring = TransientRing::new(2);
        let payload: Vec<u8> = (0..96u8).collect();
        let buf = ring.write(&device, 0, &payload).expect("write");
        assert_eq!(read_back(&buf, payload.len()), payload);

        // Rewriting the same slot replaces its contents rather than appending.
        let next = vec![0xABu8; 32];
        let buf = ring.write(&device, 0, &next).expect("rewrite");
        assert_eq!(read_back(&buf, next.len()), next);
    }

    // Two frames in flight must not share storage: the whole point of the ring
    // is that overwriting this frame's slot cannot touch bytes the GPU is still
    // reading for the previous frame.
    #[test]
    fn distinct_slots_get_distinct_buffers() {
        let Some(device) = device() else {
            return;
        };
        let mut ring = TransientRing::new(2);
        let a = ring.write(&device, 0, &[1u8; 48]).expect("slot 0");
        let b = ring.write(&device, 1, &[2u8; 48]).expect("slot 1");
        assert!(!std::ptr::eq(&*a, &*b));
        assert_eq!(read_back(&a, 48), vec![1u8; 48]);
        assert_eq!(read_back(&b, 48), vec![2u8; 48]);
    }

    // Slot indices wrap, so a monotonically-increasing frame counter can be
    // handed in directly: frame `R` and frame `R + depth` land on one buffer.
    #[test]
    fn slot_indices_wrap_modulo_depth() {
        let Some(device) = device() else {
            return;
        };
        let mut ring = TransientRing::new(2);
        let a = ring.write(&device, 0, &[1u8; 16]).expect("frame 0");
        let c = ring.write(&device, 2, &[3u8; 16]).expect("frame 2");
        assert!(std::ptr::eq(&*a, &*c), "frame 2 must reuse slot 0");
    }

    // A slot that outgrows its buffer reallocates, and the new one holds the
    // larger payload; a later smaller write reuses it rather than shrinking.
    #[test]
    fn a_slot_grows_once_and_never_shrinks() {
        let Some(device) = device() else {
            return;
        };
        let mut ring = TransientRing::new(1);
        let small = ring.write(&device, 0, &[0u8; 16]).expect("small");
        let small_len = small.length();
        let big = ring.write(&device, 0, &[9u8; 4096]).expect("big");
        assert!(big.length() >= 4096);
        assert!(big.length() > small_len);
        let shrunk = ring.write(&device, 0, &[5u8; 8]).expect("small again");
        assert!(
            std::ptr::eq(&*big, &*shrunk),
            "a grown slot must not be reallocated by a smaller write"
        );
    }
}
