//! Free-list allocator for backend draw-object slots. A backend appends draw
//! objects into a single `Vec` and stores raw indices into it on each entity's
//! RenderHandle, so a despawned object's slot cannot be compacted away without
//! invalidating every later index. Instead the allocator hands out a vacated
//! slot before growing the vec: `retire` pushes a freed index, the next runtime
//! spawn pops it. Streamed chunks were the first consumer (one freed chunk's
//! slot reused by the next); runtime entity spawn/despawn is the second. All
//! three backends (Metal, DirectX, Vulkan) route their draw-slot allocation
//! through this.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::DrawObject;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Where a newly allocated draw record lands.
pub enum SlotAlloc {
    /// Reuse this vacated slot: overwrite the existing draw_objects entry.
    Reuse(usize),
    /// No free slot was available: append at this index (== the prior length).
    Append(usize),
}

#[derive(Debug, Default)]
/// Hands out draw-record slots, reusing vacated ones before growing.
pub struct DrawSlotAllocator {
    free: Vec<usize>,
    len: usize,
}

impl DrawSlotAllocator {
    /// Start with `len` slots already in use (the draw objects built at init).
    pub fn with_len(len: usize) -> Self {
        Self {
            free: Vec::new(),
            len,
        }
    }

    /// Hand out a slot: a vacated one if any is free, else the next new index.
    /// The caller writes its draw object at the returned slot and, on Append,
    /// grows whatever side tables run parallel to draw_objects.
    pub fn allocate(&mut self) -> SlotAlloc {
        if let Some(slot) = self.free.pop() {
            SlotAlloc::Reuse(slot)
        } else {
            let idx = self.len;
            self.len += 1;
            SlotAlloc::Append(idx)
        }
    }

    /// Return a slot to the free list for a later allocate to reuse.
    pub fn free(&mut self, slot: usize) {
        self.free.push(slot);
    }
}

/// The geometry a retired chunk slot leaves free, as byte offsets and lengths
/// into the shared vertex and index buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRegion {
    /// Byte offset of the freed vertex range.
    pub vertex_offset: u64,
    /// Length of the freed vertex range in bytes.
    pub vertex_bytes: u64,
    /// Byte offset of the freed index range.
    pub index_offset: u64,
    /// Length of the freed index range in bytes.
    pub index_bytes: u64,
}

/// Rewrite a resident chunk's model matrix.
///
/// Used by camera-relative rendering: when the camera crosses into a new chunk
/// the render origin follows it, so every resident chunk is rebased onto the
/// new origin. Only the model matrix changes, the geometry stays where it was
/// uploaded. The previous-frame model is left untouched so the velocity
/// pre-pass still diffs against the origin the chunk was last rendered with:
/// the rebase is exact, so a stationary chunk shows zero motion across an
/// origin shift.
pub fn set_chunk_model(
    objects: &mut [DrawObject],
    draw_idx: usize,
    model: [[f32; 4]; 4],
) -> Result<(), String> {
    let obj = objects
        .get_mut(draw_idx)
        .ok_or_else(|| format!("set_chunk_model: draw object {} out of range", draw_idx))?;
    obj.model = model;
    Ok(())
}

/// Mark a streamed chunk's slot invisible and non-resident, and report the
/// geometry region it frees. The slot stays in the draw array so every later
/// index keeps its meaning; every pass skips it. The caller returns the region
/// to its own vertex / index allocators, which is where the retire frame that
/// keeps an in-flight submission from reading a reused range belongs.
pub fn retire_chunk_slot(
    objects: &mut [DrawObject],
    draw_idx: usize,
) -> Result<ChunkRegion, String> {
    let obj = objects
        .get_mut(draw_idx)
        .ok_or_else(|| format!("remove_chunk_mesh: draw object {} out of range", draw_idx))?;
    let region = ChunkRegion {
        vertex_offset: obj.vertex_offset as u64,
        vertex_bytes: (obj.vertex_count * core::mem::size_of::<Vertex>()) as u64,
        index_offset: (obj.index_offset * core::mem::size_of::<u32>()) as u64,
        index_bytes: (obj.index_count * core::mem::size_of::<u32>()) as u64,
    };
    obj.visible = false;
    obj.resident = false;
    Ok(region)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_past_initial_len_then_reuses_freed_slots() {
        let mut alloc = DrawSlotAllocator::with_len(3);
        // No free slots yet: allocation appends past the initial three.
        assert_eq!(alloc.allocate(), SlotAlloc::Append(3));
        assert_eq!(alloc.allocate(), SlotAlloc::Append(4));

        // Freeing a slot makes the next allocation reuse it instead of growing.
        alloc.free(3);
        assert_eq!(alloc.allocate(), SlotAlloc::Reuse(3));

        // The reuse did not advance the high-water mark: with the free list
        // empty again, allocation resumes appending at 5 (not 6).
        assert_eq!(alloc.allocate(), SlotAlloc::Append(5));
    }

    #[test]
    fn freed_slots_pop_in_lifo_order() {
        let mut alloc = DrawSlotAllocator::with_len(10);
        alloc.free(4);
        alloc.free(7);
        assert_eq!(alloc.allocate(), SlotAlloc::Reuse(7));
        assert_eq!(alloc.allocate(), SlotAlloc::Reuse(4));
        assert_eq!(alloc.allocate(), SlotAlloc::Append(10));
    }

    #[test]
    fn set_chunk_model_rebases_the_slot() {
        let mut objects = alloc::vec![crate::test_support::draw_object()];
        let model = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [4.0, 5.0, 6.0, 1.0],
        ];
        assert!(set_chunk_model(&mut objects, 0, model).is_ok());
        assert_eq!(objects[0].model, model);
    }

    #[test]
    fn set_chunk_model_reports_an_out_of_range_slot() {
        let mut objects = alloc::vec![crate::test_support::draw_object()];
        let err = set_chunk_model(&mut objects, 3, [[0.0; 4]; 4]).unwrap_err();
        assert!(err.contains('3'), "{err}");
    }

    #[test]
    fn retiring_a_chunk_slot_reports_its_byte_region() {
        let mut objects = alloc::vec![crate::test_support::draw_object()];
        objects[0].vertex_offset = 128;
        objects[0].vertex_count = 4;
        objects[0].index_offset = 6;
        objects[0].index_count = 12;
        let region = retire_chunk_slot(&mut objects, 0).expect("slot 0 exists");
        assert_eq!(
            region,
            ChunkRegion {
                vertex_offset: 128,
                vertex_bytes: 4 * core::mem::size_of::<Vertex>() as u64,
                index_offset: 24,
                index_bytes: 48,
            }
        );
    }

    #[test]
    fn retiring_a_chunk_slot_hides_it_and_drops_residency() {
        let mut objects = alloc::vec![crate::test_support::draw_object()];
        objects[0].visible = true;
        objects[0].resident = true;
        assert!(retire_chunk_slot(&mut objects, 0).is_ok());
        assert!(!objects[0].visible);
        assert!(!objects[0].resident);
    }

    #[test]
    fn retiring_an_out_of_range_slot_reports_it() {
        let mut objects: alloc::vec::Vec<DrawObject> = alloc::vec::Vec::new();
        let err = retire_chunk_slot(&mut objects, 2).unwrap_err();
        assert!(err.contains('2'), "{err}");
    }
}
