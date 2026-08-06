// src/gfx/range_alloc.rs
//
// A byte-range sub-allocator: it hands out offsets into a larger span it does
// not own. Streamed meshes place their geometry in the renderer's shared vertex
// and index buffers with it, and `block_alloc` stacks it into a block pool to
// place resources inside device-memory blocks.
//
// Mesh streaming evicts and re-uploads geometry after init. Before this, every
// streamed mesh re-filled its fixed build-time region, so the buffers had to
// be sized for every streamed mesh at once. `RangeAllocator` lets a streamed
// mesh be placed at any free block of the right size, so an evicted mesh's
// space can be reused by a different one -- the prerequisite for streaming
// more mesh geometry than the buffers hold at once.
//
// This is pure policy: `core` + `alloc` only (just `Vec`), no backend types,
// no I/O, no threads. It lives alongside `gfx::streaming` for the same reason
// -- a future `no_std` client runtime can keep it unchanged.
//
// Free space is a sorted, coalesced free list; allocation is best-fit, so an
// exact-size block is consumed whole with no split. Frees are *deferred*: a
// region freed for frame N is not handed back out until `reclaim` runs for a
// frame at or past the caller-supplied `retire_frame`. The caller passes
// `retire_frame = frame + frames_in_flight` for a runtime eviction, so a
// region a still-in-flight command buffer references is never overwritten;
// at init it passes 0, since nothing has been drawn yet.

// A contiguous free byte range `[offset, offset + size)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Block {
    offset: u64,
    size: u64,
}

// Round `value` up to the next multiple of `align`, which must be non-zero.
fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

// A freed region awaiting reclaim once its `retire_frame` has passed.
#[derive(Clone, Copy, Debug)]
struct Pending {
    offset: u64,
    size: u64,
    retire_frame: u64,
}

// Byte-range sub-allocator -- see the module comment.
pub struct RangeAllocator {
    // free blocks, kept sorted by offset and coalesced
    free: Vec<Block>,
    // frees not yet safe to hand back out
    pending: Vec<Pending>,
}

impl RangeAllocator {
    // An empty allocator: no free space until regions are added.
    pub fn new() -> Self {
        Self {
            free: Vec::new(),
            pending: Vec::new(),
        }
    }

    // Allocate `size` bytes, returning the chosen offset, or `None` if no free
    // block is large enough.
    //
    // Best-fit: the smallest sufficient block is chosen, so a block matching
    // the request exactly is consumed whole with no fragmentation.
    pub fn alloc(&mut self, size: u64) -> Option<u64> {
        self.alloc_aligned(size, 1)
    }

    // Allocate `size` bytes at an offset that is a multiple of `align`, or
    // `None` if no free block can host the request.
    //
    // Best-fit on the bytes actually wasted, so alignment padding counts
    // against a candidate block rather than being invisible to the choice. Any
    // padding skipped ahead of the returned offset stays on the free list, so
    // `free` takes back exactly the `[offset, offset + size)` handed out here.
    pub fn alloc_aligned(&mut self, size: u64, align: u64) -> Option<u64> {
        if size == 0 {
            return Some(0);
        }
        let align = align.max(1);
        let mut best: Option<(usize, u64, u64)> = None;
        for (i, b) in self.free.iter().enumerate() {
            let offset = align_up(b.offset, align);
            let pad = offset - b.offset;
            let Some(usable) = b.size.checked_sub(pad) else {
                continue;
            };
            let Some(waste) = usable.checked_sub(size) else {
                continue;
            };
            if best.is_none_or(|(_, _, best_waste)| waste < best_waste) {
                best = Some((i, offset, waste));
            }
        }
        let (i, offset, _) = best?;
        let block = self.free[i];
        let pad = offset - block.offset;
        let tail_offset = offset + size;
        let tail_size = block.offset + block.size - tail_offset;
        // Rewrite the chosen block as its leading padding (if any) and re-insert
        // the trailing remainder after it, keeping `free` sorted by offset.
        match (pad, tail_size) {
            (0, 0) => {
                self.free.remove(i);
            }
            (0, _) => {
                self.free[i] = Block {
                    offset: tail_offset,
                    size: tail_size,
                };
            }
            (_, 0) => {
                self.free[i] = Block {
                    offset: block.offset,
                    size: pad,
                };
            }
            _ => {
                self.free[i] = Block {
                    offset: block.offset,
                    size: pad,
                };
                self.free.insert(
                    i + 1,
                    Block {
                        offset: tail_offset,
                        size: tail_size,
                    },
                );
            }
        }
        Some(offset)
    }

    // Queue `[offset, offset + size)` for release. It becomes allocatable once
    // `reclaim` runs for a frame at or past `retire_frame`.
    pub fn free(&mut self, offset: u64, size: u64, retire_frame: u64) {
        if size == 0 {
            return;
        }
        self.pending.push(Pending {
            offset,
            size,
            retire_frame,
        });
    }

    // Move every pending free whose `retire_frame <= current_frame` into the
    // free list. Call once before allocating in a frame.
    pub fn reclaim(&mut self, current_frame: u64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].retire_frame <= current_frame {
                let p = self.pending.swap_remove(i);
                self.insert_free(Block {
                    offset: p.offset,
                    size: p.size,
                });
            } else {
                i += 1;
            }
        }
    }

    // Total bytes available for allocation right now (pending frees excluded).
    pub fn free_bytes(&self) -> u64 {
        self.free.iter().map(|b| b.size).sum()
    }

    // Number of distinct free blocks -- a fragmentation gauge for diagnostics.
    pub fn free_block_count(&self) -> usize {
        self.free.len()
    }

    // Insert a block, keeping `free` sorted by offset and coalescing it with
    // either adjacent neighbour so the largest possible blocks stay available.
    fn insert_free(&mut self, block: Block) {
        let pos = self.free.partition_point(|b| b.offset < block.offset);
        debug_assert!(
            pos == self.free.len() || block.offset + block.size <= self.free[pos].offset,
            "RangeAllocator: freed block overlaps an existing free block"
        );
        debug_assert!(
            pos == 0 || self.free[pos - 1].offset + self.free[pos - 1].size <= block.offset,
            "RangeAllocator: freed block overlaps an existing free block"
        );
        self.free.insert(pos, block);
        // merge forward: absorb every following block this one now touches
        while pos + 1 < self.free.len()
            && self.free[pos].offset + self.free[pos].size == self.free[pos + 1].offset
        {
            let next_size = self.free[pos + 1].size;
            self.free[pos].size += next_size;
            self.free.remove(pos + 1);
        }
        // merge backward: fold this block into the previous if they touch
        if pos > 0 && self.free[pos - 1].offset + self.free[pos - 1].size == self.free[pos].offset {
            let this_size = self.free[pos].size;
            self.free[pos - 1].size += this_size;
            self.free.remove(pos);
        }
    }
}

impl Default for RangeAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Seed the allocator with immediately-available free space, the way init
    // mesh-eviction does (a deferred free with retire_frame 0, then reclaim).
    fn seed(a: &mut RangeAllocator, offset: u64, size: u64) {
        a.free(offset, size, 0);
        a.reclaim(0);
    }

    #[test]
    fn alloc_from_a_single_region_shaves_the_front() {
        let mut a = RangeAllocator::new();
        seed(&mut a, 0, 300);
        assert_eq!(a.alloc(100), Some(0));
        assert_eq!(a.alloc(100), Some(100));
        assert_eq!(a.alloc(100), Some(200));
        assert_eq!(a.alloc(1), None);
        assert_eq!(a.free_bytes(), 0);
    }

    #[test]
    fn alloc_returns_none_when_no_block_is_large_enough() {
        let mut a = RangeAllocator::new();
        seed(&mut a, 0, 40);
        seed(&mut a, 100, 40);
        // 80 bytes free, but the largest contiguous block is only 40
        assert_eq!(a.alloc(50), None);
        assert_eq!(a.alloc(40), Some(0));
    }

    #[test]
    fn best_fit_consumes_an_exact_size_block_whole() {
        let mut a = RangeAllocator::new();
        seed(&mut a, 0, 200);
        seed(&mut a, 500, 100);
        seed(&mut a, 900, 50);
        // a request of 100 takes the exact-size block, not a slice of the 200
        assert_eq!(a.alloc(100), Some(500));
        assert_eq!(a.free_block_count(), 2);
        assert_eq!(a.free_bytes(), 250);
    }

    #[test]
    fn adjacent_regions_coalesce_into_one_block() {
        let mut a = RangeAllocator::new();
        seed(&mut a, 100, 50);
        seed(&mut a, 0, 100); // touches the [100,150) block from the left
        seed(&mut a, 150, 50); // touches it from the right
        assert_eq!(a.free_block_count(), 1);
        // the merged [0,200) block satisfies a request larger than any piece
        assert_eq!(a.alloc(200), Some(0));
    }

    #[test]
    fn freed_region_is_withheld_until_its_retire_frame() {
        let mut a = RangeAllocator::new();
        seed(&mut a, 0, 100);
        assert_eq!(a.alloc(100), Some(0));
        // freed for reuse only from frame 5 onward
        a.free(0, 100, 5);
        a.reclaim(4);
        assert_eq!(a.alloc(100), None); // still withheld at frame 4
        a.reclaim(5);
        assert_eq!(a.alloc(100), Some(0)); // available at frame 5
    }

    #[test]
    fn reclaimed_region_coalesces_with_neighbours() {
        let mut a = RangeAllocator::new();
        seed(&mut a, 0, 300);
        let first = a.alloc(100).unwrap();
        let second = a.alloc(100).unwrap();
        assert_eq!((first, second), (0, 100));
        // free both; once reclaimed they merge back with the [200,300) tail
        a.free(first, 100, 1);
        a.free(second, 100, 1);
        a.reclaim(1);
        assert_eq!(a.free_block_count(), 1);
        assert_eq!(a.alloc(300), Some(0));
    }

    #[test]
    fn aligned_alloc_rounds_the_offset_up() {
        let mut a = RangeAllocator::new();
        seed(&mut a, 4, 500);
        // the block starts at 4, so a 256-aligned request skips to 256
        assert_eq!(a.alloc_aligned(100, 256), Some(256));
    }

    #[test]
    fn alignment_padding_stays_allocatable() {
        let mut a = RangeAllocator::new();
        seed(&mut a, 0, 1024);
        assert_eq!(a.alloc_aligned(64, 1), Some(0));
        // next 256-aligned offset is 256, leaving [64,256) skipped over
        assert_eq!(a.alloc_aligned(64, 256), Some(256));
        // that skipped padding is still free, not leaked
        assert_eq!(a.alloc_aligned(192, 1), Some(64));
        assert_eq!(a.free_bytes(), 1024 - 64 - 64 - 192);
    }

    #[test]
    fn freeing_an_aligned_alloc_returns_exactly_what_was_handed_out() {
        let mut a = RangeAllocator::new();
        seed(&mut a, 0, 1024);
        let off = a.alloc_aligned(100, 256).unwrap();
        assert_eq!(off, 0);
        let second = a.alloc_aligned(100, 256).unwrap();
        assert_eq!(second, 256);
        // free takes back the handed-out range, and once both are back the
        // whole span coalesces to one block again
        a.free(off, 100, 0);
        a.free(second, 100, 0);
        a.reclaim(0);
        assert_eq!(a.free_block_count(), 1);
        assert_eq!(a.free_bytes(), 1024);
    }

    #[test]
    fn best_fit_counts_alignment_padding_as_waste() {
        let mut a = RangeAllocator::new();
        // [0,140) would need 116 bytes of padding to host a 256-aligned 24;
        // [512,32) hosts it with none, so it is the better fit despite being
        // the smaller block.
        seed(&mut a, 0, 140);
        seed(&mut a, 512, 32);
        assert_eq!(a.alloc_aligned(24, 256), Some(512));
    }

    #[test]
    fn aligned_alloc_fails_when_padding_pushes_it_past_the_block() {
        let mut a = RangeAllocator::new();
        // 200 bytes free from offset 100, but a 256-aligned 100 needs [256,356)
        seed(&mut a, 100, 200);
        assert_eq!(a.alloc_aligned(100, 256), None);
        // the same request without alignment fits
        assert_eq!(a.alloc_aligned(100, 1), Some(100));
    }

    #[test]
    fn evict_and_reuse_places_a_different_mesh_in_freed_space() {
        // mesh A occupies [0,128), mesh B [128,64); evict A, stream C(128) in
        let mut a = RangeAllocator::new();
        seed(&mut a, 0, 128);
        seed(&mut a, 128, 64);
        let a_off = a.alloc(128).unwrap();
        let b_off = a.alloc(64).unwrap();
        assert_eq!((a_off, b_off), (0, 128));
        // evict A at frame 10 with 2 frames in flight
        a.free(a_off, 128, 12);
        a.reclaim(12);
        // C reuses A's freed slot
        assert_eq!(a.alloc(128), Some(0));
    }
}
