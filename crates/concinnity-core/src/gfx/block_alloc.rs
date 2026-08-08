// src/gfx/block_alloc.rs
//
// A block pool: many small resources placed inside a few large blocks.
//
// A graphics API caps how many discrete allocations a device may hold at once
// (Vulkan's `maxMemoryAllocationCount` is commonly 4096), and every allocation
// carries driver-side page and bookkeeping overhead besides. A backend that
// allocates once per resource spends that budget on the resource count rather
// than the byte count, and runs out on a world the byte budget would have held
// comfortably. Placing resources inside a handful of blocks decouples the two.
//
// This is pure policy: `core` + `alloc` only, no backend types, no device
// handles, no I/O. It decides *which block and what offset*; the caller owns
// the blocks and performs the API-specific bind. `RangeAllocator` places
// resources within one block, so blocks inherit its best-fit placement,
// coalescing free list, and deferred frees keyed on a retire frame.
//
// The pool never allocates a block itself. `alloc` returns `None` when nothing
// fits, and the caller creates a block of `block_size_for` bytes, hands it back
// via `add_block`, and retries -- which is what keeps the device call out of
// this layer.

use super::range_alloc::RangeAllocator;

// Where a resource was placed: which block, and the byte offset within it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub block: usize,
    pub offset: u64,
}

// One block's placement state. The caller holds the matching device allocation
// and indexes it by the block's position in `blocks`.
struct BlockState {
    size: u64,
    ranges: RangeAllocator,
    in_use: u64,
    // A block sized to host one oversized resource rather than to be shared.
    // Never chosen for a later request, so an outsized resource cannot strand
    // its remainder as unusable slack in a general-purpose block.
    dedicated: bool,
}

impl BlockState {
    fn new(size: u64, dedicated: bool) -> Self {
        let mut ranges = RangeAllocator::new();
        // The whole block starts free: seed it as an already-retired free.
        ranges.free(0, size, 0);
        ranges.reclaim(0);
        Self {
            size,
            ranges,
            in_use: 0,
            dedicated,
        }
    }

    // No live resource and no free awaiting reclaim, so the caller may release
    // the backing device allocation.
    fn is_empty(&self) -> bool {
        self.in_use == 0 && self.ranges.free_bytes() == self.size
    }
}

// A pool of blocks that resources are placed inside. See the module comment.
pub struct BlockAllocator {
    // `None` marks a slot whose block was released; positions stay stable so a
    // live `Placement` keeps naming its own block.
    blocks: Vec<Option<BlockState>>,
    block_size: u64,
}

impl BlockAllocator {
    // A pool with no blocks yet, which will ask for `block_size`-byte blocks.
    pub fn new(block_size: u64) -> Self {
        Self {
            blocks: Vec::new(),
            block_size: block_size.max(1),
        }
    }

    // Place `size` bytes at an `align`-aligned offset in an existing shared
    // block, or `None` when none can host it. On `None` the caller creates a
    // block of `block_size_for` bytes, registers it with `add_block`, and
    // places into that block with `alloc_in`.
    //
    // Dedicated blocks are skipped: they are sized for one resource, and
    // letting a later small request settle in one would keep the whole block
    // alive long after the resource it was created for is gone.
    pub fn alloc(&mut self, size: u64, align: u64) -> Option<Placement> {
        for (block, state) in self.blocks.iter_mut().enumerate() {
            let Some(state) = state.as_mut() else {
                continue;
            };
            if state.dedicated {
                continue;
            }
            if let Some(offset) = state.ranges.alloc_aligned(size, align) {
                state.in_use += size;
                return Some(Placement { block, offset });
            }
        }
        None
    }

    // Place `size` bytes in `block` specifically. How a caller fills the block
    // it just added, which `alloc` will not choose when that block is
    // dedicated.
    pub fn alloc_in(&mut self, block: usize, size: u64, align: u64) -> Option<Placement> {
        let state = self.blocks.get_mut(block)?.as_mut()?;
        let offset = state.ranges.alloc_aligned(size, align)?;
        state.in_use += size;
        Some(Placement { block, offset })
    }

    // How large a block must be to host a `size`-byte request at `align`. The
    // standard block size normally, grown to fit a request too large for one.
    // The alignment slack is included so a block sized for a request can always
    // place it, whatever offset the block ends up starting at.
    pub fn block_size_for(&self, size: u64, align: u64) -> u64 {
        let needed = size.saturating_add(align.max(1).saturating_sub(1));
        needed.max(self.block_size)
    }

    // Register a block of `size` bytes the caller just created, returning the
    // index `Placement::block` will name. A block larger than the pool's
    // standard size is dedicated: it hosts the one request it was created for
    // and is never shared.
    pub fn add_block(&mut self, size: u64) -> usize {
        let dedicated = size > self.block_size;
        let state = BlockState::new(size, dedicated);
        // Reuse a released slot when one is free, so a pool that churns whole
        // blocks does not grow `blocks` without bound.
        if let Some(index) = self.blocks.iter().position(Option::is_none) {
            self.blocks[index] = Some(state);
            index
        } else {
            self.blocks.push(Some(state));
            self.blocks.len() - 1
        }
    }

    // Release `[offset, offset + size)` in `block`. The bytes stop counting as
    // in use immediately but are not placed again until `reclaim` runs for a
    // frame at or past `retire_frame`, so a resource a command buffer still
    // references is never overwritten.
    pub fn free(&mut self, placement: Placement, size: u64, retire_frame: u64) {
        let Some(Some(state)) = self.blocks.get_mut(placement.block) else {
            return;
        };
        state.ranges.free(placement.offset, size, retire_frame);
        state.in_use = state.in_use.saturating_sub(size);
    }

    // Make every free whose `retire_frame` has passed placeable again.
    pub fn reclaim(&mut self, current_frame: u64) {
        for state in self.blocks.iter_mut().flatten() {
            state.ranges.reclaim(current_frame);
        }
    }

    // Take every block that now holds nothing, clearing its slot and returning
    // the indices so the caller can release the backing device allocations. A
    // block with a free still awaiting reclaim is retained, since a command
    // buffer may still reference it.
    pub fn take_empty_blocks(&mut self) -> Vec<usize> {
        let mut released = Vec::new();
        for (index, slot) in self.blocks.iter_mut().enumerate() {
            if slot.as_ref().is_some_and(BlockState::is_empty) {
                *slot = None;
                released.push(index);
            }
        }
        released
    }

    // Bytes held in blocks, whether or not a resource occupies them. What the
    // device has actually committed.
    pub fn reserved_bytes(&self) -> u64 {
        self.blocks.iter().flatten().map(|b| b.size).sum()
    }

    // Bytes live resources occupy. The gap to `reserved_bytes` is the pool's
    // slack: alignment padding, fragmentation, and unfilled block tails.
    pub fn in_use_bytes(&self) -> u64 {
        self.blocks.iter().flatten().map(|b| b.in_use).sum()
    }

    // Live blocks, i.e. how many device allocations this pool is holding.
    pub fn block_count(&self) -> usize {
        self.blocks.iter().flatten().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: u64 = 1024;

    // Place `size` bytes, creating a block if the pool needs one. Mirrors the
    // caller's alloc-or-grow loop without any device work.
    fn place(pool: &mut BlockAllocator, size: u64, align: u64) -> Placement {
        if let Some(p) = pool.alloc(size, align) {
            return p;
        }
        let block = pool.add_block(pool.block_size_for(size, align));
        pool.alloc_in(block, size, align)
            .expect("fresh block must host it")
    }

    #[test]
    fn many_resources_share_one_block() {
        let mut pool = BlockAllocator::new(BLOCK);
        for _ in 0..16 {
            place(&mut pool, 64, 1);
        }
        // 16 x 64 fills the block exactly, so one device allocation backs all
        // sixteen resources.
        assert_eq!(pool.block_count(), 1);
        assert_eq!(pool.in_use_bytes(), 1024);
        assert_eq!(pool.reserved_bytes(), 1024);
    }

    #[test]
    fn a_second_block_opens_only_when_the_first_is_full() {
        let mut pool = BlockAllocator::new(BLOCK);
        for _ in 0..16 {
            place(&mut pool, 64, 1);
        }
        assert_eq!(pool.block_count(), 1);
        let overflow = place(&mut pool, 64, 1);
        assert_eq!(pool.block_count(), 2);
        assert_eq!(overflow.block, 1);
        assert_eq!(overflow.offset, 0);
    }

    #[test]
    fn an_oversized_request_gets_its_own_dedicated_block() {
        let mut pool = BlockAllocator::new(BLOCK);
        let big = place(&mut pool, 4096, 1);
        assert_eq!(pool.reserved_bytes(), 4096);
        // The dedicated block is never shared, so a later small request opens a
        // standard block instead of stranding the remainder of a huge one.
        let small = place(&mut pool, 64, 1);
        assert_ne!(small.block, big.block);
        assert_eq!(pool.reserved_bytes(), 4096 + BLOCK);
    }

    #[test]
    fn freed_space_is_reused_within_the_same_block() {
        let mut pool = BlockAllocator::new(BLOCK);
        let first = place(&mut pool, 512, 1);
        let second = place(&mut pool, 512, 1);
        assert_eq!(pool.block_count(), 1);
        pool.free(first, 512, 0);
        pool.reclaim(0);
        // the reused slot is the freed one, still in the original block
        let third = place(&mut pool, 512, 1);
        assert_eq!(third, first);
        assert_eq!(pool.block_count(), 1);
        assert_ne!(third, second);
    }

    #[test]
    fn a_free_is_withheld_until_its_retire_frame() {
        let mut pool = BlockAllocator::new(BLOCK);
        let only = place(&mut pool, 1024, 1);
        pool.free(only, 1024, 5);
        pool.reclaim(4);
        // still in flight at frame 4, so the pool opens a second block
        assert!(pool.alloc(1024, 1).is_none());
        pool.reclaim(5);
        assert_eq!(pool.alloc(1024, 1), Some(only));
    }

    #[test]
    fn in_use_drops_on_free_while_reserved_holds() {
        let mut pool = BlockAllocator::new(BLOCK);
        let p = place(&mut pool, 256, 1);
        assert_eq!(pool.in_use_bytes(), 256);
        assert_eq!(pool.reserved_bytes(), BLOCK);
        pool.free(p, 256, 0);
        // the resource is gone but the block is still committed to the device
        assert_eq!(pool.in_use_bytes(), 0);
        assert_eq!(pool.reserved_bytes(), BLOCK);
    }

    #[test]
    fn alignment_is_honoured_within_a_block() {
        let mut pool = BlockAllocator::new(BLOCK);
        place(&mut pool, 8, 1);
        let aligned = place(&mut pool, 64, 256);
        assert_eq!(aligned.offset % 256, 0);
        assert_eq!(pool.block_count(), 1);
    }

    #[test]
    fn a_block_sized_for_a_request_can_always_place_it() {
        // A 256-aligned request of exactly the block size needs headroom for
        // the alignment, or the fresh block could not host what it was sized
        // for. `block_size_for` accounts for that.
        let mut pool = BlockAllocator::new(BLOCK);
        place(&mut pool, 64, 1);
        let p = place(&mut pool, BLOCK, 256);
        assert_eq!(p.offset % 256, 0);
    }

    #[test]
    fn an_emptied_block_is_handed_back_for_release() {
        let mut pool = BlockAllocator::new(BLOCK);
        let a = place(&mut pool, 512, 1);
        let b = place(&mut pool, 512, 1);
        pool.free(a, 512, 0);
        // b is still live, so nothing is released yet
        assert!(pool.take_empty_blocks().is_empty());
        pool.free(b, 512, 0);
        // the frees have not retired yet, so the block is still held
        assert!(pool.take_empty_blocks().is_empty());
        pool.reclaim(0);
        assert_eq!(pool.take_empty_blocks(), vec![0]);
        assert_eq!(pool.block_count(), 0);
        assert_eq!(pool.reserved_bytes(), 0);
    }

    #[test]
    fn a_released_slot_is_reused_by_the_next_block() {
        let mut pool = BlockAllocator::new(BLOCK);
        let first = place(&mut pool, 1024, 1);
        pool.free(first, 1024, 0);
        pool.reclaim(0);
        assert_eq!(pool.take_empty_blocks(), vec![0]);
        // the vacated slot is refilled rather than the block list growing
        let next = place(&mut pool, 256, 1);
        assert_eq!(next.block, 0);
        assert_eq!(pool.block_count(), 1);
    }

    #[test]
    fn freeing_an_unknown_block_is_ignored() {
        let mut pool = BlockAllocator::new(BLOCK);
        let p = place(&mut pool, 256, 1);
        pool.free(p, 256, 0);
        pool.reclaim(0);
        pool.take_empty_blocks();
        // the block behind `p` is gone; freeing it again must not panic or
        // corrupt the pool's accounting
        pool.free(p, 256, 0);
        assert_eq!(pool.in_use_bytes(), 0);
        assert_eq!(pool.block_count(), 0);
    }

    #[test]
    fn resource_count_does_not_drive_block_count() {
        // The property the pool exists for: 4096 resources that fit in a few
        // blocks cost a few device allocations, not 4096.
        let mut pool = BlockAllocator::new(64 * 1024);
        for _ in 0..4096 {
            place(&mut pool, 256, 256);
        }
        assert_eq!(pool.in_use_bytes(), 4096 * 256);
        assert_eq!(pool.block_count(), 16);
    }
}
