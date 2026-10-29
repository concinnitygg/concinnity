//! One pool of a backend's device allocator: the `BlockAllocator` placement
//! state, the device block behind each of its slots, and the ladder the next
//! block's size climbs. The backend owns block creation and the bind; this owns
//! which block a request lands in and when a block is opened or released.

use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

use super::block_alloc::BlockAllocator;
pub use super::block_alloc::Placement;

/// How a pool's blocks grow as it fills.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockGrowth {
    /// Size of the first block.
    pub first: u64,
    /// Largest standard block. A request too large for one gets a block sized
    /// to itself, which the placement marks dedicated.
    pub max: u64,
    /// How many times the standard size doubles, one step per live block.
    pub doublings: u32,
    /// Every block size is a multiple of this; 1 for none.
    pub granularity: u64,
}

impl BlockGrowth {
    /// The ladder every backend's allocator climbs: a 4 MiB first block, so a
    /// small world commits megabytes rather than a full-size block per pool,
    /// doubling to 64 MiB, so a heavy one holds its persistent set in a
    /// handful of blocks. A backend whose API places resources on a coarser
    /// grid sets `granularity`.
    pub const STANDARD: Self = Self {
        first: 4 * 1024 * 1024,
        max: 64 * 1024 * 1024,
        doublings: 4,
        granularity: 1,
    };

    /// The size of the next block for a pool holding `live_blocks` blocks, to
    /// host `size` bytes at `align`. Keyed off the live count, so a pool that
    /// empties out restarts the ladder.
    pub fn block_bytes(&self, live_blocks: usize, size: u64, align: u64) -> u64 {
        let step = u32::saturating_from(live_blocks).min(self.doublings);
        let grown = self
            .first
            .saturating_mul(1u64.checked_shl(step).unwrap_or(u64::MAX))
            .min(self.max);
        let needed = size.saturating_add(align.max(1).saturating_sub(1));
        let granularity = self.granularity.max(1);
        needed
            .max(grown)
            .div_ceil(granularity)
            .saturating_mul(granularity)
    }
}

/// A reservation [`BlockPool::reserve`] made.
#[derive(Debug)]
pub struct Reserved<'a, B> {
    /// Where the request sits.
    pub placement: Placement,
    /// The block it sits in.
    pub block: &'a B,
    /// The block was opened for this request, so no resource used its range
    /// before.
    pub fresh: bool,
}

/// Why [`BlockPool::reserve`] failed.
#[derive(Debug, PartialEq, Eq)]
pub enum ReserveError<E> {
    /// Opening a block failed on the device.
    Open(E),
    /// The placement named a block slot whose block was released.
    Released,
    /// A block opened to the request's size could not host it.
    Unhosted,
}

/// The blocks of one pool, named by index the way `BlockAllocator` names
/// them. `B` is the backend's device block (a heap, a memory allocation).
pub struct BlockPool<B> {
    placement: BlockAllocator,
    blocks: Vec<Option<B>>,
    growth: BlockGrowth,
}

impl<B> BlockPool<B> {
    /// An empty pool growing by `growth`.
    pub fn new(growth: BlockGrowth) -> Self {
        Self {
            placement: BlockAllocator::new(growth.max),
            blocks: Vec::new(),
            growth,
        }
    }

    /// How large the next block would be to host `size` bytes at `align`.
    pub fn next_block_bytes(&self, size: u64, align: u64) -> u64 {
        self.growth
            .block_bytes(self.placement.block_count(), size, align)
    }

    /// Place `size` bytes at `align` in an existing block, or in a block
    /// `open` creates at the size it is given when none can host them.
    ///
    /// # Errors
    ///
    /// Returns an error if `open` fails, a block opened to the request's size
    /// cannot host it, or the placement names a released block.
    pub fn reserve<E>(
        &mut self,
        size: u64,
        align: u64,
        open: impl FnOnce(u64) -> Result<B, E>,
    ) -> Result<Reserved<'_, B>, ReserveError<E>> {
        if let Some(placement) = self.placement.alloc(size, align) {
            let block = self.block(placement.block).ok_or(ReserveError::Released)?;
            return Ok(Reserved {
                placement,
                block,
                fresh: false,
            });
        }
        let bytes = self.next_block_bytes(size, align);
        let block = open(bytes).map_err(ReserveError::Open)?;
        let index = self.add_block(bytes, block);
        let placement = self
            .placement
            .alloc_in(index, size, align)
            .ok_or(ReserveError::Unhosted)?;
        let block = self.block(index).ok_or(ReserveError::Released)?;
        Ok(Reserved {
            placement,
            block,
            fresh: true,
        })
    }

    /// Hand `block`, of `bytes` bytes, to the pool, returning its index.
    pub fn add_block(&mut self, bytes: u64, block: B) -> usize {
        let index = self.placement.add_block(bytes);
        if index == self.blocks.len() {
            self.blocks.push(Some(block));
        } else {
            self.blocks[index] = Some(block);
        }
        index
    }

    /// The block at `index`, unless it was released.
    pub fn block(&self, index: usize) -> Option<&B> {
        self.blocks.get(index)?.as_ref()
    }

    /// Place `size` bytes at `align` in an existing block.
    pub fn alloc(&mut self, size: u64, align: u64) -> Option<Placement> {
        self.placement.alloc(size, align)
    }

    /// Place `size` bytes at `align` in block `index` alone.
    pub fn alloc_in(&mut self, index: usize, size: u64, align: u64) -> Option<Placement> {
        self.placement.alloc_in(index, size, align)
    }

    /// Return a range, placeable again once frame `retire` is reclaimed.
    pub fn free(&mut self, placement: Placement, size: u64, retire: u64) {
        self.placement.free(placement, size, retire);
    }

    /// Make every free retired by `frame` placeable again.
    pub fn reclaim(&mut self, frame: u64) {
        self.placement.reclaim(frame);
    }

    /// Release every block that now holds nothing, handing each to `release`
    /// with its index.
    pub fn release_empty(&mut self, mut release: impl FnMut(usize, B)) {
        for index in self.placement.take_empty_blocks() {
            if let Some(block) = self.blocks.get_mut(index).and_then(Option::take) {
                release(index, block);
            }
        }
    }

    /// Take every block out of the pool, for a device teardown.
    pub fn drain(&mut self) -> impl Iterator<Item = B> + '_ {
        self.blocks.iter_mut().filter_map(Option::take)
    }

    /// Bytes the device has committed to this pool's blocks.
    pub fn reserved_bytes(&self) -> u64 {
        self.placement.reserved_bytes()
    }

    /// Bytes live resources occupy.
    pub fn in_use_bytes(&self) -> u64 {
        self.placement.in_use_bytes()
    }

    /// Live blocks.
    pub fn block_count(&self) -> usize {
        self.placement.block_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const MIB: u64 = 1024 * 1024;

    fn growth(first: u64, doublings: u32, granularity: u64) -> BlockGrowth {
        BlockGrowth {
            first,
            max: 64 * MIB,
            doublings,
            granularity,
        }
    }

    fn ladder(g: BlockGrowth, steps: usize) -> Vec<u64> {
        let mut pool = BlockPool::new(g);
        (0..steps)
            .map(|i| {
                let bytes = pool.next_block_bytes(1024, 256);
                pool.add_block(bytes, i);
                bytes
            })
            .collect()
    }

    #[test]
    fn the_standard_ladder_doubles_from_4_mib_to_the_cap() {
        assert_eq!(
            ladder(BlockGrowth::STANDARD, 6),
            vec![4 * MIB, 8 * MIB, 16 * MIB, 32 * MIB, 64 * MIB, 64 * MIB]
        );
        // A shorter ladder stops doubling at its own step count.
        assert_eq!(
            ladder(growth(8 * MIB, 2, 1), 4),
            vec![8 * MIB, 16 * MIB, 32 * MIB, 32 * MIB]
        );
    }

    #[test]
    fn an_oversized_request_sizes_its_own_block_with_alignment_slack() {
        let pool: BlockPool<()> = BlockPool::new(growth(4 * MIB, 4, 1));
        let huge = 64 * MIB * 2;
        assert_eq!(pool.next_block_bytes(huge, 1), huge);
        assert_eq!(pool.next_block_bytes(huge, 4096), huge + 4095);
    }

    #[test]
    fn block_sizes_stay_on_the_granularity() {
        let granule = 64 * 1024;
        let pool: BlockPool<()> = BlockPool::new(growth(4 * MIB, 4, granule));
        for size in [1u64, 100, granule + 1, 64 * MIB * 2 + 7] {
            assert_eq!(pool.next_block_bytes(size, 256) % granule, 0, "{size}");
        }
    }

    #[test]
    fn reserve_reuses_a_block_before_opening_one() {
        let mut pool = BlockPool::new(growth(MIB, 4, 1));
        let mut opened = Vec::new();
        let first = pool
            .reserve(1024, 16, |bytes| {
                opened.push(bytes);
                Ok::<_, ()>('a')
            })
            .unwrap();
        assert!(first.fresh);
        assert_eq!(*first.block, 'a');
        let second = pool.reserve(1024, 16, |_| Ok::<_, ()>('b')).unwrap();
        assert!(!second.fresh);
        assert_eq!(*second.block, 'a');
        assert_eq!(opened, vec![MIB]);
        assert_eq!(pool.block_count(), 1);
        assert_eq!(pool.in_use_bytes(), 2048);
    }

    #[test]
    fn a_failed_open_adds_no_block() {
        let mut pool: BlockPool<u8> = BlockPool::new(growth(MIB, 4, 1));
        assert_eq!(
            pool.reserve(10, 1, |_| Err("no memory")).unwrap_err(),
            ReserveError::Open("no memory")
        );
        assert_eq!(pool.block_count(), 0);
    }

    #[test]
    fn an_emptied_block_is_released_and_the_ladder_restarts() {
        let mut pool = BlockPool::new(growth(MIB, 4, 1));
        let p = pool.reserve(10, 1, |_| Ok::<_, ()>(7u8)).unwrap().placement;
        pool.free(p, 10, 3);
        let mut released = Vec::new();
        pool.release_empty(|i, b| released.push((i, b)));
        assert!(
            released.is_empty(),
            "the free still awaits its retire frame"
        );
        pool.reclaim(3);
        pool.release_empty(|i, b| released.push((i, b)));
        assert_eq!(released, vec![(0, 7)]);
        assert_eq!(pool.block(0), None);
        assert_eq!(pool.next_block_bytes(10, 1), MIB);
    }

    #[test]
    fn drain_takes_every_block() {
        let mut pool = BlockPool::new(growth(MIB, 4, 1));
        pool.add_block(MIB, 1);
        pool.add_block(MIB, 2);
        assert_eq!(pool.drain().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(pool.block(0), None);
    }
}
