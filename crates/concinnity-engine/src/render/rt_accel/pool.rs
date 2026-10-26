//! Spare BLAS storage a topology refresh builds into instead of allocating.
//!
//! Streamed geometry joins and leaves the BVH in steady churn, and the pieces
//! that leave are usually the same size as the ones that join: every planet
//! tile is one grid. A BLAS a refresh orphans is kept here once no TLAS can
//! reach it, and handed back for the next build that fits, so the churn costs a
//! rebuild into existing storage rather than a device allocation and a free.

use alloc::vec::Vec;

/// Spares held at most. An orphan offered past it is handed back to be freed.
pub const BLAS_POOL_CAP: usize = 128;

/// Update ticks a spare is held before it is handed back to be freed, so a
/// churn that stops does not keep its storage.
pub const BLAS_POOL_MAX_AGE: u64 = 600;

struct Spare<B> {
    blas: B,
    bytes: u64,
    // The update tick from which no TLAS references it.
    released: u64,
}

/// Orphaned BLAS waiting to be rebuilt in place. `B` is the backend's BLAS
/// handle; the pool knows each one's size in bytes and when it stopped being
/// traced, and nothing about the API.
pub struct BlasPool<B> {
    spares: Vec<Spare<B>>,
}

impl<B> Default for BlasPool<B> {
    fn default() -> Self {
        Self { spares: Vec::new() }
    }
}

impl<B> BlasPool<B> {
    /// Offer `blas`, which holds `bytes` and which no TLAS live from update
    /// tick `released` on references. Hands it back when the pool is full, for
    /// the caller to free.
    pub fn give(&mut self, blas: B, bytes: u64, released: u64) -> Option<B> {
        if self.spares.len() >= BLAS_POOL_CAP {
            return Some(blas);
        }
        self.spares.push(Spare {
            blas,
            bytes,
            released,
        });
        None
    }

    /// A spare that holds `bytes` without wasting more than as much again, and
    /// that no frame in flight can still trace on tick `now` with `depth`
    /// frames in flight, with the bytes it holds. The tightest fit wins.
    pub fn take(&mut self, bytes: u64, now: u64, depth: u64) -> Option<(B, u64)> {
        let fits = |s: &Spare<B>| {
            s.bytes >= bytes
                && s.bytes <= bytes.saturating_mul(2).max(1)
                && s.released.saturating_add(depth) <= now
        };
        let best = self
            .spares
            .iter()
            .enumerate()
            .filter(|(_, s)| fits(s))
            .min_by_key(|(_, s)| s.bytes)
            .map(|(i, _)| i)?;
        let spare = self.spares.swap_remove(best);
        Some((spare.blas, spare.bytes))
    }

    /// One spare held past [`BLAS_POOL_MAX_AGE`] by tick `now`, for the caller
    /// to free, or `None` when none has.
    pub fn pop_expired(&mut self, now: u64) -> Option<B> {
        let i = self
            .spares
            .iter()
            .position(|s| s.released.saturating_add(BLAS_POOL_MAX_AGE) <= now)?;
        Some(self.spares.swap_remove(i).blas)
    }

    /// Every spare, for teardown once the device is idle.
    pub fn drain(&mut self) -> impl Iterator<Item = B> + '_ {
        self.spares.drain(..).map(|s| s.blas)
    }

    /// How many spares are held.
    pub fn len(&self) -> usize {
        self.spares.len()
    }

    /// Whether no spare is held.
    pub fn is_empty(&self) -> bool {
        self.spares.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spare_waits_out_the_frames_in_flight() {
        let mut pool = BlasPool::default();
        assert!(pool.give(7u32, 100, 10).is_none());
        assert_eq!(pool.take(100, 11, 2), None);
        assert_eq!(pool.take(100, 12, 2), Some((7, 100)));
        assert!(pool.is_empty());
    }

    #[test]
    fn a_spare_must_fit_without_wasting_more_than_it_holds() {
        let mut pool = BlasPool::default();
        pool.give(1u32, 100, 0);
        assert_eq!(pool.take(101, 10, 1), None, "too small");
        assert_eq!(pool.take(49, 10, 1), None, "too large");
        assert_eq!(pool.take(50, 10, 1), Some((1, 100)));
    }

    #[test]
    fn the_tightest_fit_is_taken() {
        let mut pool = BlasPool::default();
        pool.give(1u32, 180, 0);
        pool.give(2u32, 120, 0);
        pool.give(3u32, 150, 0);
        assert_eq!(pool.take(100, 10, 1), Some((2, 120)));
        assert_eq!(pool.take(100, 10, 1), Some((3, 150)));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn a_full_pool_hands_the_orphan_back() {
        let mut pool = BlasPool::default();
        for i in 0..BLAS_POOL_CAP {
            assert!(pool.give(i, 100, 0).is_none());
        }
        assert_eq!(pool.give(999, 100, 0), Some(999));
        assert_eq!(pool.len(), BLAS_POOL_CAP);
    }

    #[test]
    fn spares_expire_after_their_age() {
        let mut pool = BlasPool::default();
        pool.give(1u32, 100, 5);
        pool.give(2u32, 100, 50);
        assert_eq!(pool.pop_expired(5 + BLAS_POOL_MAX_AGE - 1), None);
        assert_eq!(pool.pop_expired(5 + BLAS_POOL_MAX_AGE), Some(1));
        assert_eq!(pool.pop_expired(5 + BLAS_POOL_MAX_AGE), None);
        assert_eq!(pool.drain().collect::<Vec<_>>(), [2]);
    }
}
