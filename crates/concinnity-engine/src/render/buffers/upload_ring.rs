//! A per-frame-in-flight ring of host-visible buffers that transient geometry
//! (text quads, line ribbons) is bump-allocated into. Each frame reserves its
//! slot for the bytes it will push, growing the slot's buffer when it is too
//! small, then pushes each block at the ring's alignment. A slot is only reused
//! once its frame's fence has retired, so a push never races a GPU read.

use core::cell::RefCell;

use alloc::vec::Vec;

use super::align_up;
use crate::render::buffers::growth::grow_capacity;

/// A backend's persistently mapped upload buffer.
pub trait UploadBuffer {
    /// Copy `bytes` into the buffer at `offset`. The ring only calls this with
    /// a range inside the capacity the buffer was created at.
    fn write(&self, offset: u64, bytes: &[u8]);
}

/// A push that did not fit the bytes its frame reserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingOverflow {
    /// End of the block that did not fit.
    pub need: u64,
    /// What the frame reserved.
    pub reserved: u64,
}

struct Slot<B> {
    buffer: Option<B>,
    capacity: u64,
    cursor: u64,
}

/// One upload buffer per frame in flight, each bump-allocated from its start.
pub struct UploadRing<B> {
    slots: Vec<RefCell<Slot<B>>>,
    align: u64,
    min_capacity: u64,
}

impl<B: UploadBuffer> UploadRing<B> {
    /// A ring of `frames` empty slots (at least one), pushing at `align` into
    /// buffers of at least `min_capacity` bytes.
    pub fn new(frames: usize, align: u64, min_capacity: u64) -> Self {
        Self {
            slots: (0..frames.max(1))
                .map(|_| {
                    RefCell::new(Slot {
                        buffer: None,
                        capacity: 0,
                        cursor: 0,
                    })
                })
                .collect(),
            align,
            min_capacity,
        }
    }

    /// Start frame `frame`'s slot over and make room for `needed` bytes,
    /// replacing its buffer with one `create` makes at the grown capacity when
    /// it is too small. The fence for `frame` has retired, so the old buffer
    /// can go.
    ///
    /// # Errors
    ///
    /// Returns the error from `create` when it fails.
    pub fn reserve<E>(
        &self,
        frame: usize,
        needed: u64,
        create: impl FnOnce(u64) -> Result<B, E>,
    ) -> Result<(), E> {
        let mut slot = self.slot(frame).borrow_mut();
        slot.cursor = 0;
        let Some(capacity) = grow_capacity(slot.capacity, needed, self.min_capacity) else {
            return Ok(());
        };
        slot.buffer = Some(create(capacity)?);
        slot.capacity = capacity;
        Ok(())
    }

    /// Copy `bytes` into frame `frame`'s slot at the next aligned offset, and
    /// hand `locate` the slot's buffer and that offset for the bind.
    ///
    /// # Errors
    ///
    /// Returns [`RingOverflow`] if `bytes` does not fit the room reserved for
    /// the frame.
    pub fn push<T>(
        &self,
        frame: usize,
        bytes: &[u8],
        locate: impl FnOnce(&B, u64) -> T,
    ) -> Result<T, RingOverflow> {
        let mut slot = self.slot(frame).borrow_mut();
        let offset = align_up(slot.cursor, self.align);
        let end = offset + bytes.len() as u64;
        let reserved = slot.capacity;
        let Some(buffer) = slot.buffer.as_ref().filter(|_| end <= reserved) else {
            return Err(RingOverflow {
                need: end,
                reserved,
            });
        };
        buffer.write(offset, bytes);
        let located = locate(buffer, offset);
        slot.cursor = end;
        Ok(located)
    }

    /// Drop every slot's buffer.
    pub fn clear(&self) {
        for slot in &self.slots {
            let mut slot = slot.borrow_mut();
            slot.buffer = None;
            slot.capacity = 0;
            slot.cursor = 0;
        }
    }

    fn slot(&self, frame: usize) -> &RefCell<Slot<B>> {
        &self.slots[frame % self.slots.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // A buffer that records every write, standing in for a mapped one.
    struct Recorded {
        capacity: u64,
        writes: RefCell<Vec<(u64, usize)>>,
    }

    impl UploadBuffer for Recorded {
        fn write(&self, offset: u64, bytes: &[u8]) {
            assert!(offset + bytes.len() as u64 <= self.capacity);
            self.writes.borrow_mut().push((offset, bytes.len()));
        }
    }

    #[expect(
        clippy::unnecessary_wraps,
        reason = "stands in for the GPU buffer allocator, which can fail"
    )]
    fn make(capacity: u64) -> Result<Recorded, ()> {
        Ok(Recorded {
            capacity,
            writes: RefCell::new(Vec::new()),
        })
    }

    #[test]
    fn pushes_land_at_aligned_offsets_within_the_reservation() {
        let ring = UploadRing::new(2, 256, 1024);
        ring.reserve(0, 600, make).unwrap();
        let a = ring.push(0, &[0; 100], |_, o| o).unwrap();
        let b = ring.push(0, &[0; 12], |_, o| o).unwrap();
        assert_eq!((a, b), (0, 256));
        let c = ring.push(0, &[0; 1], |b, o| (b.capacity, o)).unwrap();
        assert_eq!(c, (1024, 512));
    }

    #[test]
    fn a_push_past_the_reservation_overflows() {
        let ring = UploadRing::new(1, 16, 32);
        ring.reserve(0, 32, make).unwrap();
        assert_eq!(
            ring.push(0, &[0; 40], |_, o| o),
            Err(RingOverflow {
                need: 40,
                reserved: 32
            })
        );
        // An unreserved slot has nothing to push into.
        let cold: UploadRing<Recorded> = UploadRing::new(1, 16, 32);
        assert_eq!(
            cold.push(0, &[0; 1], |_, o| o),
            Err(RingOverflow {
                need: 1,
                reserved: 0
            })
        );
    }

    #[test]
    fn reserve_restarts_the_cursor_and_grows_only_when_short() {
        let ring = UploadRing::new(2, 16, 64);
        let mut created = vec![];
        ring.reserve(1, 10, |c| {
            created.push(c);
            make(c)
        })
        .unwrap();
        ring.push(1, &[0; 10], |_, o| o).unwrap();
        ring.reserve(1, 20, |c| {
            created.push(c);
            make(c)
        })
        .unwrap();
        assert_eq!(ring.push(1, &[0; 5], |_, o| o), Ok(0), "cursor restarted");
        ring.reserve(1, 1000, |c| {
            created.push(c);
            make(c)
        })
        .unwrap();
        assert_eq!(created.len(), 2, "{created:?}");
        assert!(created[1] >= 1000);
        // Frame indices wrap onto the ring.
        assert_eq!(ring.push(3, &[0; 5], |_, o| o), Ok(0));
    }

    #[test]
    fn clear_drops_every_buffer() {
        let ring = UploadRing::new(1, 16, 64);
        ring.reserve(0, 10, make).unwrap();
        ring.clear();
        assert!(ring.push(0, &[0; 1], |_, o| o).is_err());
    }
}
