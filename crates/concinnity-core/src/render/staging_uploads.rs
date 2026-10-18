//! The staging side of recorded geometry uploads: one mapped buffer under a
//! [`StagingRing`], the frame tick its batches retire against, and how the
//! buffer is replaced when a burst outgrows it. The backend records and submits
//! the copies; this decides where each upload's bytes are staged.

use crate::render::staging::{STAGING_ALIGN, StagingRing, grown_capacity, reserved_capacity};
use crate::render::upload_ring::UploadBuffer;

/// Where one upload's bytes were staged.
#[derive(Debug)]
pub struct Staged<'a, B> {
    /// The staging buffer the copy reads from.
    pub buffer: &'a B,
    /// The bytes' offset in it.
    pub src_offset: u64,
    /// The ring was replaced to fit this upload: its capacity before and after.
    /// A previous capacity of 0 is the ring's first creation; any other growth
    /// is logged.
    pub grew: Option<(u64, u64)>,
}

/// Why [`StagingUploads::stage`] could not stage an upload.
#[derive(Debug, PartialEq, Eq)]
pub enum StageError<E> {
    /// Creating a staging buffer failed on the device.
    Create(E),
    /// A ring grown to fit the upload still refused it.
    Refused,
}

/// A staging buffer and its ring, ticked once per frame.
pub struct StagingUploads<B> {
    staging: Option<(B, StagingRing)>,
    tick: u64,
    depth: u64,
}

impl<B: UploadBuffer> StagingUploads<B> {
    /// No staging buffer yet, retiring a submitted batch once
    /// `frames_in_flight + 1` ticks have passed.
    pub fn new(frames_in_flight: usize) -> Self {
        Self {
            staging: None,
            tick: 0,
            depth: frames_in_flight as u64 + 1,
        }
    }

    /// The current frame tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Ticks a submission stays in flight.
    pub fn depth(&self) -> u64 {
        self.depth
    }

    /// The staging buffer, once one exists.
    pub fn buffer(&self) -> Option<&B> {
        self.staging.as_ref().map(|(buffer, _)| buffer)
    }

    /// The ring's capacity, 0 before the first buffer.
    pub fn capacity(&self) -> u64 {
        self.staging.as_ref().map_or(0, |(_, ring)| ring.capacity())
    }

    /// Advance the tick and release the batches it retired.
    pub fn begin_frame(&mut self) {
        self.tick += 1;
        if let Some((_, ring)) = self.staging.as_mut() {
            ring.reclaim(self.tick);
        }
    }

    /// Make room ahead of need for a stream whose single frame may stage up to
    /// `expected` bytes, replacing the buffer with one `create` makes when the
    /// ring is smaller.
    pub fn reserve<E>(
        &mut self,
        expected: u64,
        create: impl FnOnce(u64) -> Result<B, E>,
    ) -> Result<(), E> {
        let capacity = reserved_capacity(expected);
        if self
            .staging
            .as_ref()
            .is_none_or(|(_, ring)| ring.capacity() < capacity)
        {
            self.staging = Some((create(capacity)?, StagingRing::new(capacity)));
        }
        Ok(())
    }

    /// Stage `bytes`, growing the ring through `create` when it has no room.
    /// `None` for an empty upload, which stages nothing.
    pub fn stage<E>(
        &mut self,
        bytes: &[u8],
        create: impl FnOnce(u64) -> Result<B, E>,
    ) -> Result<Option<Staged<'_, B>>, StageError<E>> {
        if bytes.is_empty() {
            return Ok(None);
        }
        let size = bytes.len() as u64;
        let mut grew = None;
        let mut offset = self.place(size);
        if offset.is_none() {
            let current = self.staging.as_ref().map_or(0, |(_, r)| r.capacity());
            let capacity = grown_capacity(current, size);
            let buffer = create(capacity).map_err(StageError::Create)?;
            self.staging = Some((buffer, StagingRing::new(capacity)));
            grew = Some((current, capacity));
            if current > 0 {
                tracing::info!(
                    from_kib = current / 1024,
                    to_kib = capacity / 1024,
                    write_kib = size / 1024,
                    "geometry staging ring grew"
                );
            }
            offset = self.place(size);
        }
        let src_offset = offset.ok_or(StageError::Refused)?;
        let (buffer, _) = self.staging.as_ref().ok_or(StageError::Refused)?;
        buffer.write(src_offset, bytes);
        Ok(Some(Staged {
            buffer,
            src_offset,
            grew,
        }))
    }

    /// Close the placements staged since the last submit into a batch, which
    /// the ring holds until the submission has retired.
    pub fn seal(&mut self) {
        let retire_at = self.tick + self.depth;
        if let Some((_, ring)) = self.staging.as_mut() {
            ring.seal(retire_at);
        }
    }

    /// Drop the staging buffer.
    pub fn clear(&mut self) {
        self.staging = None;
    }

    fn place(&mut self, size: u64) -> Option<u64> {
        self.staging
            .as_mut()
            .and_then(|(_, ring)| ring.alloc(size, STAGING_ALIGN))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::staging::STAGING_MIN_CAPACITY;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    #[derive(Debug)]
    struct Buffer {
        capacity: u64,
        writes: RefCell<Vec<u64>>,
    }

    impl UploadBuffer for Buffer {
        fn write(&self, offset: u64, bytes: &[u8]) {
            assert!(offset + bytes.len() as u64 <= self.capacity);
            self.writes.borrow_mut().push(offset);
        }
    }

    fn make(capacity: u64) -> Result<Buffer, ()> {
        Ok(Buffer {
            capacity,
            writes: RefCell::new(Vec::new()),
        })
    }

    #[test]
    fn an_empty_upload_stages_nothing() {
        let mut s = StagingUploads::new(2);
        assert!(s.stage(&[], make).unwrap().is_none());
        assert!(s.staging.is_none());
    }

    #[test]
    fn the_first_upload_creates_the_ring_and_later_ones_share_it() {
        let mut s = StagingUploads::new(2);
        let first = s.stage(&[1; 10], make).unwrap().unwrap();
        assert_eq!(first.grew, Some((0, STAGING_MIN_CAPACITY)));
        assert_eq!(first.src_offset, 0);
        let second = s.stage(&[1; 10], make).unwrap().unwrap();
        assert_eq!(second.grew, None);
        assert_eq!(second.src_offset, STAGING_ALIGN);
        assert_eq!(*second.buffer.writes.borrow(), [0, STAGING_ALIGN]);
    }

    #[test]
    fn a_burst_past_the_ring_grows_it() {
        let mut s = StagingUploads::new(1);
        s.reserve(0, make).unwrap();
        let big = alloc::vec![0u8; (STAGING_MIN_CAPACITY + 1) as usize];
        let staged = s.stage(&big, make).unwrap().unwrap();
        let (from, to) = staged.grew.expect("grew");
        assert_eq!(from, STAGING_MIN_CAPACITY);
        assert!(to > from && to.is_power_of_two());
    }

    #[test]
    fn a_sealed_batch_frees_its_bytes_after_the_depth() {
        let mut s = StagingUploads::new(1);
        s.reserve(0, make).unwrap();
        let half = alloc::vec![0u8; (STAGING_MIN_CAPACITY / 2) as usize];
        s.stage(&half, make).unwrap();
        s.stage(&half, make).unwrap();
        s.seal();
        // The ring is full until the batch retires two ticks later.
        assert_eq!(s.depth(), 2);
        s.begin_frame();
        assert_eq!(s.place(16), None);
        s.begin_frame();
        assert_eq!(s.tick(), 2);
        assert!(s.place(16).is_some());
    }

    #[test]
    fn a_failed_create_is_reported() {
        let mut s: StagingUploads<Buffer> = StagingUploads::new(1);
        assert_eq!(
            s.stage(&[0; 4], |_| Err("oom")).unwrap_err(),
            StageError::Create("oom")
        );
    }
}
