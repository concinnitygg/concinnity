// Payload residency: which blob payload sections are in memory right now.
//
// This is runtime memory policy, not format, so it sits here rather than in the
// concinnity-blob format crate -- it deals in file paths and lazy disk reads,
// both of which that crate is deliberately free of.

use concinnity_asset::PayloadLocator;

use crate::result::CnResult;

// State of one blob file's payload section.
//
// `Unloaded` is the lazy state of an overflow blob: its file is on disk but has
// not been read yet. `Loaded` holds the resident bytes. `Released` means a
// system deliberately freed the payload after consuming it -- reads then error
// rather than reload, since the data is known to be no longer needed.
enum BlobSlot {
    // overflow blob not yet read; the String is its file path
    Unloaded(String),
    // payload section resident in memory
    Loaded(Vec<u8>),
    // payload deliberately released after use; reads error, no reload
    Released,
}

// Holds the raw payload sections of each blob file.
//
// Indexed by `PayloadLocator::blob_index`. Blob 0's payload section is loaded
// eagerly by `load_raw()` -- it carries the defs and the primary payloads and
// is needed immediately. Overflow blobs (1, 2, ...) start `Unloaded` and are
// read from disk on demand the first time a locator references them, so a
// large world does not pay the RAM (or I/O) cost of every overflow blob at
// startup.
//
// Systems call `release(blob_index)` after consuming a blob's payloads (e.g.
// after uploading SPIR-V to the GPU) so the memory is freed promptly.
pub struct BlobData {
    // slots[i] is the payload state of blob i
    slots: Vec<BlobSlot>,
    // True when the payloads came from blob files on disk (the `cn run` path).
    // False for in-memory builds (`cn debug`) and empty stores. The
    // asset-streaming subsystem reads this to decide whether a streamed
    // payload can be re-read from its blob file on demand instead of held
    // RAM-resident.
    disk_backed: bool,
}

impl BlobData {
    // Build an in-memory store where every section is already resident. Used
    // by the `cn debug` path, which compiles payloads in memory with no blob
    // files, so there is nothing to lazily load. A `None` section is treated
    // as already released.
    pub fn new(payload_sections: Vec<Option<Vec<u8>>>) -> Self {
        let slots = payload_sections
            .into_iter()
            .map(|s| match s {
                Some(bytes) => BlobSlot::Loaded(bytes),
                None => BlobSlot::Released,
            })
            .collect();
        Self {
            slots,
            disk_backed: false,
        }
    }

    // empty store for worlds with no compiled payloads (tests, runtime-only worlds)
    pub fn empty() -> Self {
        Self {
            slots: Vec::new(),
            disk_backed: false,
        }
    }

    // Build a disk-backed store: blob 0's section is already resident, each
    // overflow path stays `Unloaded` until a locator first reaches it.
    pub(super) fn from_blob_files(blob0_payload: Vec<u8>, overflow_paths: Vec<String>) -> Self {
        let mut slots = Vec::with_capacity(overflow_paths.len() + 1);
        slots.push(BlobSlot::Loaded(blob0_payload));
        slots.extend(overflow_paths.into_iter().map(BlobSlot::Unloaded));
        Self {
            slots,
            disk_backed: true,
        }
    }

    // true when the payloads were loaded from blob files on disk, so a
    // streamed payload can be re-read from disk rather than kept in RAM
    pub fn disk_backed(&self) -> bool {
        self.disk_backed
    }

    // read the bytes for a given locator
    //
    // An `Unloaded` overflow blob is read from its file on first access and
    // becomes `Loaded`. Errors if the locator is out of range, the blob was
    // released, or the on-demand load fails.
    pub fn read(&mut self, locator: &PayloadLocator) -> Result<&[u8], CnResult> {
        let idx = locator.blob_index as usize;
        let slot = self.slots.get_mut(idx).ok_or_else(|| {
            tracing::error!("BlobData: blob {} is out of range", locator.blob_index);
            CnResult::FileIo
        })?;
        if let BlobSlot::Unloaded(path) = slot {
            tracing::debug!(
                "BlobData: lazily loading overflow blob {}",
                locator.blob_index
            );
            let bytes = super::read_payload_section(&path.clone())?;
            *slot = BlobSlot::Loaded(bytes);
        }

        let section = match &self.slots[idx] {
            BlobSlot::Loaded(bytes) => bytes,
            BlobSlot::Released => {
                tracing::error!("BlobData: blob {} has been released", locator.blob_index);
                return Err(CnResult::FileIo);
            }
            // Unreachable: an Unloaded slot was loaded just above.
            BlobSlot::Unloaded(_) => return Err(CnResult::FileIo),
        };

        let start = locator.offset as usize;
        let end = start.checked_add(locator.len as usize).ok_or_else(|| {
            tracing::error!(
                "BlobData: payload slice offset {} + len {} overflows in blob {}",
                start,
                locator.len,
                locator.blob_index
            );
            CnResult::FileIo
        })?;
        section.get(start..end).ok_or_else(|| {
            tracing::error!(
                "BlobData: payload slice [{}, {}) out of bounds in blob {} (len={})",
                start,
                end,
                locator.blob_index,
                section.len()
            );
            CnResult::FileIo
        })
    }

    // release a blob's in-memory payload once all systems that need it have
    // finished consuming it (e.g. after GPU upload)
    //
    // subsequent `read()` calls for locators in this blob return an error
    // rather than reloading -- the data is known to no longer be needed -- so
    // only call this once you are sure no other system needs it
    pub fn release(&mut self, blob_index: u32) {
        if let Some(slot) = self.slots.get_mut(blob_index as usize)
            && !matches!(slot, BlobSlot::Released)
        {
            tracing::debug!("BlobData: releasing payload for blob {}", blob_index);
            *slot = BlobSlot::Released;
        }
    }

    // true if the blob's payload is resident in memory right now; an
    // `Unloaded` overflow blob reports false until its first read
    pub fn is_loaded(&self, blob_index: u32) -> bool {
        matches!(
            self.slots.get(blob_index as usize),
            Some(BlobSlot::Loaded(_))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_blob::{BlobMeta, encode_cnb};

    fn locator(blob_index: u32, offset: u64, len: u64) -> PayloadLocator {
        PayloadLocator {
            blob_index,
            offset,
            len,
        }
    }

    #[test]
    fn disk_backed_defaults_false() {
        assert!(!BlobData::empty().disk_backed());
        assert!(!BlobData::new(vec![Some(vec![1, 2, 3])]).disk_backed());
    }

    #[test]
    fn from_blob_files_is_disk_backed_with_blob0_resident() {
        let bd = BlobData::from_blob_files(b"primary".to_vec(), vec!["1".into(), "2".into()]);
        assert!(bd.disk_backed());
        assert!(bd.is_loaded(0));
        // overflow slots stay deferred until first read
        assert!(!bd.is_loaded(1));
        assert!(!bd.is_loaded(2));
    }

    #[test]
    fn read_lazily_loads_an_unloaded_overflow_blob() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("1").to_string_lossy().into_owned();
        let image = encode_cnb(&BlobMeta::default(), b"hello world").unwrap();
        std::fs::write(&path, image).expect("write blob");

        let mut bd = BlobData::from_blob_files(Vec::new(), vec![path]);
        assert!(!bd.is_loaded(1));

        assert_eq!(bd.read(&locator(1, 6, 5)).expect("read ok"), b"world");
        // the lazy load promoted the slot to resident
        assert!(bd.is_loaded(1));
    }

    #[test]
    fn read_errors_when_a_deferred_overflow_blob_is_missing() {
        let mut bd = BlobData::from_blob_files(Vec::new(), vec!["/nonexistent/cn/1".into()]);
        assert_eq!(bd.read(&locator(1, 0, 1)), Err(CnResult::FileIo));
    }

    #[test]
    fn read_errors_on_released_blob() {
        // a `None` section is treated as already released
        let mut bd = BlobData::new(vec![None]);
        assert!(bd.read(&locator(0, 0, 1)).is_err());
    }

    #[test]
    fn release_then_read_errors() {
        let mut bd = BlobData::new(vec![Some(b"abcd".to_vec())]);
        assert_eq!(bd.read(&locator(0, 0, 2)).expect("read ok"), b"ab");
        bd.release(0);
        assert!(!bd.is_loaded(0));
        assert!(bd.read(&locator(0, 0, 2)).is_err());
    }

    #[test]
    fn read_errors_on_out_of_range_blob() {
        let mut bd = BlobData::empty();
        assert!(bd.read(&locator(3, 0, 1)).is_err());
    }

    #[test]
    fn read_errors_when_the_locator_runs_past_the_section() {
        let mut bd = BlobData::new(vec![Some(b"abcd".to_vec())]);
        assert!(bd.read(&locator(0, 2, 99)).is_err());
        assert!(bd.read(&locator(0, u64::MAX, 1)).is_err());
    }
}
