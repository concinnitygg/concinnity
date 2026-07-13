// The blob read half: load blob binaries back into memory. This is the only
// half a runtime build links; the encode half lives in write.rs behind the
// `write` feature.

use crate::schema::{BlobAssetDef, BlobMeta, ResourceRecord};
use crate::{BLOB_MAGIC, BLOB_VERSION, HEADER_SIZE};
use concinnity_asset::PayloadLocator;
use std::fs;

// Why a blob operation failed. Every failure site logs its detail via tracing
// before returning, so the value records only the class; callers map it onto
// their own result types (concinnity-core folds both onto CnResult::FileIo).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobError {
    // the file could not be read
    Io,
    // the bytes are not a valid blob (magic/version/truncation/decode)
    Format,
}

// State of one blob file's payload section.
//
// `Unloaded` is the lazy state of an overflow blob: its file is on disk but
// has not been read yet. `Loaded` holds the resident bytes. `Released` means
// a system deliberately freed the payload after consuming it -- reads then
// error rather than reload, since the data is known to be no longer needed.
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
    pub fn read(&mut self, locator: &PayloadLocator) -> Result<&[u8], BlobError> {
        let idx = locator.blob_index as usize;
        let slot = self.slots.get_mut(idx).ok_or_else(|| {
            tracing::error!("BlobData: blob {} is out of range", locator.blob_index);
            BlobError::Io
        })?;
        if let BlobSlot::Unloaded(path) = slot {
            tracing::debug!(
                "BlobData: lazily loading overflow blob {}",
                locator.blob_index
            );
            let bytes = read_payload_section(&path.clone())?;
            *slot = BlobSlot::Loaded(bytes);
        }

        let section = match &self.slots[idx] {
            BlobSlot::Loaded(bytes) => bytes,
            BlobSlot::Released => {
                tracing::error!("BlobData: blob {} has been released", locator.blob_index);
                return Err(BlobError::Io);
            }
            // Unreachable: an Unloaded slot was loaded just above.
            BlobSlot::Unloaded(_) => return Err(BlobError::Io),
        };

        let start = locator.offset as usize;
        let end = start.checked_add(locator.len as usize).ok_or_else(|| {
            tracing::error!(
                "BlobData: payload slice offset {} + len {} overflows in blob {}",
                start,
                locator.len,
                locator.blob_index
            );
            BlobError::Format
        })?;
        section.get(start..end).ok_or_else(|| {
            tracing::error!(
                "BlobData: payload slice [{}, {}) out of bounds in blob {} (len={})",
                start,
                end,
                locator.blob_index,
                section.len()
            );
            BlobError::Format
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

// Read and deserialize a blob's metadata section (component defs + resource
// records). Returns (meta, payload_start_offset).
pub fn read_cnb(path: &str) -> Result<(BlobMeta, usize), BlobError> {
    let data = fs::read(path).map_err(|e| {
        tracing::error!("Failed to read {}: {}", path, e);
        BlobError::Io
    })?;

    if data.len() < HEADER_SIZE {
        tracing::error!("{}: file too short ({} bytes)", path, data.len());
        return Err(BlobError::Format);
    }

    let magic = &data[0..4];
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let meta_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;

    if magic != BLOB_MAGIC {
        tracing::error!("Bad magic in {}: {:?}", path, magic);
        return Err(BlobError::Format);
    }
    if version != BLOB_VERSION {
        tracing::error!(
            "Version mismatch in {} (got {}, want {})",
            path,
            version,
            BLOB_VERSION
        );
        return Err(BlobError::Format);
    }

    let meta_end = HEADER_SIZE + meta_len;
    if data.len() < meta_end {
        tracing::error!("{}: truncated metadata section", path);
        return Err(BlobError::Format);
    }

    let meta = postcard::from_bytes(&data[HEADER_SIZE..meta_end]).map_err(|e| {
        tracing::error!("Failed to deserialize metadata from {}: {}", path, e);
        BlobError::Format
    })?;

    Ok((meta, meta_end))
}

// Byte offset within a blob file at which its payload section begins, i.e.
// just past the 16-byte header and the metadata section. Reads only the
// header, so it is cheap to call without loading the whole file -- the
// disk-backed streaming source uses it to turn a `PayloadLocator` offset
// (relative to the payload section) into an absolute file offset.
pub fn payload_section_start(path: &str) -> Result<u64, BlobError> {
    use std::io::Read;
    let mut file = fs::File::open(path).map_err(|e| {
        tracing::error!("Failed to open {}: {}", path, e);
        BlobError::Io
    })?;
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header).map_err(|e| {
        tracing::error!("Failed to read header of {}: {}", path, e);
        BlobError::Io
    })?;
    if header[0..4] != BLOB_MAGIC {
        tracing::error!("Bad magic in {}: {:?}", path, &header[0..4]);
        return Err(BlobError::Format);
    }
    let meta_len = u64::from_le_bytes(header[8..16].try_into().unwrap());
    Ok(HEADER_SIZE as u64 + meta_len)
}

// Read just the payload section of a .cnb file into memory
fn read_payload_section(path: &str) -> Result<Vec<u8>, BlobError> {
    let data = fs::read(path).map_err(|e| {
        tracing::error!("Failed to read {}: {}", path, e);
        BlobError::Io
    })?;
    if data.len() < HEADER_SIZE {
        return Ok(Vec::new());
    }
    let meta_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    let payload_start = HEADER_SIZE + meta_len;
    Ok(data.get(payload_start..).unwrap_or(&[]).to_vec())
}

// Load the primary blob's metadata and the `BlobData` payload store, without
// resolving defs into runtime components (that resolution depends on the
// runtime registry, so it lives in the engine's `blob::load` shim).
//
// `blob_path` maps a blob index to its file path -- the caller owns the disk
// layout. Only blob 0's payload section is read into memory here; overflow
// blobs (index >= 1) start `Unloaded` and `BlobData::read()` reads each from
// disk the first time a locator references it.
pub fn load_raw(
    blob_path: impl Fn(u32) -> String,
) -> Result<(Vec<BlobAssetDef>, Vec<ResourceRecord>, BlobData), BlobError> {
    let (meta, _payload_start) = read_cnb(&blob_path(0))?;

    // determine how many distinct blob indices are referenced so we know
    // which overflow files exist. Both the component defs and the resource
    // records address the payload section, so scan both streams.
    let max_blob_index = meta
        .defs
        .iter()
        .filter_map(|d| d.payload.as_ref())
        .chain(meta.resources.iter().filter_map(|r| r.payload.as_ref()))
        .map(|p| p.blob_index)
        .max()
        .unwrap_or(0);

    // Blob 0 is read eagerly -- it is needed immediately. Overflow blobs are
    // left `Unloaded`; `BlobData::read()` pulls each from disk on first use.
    let mut slots: Vec<BlobSlot> = Vec::with_capacity(max_blob_index as usize + 1);
    let blob0_payload = read_payload_section(&blob_path(0))?;
    tracing::debug!("Loaded blob 0 payload ({} bytes)", blob0_payload.len());
    slots.push(BlobSlot::Loaded(blob0_payload));
    for idx in 1..=max_blob_index {
        slots.push(BlobSlot::Unloaded(blob_path(idx)));
    }

    // these sections came from blob files on disk, so the streaming subsystem
    // may re-read a payload from disk instead of holding it RAM-resident
    let blob_data = BlobData {
        slots,
        disk_backed: true,
    };

    Ok((meta.defs, meta.resources, blob_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{AssetKind, RecordKind, ResourceKind};

    // A distinct temp path per test tag so parallel tests never collide.
    fn tmp_path(tag: &str) -> String {
        std::env::temp_dir()
            .join(format!("cn_blob_{}_{}.cnb", tag, std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    // A header-only .cnb byte image with an empty metadata section and the
    // given payload, built by hand so the corruption tests below can poke at
    // raw bytes independently of the write half.
    fn cnb_bytes(payload: &[u8]) -> Vec<u8> {
        let meta_len: u64 = 0;
        let mut data = Vec::with_capacity(HEADER_SIZE + payload.len());
        data.extend_from_slice(&BLOB_MAGIC);
        data.extend_from_slice(&BLOB_VERSION.to_le_bytes());
        data.extend_from_slice(&meta_len.to_le_bytes());
        data.extend_from_slice(payload);
        data
    }

    #[test]
    fn payload_section_start_skips_header_and_meta() {
        let path = tmp_path("section");
        fs::write(&path, cnb_bytes(b"payloadbytes")).expect("write blob");

        let start = payload_section_start(&path).expect("section start");
        // the payload must sit exactly at the reported offset
        let data = fs::read(&path).unwrap();
        assert_eq!(&data[start as usize..], b"payloadbytes");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn payload_section_start_rejects_bad_magic() {
        let path = tmp_path("badmagic");
        fs::write(&path, vec![0u8; HEADER_SIZE]).unwrap();
        assert_eq!(payload_section_start(&path), Err(BlobError::Format));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn blob_data_disk_backed_defaults_false() {
        assert!(!BlobData::empty().disk_backed());
        assert!(!BlobData::new(vec![Some(vec![1, 2, 3])]).disk_backed());
    }

    #[test]
    fn read_lazily_loads_an_unloaded_overflow_blob() {
        let path = tmp_path("lazy");
        fs::write(&path, cnb_bytes(b"hello world")).expect("write blob");

        let mut bd = BlobData {
            slots: vec![
                BlobSlot::Loaded(Vec::new()),     // blob 0
                BlobSlot::Unloaded(path.clone()), // blob 1, not yet read
            ],
            disk_backed: true,
        };
        assert!(!bd.is_loaded(1));

        let loc = PayloadLocator {
            blob_index: 1,
            offset: 6,
            len: 5,
        };
        assert_eq!(bd.read(&loc).expect("read ok"), b"world");
        // the lazy load promoted the slot to resident
        assert!(bd.is_loaded(1));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn read_errors_on_released_blob() {
        // a `None` section is treated as already released
        let mut bd = BlobData::new(vec![None]);
        let loc = PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 1,
        };
        assert!(bd.read(&loc).is_err());
    }

    #[test]
    fn release_then_read_errors() {
        let mut bd = BlobData::new(vec![Some(b"abcd".to_vec())]);
        let loc = PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 2,
        };
        assert_eq!(bd.read(&loc).expect("read ok"), b"ab");
        bd.release(0);
        assert!(!bd.is_loaded(0));
        assert!(bd.read(&loc).is_err());
    }

    #[test]
    fn read_errors_on_out_of_range_blob() {
        let mut bd = BlobData::empty();
        let loc = PayloadLocator {
            blob_index: 3,
            offset: 0,
            len: 1,
        };
        assert!(bd.read(&loc).is_err());
    }

    #[test]
    fn read_cnb_parses_a_valid_header_and_metadata() {
        let meta = BlobMeta {
            defs: vec![BlobAssetDef {
                name: Some(concinnity_asset::AssetId(1)),
                kind: AssetKind::Component,
                record: RecordKind::Authored,
                discriminant: 7,
                args_bytes: vec![1, 2, 3],
                payload: None,
            }],
            resources: vec![ResourceRecord {
                resource_kind: ResourceKind::AudioClip as u8,
                handle: 0,
                payload: Some(PayloadLocator {
                    blob_index: 0,
                    offset: 0,
                    len: 7,
                }),
                data_bytes: Vec::new(),
            }],
        };
        let path = tmp_path("read_ok");
        crate::write::write_cnb(&meta, b"payload", &path).unwrap();

        let (got, payload_start) = read_cnb(&path).expect("read");
        assert_eq!(got.defs.len(), 1);
        assert_eq!(got.defs[0].discriminant, 7);
        // The resource stream round-trips alongside the component defs.
        assert_eq!(got.resources.len(), 1);
        assert_eq!(
            got.resources[0].resource_kind,
            ResourceKind::AudioClip as u8
        );
        assert_eq!(got.resources[0].handle, 0);
        // The returned offset points exactly at the payload section.
        let raw = fs::read(&path).unwrap();
        assert_eq!(&raw[payload_start..], b"payload");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn read_cnb_rejects_short_bad_magic_and_wrong_version() {
        let short = tmp_path("short");
        fs::write(&short, vec![0u8; HEADER_SIZE - 1]).unwrap();
        assert_eq!(read_cnb(&short), Err(BlobError::Format));
        let _ = fs::remove_file(&short);

        // Correct length but zeroed magic.
        let magic = tmp_path("magic");
        fs::write(&magic, vec![0u8; HEADER_SIZE]).unwrap();
        assert_eq!(read_cnb(&magic), Err(BlobError::Format));
        let _ = fs::remove_file(&magic);

        // Valid magic, corrupted version.
        let mut bytes = cnb_bytes(b"");
        bytes[4] = 99;
        let ver = tmp_path("version");
        fs::write(&ver, &bytes).unwrap();
        assert_eq!(read_cnb(&ver), Err(BlobError::Format));
        let _ = fs::remove_file(&ver);
    }

    #[test]
    fn read_cnb_rejects_a_truncated_meta_section() {
        // The header claims more metadata bytes than the file actually holds.
        let mut data = Vec::new();
        data.extend_from_slice(&BLOB_MAGIC);
        data.extend_from_slice(&BLOB_VERSION.to_le_bytes());
        data.extend_from_slice(&100u64.to_le_bytes());
        data.extend_from_slice(b"short");
        let path = tmp_path("truncated");
        fs::write(&path, &data).unwrap();
        assert_eq!(read_cnb(&path), Err(BlobError::Format));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn read_cnb_errors_on_a_missing_file() {
        assert_eq!(
            read_cnb("/nonexistent/cn/blob/path.cnb"),
            Err(BlobError::Io)
        );
    }

    #[test]
    fn read_payload_section_returns_empty_for_a_short_file() {
        let path = tmp_path("payload_short");
        fs::write(&path, vec![0u8; HEADER_SIZE - 1]).unwrap();
        assert!(read_payload_section(&path).unwrap().is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_raw_reads_blob0_eagerly_and_defers_overflow() {
        let dir = tempfile::tempdir().unwrap();
        let path_for = |idx: u32| {
            dir.path()
                .join(idx.to_string())
                .to_string_lossy()
                .into_owned()
        };

        // Blob 0: one def whose payload lives in overflow blob 1.
        let meta = BlobMeta {
            defs: vec![BlobAssetDef {
                name: None,
                kind: AssetKind::Component,
                record: RecordKind::Baked,
                discriminant: 1,
                args_bytes: Vec::new(),
                payload: Some(PayloadLocator {
                    blob_index: 1,
                    offset: 0,
                    len: 8,
                }),
            }],
            resources: Vec::new(),
        };
        crate::write::write_cnb(&meta, b"primary", &path_for(0)).unwrap();
        crate::write::write_cnb(&BlobMeta::default(), b"overflow", &path_for(1)).unwrap();

        let (defs, resources, mut bd) = load_raw(path_for).expect("load");
        assert_eq!(defs.len(), 1);
        assert!(resources.is_empty());
        assert!(bd.disk_backed());
        // Blob 0 resident, blob 1 deferred until its first read.
        assert!(bd.is_loaded(0));
        assert!(!bd.is_loaded(1));
        let loc = defs[0].payload.clone().unwrap();
        assert_eq!(bd.read(&loc).expect("overflow read"), b"overflow");
        assert!(bd.is_loaded(1));
    }
}
