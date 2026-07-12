// The blob WRITE half (build output): pack compiled payloads + the def table
// into .cnb files and emit world-lock.json. The READ half (BlobData, load_raw,
// read_cnb, payload_section_start, load_defs) stays in concinnity-core and is
// re-exported here so callers can keep using `blob::...` uniformly.

use std::fs;

use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use concinnity_core::blob::{BLOB_MAGIC, BLOB_VERSION, HEADER_SIZE, LOCK_PATH};
use concinnity_core::ecs::{BlobAssetDef, BlobMeta, PayloadLocator, ResourceRecord};

// Re-export the read side from core so `crate::blob::{BlobData, load_raw, ...}`
// resolves for build-crate consumers that read blobs back. `blob_path` is shared
// (used by write_blobs below and by readers).
pub use concinnity_core::blob::{
    BlobData, blob_path, load_defs, load_raw, payload_section_start, read_cnb,
};

use serde::{Deserialize, Serialize};

// Per-blob entry in the lock file
#[derive(Debug, Serialize, Deserialize)]
pub struct BlobEntry {
    pub path: String,
    pub checksum: String,
    pub payload_bytes: u64,
}

// The resolved build record written alongside the binary blobs
// Human-readable; owned by the build, not the user
#[derive(Debug, Serialize, Deserialize)]
pub struct BlobLock {
    // Engine version the blob was built with; injected defaults come from the
    // engine, so their content can change across versions.
    pub engine_version: String,
    pub built_at: String,
    pub blobs: Vec<BlobEntry>,
    pub assets: Vec<LockedAsset>,
    // Assets the build added that have no world.jsonl line (companions and
    // engine defaults). Each entry carries its full args so it can be copied
    // into world.jsonl verbatim as an override.
    pub injected: Vec<LockedInjection>,
}

// One asset as recorded in the lock file
#[derive(Debug, Serialize, Deserialize)]
pub struct LockedAsset {
    pub name: String,
    pub kind: String,
    pub discriminant: u8,
    // sha-256 of the asset's serialized args_bytes
    pub args_hash: String,
    // which blob holds this asset's payload, if any
    pub payload_blob: Option<u32>,
}

// One injected asset as recorded in the lock file
#[derive(Debug, Serialize, Deserialize)]
pub struct LockedInjection {
    pub name: String,
    #[serde(rename = "type")]
    pub asset_type: String,
    pub args: serde_json::Value,
    pub injected_by: String,
}

// The result of a build pack: the blobs written and the path of each
pub struct PackResult {
    pub blob_paths: Vec<String>,
}

// Pack the metadata (component defs + resource records) and their payloads into
// one or more blobs. The full metadata rides in blob 0; overflow blobs carry an
// empty metadata section and pure payload bytes.
pub fn write_blobs(
    defs: &[BlobAssetDef],
    resources: &[ResourceRecord],
    blob_payloads: &[Vec<u8>],
) -> std::io::Result<PackResult> {
    fs::create_dir_all(concinnity_core::paths::data_dir())?;

    let primary_meta = || BlobMeta {
        defs: defs.to_vec(),
        resources: resources.to_vec(),
    };

    let mut blob_paths = Vec::new();

    for (idx, payload) in blob_payloads.iter().enumerate() {
        let path = blob_path(idx as u32);
        let meta = if idx == 0 {
            primary_meta()
        } else {
            BlobMeta::default()
        };
        write_cnb(&meta, payload, &path)?;
        blob_paths.push(path);
    }

    if blob_payloads.is_empty() {
        let primary = blob_path(0);
        write_cnb(&primary_meta(), &[], &primary)?;
        blob_paths.push(primary);
    }

    Ok(PackResult { blob_paths })
}

// Write a single blob file
fn write_cnb(meta: &BlobMeta, payload: &[u8], path: &str) -> std::io::Result<()> {
    let meta_bytes = postcard::to_stdvec(meta).map_err(|e| std::io::Error::other(e.to_string()))?;

    let meta_len = meta_bytes.len() as u64;

    let mut data = Vec::with_capacity(HEADER_SIZE + meta_bytes.len() + payload.len());
    data.extend_from_slice(&BLOB_MAGIC);
    data.extend_from_slice(&BLOB_VERSION.to_le_bytes());
    data.extend_from_slice(&meta_len.to_le_bytes());
    data.extend_from_slice(&meta_bytes);
    data.extend_from_slice(payload);

    fs::write(path, &data)
}

// PayloadPacker (build step)
pub struct PayloadPacker {
    max_blob_bytes: u64,
    blobs: Vec<Vec<u8>>,
    current_blob: u32,
    current_offset: u64,
}

impl PayloadPacker {
    pub fn new(max_blob_bytes: u64) -> Self {
        Self {
            max_blob_bytes,
            blobs: vec![Vec::new()],
            current_blob: 0,
            current_offset: 0,
        }
    }

    pub fn push(&mut self, data: &[u8]) -> PayloadLocator {
        let len = data.len() as u64;

        if self.current_offset > 0 && self.current_offset + len > self.max_blob_bytes {
            self.blobs.push(Vec::new());
            self.current_blob += 1;
            self.current_offset = 0;
        }

        let offset = self.current_offset;
        self.blobs[self.current_blob as usize].extend_from_slice(data);
        self.current_offset += len;

        PayloadLocator {
            blob_index: self.current_blob,
            offset,
            len,
        }
    }

    pub fn finish(self) -> Vec<Vec<u8>> {
        self.blobs
    }
}

// Lock file
pub fn write_lock(
    named_defs: &[(&str, &BlobAssetDef)],
    injected: &[crate::world::InjectedAsset],
    blob_paths: &[String],
) -> std::io::Result<()> {
    let mut blobs = Vec::new();
    for path in blob_paths {
        let data = fs::read(path).unwrap_or_default();
        let payload_bytes = if data.len() > HEADER_SIZE {
            let defs_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
            let payload_start = HEADER_SIZE + defs_len;
            (data.len().saturating_sub(payload_start)) as u64
        } else {
            0
        };
        blobs.push(BlobEntry {
            path: path.clone(),
            checksum: checksum(&data),
            payload_bytes,
        });
    }

    let assets = named_defs
        .iter()
        .map(|(name, def)| LockedAsset {
            name: name.to_string(),
            kind: format!("{:?}", def.kind),
            discriminant: def.discriminant,
            args_hash: checksum(&def.args_bytes),
            payload_blob: def.payload.as_ref().map(|p| p.blob_index),
        })
        .collect();

    let lock = BlobLock {
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        built_at: now_iso8601(),
        blobs,
        assets,
        injected: injected
            .iter()
            .map(|i| LockedInjection {
                name: i.name.clone(),
                asset_type: i.asset_type.clone(),
                args: i.args.clone(),
                injected_by: i.injected_by.to_string(),
            })
            .collect(),
    };

    fs::write(LOCK_PATH, serde_json::to_string_pretty(&lock)?)
}

// Helpers
fn checksum(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

fn now_iso8601() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .expect("time format")
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::{AssetKind, RecordKind};

    fn def(discriminant: u8, args_bytes: Vec<u8>) -> BlobAssetDef {
        BlobAssetDef {
            name: None,
            kind: AssetKind::Component,
            record: RecordKind::Authored,
            discriminant,
            args_bytes,
            payload: None,
        }
    }

    #[test]
    fn packer_appends_within_the_limit() {
        let mut p = PayloadPacker::new(100);
        let a = p.push(&[1, 2, 3]);
        let b = p.push(&[4, 5]);

        assert_eq!((a.blob_index, a.offset, a.len), (0, 0, 3));
        assert_eq!((b.blob_index, b.offset, b.len), (0, 3, 2));
        assert_eq!(p.finish(), vec![vec![1, 2, 3, 4, 5]]);
    }

    #[test]
    fn packer_rolls_to_a_new_blob_at_the_limit() {
        let mut p = PayloadPacker::new(4);
        let a = p.push(&[1, 2, 3]);
        // 3 + 2 exceeds the 4-byte cap, so this payload starts blob 1.
        let b = p.push(&[4, 5]);

        assert_eq!(a.blob_index, 0);
        assert_eq!((b.blob_index, b.offset, b.len), (1, 0, 2));
        assert_eq!(p.finish(), vec![vec![1, 2, 3], vec![4, 5]]);
    }

    #[test]
    fn packer_keeps_an_oversized_payload_in_an_empty_blob() {
        // A single payload larger than the cap cannot be split, so it stays
        // in the current (empty) blob rather than rolling forever.
        let mut p = PayloadPacker::new(4);
        let a = p.push(&[7; 10]);
        assert_eq!((a.blob_index, a.offset, a.len), (0, 0, 10));

        // The next payload rolls to a fresh blob.
        let b = p.push(&[1]);
        assert_eq!((b.blob_index, b.offset), (1, 0));
    }

    #[test]
    fn packer_zero_length_payload_gets_a_valid_locator() {
        let mut p = PayloadPacker::new(8);
        let a = p.push(&[]);
        let b = p.push(&[1]);
        assert_eq!((a.blob_index, a.offset, a.len), (0, 0, 0));
        assert_eq!((b.blob_index, b.offset, b.len), (0, 0, 1));
    }

    fn meta(defs: Vec<BlobAssetDef>, resources: Vec<ResourceRecord>) -> BlobMeta {
        BlobMeta { defs, resources }
    }

    #[test]
    fn write_cnb_round_trips_defs_resources_and_payload() {
        use concinnity_core::ecs::{PayloadLocator, ResourceKind};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("0.cnb");
        let path = path.to_str().unwrap();

        let resources = vec![ResourceRecord {
            resource_kind: ResourceKind::AudioClip as u8,
            handle: 0,
            payload: Some(PayloadLocator {
                blob_index: 0,
                offset: 0,
                len: 3,
            }),
            data_bytes: Vec::new(),
        }];
        let m = meta(vec![def(3, vec![1, 2]), def(9, vec![])], resources);
        let payload = [0xAA, 0xBB, 0xCC];
        write_cnb(&m, &payload, path).unwrap();

        let (read_meta, payload_start) = read_cnb(path).expect("read back");
        assert_eq!(read_meta.defs.len(), 2);
        assert_eq!(read_meta.defs[0].discriminant, 3);
        assert_eq!(read_meta.defs[0].args_bytes, vec![1, 2]);
        assert_eq!(read_meta.defs[1].discriminant, 9);
        assert!(read_meta.defs[1].args_bytes.is_empty());
        // The resource stream round-trips alongside the component defs.
        assert_eq!(read_meta.resources.len(), 1);
        assert_eq!(
            read_meta.resources[0].resource_kind,
            ResourceKind::AudioClip as u8
        );

        // The payload section starts right after the header + metadata table and
        // holds exactly the bytes we packed.
        let data = fs::read(path).unwrap();
        assert_eq!(&data[payload_start..], &payload);
        assert_eq!(
            payload_section_start(path).expect("header-only offset"),
            payload_start as u64
        );
    }

    #[test]
    fn write_cnb_with_no_metadata_and_no_payload_is_readable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.cnb");
        let path = path.to_str().unwrap();

        write_cnb(&BlobMeta::default(), &[], path).unwrap();

        let (m, payload_start) = read_cnb(path).expect("read back");
        assert!(m.defs.is_empty());
        assert!(m.resources.is_empty());
        assert_eq!(fs::read(path).unwrap().len(), payload_start);
    }

    #[test]
    fn write_cnb_emits_magic_and_version_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hdr.cnb");
        let path = path.to_str().unwrap();

        write_cnb(&BlobMeta::default(), &[1], path).unwrap();

        let data = fs::read(path).unwrap();
        assert_eq!(&data[0..4], &BLOB_MAGIC);
        assert_eq!(
            u32::from_le_bytes(data[4..8].try_into().unwrap()),
            BLOB_VERSION
        );
        let defs_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        assert_eq!(data.len(), HEADER_SIZE + defs_len + 1);
    }

    #[test]
    fn read_cnb_rejects_a_non_blob_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.cnb");
        fs::write(&path, b"this is not a blob file at all").unwrap();
        assert!(read_cnb(path.to_str().unwrap()).is_err());
        assert!(payload_section_start(path.to_str().unwrap()).is_err());
    }

    #[test]
    fn checksum_matches_known_sha256_vectors() {
        assert_eq!(
            checksum(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            checksum(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn now_iso8601_is_rfc3339_parseable() {
        let stamp = now_iso8601();
        assert!(OffsetDateTime::parse(&stamp, &Rfc3339).is_ok());
    }

    #[test]
    fn blob_lock_serializes_injected_type_field_as_type() {
        // The lock file is read by humans and tools; the serde rename on
        // LockedInjection keeps the JSON key `type`, matching world.jsonl.
        let lock = BlobLock {
            engine_version: "0.0.0".to_string(),
            built_at: "2026-01-01T00:00:00Z".to_string(),
            blobs: vec![BlobEntry {
                path: "data/0".to_string(),
                checksum: "00".to_string(),
                payload_bytes: 4,
            }],
            assets: vec![],
            injected: vec![LockedInjection {
                name: "debug_hud".to_string(),
                asset_type: "DebugHud".to_string(),
                args: serde_json::json!({}),
                injected_by: "engine".to_string(),
            }],
        };
        let json = serde_json::to_value(&lock).unwrap();
        assert_eq!(json["injected"][0]["type"], "DebugHud");
        assert!(json["injected"][0].get("asset_type").is_none());

        let back: BlobLock = serde_json::from_value(json).unwrap();
        assert_eq!(back.injected[0].asset_type, "DebugHud");
        assert_eq!(back.blobs[0].payload_bytes, 4);
    }
}
