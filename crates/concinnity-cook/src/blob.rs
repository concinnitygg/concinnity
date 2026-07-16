// The blob WRITE side (build output): pack compiled payloads + the def table
// into .cnb files and emit world-lock.json. The file format itself -- the
// header, the record schema, and the single-file encoder `write_cnb` -- is
// owned by the concinnity-blob crate (its encode half sits behind the `write`
// feature only this crate enables); this file owns the packing POLICY (payload
// distribution across overflow blobs, the size ceiling) and the lock.

use std::fs;

use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use concinnity_blob::{HEADER_SIZE, write_cnb};
use concinnity_core::ecs::{BlobAssetDef, BlobMeta, PayloadLocator, ResourceRecord};

// Re-export the read side from core so `crate::blob::{BlobData, load_raw, ...}`
// resolves for build-crate consumers that read blobs back. `blob_path` is shared
// (used by write_blobs below and by readers).
pub use concinnity_core::blob::{
    BlobData, blob_path, load_defs, load_raw, payload_section_start, read_cnb,
};

use serde::{Deserialize, Serialize};

// The build record written beside the blobs. Provenance metadata, not part of
// the .cnb format, so it lives here rather than in concinnity-blob.
pub const LOCK_PATH: &str = "world-lock.json";

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
    // Assets compiled into the blob's resource stream (addressed by a per-kind
    // handle) rather than the component def table.
    #[serde(default)]
    pub resources: Vec<LockedResource>,
    // Assets the build added that have no world.jsonl line (companions and
    // engine defaults). Each entry carries its full args so it can be copied
    // into world.jsonl verbatim as an override.
    pub injected: Vec<LockedInjection>,
    // Generated assets the world declares its own copy of. The copy won and the
    // generated entry was dropped, so the source file no longer drives these;
    // recorded to make that override visible rather than silent.
    #[serde(default)]
    pub shadowed: Vec<LockedShadow>,
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

// One generated asset the world overrides with its own copy. Carries no args:
// the copy that won is the world.jsonl line of the same name.
#[derive(Debug, Serialize, Deserialize)]
pub struct LockedShadow {
    pub name: String,
    #[serde(rename = "type")]
    pub asset_type: String,
    pub generated_by: String,
}

// One resource-stream asset as recorded in the lock file. Resource assets have
// left the component def table, so they are recorded with their per-kind handle
// instead of a component discriminant. `payload_blob` is None for a data
// resource (bytes ride inline in the record).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockedResource {
    pub name: String,
    pub kind: String,
    pub handle: u32,
    // sha-256 of the asset's authored args JSON
    pub args_hash: String,
    // which blob holds this resource's payload, if any
    pub payload_blob: Option<u32>,
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
    resources: &[LockedResource],
    injected: &[crate::world::InjectedAsset],
    shadowed: &[crate::world::ShadowedAsset],
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
        resources: resources.to_vec(),
        injected: injected
            .iter()
            .map(|i| LockedInjection {
                name: i.name.clone(),
                asset_type: i.asset_type.clone(),
                args: i.args.clone(),
                injected_by: i.injected_by.to_string(),
            })
            .collect(),
        shadowed: shadowed
            .iter()
            .map(|s| LockedShadow {
                name: s.name.clone(),
                asset_type: s.asset_type.clone(),
                generated_by: s.generated_by.clone(),
            })
            .collect(),
    };

    fs::write(LOCK_PATH, serde_json::to_string_pretty(&lock)?)
}

// Helpers
pub(crate) fn checksum(data: &[u8]) -> String {
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

    // The write_cnb round-trip tests live in the concinnity-blob crate with the
    // encoder; here the tests cover cook's packing policy and the lock.
    // (write_blobs itself writes through the process-global data-dir anchor, so
    // it is exercised by `cn build`, not unit-tested.)

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
            resources: vec![LockedResource {
                name: "clip".to_string(),
                kind: "AudioClip".to_string(),
                handle: 0,
                args_hash: "00".to_string(),
                payload_blob: Some(0),
            }],
            injected: vec![LockedInjection {
                name: "debug_hud".to_string(),
                asset_type: "DebugHud".to_string(),
                args: serde_json::json!({}),
                injected_by: "engine".to_string(),
            }],
            shadowed: vec![LockedShadow {
                name: "bistro_mat_wood".to_string(),
                asset_type: "Material".to_string(),
                generated_by: "bistro".to_string(),
            }],
        };
        let json = serde_json::to_value(&lock).unwrap();
        assert_eq!(json["injected"][0]["type"], "DebugHud");
        assert!(json["injected"][0].get("asset_type").is_none());
        // LockedShadow carries the same rename, and names what it overrides.
        assert_eq!(json["shadowed"][0]["type"], "Material");
        assert_eq!(json["shadowed"][0]["generated_by"], "bistro");

        let back: BlobLock = serde_json::from_value(json).unwrap();
        assert_eq!(back.injected[0].asset_type, "DebugHud");
        assert_eq!(back.blobs[0].payload_bytes, 4);
        assert_eq!(back.resources[0].kind, "AudioClip");
        assert_eq!(back.shadowed[0].name, "bistro_mat_wood");
    }

    #[test]
    fn blob_lock_reads_a_lock_without_its_optional_fields() {
        // Locks written before resource and shadow provenance landed have no
        // `resources` / `shadowed` key; reading one must not fail.
        let json = serde_json::json!({
            "engine_version": "0.0.0",
            "built_at": "2026-01-01T00:00:00Z",
            "blobs": [],
            "assets": [],
            "injected": [],
        });
        let back: BlobLock = serde_json::from_value(json).unwrap();
        assert!(back.resources.is_empty());
        assert!(back.shadowed.is_empty());
    }
}
