// The blob encode half, behind the `write` feature: only the compile pipeline
// (concinnity-cook) enables it, so a runtime build cannot write blobs. Packing
// policy (payload distribution across overflow blobs, the size ceiling) is
// cook's; this file owns only the byte format a single blob file carries.

use crate::HEADER_SIZE;
use crate::schema::BlobMeta;
use crate::{BLOB_MAGIC, BLOB_VERSION};
use std::fs;

// Write a single blob file: the 16-byte header, the postcard-serialized
// metadata block, then the raw payload section.
pub fn write_cnb(meta: &BlobMeta, payload: &[u8], path: &str) -> std::io::Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::{payload_section_start, read_cnb};
    use crate::schema::{AssetKind, BlobAssetDef, RecordKind, ResourceKind, ResourceRecord};
    use concinnity_asset::PayloadLocator;

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
    fn write_cnb_round_trips_defs_resources_and_payload() {
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
        let m = BlobMeta {
            defs: vec![def(3, vec![1, 2]), def(9, vec![])],
            resources,
        };
        let payload = [0xAA, 0xBB, 0xCC];
        write_cnb(&m, &payload, path).unwrap();

        let (read_meta, payload_start) = read_cnb(path).expect("read back");
        assert_eq!(read_meta, m);

        // The payload section starts right after the header + metadata block and
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
        let meta_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        assert_eq!(data.len(), HEADER_SIZE + meta_len + 1);
    }

    #[test]
    fn read_cnb_rejects_a_non_blob_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.cnb");
        fs::write(&path, b"this is not a blob file at all").unwrap();
        assert!(read_cnb(path.to_str().unwrap()).is_err());
        assert!(payload_section_start(path.to_str().unwrap()).is_err());
    }
}
