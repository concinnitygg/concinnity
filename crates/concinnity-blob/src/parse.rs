// The blob decode half: turn blob bytes back into metadata. Reading those bytes
// off disk is the caller's job, so the payload residency store and the
// `.concinnity/data/` layout both live in concinnity-core.

use crate::error::BlobError;
use crate::schema::BlobMeta;
use crate::{BLOB_MAGIC, BLOB_VERSION, HEADER_SIZE};

// Parse a blob image's header and metadata block. Returns the metadata and the
// offset at which the payload section begins.
pub fn parse_cnb(data: &[u8]) -> Result<(BlobMeta, usize), BlobError> {
    let meta_len = parse_header(data)? as usize;

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != BLOB_VERSION {
        return Err(BlobError::UnsupportedVersion(version));
    }

    let meta_end = HEADER_SIZE
        .checked_add(meta_len)
        .ok_or(BlobError::TruncatedMeta)?;
    let meta_bytes = data
        .get(HEADER_SIZE..meta_end)
        .ok_or(BlobError::TruncatedMeta)?;
    let meta = postcard::from_bytes(meta_bytes).map_err(|_| BlobError::Decode)?;
    Ok((meta, meta_end))
}

// Payload-section offset read from the header alone, so a caller holding only
// the first HEADER_SIZE bytes can turn a `PayloadLocator` offset into an
// absolute file offset without loading the image.
pub fn parse_payload_section_start(header: &[u8]) -> Result<u64, BlobError> {
    Ok(HEADER_SIZE as u64 + parse_header(header)?)
}

// The payload section of a full blob image.
//
// Infallible and lenient: an image too short to hold a header, or one whose
// header points past its end, yields an empty section. Overflow blobs carry no
// metadata and reach here without a magic or version check.
pub fn payload_section(data: &[u8]) -> &[u8] {
    if data.len() < HEADER_SIZE {
        return &[];
    }
    let meta_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    HEADER_SIZE
        .checked_add(meta_len)
        .and_then(|start| data.get(start..))
        .unwrap_or(&[])
}

// Validate magic and return the declared metadata length. The version is
// checked only where the metadata is actually decoded.
fn parse_header(data: &[u8]) -> Result<u64, BlobError> {
    if data.len() < HEADER_SIZE {
        return Err(BlobError::TooShort);
    }
    if data[0..4] != BLOB_MAGIC {
        return Err(BlobError::BadMagic);
    }
    Ok(u64::from_le_bytes(data[8..16].try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode_cnb;
    use crate::schema::{AssetKind, BlobAssetDef, ResourceKind, ResourceRecord};
    use alloc::vec;
    use alloc::vec::Vec;
    use concinnity_asset::PayloadLocator;

    fn def(discriminant: u8, args_bytes: Vec<u8>) -> BlobAssetDef {
        BlobAssetDef {
            name: None,
            kind: AssetKind::Component,
            discriminant,
            args_bytes,
            payload: None,
        }
    }

    fn meta() -> BlobMeta {
        let defs = vec![def(3, vec![1, 2]), def(9, vec![])];
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
        let manifest = crate::WorldManifest::from_records(&defs, &resources);
        BlobMeta {
            defs,
            resources,
            manifest,
        }
    }

    #[test]
    fn encode_round_trips_defs_resources_and_payload() {
        let m = meta();
        let payload = [0xAA, 0xBB, 0xCC];
        let image = encode_cnb(&m, &payload).unwrap();

        let (got, payload_start) = parse_cnb(&image).expect("parse");
        assert_eq!(got, m);
        assert_eq!(&image[payload_start..], &payload);
        assert_eq!(
            parse_payload_section_start(&image).unwrap(),
            payload_start as u64
        );
        assert_eq!(payload_section(&image), &payload);
    }

    #[test]
    fn encode_with_no_metadata_and_no_payload_is_parseable() {
        let image = encode_cnb(&BlobMeta::default(), &[]).unwrap();
        let (m, payload_start) = parse_cnb(&image).expect("parse");
        assert!(m.defs.is_empty());
        assert!(m.resources.is_empty());
        assert_eq!(image.len(), payload_start);
        assert_eq!(payload_section(&image), &[] as &[u8]);
    }

    #[test]
    fn encode_emits_magic_and_version_header() {
        let image = encode_cnb(&BlobMeta::default(), &[1]).unwrap();
        assert_eq!(&image[0..4], &BLOB_MAGIC);
        assert_eq!(
            u32::from_le_bytes(image[4..8].try_into().unwrap()),
            BLOB_VERSION
        );
        let meta_len = u64::from_le_bytes(image[8..16].try_into().unwrap()) as usize;
        assert_eq!(image.len(), HEADER_SIZE + meta_len + 1);
    }

    #[test]
    fn parse_rejects_short_bad_magic_and_wrong_version() {
        assert_eq!(parse_cnb(&[0u8; HEADER_SIZE - 1]), Err(BlobError::TooShort));
        assert_eq!(parse_cnb(&[0u8; HEADER_SIZE]), Err(BlobError::BadMagic));

        let mut wrong_version = encode_cnb(&BlobMeta::default(), &[]).unwrap();
        wrong_version[4..8].copy_from_slice(&(BLOB_VERSION + 1).to_le_bytes());
        assert_eq!(
            parse_cnb(&wrong_version),
            Err(BlobError::UnsupportedVersion(BLOB_VERSION + 1))
        );
    }

    #[test]
    fn parse_rejects_a_truncated_meta_section() {
        let image = encode_cnb(&meta(), &[]).unwrap();
        assert_eq!(
            parse_cnb(&image[..image.len() - 1]),
            Err(BlobError::TruncatedMeta)
        );
    }

    #[test]
    fn parse_rejects_a_non_blob_image() {
        let garbage = b"this is not a blob file at all";
        assert_eq!(parse_cnb(garbage), Err(BlobError::BadMagic));
        assert_eq!(
            parse_payload_section_start(garbage),
            Err(BlobError::BadMagic)
        );
    }

    #[test]
    fn payload_section_is_empty_for_a_headerless_or_overrun_image() {
        assert_eq!(payload_section(&[]), &[] as &[u8]);
        assert_eq!(payload_section(&[0u8; HEADER_SIZE - 1]), &[] as &[u8]);

        // header declaring more metadata than the image carries
        let mut overrun = [0u8; HEADER_SIZE];
        overrun[0..4].copy_from_slice(&BLOB_MAGIC);
        overrun[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(payload_section(&overrun), &[] as &[u8]);
    }

    // Overflow blobs carry no metadata and are read for payload only, so the
    // section read must not depend on a valid magic.
    #[test]
    fn payload_section_ignores_magic() {
        let mut image = encode_cnb(&BlobMeta::default(), b"overflow").unwrap();
        image[0..4].copy_from_slice(b"XXXX");
        assert_eq!(payload_section(&image), b"overflow");
    }
}
