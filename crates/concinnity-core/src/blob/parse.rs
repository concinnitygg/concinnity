// The blob decode half: turn blob bytes back into metadata. Reading those bytes
// off disk is the caller's job, so the payload residency store lives in
// concinnity-core and the state root's `data/` layout in `concinnity_host::store`.

use crate::blob::HEADER_SIZE;
use crate::blob::error::BlobError;
use crate::blob::frame::{FrameError, decode_exact};
use crate::blob::kind::BlobKind;

/// Parse a blob image's header and metadata block. Returns the metadata and the
/// offset at which the payload section begins.
///
/// `K` decides both the magic the header must carry and the type the metadata
/// block decodes into, so an image written for another kind is rejected as
/// [`BlobError::BadMagic`] rather than decoded into the wrong shape.
///
/// `validity` is what the header's token must equal for the image to be
/// readable by this build. What it means belongs to the kind: for
/// [`BlobMeta`](crate::blob::BlobMeta) it is a schema version, and runtime
/// callers pass `concinnity_core::SCHEMA_VERSION`.
pub fn parse_cnb<K: BlobKind>(validity: u32, data: &[u8]) -> Result<(K, usize), BlobError> {
    let meta_len = parse_header::<K>(data)? as usize;

    let stored = le_u32(data, 4).ok_or(BlobError::TooShort)?;
    if stored != validity {
        return Err(BlobError::ValidityMismatch(stored));
    }

    let meta_end = HEADER_SIZE
        .checked_add(meta_len)
        .ok_or(BlobError::TruncatedMeta)?;
    let meta_bytes = data
        .get(HEADER_SIZE..meta_end)
        .ok_or(BlobError::TruncatedMeta)?;
    let meta = decode_exact(meta_bytes).map_err(|e| match e {
        FrameError::Decode(_) => BlobError::Decode,
        FrameError::Trailing(n) => BlobError::TrailingMeta(n),
    })?;
    Ok((meta, meta_end))
}

/// Payload-section offset read from the header alone, so a caller holding only
/// the first HEADER_SIZE bytes can turn a `PayloadLocator` offset into an
/// absolute file offset without loading the image.
pub fn parse_payload_section_start<K: BlobKind>(header: &[u8]) -> Result<u64, BlobError> {
    Ok(HEADER_SIZE as u64 + parse_header::<K>(header)?)
}

/// The payload section of a full blob image.
///
/// Infallible and lenient: an image too short to hold a header, or one whose
/// header points past its end, yields an empty section. Overflow blobs carry no
/// metadata and reach here without a magic or validity check, so this is the one
/// container read that needs no kind.
pub fn payload_section(data: &[u8]) -> &[u8] {
    let Some(meta_len) = le_u64(data, 8) else {
        return &[];
    };
    let meta_len = meta_len as usize;
    HEADER_SIZE
        .checked_add(meta_len)
        .and_then(|start| data.get(start..))
        .unwrap_or(&[])
}

// Validate the kind's magic and return the declared metadata length. The
// validity token is checked only where the metadata is actually decoded.
fn parse_header<K: BlobKind>(data: &[u8]) -> Result<u64, BlobError> {
    if data.len() < HEADER_SIZE {
        return Err(BlobError::TooShort);
    }
    if data.get(..4) != Some(&K::MAGIC[..]) {
        return Err(BlobError::BadMagic);
    }
    le_u64(data, 8).ok_or(BlobError::TooShort)
}

// The little-endian u32 at `at`, or `None` if the buffer ends first.
fn le_u32(data: &[u8], at: usize) -> Option<u32> {
    data.get(at..)?
        .first_chunk::<4>()
        .copied()
        .map(u32::from_le_bytes)
}

// The little-endian u64 at `at`, or `None` if the buffer ends first.
fn le_u64(data: &[u8], at: usize) -> Option<u64> {
    data.get(at..)?
        .first_chunk::<8>()
        .copied()
        .map(u64::from_le_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::cache::CacheMeta;
    use crate::blob::encode::encode_cnb;
    use crate::blob::schema::{AssetKind, BlobAssetDef, BlobMeta, ResourceKind, ResourceRecord};
    use crate::blob::{BLOB_MAGIC, kind::BlobKind};
    use crate::ecs::PayloadLocator;
    use alloc::vec;
    use alloc::vec::Vec;

    // Any value both sides agree on exercises the header check; the real
    // one is `crate::SCHEMA_VERSION`.
    const TEST_SCHEMA_VERSION: u32 = 0x1234_5678;

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
        let manifest = crate::blob::WorldManifest::from_records(&defs, &resources);
        BlobMeta {
            defs,
            resources,
            manifest,
            scene_groups: Vec::new(),
            mesh_bounds: Vec::new(),
            physics_budget: None,
        }
    }

    #[test]
    fn encode_round_trips_defs_resources_and_payload() {
        let m = meta();
        let payload = [0xAA, 0xBB, 0xCC];
        let image = encode_cnb(TEST_SCHEMA_VERSION, &m, &payload).unwrap();

        let (got, payload_start) =
            parse_cnb::<BlobMeta>(TEST_SCHEMA_VERSION, &image).expect("parse");
        assert_eq!(got, m);
        assert_eq!(&image[payload_start..], &payload);
        assert_eq!(
            parse_payload_section_start::<BlobMeta>(&image).unwrap(),
            payload_start as u64
        );
        assert_eq!(payload_section(&image), &payload);
    }

    #[test]
    fn encode_with_no_metadata_and_no_payload_is_parseable() {
        let image = encode_cnb(TEST_SCHEMA_VERSION, &BlobMeta::default(), &[]).unwrap();
        let (m, payload_start) = parse_cnb::<BlobMeta>(TEST_SCHEMA_VERSION, &image).expect("parse");
        assert!(m.defs.is_empty());
        assert!(m.resources.is_empty());
        assert_eq!(image.len(), payload_start);
        assert_eq!(payload_section(&image), &[] as &[u8]);
    }

    #[test]
    fn encode_emits_magic_and_validity_token_header() {
        let image = encode_cnb(TEST_SCHEMA_VERSION, &BlobMeta::default(), &[1]).unwrap();
        assert_eq!(&image[0..4], &BLOB_MAGIC);
        assert_eq!(
            u32::from_le_bytes(image[4..8].try_into().unwrap()),
            TEST_SCHEMA_VERSION
        );
        let meta_len = u64::from_le_bytes(image[8..16].try_into().unwrap()) as usize;
        assert_eq!(image.len(), HEADER_SIZE + meta_len + 1);
    }

    // The reason the magic hangs off the kind: a container written for one kind
    // must not decode as another, even when both metadata types would accept
    // the bytes.
    #[test]
    fn a_container_of_another_kind_is_rejected() {
        let other = encode_cnb(TEST_SCHEMA_VERSION, &CacheMeta::default(), &[]).unwrap();
        assert_eq!(&other[0..4], &CacheMeta::MAGIC);
        assert_eq!(
            parse_cnb::<BlobMeta>(TEST_SCHEMA_VERSION, &other),
            Err(BlobError::BadMagic)
        );
        assert_eq!(
            parse_payload_section_start::<BlobMeta>(&other),
            Err(BlobError::BadMagic)
        );

        let world = encode_cnb(TEST_SCHEMA_VERSION, &meta(), &[]).unwrap();
        assert_eq!(
            parse_cnb::<CacheMeta>(TEST_SCHEMA_VERSION, &world),
            Err(BlobError::BadMagic)
        );
    }

    #[test]
    fn parse_rejects_short_bad_magic_and_validity_mismatch() {
        assert_eq!(
            parse_cnb::<BlobMeta>(TEST_SCHEMA_VERSION, &[0u8; HEADER_SIZE - 1]),
            Err(BlobError::TooShort)
        );
        assert_eq!(
            parse_cnb::<BlobMeta>(TEST_SCHEMA_VERSION, &[0u8; HEADER_SIZE]),
            Err(BlobError::BadMagic)
        );

        let mut mismatched = encode_cnb(TEST_SCHEMA_VERSION, &BlobMeta::default(), &[]).unwrap();
        let stored = TEST_SCHEMA_VERSION.wrapping_add(1);
        mismatched[4..8].copy_from_slice(&stored.to_le_bytes());
        assert_eq!(
            parse_cnb::<BlobMeta>(TEST_SCHEMA_VERSION, &mismatched),
            Err(BlobError::ValidityMismatch(stored))
        );
    }

    // A meta block written by a schema carrying more than this build reads back
    // decodes cleanly under plain postcard, leaving the tail unread. Widening
    // the block without changing its content is that shape.
    #[test]
    fn parse_rejects_a_meta_section_with_unread_bytes() {
        let image = encode_cnb(TEST_SCHEMA_VERSION, &meta(), &[]).unwrap();
        let meta_len = u64::from_le_bytes(image[8..16].try_into().unwrap());

        let mut widened = image.clone();
        widened[8..16].copy_from_slice(&(meta_len + 2).to_le_bytes());
        widened.extend_from_slice(&[0, 0]);

        assert_eq!(
            parse_cnb::<BlobMeta>(TEST_SCHEMA_VERSION, &widened),
            Err(BlobError::TrailingMeta(2))
        );
    }

    #[test]
    fn parse_rejects_a_truncated_meta_section() {
        let image = encode_cnb(TEST_SCHEMA_VERSION, &meta(), &[]).unwrap();
        assert_eq!(
            parse_cnb::<BlobMeta>(TEST_SCHEMA_VERSION, &image[..image.len() - 1]),
            Err(BlobError::TruncatedMeta)
        );
    }

    #[test]
    fn parse_rejects_a_non_blob_image() {
        let garbage = b"this is not a blob file at all";
        assert_eq!(
            parse_cnb::<BlobMeta>(TEST_SCHEMA_VERSION, garbage),
            Err(BlobError::BadMagic)
        );
        assert_eq!(
            parse_payload_section_start::<BlobMeta>(garbage),
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
        let mut image = encode_cnb(TEST_SCHEMA_VERSION, &BlobMeta::default(), b"overflow").unwrap();
        image[0..4].copy_from_slice(b"XXXX");
        assert_eq!(payload_section(&image), b"overflow");
    }
}
