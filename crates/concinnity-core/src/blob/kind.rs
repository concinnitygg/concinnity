// What a .cnb container admits. The metadata type is the container's whole
// contract -- header layout is shared, the meta type decides what the file can
// hold -- so the magic that identifies a file belongs to that type rather than
// to the encode and parse functions. Binding them here makes reading one kind
// of container as another a type error rather than a runtime check.

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::blob::schema::BlobMeta;

/// A `.cnb` container kind: a metadata type paired with the magic bytes that
/// identify a file carrying it.
///
/// Every kind shares the 16-byte header and the payload section after it; only
/// the metadata block and the meaning of the header's validity token differ:
/// [`BlobMeta`] is the cooked world, [`CacheMeta`](crate::blob::CacheMeta) a
/// segment of regenerable cache.
/// [`encode_cnb`](crate::blob::encode_cnb) and
/// [`parse_cnb`](crate::blob::parse_cnb) are generic over this trait, so a file
/// written for one kind cannot be parsed as another: the magic will not match.
pub trait BlobKind: Serialize + DeserializeOwned {
    /// The four bytes a file of this kind starts with.
    const MAGIC: [u8; 4];
}

impl BlobKind for BlobMeta {
    const MAGIC: [u8; 4] = crate::blob::BLOB_MAGIC;
}
