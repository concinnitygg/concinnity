// What a .cnb container admits. The metadata type owns the magic, so reading
// one kind of container as another is a type error.

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
