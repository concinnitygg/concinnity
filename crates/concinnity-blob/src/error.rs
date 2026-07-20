// Why a blob image did not parse or encode.
//
// Each variant carries what a caller needs to report the failure. The crate
// does no logging of its own: it never knows which file the bytes came from, so
// the caller that opened the file owns the diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobError {
    // fewer bytes than the fixed header
    TooShort,
    // leading bytes are not BLOB_MAGIC
    BadMagic,
    // built against a different SCHEMA_HASH; carries the hash found
    SchemaMismatch(u32),
    // header promises more metadata than the image holds
    TruncatedMeta,
    // metadata block is not decodable postcard
    Decode,
    // metadata could not be serialized
    Encode,
}
