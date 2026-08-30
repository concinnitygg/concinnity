/// Why a blob image did not parse or encode.
///
/// Each variant carries what a caller needs to report the failure. The crate
/// does no logging of its own: it never knows which file the bytes came from, so
/// the caller that opened the file owns the diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobError {
    /// fewer bytes than the fixed header
    TooShort,
    /// leading bytes are not the kind's magic
    BadMagic,
    /// the header's validity token is not the one the kind requires; carries
    /// the token found
    ValidityMismatch(u32),
    /// header promises more metadata than the image holds
    TruncatedMeta,
    /// metadata block is not decodable postcard
    Decode,
    /// metadata decoded without reading the whole block; carries the number of
    /// bytes left over
    TrailingMeta(usize),
    /// metadata could not be serialized
    Encode,
}
