use thiserror::Error;

/// Why a blob image did not parse or encode.
///
/// Each variant carries what a caller needs to report the failure. The crate
/// does no logging of its own: it never knows which file the bytes came from, so
/// the caller that opened the file owns the diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BlobError {
    /// fewer bytes than the fixed header
    #[error("fewer bytes than the {} a blob header takes", super::HEADER_SIZE)]
    TooShort,
    /// leading bytes are not the kind's magic
    #[error("the leading bytes are not this blob kind's magic")]
    BadMagic,
    /// the header's validity token is not the one the kind requires; carries
    /// the token found
    #[error(
        "validity token {0} is not the one this build writes, so the data was built by a different version of the engine"
    )]
    ValidityMismatch(u32),
    /// header promises more metadata than the image holds
    #[error("the header promises more metadata than the image holds")]
    TruncatedMeta,
    /// metadata block is not decodable postcard
    #[error("the metadata block did not deserialize")]
    Decode,
    /// metadata decoded without reading the whole block; carries the number of
    /// bytes left over
    #[error(
        "the metadata left {0} unread bytes, so the data was built by a different version of the engine"
    )]
    TrailingMeta(usize),
    /// metadata could not be serialized
    #[error("the metadata could not be serialized")]
    Encode,
}
