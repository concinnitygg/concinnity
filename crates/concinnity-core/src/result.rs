//! The engine's flat result code, shared by every crate that reports a
//! recoverable failure across an API or FFI seam.

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
/// The engine's flat result code, returned across the API and FFI seams.
pub enum CnResult {
    #[error("Success")]
    /// The call succeeded.
    Success = 0,

    #[error("Invalid asset type")]
    /// The asset type name is not in the registry.
    AssetInvalidType,

    /// Generic
    #[error("Invalid state")]
    InvalidState,
    #[error("Invalid argument")]
    /// An argument was outside its accepted range.
    InvalidArgument,

    #[error("File I/O error")]
    /// A file could not be read or written.
    FileIo,

    #[error("No state directory installed")]
    /// Project state was read before any host anchored the state tree. See
    /// `concinnity_store::paths::set_state_dir`.
    NoStateRoot,
}

// Baked blob records are postcard; a decode failure means the record and the
// component schema disagree (a stale blob survives the version check instead
// of reaching here).
impl From<postcard::Error> for CnResult {
    fn from(e: postcard::Error) -> Self {
        tracing::error!("postcard deserialization error: {}", e);
        CnResult::InvalidArgument
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::{String, ToString};

    #[test]
    fn display_messages_are_stable() {
        assert_eq!(CnResult::Success.to_string(), "Success");
        assert_eq!(CnResult::AssetInvalidType.to_string(), "Invalid asset type");
        assert_eq!(CnResult::InvalidState.to_string(), "Invalid state");
        assert_eq!(CnResult::InvalidArgument.to_string(), "Invalid argument");
        assert_eq!(CnResult::FileIo.to_string(), "File I/O error");
        assert_eq!(
            CnResult::NoStateRoot.to_string(),
            "No state directory installed"
        );
    }

    #[test]
    fn postcard_errors_map_to_invalid_argument() {
        let e = postcard::from_bytes::<String>(&[0xff]).unwrap_err();
        assert_eq!(CnResult::from(e), CnResult::InvalidArgument);
    }
}
