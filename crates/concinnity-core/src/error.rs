//! The engine's flat error code, shared by every crate that reports a
//! recoverable failure across an API or FFI seam.

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
/// The engine's flat error code, carried as the `Err` half across the API and
/// FFI seams.
pub enum CnError {
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
    /// Project state was read by a caller that was handed no state tree. See
    /// `concinnity_host::store::paths::StateTree`.
    NoStateRoot,
}

// Baking a component into its blob record serializes it with postcard.
impl From<postcard::Error> for CnError {
    fn from(_: postcard::Error) -> Self {
        CnError::InvalidArgument
    }
}

// Reading one back reads a length-delimited frame; a failure means the record
// and the component schema disagree (a stale blob survives the version check
// instead of reaching here).
impl From<crate::blob::FrameError> for CnError {
    fn from(_: crate::blob::FrameError) -> Self {
        CnError::InvalidArgument
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::{String, ToString};

    #[test]
    fn display_messages_are_stable() {
        assert_eq!(CnError::AssetInvalidType.to_string(), "Invalid asset type");
        assert_eq!(CnError::InvalidState.to_string(), "Invalid state");
        assert_eq!(CnError::InvalidArgument.to_string(), "Invalid argument");
        assert_eq!(CnError::FileIo.to_string(), "File I/O error");
        assert_eq!(
            CnError::NoStateRoot.to_string(),
            "No state directory installed"
        );
    }

    #[test]
    fn frame_errors_map_to_invalid_argument() {
        let bad = crate::blob::decode_exact::<String>(&[0xff]).unwrap_err();
        assert_eq!(CnError::from(bad), CnError::InvalidArgument);

        let trailing = crate::blob::decode_exact::<u8>(&[1, 2]).unwrap_err();
        assert_eq!(CnError::from(trailing), CnError::InvalidArgument);
    }
}
