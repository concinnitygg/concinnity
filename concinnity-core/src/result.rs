// src/result.rs
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CnResult {
    #[error("Success")]
    #[allow(dead_code)]
    Success = 0,

    #[error("Invalid asset type")]
    AssetInvalidType,

    // Generic
    #[error("Invalid state")]
    InvalidState,
    #[error("Invalid argument")]
    InvalidArgument,

    #[error("File I/O error")]
    FileIo,
}

impl From<std::io::Error> for CnResult {
    fn from(_e: std::io::Error) -> Self {
        CnResult::FileIo
    }
}

impl From<serde_json::Error> for CnResult {
    fn from(e: serde_json::Error) -> Self {
        tracing::error!("JSON deserialization error: {}", e);
        CnResult::InvalidArgument // or better: CnResult::JsonError (see note below)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_stable() {
        assert_eq!(CnResult::Success.to_string(), "Success");
        assert_eq!(CnResult::AssetInvalidType.to_string(), "Invalid asset type");
        assert_eq!(CnResult::InvalidState.to_string(), "Invalid state");
        assert_eq!(CnResult::InvalidArgument.to_string(), "Invalid argument");
        assert_eq!(CnResult::FileIo.to_string(), "File I/O error");
    }

    #[test]
    fn io_errors_map_to_file_io() {
        let e = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        assert_eq!(CnResult::from(e), CnResult::FileIo);
    }

    #[test]
    fn json_errors_map_to_invalid_argument() {
        let e = serde_json::from_str::<u32>("not json").unwrap_err();
        assert_eq!(CnResult::from(e), CnResult::InvalidArgument);
    }
}
