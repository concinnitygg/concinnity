//! The typed error vocabulary of the `RenderBackend` boundary. Backends map
//! their native failure codes (VkResult, HRESULT, MTLCommandBuffer status) into
//! these classes at the detection sites; the frame loop dispatches recovery
//! policy on the class, never on prose. `Other` carries legacy string errors so
//! interior call sites can migrate incrementally.

use thiserror::Error;

/// Why the GPU device stopped servicing work, as reported by the backend API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DeviceLostReason {
    /// The physical device left (unplugged eGPU, driver upgrade teardown).
    #[error("device removed")]
    Removed,
    /// The device reset underneath the app (TDR without a hang verdict).
    #[error("device reset")]
    Reset,
    /// The OS killed the device after deciding our workload hung it.
    #[error("device hung")]
    Hung,
    /// The presentation surface died; the device may be healthy, but the
    /// backend cannot present without recreating the surface.
    #[error("surface lost")]
    SurfaceLost,
    /// The backend reported loss without a usable reason code.
    #[error("unknown")]
    Unknown,
}

/// A failure crossing the `RenderBackend` boundary, classified for recovery.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RenderError {
    /// The device is gone; no further GPU work can succeed on it. `detail` is
    /// backend prose for the log (e.g. the `GetDeviceRemovedReason` message).
    #[error("device lost ({reason}): {detail}")]
    DeviceLost {
        /// Why the device was lost.
        reason: DeviceLostReason,
        /// Backend prose for the log.
        detail: String,
    },
    /// A GPU allocation failed for lack of device memory.
    #[error("out of device memory: {0}")]
    OutOfDeviceMemory(String),
    /// The swapchain no longer matches the surface; the frame did not present.
    /// Transient: the backend recreates the swapchain and the next frame
    /// normally succeeds.
    #[error("swapchain out of date")]
    SwapchainOutOfDate,
    /// A shader failed to compile or link into a pipeline.
    #[error("shader compile: {0}")]
    ShaderCompile(String),
    /// An unclassified failure carrying the original message.
    #[error("{0}")]
    Other(String),
}

/// A backend call's result.
pub type RenderResult<T> = Result<T, RenderError>;

impl From<String> for RenderError {
    fn from(message: String) -> Self {
        RenderError::Other(message)
    }
}

impl From<&str> for RenderError {
    fn from(message: &str) -> Self {
        RenderError::Other(message.to_string())
    }
}

// Bridge for interior call sites still reporting `Result<_, String>`: a typed
// error crossing one decays to its message, so a detection site can go typed
// before every caller above it has migrated.
impl From<RenderError> for String {
    fn from(error: RenderError) -> Self {
        error.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_coerces_to_other() {
        fn fails() -> RenderResult<()> {
            Err::<(), String>("boom".to_string())?;
            Ok(())
        }
        assert_eq!(fails(), Err(RenderError::Other("boom".to_string())));
    }

    #[test]
    fn display_includes_reason_and_detail() {
        let e = RenderError::DeviceLost {
            reason: DeviceLostReason::Hung,
            detail: "queue submit".to_string(),
        };
        assert_eq!(e.to_string(), "device lost (device hung): queue submit");
    }
}
