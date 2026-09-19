//! The typed error vocabulary of the `RenderBackend` boundary. Backends map
//! their native failure codes (VkResult, HRESULT, MTLCommandBuffer status) into
//! these classes at the detection sites; the frame loop dispatches recovery
//! policy on the class, never on prose. `Other` is the unclassified bucket: a
//! failure no detection site classified.

use alloc::format;
use alloc::string::String;
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
    /// The backend does not implement the operation. Nothing was changed, so
    /// a caller that can proceed without it may treat this as a skip.
    #[error("{op}: not supported on this backend")]
    Unsupported {
        /// The trait method that was called.
        op: &'static str,
    },
    /// An unclassified failure carrying the original message.
    #[error("{0}")]
    Other(String),
}

/// A backend call's result.
pub type RenderResult<T> = Result<T, RenderError>;

impl RenderError {
    /// Whether the device failed rather than the step that reported it: loss or
    /// memory exhaustion. A caller that tolerates a failed step must still
    /// propagate these so the frame loop's recovery policy sees them.
    pub fn is_device_failure(&self) -> bool {
        matches!(
            self,
            RenderError::DeviceLost { .. } | RenderError::OutOfDeviceMemory(_)
        )
    }

    /// Prefix the message with `what` (the resource or step that failed),
    /// keeping the class so recovery policy still sees it.
    pub fn context(self, what: impl core::fmt::Display) -> Self {
        match self {
            RenderError::DeviceLost { reason, detail } => RenderError::DeviceLost {
                reason,
                detail: format!("{what}: {detail}"),
            },
            RenderError::OutOfDeviceMemory(m) => {
                RenderError::OutOfDeviceMemory(format!("{what}: {m}"))
            }
            RenderError::ShaderCompile(m) => RenderError::ShaderCompile(format!("{what}: {m}")),
            RenderError::Other(m) => RenderError::Other(format!("{what}: {m}")),
            e @ (RenderError::SwapchainOutOfDate | RenderError::Unsupported { .. }) => e,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn display_includes_reason_and_detail() {
        let e = RenderError::DeviceLost {
            reason: DeviceLostReason::Hung,
            detail: "queue submit".to_string(),
        };
        assert_eq!(e.to_string(), "device lost (device hung): queue submit");
    }

    #[test]
    fn unsupported_display_names_the_operation() {
        let e = RenderError::Unsupported { op: "add_decal" };
        assert_eq!(e.to_string(), "add_decal: not supported on this backend");
    }

    #[test]
    fn device_failure_covers_loss_and_memory_only() {
        let lost = RenderError::DeviceLost {
            reason: DeviceLostReason::Removed,
            detail: String::new(),
        };
        assert!(lost.is_device_failure());
        assert!(RenderError::OutOfDeviceMemory(String::new()).is_device_failure());
        assert!(!RenderError::ShaderCompile(String::new()).is_device_failure());
        assert!(!RenderError::Other(String::new()).is_device_failure());
        assert!(!RenderError::SwapchainOutOfDate.is_device_failure());
        assert!(!RenderError::Unsupported { op: "x" }.is_device_failure());
    }

    #[test]
    fn context_prefixes_the_message_and_keeps_the_class() {
        let oom = RenderError::OutOfDeviceMemory("create_image".to_string()).context("hiz image");
        assert_eq!(
            oom,
            RenderError::OutOfDeviceMemory("hiz image: create_image".to_string())
        );
        let other = RenderError::Other("boom".to_string()).context(format_args!("texture[{}]", 3));
        assert_eq!(other, RenderError::Other("texture[3]: boom".to_string()));
        let lost = RenderError::DeviceLost {
            reason: DeviceLostReason::Hung,
            detail: "submit".to_string(),
        }
        .context("frame");
        assert_eq!(
            lost,
            RenderError::DeviceLost {
                reason: DeviceLostReason::Hung,
                detail: "frame: submit".to_string(),
            }
        );
        assert_eq!(
            RenderError::SwapchainOutOfDate.context("present"),
            RenderError::SwapchainOutOfDate
        );
    }
}
