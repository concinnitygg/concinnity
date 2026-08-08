// src/metal/error.rs
//
// Classify a faulted MTLCommandBuffer's NSError into the RenderError boundary
// vocabulary. GPU failures on Metal surface asynchronously on completed
// command buffers, so the completion handler classifies here and parks the
// result on the context for the next draw_frame to report.

use crate::gfx::error::{DeviceLostReason, RenderError};
use objc2_foundation::NSError;
use objc2_metal::MTLCommandBufferError;

// Pure mapping from a command-buffer error code to the boundary class, so the
// table is testable without a GPU. `detail` is the NSError's description.
pub(super) fn classify_command_buffer_error(code: usize, detail: String) -> RenderError {
    match MTLCommandBufferError(code) {
        MTLCommandBufferError::DeviceRemoved => RenderError::DeviceLost {
            reason: DeviceLostReason::Removed,
            detail,
        },
        MTLCommandBufferError::Timeout => RenderError::DeviceLost {
            reason: DeviceLostReason::Hung,
            detail,
        },
        // A page fault or stack overflow makes the OS ignore every later
        // submission from this process, so the device is effectively lost.
        MTLCommandBufferError::PageFault | MTLCommandBufferError::StackOverflow => {
            RenderError::DeviceLost {
                reason: DeviceLostReason::Hung,
                detail,
            }
        }
        MTLCommandBufferError::AccessRevoked => RenderError::DeviceLost {
            reason: DeviceLostReason::Reset,
            detail,
        },
        MTLCommandBufferError::OutOfMemory => RenderError::OutOfDeviceMemory(detail),
        _ => RenderError::Other(detail),
    }
}

// Classify a completed command buffer's NSError. Only errors in the Metal
// command-buffer domain carry a meaningful code; anything else stays `Other`.
pub(super) fn classify_ns_error(error: &NSError) -> RenderError {
    let detail = error.localizedDescription().to_string();
    let in_metal_domain = &*error.domain() == unsafe { objc2_metal::MTLCommandBufferErrorDomain };
    if in_metal_domain {
        classify_command_buffer_error(error.code() as usize, detail)
    } else {
        RenderError::Other(detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(code: MTLCommandBufferError) -> RenderError {
        classify_command_buffer_error(code.0, "gpu fault".to_string())
    }

    #[test]
    fn device_removed_is_device_lost_removed() {
        assert_eq!(
            classify(MTLCommandBufferError::DeviceRemoved),
            RenderError::DeviceLost {
                reason: DeviceLostReason::Removed,
                detail: "gpu fault".to_string(),
            }
        );
    }

    #[test]
    fn timeout_and_page_fault_are_device_lost_hung() {
        for code in [
            MTLCommandBufferError::Timeout,
            MTLCommandBufferError::PageFault,
            MTLCommandBufferError::StackOverflow,
        ] {
            assert_eq!(
                classify(code),
                RenderError::DeviceLost {
                    reason: DeviceLostReason::Hung,
                    detail: "gpu fault".to_string(),
                }
            );
        }
    }

    #[test]
    fn access_revoked_is_device_lost_reset() {
        assert_eq!(
            classify(MTLCommandBufferError::AccessRevoked),
            RenderError::DeviceLost {
                reason: DeviceLostReason::Reset,
                detail: "gpu fault".to_string(),
            }
        );
    }

    #[test]
    fn out_of_memory_is_typed_oom() {
        assert_eq!(
            classify(MTLCommandBufferError::OutOfMemory),
            RenderError::OutOfDeviceMemory("gpu fault".to_string())
        );
    }

    #[test]
    fn unrecognized_codes_stay_other() {
        for code in [
            MTLCommandBufferError::None,
            MTLCommandBufferError::Internal,
            MTLCommandBufferError::InvalidResource,
            MTLCommandBufferError::Memoryless,
            MTLCommandBufferError::NotPermitted,
        ] {
            assert_eq!(classify(code), RenderError::Other("gpu fault".to_string()));
        }
    }
}
