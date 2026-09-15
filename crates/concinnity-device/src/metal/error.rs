//! Classify a faulted MTLCommandBuffer's NSError into the RenderError boundary
//! vocabulary. GPU failures on Metal surface asynchronously on completed
//! command buffers, so the completion handler classifies here and parks the
//! result on the context for the next draw_frame to report.

use concinnity_core::render::error::{DeviceLostReason, RenderError, RenderResult};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSError;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferError, MTLCommandBufferStatus};

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

// A nil result from a Metal allocation call (`newBuffer*`, `newTexture*`,
// `newHeap*`, an indirect command buffer), which Metal returns when the device
// cannot back the request.
pub(super) fn allocation_failed(what: impl core::fmt::Display) -> RenderError {
    RenderError::OutOfDeviceMemory(format!("failed to allocate {what}"))
}

// Classify a completed command buffer's NSError. Only errors in the Metal
// command-buffer domain carry a meaningful code; anything else stays `Other`.
pub(super) fn classify_ns_error(error: &NSError) -> RenderError {
    let detail = error.localizedDescription().to_string();
    // SAFETY: `MTLCommandBufferErrorDomain` is a framework-owned static NSString that outlives this
    // comparison.
    let in_metal_domain = &*error.domain() == unsafe { objc2_metal::MTLCommandBufferErrorDomain };
    if in_metal_domain {
        classify_command_buffer_error(error.code() as usize, detail)
    } else {
        RenderError::Other(detail)
    }
}

// The outcome of a command buffer that has finished executing: `Ok` unless it
// faulted, in which case its NSError is classified. A faulted buffer that
// carries no error object stays `Other`.
pub(super) fn completed_command_buffer(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    what: impl core::fmt::Display,
) -> RenderResult<()> {
    if cmd.status() != MTLCommandBufferStatus::Error {
        return Ok(());
    }
    let stage = format!("{what} faulted on the GPU");
    Err(match cmd.error() {
        Some(error) => classify_ns_error(&error).context(stage),
        None => RenderError::Other(stage),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLCommandQueue as _, MTLDevice as _};

    #[test]
    fn a_command_buffer_that_completed_cleanly_is_ok() {
        let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
            return;
        };
        let queue = device.newCommandQueue().expect("command queue");
        let cmd = queue.commandBuffer().expect("command buffer");
        cmd.commit();
        cmd.waitUntilCompleted();
        assert_eq!(completed_command_buffer(&cmd, "empty buffer"), Ok(()));
    }

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
    fn allocation_failure_is_typed_oom() {
        assert_eq!(
            allocation_failed(format_args!("bloom mip {} texture", 2)),
            RenderError::OutOfDeviceMemory("failed to allocate bloom mip 2 texture".to_string())
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
