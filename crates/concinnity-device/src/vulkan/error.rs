// src/vulkan/error.rs
//
// Classify VkResult failure codes into the RenderError boundary vocabulary.
// Applied at the sites where the classes matter for recovery: fence waits,
// swapchain acquire/present, queue submit, and device-memory allocation.

use ash::vk;

use crate::gfx::error::{DeviceLostReason, RenderError};

// Pure mapping from a VkResult to the boundary class, testable without a GPU.
// `context` names the failing call for the log. Vulkan reports loss as one
// code with no removed/reset/hung verdict, so the reason stays `Unknown`.
pub(super) fn map_vk_result(result: vk::Result, context: &str) -> RenderError {
    match result {
        vk::Result::ERROR_DEVICE_LOST => RenderError::DeviceLost {
            reason: DeviceLostReason::Unknown,
            detail: format!("{context}: {result:?}"),
        },
        vk::Result::ERROR_SURFACE_LOST_KHR => RenderError::DeviceLost {
            reason: DeviceLostReason::SurfaceLost,
            detail: format!("{context}: {result:?}"),
        },
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            RenderError::OutOfDeviceMemory(format!("{context}: {result:?}"))
        }
        vk::Result::ERROR_OUT_OF_DATE_KHR => RenderError::SwapchainOutOfDate,
        _ => RenderError::Other(format!("{context}: {result:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_lost_maps_with_unknown_reason() {
        assert_eq!(
            map_vk_result(vk::Result::ERROR_DEVICE_LOST, "queue submit"),
            RenderError::DeviceLost {
                reason: DeviceLostReason::Unknown,
                detail: "queue submit: ERROR_DEVICE_LOST".to_string(),
            }
        );
    }

    #[test]
    fn surface_lost_maps_to_surface_lost_reason() {
        assert_eq!(
            map_vk_result(vk::Result::ERROR_SURFACE_LOST_KHR, "present"),
            RenderError::DeviceLost {
                reason: DeviceLostReason::SurfaceLost,
                detail: "present: ERROR_SURFACE_LOST_KHR".to_string(),
            }
        );
    }

    #[test]
    fn out_of_device_memory_maps_to_typed_oom() {
        assert_eq!(
            map_vk_result(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY, "allocator"),
            RenderError::OutOfDeviceMemory("allocator: ERROR_OUT_OF_DEVICE_MEMORY".to_string())
        );
    }

    #[test]
    fn out_of_date_maps_to_swapchain_class() {
        assert_eq!(
            map_vk_result(vk::Result::ERROR_OUT_OF_DATE_KHR, "present"),
            RenderError::SwapchainOutOfDate
        );
    }

    #[test]
    fn unrecognized_codes_stay_other() {
        for code in [
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            vk::Result::ERROR_INITIALIZATION_FAILED,
            vk::Result::ERROR_FRAGMENTED_POOL,
            vk::Result::TIMEOUT,
        ] {
            assert!(matches!(map_vk_result(code, "call"), RenderError::Other(_)));
        }
    }
}
