// src/directx/error.rs
//
// Classify D3D12 / DXGI HRESULTs into the RenderError boundary vocabulary.
// Device removal is detected at Present; `classify_present_failure` refines
// the loss class with the device's own `GetDeviceRemovedReason` verdict.

use windows::Win32::Foundation::E_OUTOFMEMORY;
use windows::Win32::Graphics::Dxgi::{
    DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET,
};
use windows::core::HRESULT;

use crate::gfx::error::{DeviceLostReason, RenderError};

// Pure mapping from an HRESULT to the boundary class, testable without a GPU.
// `context` names the failing call for the log.
pub(super) fn map_hresult(hr: HRESULT, context: &str) -> RenderError {
    let detail = format!("{context}: {hr:?}");
    match hr {
        DXGI_ERROR_DEVICE_REMOVED => RenderError::DeviceLost {
            reason: DeviceLostReason::Removed,
            detail,
        },
        DXGI_ERROR_DEVICE_RESET => RenderError::DeviceLost {
            reason: DeviceLostReason::Reset,
            detail,
        },
        DXGI_ERROR_DEVICE_HUNG => RenderError::DeviceLost {
            reason: DeviceLostReason::Hung,
            detail,
        },
        E_OUTOFMEMORY => RenderError::OutOfDeviceMemory(detail),
        _ => RenderError::Other(detail),
    }
}

// Classify a failed Present. `removed_reason` is the device's
// `GetDeviceRemovedReason` verdict, which names the actual loss cause when
// Present reports the generic DXGI_ERROR_DEVICE_REMOVED.
pub(super) fn classify_present_failure(present: HRESULT, removed_reason: HRESULT) -> RenderError {
    let detail = format!("Present: {present:?}; device removed reason: {removed_reason:?}");
    match present {
        DXGI_ERROR_DEVICE_REMOVED | DXGI_ERROR_DEVICE_RESET | DXGI_ERROR_DEVICE_HUNG => {
            let reason = match removed_reason {
                DXGI_ERROR_DEVICE_HUNG => DeviceLostReason::Hung,
                DXGI_ERROR_DEVICE_RESET => DeviceLostReason::Reset,
                DXGI_ERROR_DEVICE_REMOVED => DeviceLostReason::Removed,
                _ => DeviceLostReason::Unknown,
            };
            RenderError::DeviceLost { reason, detail }
        }
        E_OUTOFMEMORY => RenderError::OutOfDeviceMemory(detail),
        _ => RenderError::Other(detail),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_removed_hresults_map_to_loss_reasons() {
        for (hr, reason) in [
            (DXGI_ERROR_DEVICE_REMOVED, DeviceLostReason::Removed),
            (DXGI_ERROR_DEVICE_RESET, DeviceLostReason::Reset),
            (DXGI_ERROR_DEVICE_HUNG, DeviceLostReason::Hung),
        ] {
            match map_hresult(hr, "call") {
                RenderError::DeviceLost { reason: r, .. } => assert_eq!(r, reason),
                other => panic!("expected DeviceLost, got {other:?}"),
            }
        }
    }

    #[test]
    fn out_of_memory_maps_to_typed_oom() {
        assert!(matches!(
            map_hresult(E_OUTOFMEMORY, "call"),
            RenderError::OutOfDeviceMemory(_)
        ));
    }

    #[test]
    fn present_failure_prefers_removed_reason_verdict() {
        match classify_present_failure(DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_HUNG) {
            RenderError::DeviceLost { reason, .. } => {
                assert_eq!(reason, DeviceLostReason::Hung);
            }
            other => panic!("expected DeviceLost, got {other:?}"),
        }
    }

    #[test]
    fn unrecognized_hresults_stay_other() {
        let hr = HRESULT(-1);
        assert!(matches!(map_hresult(hr, "call"), RenderError::Other(_)));
        assert!(matches!(
            classify_present_failure(hr, HRESULT(0)),
            RenderError::Other(_)
        ));
    }
}
