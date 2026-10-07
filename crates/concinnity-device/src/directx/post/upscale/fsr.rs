//! AMD FidelityFX FSR3 temporal upscaling for the D3D12 backend, through the
//! FidelityFX SDK's `ffx_api` (see `crate::upscale_sdk::fsr`). The runtime is
//! `amd_fidelityfx_dx12.dll`, bundled beside the executable by `build.rs` when
//! the SDK is found or else taken from PATH; a missing DLL falls through to the
//! next backend. FFX needs a newer D3D12 runtime than the OS ships, so this
//! backend is only tried in builds that bundle the Agility SDK.
//!
//! The scaler accumulates temporally itself, so the TAA resolve is bypassed
//! while it runs; the projection is jittered by FFX's own sequence rather than
//! the engine's Halton one.

use std::ffi::{CStr, c_void};

use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::{Dll, UpscaleBackend, UpscaleInputs, UpscaleOutput, UpscalerTarget};
use crate::upscale_sdk::fsr::{FfxContext, FfxDispatchHandles, ffxApiHeader};
use crate::upscale_sdk::{UpscaleCamera, UpscaleExtent};

const LIBRARY: &CStr = c"amd_fidelityfx_dx12.dll";
const LABEL: &str = "FidelityFX FSR3";

const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12: u64 = 0x000_0002;

// `ffxCreateBackendDX12Desc` (`ffx_api/dx12/ffx_api_dx12.h`).
#[repr(C)]
struct ffxCreateBackendDX12Desc {
    header: ffxApiHeader,
    device: *mut c_void,
}

// The FFX upscale context and the output texture it writes.
pub(super) struct FsrUpscaler {
    ffx: FfxContext<Dll>,
    output: UpscaleOutput,
}

// SAFETY: `FsrUpscaler` owns its FFX context and the DLL its entry points come from, neither shared:
// every entry point runs on the render thread that built them, under the same main-thread guard as
// the rest of `DxContext`. Moving the whole upscaler hands over exclusive ownership, so it is `Send`
// without being `Sync`.
unsafe impl Send for FsrUpscaler {}

impl FsrUpscaler {
    // `Ok(None)` when FFX is unavailable: the DLL or one of its entry points
    // is missing, or the context could not be created.
    pub(super) fn try_new(target: UpscalerTarget<'_>) -> RenderResult<Option<Self>> {
        let Some(library) = Dll::open(LIBRARY) else {
            if cfg!(ffx_sdk_bundled) {
                tracing::warn!(
                    "{LABEL}: amd_fidelityfx_dx12.dll was bundled at build time but failed to \
                     load at runtime; trying the next backend"
                );
            } else {
                tracing::warn!(
                    "{LABEL}: amd_fidelityfx_dx12.dll not found (build.rs did not bundle it; set \
                     CN_FIDELITYFX_SDK or put the DLL on PATH). Trying the next backend."
                );
            }
            return Ok(None);
        };
        let mut backend = ffxCreateBackendDX12Desc {
            header: ffxApiHeader::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12),
            device: target.gpu.device.as_raw(),
        };
        // SAFETY: the description names the live device, which owns every upscaler built on it and
        // so outlives the context.
        let ffx = unsafe { FfxContext::create(library, &mut backend.header, target.extent, LABEL) };
        let Some(ffx) = ffx else {
            return Ok(None);
        };
        let output =
            UpscaleOutput::create(target.gpu.device, target.extent.output, target.descriptors)?;
        Ok(Some(Self { ffx, output }))
    }
}

impl UpscaleBackend for FsrUpscaler {
    fn extent(&self) -> UpscaleExtent {
        self.ffx.extent()
    }

    fn output(&self) -> &UpscaleOutput {
        &self.output
    }

    fn jitter_offset(&self, frame_index: u32) -> [f32; 2] {
        self.ffx.jitter_offset(frame_index)
    }

    fn dispatch(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        inputs: UpscaleInputs<'_>,
        camera: UpscaleCamera,
    ) -> RenderResult<()> {
        let handles = FfxDispatchHandles {
            command_list: cmd.as_raw(),
            color: inputs.color.as_raw(),
            depth: inputs.depth.as_raw(),
            motion_vectors: inputs.motion_vectors.as_raw(),
            output: self.output.resource().as_raw(),
        };
        // SAFETY: `cmd` is recording; `encode_upscale` put the inputs in NON_PIXEL_SHADER_RESOURCE
        // (FFX's compute read) and the output in UNORDERED_ACCESS, all at this context's extent,
        // and the frame keeps every one alive until the list executes.
        unsafe { self.ffx.dispatch(handles, camera) }
    }

    fn request_history_reset(&self) {
        self.ffx.request_history_reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn ffx_dx12_backend_layout_matches_sdk_v114() {
        assert_eq!(size_of::<ffxCreateBackendDX12Desc>(), 24);
        assert_eq!(offset_of!(ffxCreateBackendDX12Desc, header), 0);
        assert_eq!(offset_of!(ffxCreateBackendDX12Desc, device), 16);
    }
}
