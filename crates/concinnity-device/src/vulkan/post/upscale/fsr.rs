//! AMD FidelityFX FSR temporal upscaling for the Vulkan backend, through the
//! FidelityFX SDK's `ffx_api` (see `crate::upscale_sdk::fsr`). The runtime is
//! `amd_fidelityfx_vk.dll` (Windows) / `libamd_fidelityfx_vk.so` (Linux),
//! loaded on demand; a missing library falls through to the next backend.
//!
//! The scaler accumulates temporally itself, so the frame graph drops the TAA
//! resolve and runs `Upscale` in its slot; the projection is jittered by FFX's
//! own sequence rather than the engine's Halton one.

use std::cell::Cell;
use std::ffi::c_void;

use ash::vk;
use ash::vk::Handle;
use concinnity_core::render::error::RenderResult;

use super::{
    OutputWrites, UpscaleInputs, UpscaleOutput, UpscalerGpu, VkUpscaleBackend, open_library,
};
use crate::upscale_sdk::fsr::{FfxContext, FfxDispatchHandles, ffxApiHeader};
use crate::upscale_sdk::{UpscaleCamera, UpscaleExtent};

const LIBRARY: &str = if cfg!(windows) {
    "amd_fidelityfx_vk.dll"
} else {
    "libamd_fidelityfx_vk.so"
};
const LABEL: &str = "FidelityFX FSR (Vulkan)";

const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK: u64 = 0x000_0003;

// `ffxCreateBackendVKDesc` (`ffx_api/vk/ffx_api_vk.h`). FFX loads its own
// Vulkan entry points through `vkDeviceProcAddr`.
#[repr(C)]
struct ffxCreateBackendVKDesc {
    header: ffxApiHeader,
    vk_device: *mut c_void,
    vk_physical_device: *mut c_void,
    vk_device_proc_addr: *mut c_void,
}

// A Vulkan handle as the pointer-sized value `ffx_api` takes.
fn raw(handle: impl Handle) -> *mut c_void {
    handle.as_raw() as usize as *mut c_void
}

// The FFX upscale context and the output image it writes.
pub(super) struct FsrUpscaler {
    ffx: FfxContext<libloading::Library>,
    output: UpscaleOutput,
    jitter: Cell<[f32; 2]>,
}

// SAFETY: `FsrUpscaler` owns its FFX context and the library its entry points come from, neither
// shared: the upscale pass is recorded by exactly one parallel-encoder worker per frame, under the
// same main-thread guard as the rest of `VkContext`. Moving the whole upscaler hands over exclusive
// ownership, so it is `Send` without being `Sync`.
unsafe impl Send for FsrUpscaler {}

impl FsrUpscaler {
    // `Ok(None)` when FFX is unavailable: the library or one of its entry
    // points is missing, or the context could not be created.
    pub(super) fn try_new(
        gpu: UpscalerGpu<'_>,
        extent: UpscaleExtent,
    ) -> RenderResult<Option<Self>> {
        let Some(library) = open_library(LIBRARY) else {
            if cfg!(ffx_sdk_bundled) {
                tracing::warn!(
                    "{LABEL}: {LIBRARY} was bundled at build time but failed to load at runtime; \
                     trying the next backend"
                );
            } else {
                tracing::warn!(
                    "{LABEL}: {LIBRARY} not found (build.rs did not bundle it; set \
                     CN_FIDELITYFX_SDK or put the library on the search path). Trying the next \
                     backend."
                );
            }
            return Ok(None);
        };
        let mut backend = ffxCreateBackendVKDesc {
            header: ffxApiHeader::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK),
            vk_device: raw(gpu.device.handle()),
            vk_physical_device: raw(gpu.physical_device),
            vk_device_proc_addr: gpu.instance.fp_v1_0().get_device_proc_addr as usize
                as *mut c_void,
        };
        // SAFETY: the description names the live device and physical device, which `VkContext`
        // keeps until after it destroys the upscaler.
        let ffx = unsafe { FfxContext::create(library, &mut backend.header, extent, LABEL) };
        let Some(ffx) = ffx else {
            return Ok(None);
        };
        let output = UpscaleOutput::create(gpu, extent.output, OutputWrites::storage())?;
        Ok(Some(Self {
            ffx,
            output,
            jitter: Cell::new([0.0, 0.0]),
        }))
    }
}

impl VkUpscaleBackend for FsrUpscaler {
    fn extent(&self) -> UpscaleExtent {
        self.ffx.extent()
    }

    fn output(&self) -> &UpscaleOutput {
        &self.output
    }

    fn jitter_offset(&self, frame_index: u32) -> [f32; 2] {
        self.ffx.jitter_offset(frame_index)
    }

    fn jitter(&self) -> &Cell<[f32; 2]> {
        &self.jitter
    }

    fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        inputs: UpscaleInputs<'_>,
        camera: UpscaleCamera,
    ) -> RenderResult<()> {
        let handles = FfxDispatchHandles {
            command_list: raw(cmd),
            color: raw(inputs.color.image),
            depth: raw(inputs.depth.image),
            motion_vectors: raw(inputs.motion.image),
            output: raw(self.output.image().image),
        };
        // SAFETY: `cmd` is recording; `encode_upscale` put the inputs in SHADER_READ_ONLY_OPTIMAL
        // (FFX's compute read) and the output in GENERAL, all at this context's extent, and the
        // frame keeps every one alive until the buffer executes.
        unsafe { self.ffx.dispatch(handles, camera) }
    }

    fn request_history_reset(&self) {
        self.ffx.request_history_reset();
    }

    fn destroy(&mut self) {
        self.ffx.destroy();
        self.output.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn ffx_vk_backend_layout_matches_sdk_v114() {
        assert_eq!(size_of::<ffxCreateBackendVKDesc>(), 40);
        assert_eq!(offset_of!(ffxCreateBackendVKDesc, header), 0);
        assert_eq!(offset_of!(ffxCreateBackendVKDesc, vk_device), 16);
        assert_eq!(offset_of!(ffxCreateBackendVKDesc, vk_physical_device), 24);
        assert_eq!(offset_of!(ffxCreateBackendVKDesc, vk_device_proc_addr), 32);
    }
}
