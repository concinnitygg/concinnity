//! Intel XeSS temporal upscaling for the D3D12 backend (see
//! `crate::upscale_sdk::xess`); runs cross-vendor (Arc XMX + the DP4a fallback
//! elsewhere). The runtime is `libxess.dll`, bundled beside the executable by
//! `build.rs` when the SDK is found or else taken from PATH; a missing DLL falls
//! through to the next backend.

use std::ffi::{CStr, c_void};
use std::ptr;

use concinnity_core::gfx::jitter;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::reactive_mask::ReactiveReader;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::{Dll, UpscaleBackend, UpscaleInputs, UpscaleOutput, UpscalerTarget};
use crate::upscale_sdk::xess::{
    XESS_RESULT_SUCCESS, XessCommonApi, XessContext, XessExecuteFrame, XessInitHead,
    xess_context_handle_t,
};
use crate::upscale_sdk::{SdkLibrary, UpscaleCamera, UpscaleExtent, entry_point};

const LIBRARY: &CStr = c"libxess.dll";
const LABEL: &str = "XeSS";

// `xess_d3d12_init_params_t` (`xess_d3d12.h`).
#[repr(C)]
struct xess_d3d12_init_params_t {
    head: XessInitHead,
    p_temp_buffer_heap: *mut c_void,
    buffer_heap_offset: u64,
    p_temp_texture_heap: *mut c_void,
    texture_heap_offset: u64,
    p_pipeline_library: *mut c_void,
}

// `xess_d3d12_execute_params_t` (`xess_d3d12.h`).
#[repr(C)]
struct xess_d3d12_execute_params_t {
    p_color_texture: *mut c_void,
    p_velocity_texture: *mut c_void,
    p_depth_texture: *mut c_void,
    p_exposure_scale_texture: *mut c_void,
    p_responsive_pixel_mask_texture: *mut c_void,
    p_output_texture: *mut c_void,
    frame: XessExecuteFrame,
    p_descriptor_heap: *mut c_void,
    descriptor_heap_offset: u32,
}

type PfnXessD3D12CreateContext =
    unsafe extern "C" fn(device: *mut c_void, out_ctx: *mut xess_context_handle_t) -> i32;
type PfnXessD3D12BuildPipelines = unsafe extern "C" fn(
    ctx: xess_context_handle_t,
    pipeline_lib: *mut c_void,
    blocking: bool,
    init_flags: u32,
) -> i32;
type PfnXessD3D12Init = unsafe extern "C" fn(
    ctx: xess_context_handle_t,
    params: *const xess_d3d12_init_params_t,
) -> i32;
type PfnXessD3D12Execute = unsafe extern "C" fn(
    ctx: xess_context_handle_t,
    cmd: *mut c_void,
    params: *const xess_d3d12_execute_params_t,
) -> i32;

struct XessD3D12Api {
    create_context: PfnXessD3D12CreateContext,
    build_pipelines: PfnXessD3D12BuildPipelines,
    init: PfnXessD3D12Init,
    execute: PfnXessD3D12Execute,
    common: XessCommonApi,
}

impl XessD3D12Api {
    fn resolve(library: &impl SdkLibrary) -> Option<Self> {
        // SAFETY: each type is the prototype `xess_d3d12.h` declares for that export.
        unsafe {
            Some(Self {
                create_context: entry_point(library, c"xessD3D12CreateContext")?,
                build_pipelines: entry_point(library, c"xessD3D12BuildPipelines")?,
                init: entry_point(library, c"xessD3D12Init")?,
                execute: entry_point(library, c"xessD3D12Execute")?,
                common: XessCommonApi::resolve(library)?,
            })
        }
    }
}

// The XeSS context, its execute entry point, and the output texture it writes.
pub(super) struct XessUpscaler {
    ctx: XessContext<Dll>,
    execute: PfnXessD3D12Execute,
    output: UpscaleOutput,
}

// SAFETY: `XessUpscaler` owns its XeSS context and the DLL its entry points come from, neither
// shared: every entry point runs on the render thread that built them, under the same main-thread
// guard as the rest of `DxContext`. Moving the whole upscaler hands over exclusive ownership, so it
// is `Send` without being `Sync`.
unsafe impl Send for XessUpscaler {}

impl XessUpscaler {
    // `Ok(None)` when XeSS is unavailable: the DLL or one of its entry points
    // is missing, or the context could not be created or initialized.
    pub(super) fn try_new(target: UpscalerTarget<'_>) -> RenderResult<Option<Self>> {
        let Some(library) = Dll::open(LIBRARY) else {
            tracing::warn!(
                "{LABEL}: libxess.dll not found (build.rs did not bundle it; set CN_XESS_SDK or \
                 put the DLL on PATH). Trying the next backend."
            );
            return Ok(None);
        };
        let Some(api) = XessD3D12Api::resolve(&library) else {
            tracing::warn!(
                "{LABEL}: libxess.dll lacks a D3D12 entry point; trying the next backend"
            );
            return Ok(None);
        };
        let extent = target.extent;

        let mut handle: xess_context_handle_t = ptr::null_mut();
        // SAFETY: the entry point came from the loaded DLL with the header's prototype, the device
        // is live, and `handle` is a live local the call fills.
        let rc = unsafe { (api.create_context)(target.gpu.device.as_raw(), &mut handle) };
        if rc != XESS_RESULT_SUCCESS || handle.is_null() {
            tracing::warn!(
                "{LABEL}: xessD3D12CreateContext returned {rc}; trying the next backend"
            );
            return Ok(None);
        }
        let ctx = XessContext::adopt(library, api.common, handle, extent);

        let head = XessInitHead::new(extent);
        // SAFETY: `ctx` holds the live context, and a null pipeline library is the header's "none".
        let rc = unsafe {
            (api.build_pipelines)(ctx.handle(), ptr::null_mut(), true, head.init_flags())
        };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!(
                "{LABEL}: xessD3D12BuildPipelines returned {rc}; trying the next backend"
            );
            return Ok(None);
        }
        let init_params = xess_d3d12_init_params_t {
            head,
            p_temp_buffer_heap: ptr::null_mut(),
            buffer_heap_offset: 0,
            p_temp_texture_heap: ptr::null_mut(),
            texture_heap_offset: 0,
            p_pipeline_library: ptr::null_mut(),
        };
        // SAFETY: `ctx` holds the live context, and `init_params` is a live local the call reads.
        let rc = unsafe { (api.init)(ctx.handle(), &init_params) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("{LABEL}: xessD3D12Init returned {rc}; trying the next backend");
            return Ok(None);
        }
        ctx.set_velocity_scale(LABEL);
        ctx.set_responsive_mask_cap(LABEL);

        let output = UpscaleOutput::create(target.gpu.device, extent.output, target.descriptors)?;
        tracing::info!("{LABEL}: context created: {extent}");
        Ok(Some(Self {
            ctx,
            execute: api.execute,
            output,
        }))
    }
}

impl UpscaleBackend for XessUpscaler {
    fn extent(&self) -> UpscaleExtent {
        self.ctx.extent()
    }

    fn output(&self) -> &UpscaleOutput {
        &self.output
    }

    // XeSS prescribes no jitter sequence; the engine's Halton (2, 3) drives
    // both the projection and the execute.
    fn jitter_offset(&self, frame_index: u32) -> [f32; 2] {
        jitter::offset(frame_index)
    }

    fn dispatch(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        inputs: UpscaleInputs<'_>,
        camera: UpscaleCamera,
    ) -> RenderResult<()> {
        let params = xess_d3d12_execute_params_t {
            p_color_texture: inputs.color.as_raw(),
            p_velocity_texture: inputs.motion_vectors.as_raw(),
            p_depth_texture: inputs.depth.as_raw(),
            p_exposure_scale_texture: ptr::null_mut(),
            p_responsive_pixel_mask_texture: inputs
                .reactive
                .map_or(ptr::null_mut(), Interface::as_raw),
            p_output_texture: self.output.resource().as_raw(),
            frame: self.ctx.frame(camera.jitter_offset),
            p_descriptor_heap: ptr::null_mut(),
            descriptor_heap_offset: 0,
        };
        // SAFETY: `ctx` holds the live context, `cmd` is recording, and `params` is a live local
        // naming resources the frame keeps alive until the list executes.
        let rc = unsafe { (self.execute)(self.ctx.handle(), cmd.as_raw(), &params) };
        if rc != XESS_RESULT_SUCCESS {
            return Err(RenderError::Other(format!(
                "xessD3D12Execute returned {rc}"
            )));
        }
        Ok(())
    }

    fn request_history_reset(&self) {
        self.ctx.request_history_reset();
    }

    fn reactive_reader(&self) -> ReactiveReader {
        ReactiveReader::Xess
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // The shared head and frame runs pin their own fields.
    #[test]
    fn xess_d3d12_layouts_match_sdk_v301() {
        type I = xess_d3d12_init_params_t;
        assert_eq!(size_of::<I>(), 64);
        assert_eq!(offset_of!(I, head), 0);
        assert_eq!(offset_of!(I, p_temp_buffer_heap), 24);
        assert_eq!(offset_of!(I, buffer_heap_offset), 32);
        assert_eq!(offset_of!(I, p_temp_texture_heap), 40);
        assert_eq!(offset_of!(I, texture_heap_offset), 48);
        assert_eq!(offset_of!(I, p_pipeline_library), 56);

        type E = xess_d3d12_execute_params_t;
        assert_eq!(size_of::<E>(), 136);
        assert_eq!(offset_of!(E, p_color_texture), 0);
        assert_eq!(offset_of!(E, p_velocity_texture), 8);
        assert_eq!(offset_of!(E, p_depth_texture), 16);
        assert_eq!(offset_of!(E, p_exposure_scale_texture), 24);
        assert_eq!(offset_of!(E, p_responsive_pixel_mask_texture), 32);
        assert_eq!(offset_of!(E, p_output_texture), 40);
        assert_eq!(offset_of!(E, frame), 48);
        assert_eq!(offset_of!(E, p_descriptor_heap), 120);
        assert_eq!(offset_of!(E, descriptor_heap_offset), 128);
    }
}
