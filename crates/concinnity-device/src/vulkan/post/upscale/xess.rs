//! Intel XeSS temporal upscaling for the Vulkan backend (see
//! `crate::upscale_sdk::xess`); runs cross-vendor (Arc XMX + the DP4a fallback
//! elsewhere). The runtime is `libxess.dll` / `libxess.so`, which carries the
//! `xessVK*` entry points beside the D3D12 ones, loaded on demand; a missing
//! library falls through to the next backend.
//!
//! Unlike FSR, XeSS needs instance and device extensions and a device feature
//! chain enabled when the device is created. Those are queried up front through
//! `XessExtQuery` (held by `UpscaleSdk`); this module creates the upscale
//! context once the device exists.

use std::cell::Cell;
use std::ffi::{CString, c_char, c_void};
use std::ptr;

use ash::vk;
use concinnity_core::gfx::jitter;
use concinnity_core::render::error::{RenderError, RenderResult};

use super::{
    ImageViewInfo, OutputWrites, UpscaleInputs, UpscaleOutput, UpscalerGpu, VkUpscaleBackend,
    copy_ext_names, open_library,
};
use crate::upscale_sdk::xess::{
    XESS_RESULT_SUCCESS, XessCommonApi, XessContext, XessExecuteFrame, XessInitHead,
    xess_context_handle_t,
};
use crate::upscale_sdk::{SdkLibrary, UpscaleCamera, UpscaleExtent, entry_point};

const LIBRARY: &str = if cfg!(windows) {
    "libxess.dll"
} else {
    "libxess.so"
};
const LABEL: &str = "XeSS (Vulkan)";

// `xess_vk_init_params_t` (`xess_vk.h`).
#[repr(C)]
struct xess_vk_init_params_t {
    head: XessInitHead,
    temp_buffer_heap: vk::DeviceMemory,
    buffer_heap_offset: u64,
    temp_texture_heap: vk::DeviceMemory,
    texture_heap_offset: u64,
    pipeline_cache: vk::PipelineCache,
}

// `xess_vk_execute_params_t` (`xess_vk.h`).
#[repr(C)]
struct xess_vk_execute_params_t {
    color_texture: ImageViewInfo,
    velocity_texture: ImageViewInfo,
    depth_texture: ImageViewInfo,
    exposure_scale_texture: ImageViewInfo,
    responsive_pixel_mask_texture: ImageViewInfo,
    output_texture: ImageViewInfo,
    frame: XessExecuteFrame,
}

// XESS_API is a bare dllimport (no explicit calling convention), so on x86_64
// `extern "C"` is the only one.
type PfnXessVKGetRequiredInstanceExtensions =
    unsafe extern "C" fn(*mut u32, *mut *const *const c_char, *mut u32) -> i32;
type PfnXessVKGetRequiredDeviceExtensions = unsafe extern "C" fn(
    vk::Instance,
    vk::PhysicalDevice,
    *mut u32,
    *mut *const *const c_char,
) -> i32;
type PfnXessVKGetRequiredDeviceFeatures =
    unsafe extern "C" fn(vk::Instance, vk::PhysicalDevice, *mut *mut c_void) -> i32;
type PfnXessVKCreateContext = unsafe extern "C" fn(
    vk::Instance,
    vk::PhysicalDevice,
    vk::Device,
    *mut xess_context_handle_t,
) -> i32;
type PfnXessVKBuildPipelines =
    unsafe extern "C" fn(xess_context_handle_t, vk::PipelineCache, bool, u32) -> i32;
type PfnXessVKInit =
    unsafe extern "C" fn(xess_context_handle_t, *const xess_vk_init_params_t) -> i32;
type PfnXessVKExecute = unsafe extern "C" fn(
    xess_context_handle_t,
    vk::CommandBuffer,
    *const xess_vk_execute_params_t,
) -> i32;

// Pre-device extension / feature queries. Held by `UpscaleSdk` across instance
// and device creation so the SDK-owned device-feature chain stays mapped
// through `vkCreateDevice`.
pub(super) struct XessExtQuery {
    get_instance_exts: PfnXessVKGetRequiredInstanceExtensions,
    get_device_exts: PfnXessVKGetRequiredDeviceExtensions,
    get_device_features: PfnXessVKGetRequiredDeviceFeatures,
    _library: libloading::Library,
}

impl XessExtQuery {
    pub(super) fn load() -> Option<Self> {
        let library = open_library(LIBRARY)?;
        // SAFETY: each type is the prototype `xess_vk.h` declares for that export, and the library
        // they come from is held alongside them.
        unsafe {
            Some(Self {
                get_instance_exts: entry_point(&library, c"xessVKGetRequiredInstanceExtensions")?,
                get_device_exts: entry_point(&library, c"xessVKGetRequiredDeviceExtensions")?,
                get_device_features: entry_point(&library, c"xessVKGetRequiredDeviceFeatures")?,
                _library: library,
            })
        }
    }

    // XeSS's required instance extensions and the minimum Vulkan API version
    // it needs (XeSS 3.x shaders use SPV_KHR_integer_dot_product, which needs a
    // Vulkan 1.3 environment). The caller raises the instance `apiVersion` to
    // at least this, clamped to loader support.
    pub(super) fn instance_extensions(&self) -> (Vec<CString>, u32) {
        let mut count: u32 = 0;
        let mut exts: *const *const c_char = ptr::null();
        let mut min_api: u32 = 0;
        // SAFETY: the entry point came from the loaded library with the header's prototype, and
        // every out-param is a live local.
        let rc = unsafe { (self.get_instance_exts)(&mut count, &mut exts, &mut min_api) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("{LABEL}: xessVKGetRequiredInstanceExtensions returned {rc}");
            return (Vec::new(), 0);
        }
        // SAFETY: `count`/`exts` are the pair the SDK just wrote on the success path above, and the
        // library owns that array for as long as it stays loaded.
        (unsafe { copy_ext_names(count, exts) }, min_api)
    }

    pub(super) fn device_extensions(
        &self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Vec<CString> {
        let mut count: u32 = 0;
        let mut exts: *const *const c_char = ptr::null();
        // SAFETY: the entry point came from the loaded library with the header's prototype, the
        // instance and physical device are live, and every out-param is a live local.
        let rc = unsafe {
            (self.get_device_exts)(instance.handle(), physical_device, &mut count, &mut exts)
        };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("{LABEL}: xessVKGetRequiredDeviceExtensions returned {rc}");
            return Vec::new();
        }
        // SAFETY: as in `instance_extensions` -- `count`/`exts` are the pair the SDK just wrote.
        unsafe { copy_ext_names(count, exts) }
    }

    // Patch the device-feature `pNext` chain with XeSS's required features and
    // return the (possibly new) chain head, to be set as `VkDeviceCreateInfo.pNext`.
    // `head` is the caller's existing chain; the memory the SDK adds is owned by
    // the library and valid while `self` lives. On failure the caller's `head`
    // is returned unchanged.
    pub(super) fn device_features(
        &self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        head: *mut c_void,
    ) -> *mut c_void {
        let mut chain = head;
        // SAFETY: the entry point came from the loaded library with the header's prototype, the
        // instance and physical device are live, and `chain` is a live local the SDK may rewrite.
        let rc =
            unsafe { (self.get_device_features)(instance.handle(), physical_device, &mut chain) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!(
                "{LABEL}: xessVKGetRequiredDeviceFeatures returned {rc}; using base features"
            );
            return head;
        }
        chain
    }
}

struct XessVkApi {
    create_context: PfnXessVKCreateContext,
    build_pipelines: PfnXessVKBuildPipelines,
    init: PfnXessVKInit,
    execute: PfnXessVKExecute,
    common: XessCommonApi,
}

impl XessVkApi {
    fn resolve(library: &impl SdkLibrary) -> Option<Self> {
        // SAFETY: each type is the prototype `xess_vk.h` declares for that export.
        unsafe {
            Some(Self {
                create_context: entry_point(library, c"xessVKCreateContext")?,
                build_pipelines: entry_point(library, c"xessVKBuildPipelines")?,
                init: entry_point(library, c"xessVKInit")?,
                execute: entry_point(library, c"xessVKExecute")?,
                common: XessCommonApi::resolve(library)?,
            })
        }
    }
}

// The XeSS context, its execute entry point, and the output image it writes.
pub(super) struct XessUpscaler {
    ctx: XessContext<libloading::Library>,
    execute: PfnXessVKExecute,
    output: UpscaleOutput,
    jitter: Cell<[f32; 2]>,
}

// SAFETY: `XessUpscaler` owns its XeSS context and the library its entry points come from, neither
// shared: the upscale pass is recorded by exactly one parallel-encoder worker per frame, under the
// same main-thread guard as the rest of `VkContext`. Moving the whole upscaler hands over exclusive
// ownership, so it is `Send` without being `Sync`.
unsafe impl Send for XessUpscaler {}

impl XessUpscaler {
    // `Ok(None)` when XeSS is unavailable: the library or one of its entry
    // points is missing, or the context could not be created or initialized
    // (as when its extensions were not enabled at device creation).
    pub(super) fn try_new(
        gpu: UpscalerGpu<'_>,
        extent: UpscaleExtent,
    ) -> RenderResult<Option<Self>> {
        let Some(library) = open_library(LIBRARY) else {
            tracing::warn!(
                "{LABEL}: {LIBRARY} not found (build.rs did not bundle it; set CN_XESS_SDK or put \
                 the library on the search path). Trying the next backend."
            );
            return Ok(None);
        };
        let Some(api) = XessVkApi::resolve(&library) else {
            tracing::warn!(
                "{LABEL}: {LIBRARY} lacks a Vulkan entry point; trying the next backend"
            );
            return Ok(None);
        };

        let mut handle: xess_context_handle_t = ptr::null_mut();
        // SAFETY: the entry point came from the loaded library with the header's prototype, the
        // instance, physical device and device are live, and `handle` is a live local it fills.
        let rc = unsafe {
            (api.create_context)(
                gpu.instance.handle(),
                gpu.physical_device,
                gpu.device.handle(),
                &mut handle,
            )
        };
        if rc != XESS_RESULT_SUCCESS || handle.is_null() {
            tracing::warn!("{LABEL}: xessVKCreateContext returned {rc}; trying the next backend");
            return Ok(None);
        }
        let ctx = XessContext::adopt(library, api.common, handle, extent);

        let head = XessInitHead::new(extent);
        let pipeline_cache = crate::vulkan::pipeline_cache::handle();
        // SAFETY: `ctx` holds the live context, and the pipeline cache is the engine's own.
        let rc =
            unsafe { (api.build_pipelines)(ctx.handle(), pipeline_cache, true, head.init_flags()) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("{LABEL}: xessVKBuildPipelines returned {rc}; trying the next backend");
            return Ok(None);
        }
        let init_params = xess_vk_init_params_t {
            head,
            temp_buffer_heap: vk::DeviceMemory::null(),
            buffer_heap_offset: 0,
            temp_texture_heap: vk::DeviceMemory::null(),
            texture_heap_offset: 0,
            pipeline_cache,
        };
        // SAFETY: `ctx` holds the live context, and `init_params` is a live local the call reads.
        let rc = unsafe { (api.init)(ctx.handle(), &init_params) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("{LABEL}: xessVKInit returned {rc}; trying the next backend");
            return Ok(None);
        }
        ctx.set_velocity_scale(LABEL);

        let output = UpscaleOutput::create(gpu, extent.output, OutputWrites::storage())?;
        tracing::info!("{LABEL}: context created: {extent}");
        Ok(Some(Self {
            ctx,
            execute: api.execute,
            output,
            jitter: Cell::new([0.0, 0.0]),
        }))
    }
}

impl VkUpscaleBackend for XessUpscaler {
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

    fn jitter(&self) -> &Cell<[f32; 2]> {
        &self.jitter
    }

    fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        inputs: UpscaleInputs<'_>,
        camera: UpscaleCamera,
    ) -> RenderResult<()> {
        let params = xess_vk_execute_params_t {
            color_texture: ImageViewInfo::of(inputs.color),
            velocity_texture: ImageViewInfo::of(inputs.motion),
            depth_texture: ImageViewInfo::of(inputs.depth),
            exposure_scale_texture: ImageViewInfo::empty(),
            responsive_pixel_mask_texture: ImageViewInfo::empty(),
            output_texture: ImageViewInfo::of(&self.output.as_upscale_image()),
            frame: self.ctx.frame(camera.jitter_offset),
        };
        // SAFETY: `ctx` holds the live context, `cmd` is recording, and `params` is a live local
        // naming images the frame keeps alive until the buffer executes.
        let rc = unsafe { (self.execute)(self.ctx.handle(), cmd, &params) };
        if rc != XESS_RESULT_SUCCESS {
            return Err(RenderError::Other(format!("xessVKExecute returned {rc}")));
        }
        Ok(())
    }

    fn request_history_reset(&self) {
        self.ctx.request_history_reset();
    }

    fn destroy(&mut self) {
        self.ctx.destroy();
        self.output.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // The shared head, frame and image-view-info layouts pin their own fields.
    #[test]
    fn xess_vk_layouts_match_sdk_v301() {
        type I = xess_vk_init_params_t;
        assert_eq!(size_of::<I>(), 64);
        assert_eq!(offset_of!(I, head), 0);
        assert_eq!(offset_of!(I, temp_buffer_heap), 24);
        assert_eq!(offset_of!(I, buffer_heap_offset), 32);
        assert_eq!(offset_of!(I, temp_texture_heap), 40);
        assert_eq!(offset_of!(I, texture_heap_offset), 48);
        assert_eq!(offset_of!(I, pipeline_cache), 56);

        type E = xess_vk_execute_params_t;
        assert_eq!(size_of::<E>(), 360);
        assert_eq!(offset_of!(E, color_texture), 0);
        assert_eq!(offset_of!(E, velocity_texture), 48);
        assert_eq!(offset_of!(E, depth_texture), 96);
        assert_eq!(offset_of!(E, exposure_scale_texture), 144);
        assert_eq!(offset_of!(E, responsive_pixel_mask_texture), 192);
        assert_eq!(offset_of!(E, output_texture), 240);
        assert_eq!(offset_of!(E, frame), 288);
    }
}
