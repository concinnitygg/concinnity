//! NVIDIA DLSS temporal upscaling for the Vulkan backend, through the raw NGX
//! API's `NVSDK_NGX_VULKAN_*` entry points (see `crate::upscale_sdk::dlss`);
//! RTX only. Compiled only when `build.rs` finds the NGX SDK, links its static
//! library and bundles `nvngx_dlss.dll` beside the executable
//! (`cfg(ngx_sdk_bundled)`).
//!
//! The Vulkan specifics are the instance and device extensions DLSS needs at
//! creation time (queried by `required_extensions`, enabled through
//! `UpscaleSdk`) and resources bound as `NVSDK_NGX_Resource_VK` through
//! `SetVoidPointer`. Exposure comes from an engine-supplied 1x1 texture of 1.0
//! rather than DLSS's auto-exposure: the scene is un-exposed until after the
//! upscale, so 1.0 is the identity.

use std::cell::Cell;
use std::ffi::{CString, c_char, c_void};
use std::ptr;

use ash::vk;
use concinnity_core::gfx::jitter;
use concinnity_core::render::error::{RenderError, RenderResult};

use super::{
    BarrierSync, ImageViewInfo, LayoutTransition, OutputWrites, UpscaleImage, UpscaleInputs,
    UpscaleOutput, UpscalerGpu, VkUpscaleBackend, copy_ext_names, image_barrier,
};
use crate::upscale_sdk::dlss::{
    ENGINE_VERSION, NVSDK_NGX_ENGINE_TYPE_CUSTOM, NVSDK_NGX_FEATURE_SUPERSAMPLING,
    NVSDK_NGX_RESULT_FAIL, NVSDK_NGX_VERSION_API, P_COLOR, P_DEPTH, P_EXPOSURE_TEXTURE,
    P_MOTION_VECTORS, P_OUTPUT, PROJECT_ID, app_data_path, ngx_succeeded, set_create_parameters,
    set_evaluate_parameters, supersampling_available,
};
use crate::upscale_sdk::{UpscaleCamera, UpscaleExtent};
use crate::vulkan::texture::{
    GpuImage, ImageSpec, create_image, create_image_view, one_shot_submit,
};

// NVSDK_NGX_Resource_VK_Type.
const NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW: i32 = 0;

const EXPOSURE_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

// `NVSDK_NGX_Resource_VK` (`nvsdk_ngx_defs_vk.h`). The C struct opens with a
// union of an image-view and a buffer description; the image view is the
// larger arm and the only one bound here, so it stands in for the union.
#[repr(C)]
struct NVSDK_NGX_Resource_VK {
    image_view_info: ImageViewInfo,
    ty: i32,
    read_write: bool,
}

impl NVSDK_NGX_Resource_VK {
    fn image(img: &UpscaleImage, read_write: bool) -> Self {
        Self {
            image_view_info: ImageViewInfo::of(img),
            ty: NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,
            read_write,
        }
    }
}

// NGX's Vulkan entry points and the resource setter they take, exported
// unmangled from the static library.
unsafe extern "C" {
    fn NVSDK_NGX_VULKAN_Init_with_ProjectID(
        project_id: *const u8,
        engine_type: i32,
        engine_version: *const u8,
        app_data_path: *const u16,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        gipa: *const c_void,
        gdpa: *const c_void,
        feature_info: *const c_void,
        sdk_version: i32,
    ) -> u32;
    fn NVSDK_NGX_VULKAN_Shutdown1(device: vk::Device) -> u32;
    fn NVSDK_NGX_VULKAN_GetCapabilityParameters(out_params: *mut *mut c_void) -> u32;
    fn NVSDK_NGX_VULKAN_DestroyParameters(params: *mut c_void) -> u32;
    // CreateFeature1 takes the device, so NGX sets its internal resources up
    // during the init command buffer rather than lazily on the first evaluate;
    // the NGX helpers prefer it whenever a device is at hand.
    fn NVSDK_NGX_VULKAN_CreateFeature1(
        device: vk::Device,
        cmd: vk::CommandBuffer,
        feature_id: i32,
        params: *const c_void,
        out_handle: *mut *mut c_void,
    ) -> u32;
    fn NVSDK_NGX_VULKAN_ReleaseFeature(handle: *mut c_void) -> u32;
    fn NVSDK_NGX_VULKAN_EvaluateFeature_C(
        cmd: vk::CommandBuffer,
        handle: *const c_void,
        params: *const c_void,
        callback: *const c_void,
    ) -> u32;
    fn NVSDK_NGX_VULKAN_RequiredExtensions(
        out_inst_count: *mut u32,
        out_inst_exts: *mut *const *const c_char,
        out_dev_count: *mut u32,
        out_dev_exts: *mut *const *const c_char,
    ) -> u32;
    fn NVSDK_NGX_Parameter_SetVoidPointer(params: *mut c_void, name: *const u8, value: *mut c_void);
}

// Instance + device extensions NGX requires, queried before instance / device
// creation. Returns `(instance_exts, device_exts)` as owned `CString`s, or
// `None` if the query fails. Static-linked, so no library load is needed.
pub(super) fn required_extensions() -> Option<(Vec<CString>, Vec<CString>)> {
    let mut inst_count: u32 = 0;
    let mut inst_exts: *const *const c_char = ptr::null();
    let mut dev_count: u32 = 0;
    let mut dev_exts: *const *const c_char = ptr::null();
    // SAFETY: the NGX entry point is statically linked and matches the SDK's declared signature;
    // every pointer it is handed points at a live local that outlives the call.
    let rc = unsafe {
        NVSDK_NGX_VULKAN_RequiredExtensions(
            &mut inst_count,
            &mut inst_exts,
            &mut dev_count,
            &mut dev_exts,
        )
    };
    if !ngx_succeeded(rc) {
        tracing::warn!("DLSS: NVSDK_NGX_VULKAN_RequiredExtensions returned {rc:#x}");
        return None;
    }
    // SAFETY: `inst_count`/`inst_exts` are the pair NGX just wrote on the success path above, and
    // the SDK owns that array.
    let inst = unsafe { copy_ext_names(inst_count, inst_exts) };
    // SAFETY: as above, for the device extension pair.
    let dev = unsafe { copy_ext_names(dev_count, dev_exts) };
    Some((inst, dev))
}

// NGX initialized on `device`, with the parameter bag and super-sampling
// feature once they exist. `release` frees whatever it holds and shuts NGX
// down, at most once; dropping the session releases it too.
struct NgxSession {
    device: vk::Device,
    params: *mut c_void,
    feature: *mut c_void,
}

impl NgxSession {
    fn release(&mut self) {
        if self.device == vk::Device::null() {
            return;
        }
        // SAFETY: the feature and the bag are each released at most once (only when non-null), and
        // the shutdown names the live device NGX was initialized on, cleared straight after so it
        // runs once.
        unsafe {
            if !self.feature.is_null() {
                NVSDK_NGX_VULKAN_ReleaseFeature(self.feature);
            }
            if !self.params.is_null() {
                NVSDK_NGX_VULKAN_DestroyParameters(self.params);
            }
            NVSDK_NGX_VULKAN_Shutdown1(self.device);
        }
        self.feature = ptr::null_mut();
        self.params = ptr::null_mut();
        self.device = vk::Device::null();
    }
}

impl Drop for NgxSession {
    fn drop(&mut self) {
        self.release();
    }
}

// The DLSS feature, the output image it writes and the exposure texture it
// reads.
pub(super) struct DlssUpscaler {
    ngx: NgxSession,
    extent: UpscaleExtent,
    output: UpscaleOutput,
    // 1x1 R32F holding 1.0, created once in GENERAL and bound every evaluate.
    exposure: GpuImage,
    jitter: Cell<[f32; 2]>,
    reset_pending: Cell<bool>,
}

// SAFETY: the NGX feature handle and parameter bag this owns are not shared:
// every entry point runs on the render thread that built them, under the same
// main-thread guard as the rest of `VkContext`. Moving the whole upscaler hands
// over exclusive ownership, so it is `Send` without being `Sync`.
unsafe impl Send for DlssUpscaler {}

impl DlssUpscaler {
    // `Ok(None)` when DLSS is unavailable: NGX failed to initialize, the GPU
    // lacks DLSS, or the feature could not be created. Feature creation records
    // onto a command buffer, so this submits a one-shot buffer to the queue.
    // Assumes the NGX extensions were enabled at device creation.
    pub(super) fn try_new(
        gpu: UpscalerGpu<'_>,
        extent: UpscaleExtent,
    ) -> RenderResult<Option<Self>> {
        let app_path = app_data_path();
        // SAFETY: the NGX entry point is statically linked and matches the SDK's declared
        // signature; every pointer it is handed points at a live local that outlives the call.
        let rc = unsafe {
            NVSDK_NGX_VULKAN_Init_with_ProjectID(
                PROJECT_ID.as_ptr(),
                NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                ENGINE_VERSION.as_ptr(),
                app_path.as_ptr(),
                gpu.instance.handle(),
                gpu.physical_device,
                gpu.device.handle(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                NVSDK_NGX_VERSION_API,
            )
        };
        if !ngx_succeeded(rc) {
            tracing::warn!(
                "DLSS (Vulkan): NVSDK_NGX_VULKAN_Init returned {rc:#x} (NGX unavailable / not \
                 RTX). Trying the next backend."
            );
            return Ok(None);
        }
        let mut ngx = NgxSession {
            device: gpu.device.handle(),
            params: ptr::null_mut(),
            feature: ptr::null_mut(),
        };

        // SAFETY: NGX is initialized, and `ngx.params` is a live field that receives the bag.
        let rc = unsafe { NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut ngx.params) };
        if !ngx_succeeded(rc) || ngx.params.is_null() {
            tracing::warn!(
                "DLSS (Vulkan): GetCapabilityParameters returned {rc:#x}; trying the next backend"
            );
            return Ok(None);
        }
        // SAFETY: `ngx.params` is the non-null bag NGX just returned.
        if !unsafe { supersampling_available(ngx.params) } {
            tracing::warn!(
                "DLSS (Vulkan): SuperSampling not available on this GPU; trying the next backend"
            );
            return Ok(None);
        }
        // SAFETY: `ngx.params` is the live bag.
        unsafe { set_create_parameters(ngx.params, extent, true) };

        // The submit is fence-waited, so the feature and its internal resources
        // are ready before the first frame's evaluate.
        let mut create_rc = NVSDK_NGX_RESULT_FAIL;
        one_shot_submit(gpu.device, gpu.command_pool, gpu.queue, |cmd| {
            // SAFETY: `cmd` is in the recording state, `ngx.params` is the live bag filled above,
            // and `ngx.feature` is a live field NGX writes the new feature into.
            create_rc = unsafe {
                NVSDK_NGX_VULKAN_CreateFeature1(
                    gpu.device.handle(),
                    cmd,
                    NVSDK_NGX_FEATURE_SUPERSAMPLING,
                    ngx.params,
                    &mut ngx.feature,
                )
            };
        })?;
        if !ngx_succeeded(create_rc) || ngx.feature.is_null() {
            tracing::warn!(
                "DLSS (Vulkan): CreateFeature returned {create_rc:#x}; trying the next backend"
            );
            return Ok(None);
        }

        let output = UpscaleOutput::create(gpu, extent.output, OutputWrites::storage_and_clear())?;
        let exposure = create_identity_exposure(gpu)?;
        tracing::info!("DLSS (Vulkan): feature created: {extent}");
        Ok(Some(Self {
            ngx,
            extent,
            output,
            exposure,
            jitter: Cell::new([0.0, 0.0]),
            reset_pending: Cell::new(true),
        }))
    }
}

// Create the 1x1 exposure texture, cleared to 1.0 and left in GENERAL (the
// layout NGX reads every resource in). Written once here and never again, so
// the one clear barrier covers every later read. SAMPLED | STORAGE usage covers
// however NGX binds it.
fn create_identity_exposure(gpu: UpscalerGpu<'_>) -> RenderResult<GpuImage> {
    let pooled = create_image(
        gpu.alloc,
        &ImageSpec {
            width: 1,
            height: 1,
            format: EXPOSURE_FORMAT,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST,
            mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            samples: vk::SampleCountFlags::TYPE_1,
        },
    )?;
    let image = pooled.image();
    let view = create_image_view(
        gpu.device,
        image,
        EXPOSURE_FORMAT,
        vk::ImageAspectFlags::COLOR,
    )?;
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let one = vk::ClearColorValue {
        float32: [1.0, 0.0, 0.0, 0.0],
    };
    one_shot_submit(gpu.device, gpu.command_pool, gpu.queue, |cmd| {
        image_barrier(
            gpu.device,
            cmd,
            image,
            vk::ImageAspectFlags::COLOR,
            LayoutTransition {
                from: vk::ImageLayout::UNDEFINED,
                to: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            BarrierSync {
                src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                src_access: vk::AccessFlags::empty(),
                dst_stage: vk::PipelineStageFlags::TRANSFER,
                dst_access: vk::AccessFlags::TRANSFER_WRITE,
            },
        );
        // SAFETY: `cmd` is a command buffer in the recording state, `image` was just transitioned
        // to TRANSFER_DST_OPTIMAL by the barrier above, and `range` names its only mip and layer.
        unsafe {
            gpu.device.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &one,
                std::slice::from_ref(&range),
            );
        }
        image_barrier(
            gpu.device,
            cmd,
            image,
            vk::ImageAspectFlags::COLOR,
            LayoutTransition {
                from: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                to: vk::ImageLayout::GENERAL,
            },
            BarrierSync {
                src_stage: vk::PipelineStageFlags::TRANSFER,
                src_access: vk::AccessFlags::TRANSFER_WRITE,
                dst_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
                dst_access: vk::AccessFlags::SHADER_READ,
            },
        );
    })?;
    Ok(GpuImage::from_pooled(pooled, view))
}

impl VkUpscaleBackend for DlssUpscaler {
    fn extent(&self) -> UpscaleExtent {
        self.extent
    }

    fn output(&self) -> &UpscaleOutput {
        &self.output
    }

    // DLSS prescribes no jitter sequence; the engine's Halton (2, 3) drives
    // both the projection and the evaluate.
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
        let exposure = UpscaleImage {
            image: self.exposure.image,
            view: self.exposure.view,
            format: EXPOSURE_FORMAT,
            width: 1,
            height: 1,
            aspect: vk::ImageAspectFlags::COLOR,
        };
        // NGX reads these through the pointers bound below while recording the
        // evaluate, so they live until it returns.
        let mut resources = [
            (P_COLOR, NVSDK_NGX_Resource_VK::image(inputs.color, false)),
            (
                P_OUTPUT,
                NVSDK_NGX_Resource_VK::image(&self.output.as_upscale_image(), true),
            ),
            (P_DEPTH, NVSDK_NGX_Resource_VK::image(inputs.depth, false)),
            (
                P_MOTION_VECTORS,
                NVSDK_NGX_Resource_VK::image(inputs.motion, false),
            ),
            (
                P_EXPOSURE_TEXTURE,
                NVSDK_NGX_Resource_VK::image(&exposure, false),
            ),
        ];
        let params = self.ngx.params;
        // SAFETY: `params` is the live bag, every name is a NUL-terminated constant, and every
        // resource description points at an element of `resources`, which outlives the evaluate.
        unsafe {
            for (name, resource) in &mut resources {
                let resource: *mut NVSDK_NGX_Resource_VK = resource;
                NVSDK_NGX_Parameter_SetVoidPointer(params, name.as_ptr(), resource.cast());
            }
            set_evaluate_parameters(
                params,
                camera.jitter_offset,
                self.reset_pending.replace(false),
                self.extent,
            );
        }
        // SAFETY: `cmd` is recording, and the feature and bag are the live ones created in
        // `try_new`.
        let rc = unsafe {
            NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd, self.ngx.feature, params, ptr::null())
        };
        if !ngx_succeeded(rc) {
            return Err(RenderError::Other(format!(
                "NVSDK_NGX_VULKAN_EvaluateFeature returned {rc:#x}"
            )));
        }
        Ok(())
    }

    fn destroy(&mut self) {
        self.ngx.release();
        self.output.release();
        self.exposure = GpuImage::null();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // The image-view arm pins its own fields in `upscale::tests`.
    #[test]
    fn ngx_vk_resource_layout_matches_sdk() {
        assert_eq!(size_of::<NVSDK_NGX_Resource_VK>(), 56);
        assert_eq!(offset_of!(NVSDK_NGX_Resource_VK, image_view_info), 0);
        assert_eq!(offset_of!(NVSDK_NGX_Resource_VK, ty), 48);
        assert_eq!(offset_of!(NVSDK_NGX_Resource_VK, read_write), 52);
        assert_eq!(NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW, 0);
    }
}
