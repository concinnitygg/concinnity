//! NVIDIA DLSS temporal upscaling for the D3D12 backend, through the raw NGX
//! API's `NVSDK_NGX_D3D12_*` entry points (see `crate::upscale_sdk::dlss`);
//! RTX only. Compiled only when `build.rs` finds the NGX SDK, links its static
//! library and bundles `nvngx_dlss.dll` beside the executable
//! (`cfg(ngx_sdk_bundled)`).

use std::ffi::c_void;
use std::ptr;

use concinnity_core::gfx::jitter;
use concinnity_core::render::dlss::DlssPreset;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::history_reset::UpscalerResetLatch;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::{UpscaleBackend, UpscaleDevice, UpscaleInputs, UpscaleOutput, UpscalerTarget};
use crate::upscale_sdk::dlss::{
    ENGINE_VERSION, NVSDK_NGX_ENGINE_TYPE_CUSTOM, NVSDK_NGX_FEATURE_SUPERSAMPLING,
    NVSDK_NGX_RESULT_FAIL, NVSDK_NGX_VERSION_API, P_COLOR, P_DEPTH, P_MOTION_VECTORS, P_OUTPUT,
    PROJECT_ID, app_data_path, create_with_preset_fallback, loaded_feature, ngx_succeeded,
    set_create_parameters, set_evaluate_parameters, supersampling_unavailable,
};
use crate::upscale_sdk::{UpscaleCamera, UpscaleExtent};

// NGX's D3D12 entry points, exported unmangled from the static library.
unsafe extern "C" {
    fn NVSDK_NGX_D3D12_Init_with_ProjectID(
        project_id: *const u8,
        engine_type: i32,
        engine_version: *const u8,
        app_data_path: *const u16,
        device: *mut c_void,
        feature_info: *const c_void,
        sdk_version: i32,
    ) -> u32;
    fn NVSDK_NGX_D3D12_Shutdown1(device: *mut c_void) -> u32;
    fn NVSDK_NGX_D3D12_GetCapabilityParameters(out_params: *mut *mut c_void) -> u32;
    fn NVSDK_NGX_D3D12_DestroyParameters(params: *mut c_void) -> u32;
    fn NVSDK_NGX_D3D12_CreateFeature(
        cmd: *mut c_void,
        feature_id: i32,
        params: *const c_void,
        out_handle: *mut *mut c_void,
    ) -> u32;
    fn NVSDK_NGX_D3D12_ReleaseFeature(handle: *mut c_void) -> u32;
    fn NVSDK_NGX_D3D12_EvaluateFeature_C(
        cmd: *mut c_void,
        handle: *const c_void,
        params: *const c_void,
        callback: *const c_void,
    ) -> u32;
    fn NVSDK_NGX_Parameter_SetD3d12Resource(params: *mut c_void, name: *const u8, res: *mut c_void);
}

// NGX initialized on `device`, with the parameter bag and super-sampling
// feature once they exist. Releases whatever it holds, then shuts NGX down, on
// drop.
struct NgxSession {
    device: ID3D12Device,
    params: *mut c_void,
    feature: *mut c_void,
}

impl Drop for NgxSession {
    fn drop(&mut self) {
        // SAFETY: the feature and the bag are each released at most once (only when non-null), and
        // the shutdown names the device NGX was initialized on, which this session still holds.
        unsafe {
            if !self.feature.is_null() {
                NVSDK_NGX_D3D12_ReleaseFeature(self.feature);
            }
            if !self.params.is_null() {
                NVSDK_NGX_D3D12_DestroyParameters(self.params);
            }
            NVSDK_NGX_D3D12_Shutdown1(self.device.as_raw());
        }
    }
}

impl NgxSession {
    // Create the super-sampling feature for `extent` running `preset`,
    // replacing any feature this session holds. `false` (logged) when NGX
    // refuses it. Creation records onto a command list, so this submits a
    // one-shot list and waits for it.
    fn create_feature(
        &mut self,
        gpu: UpscaleDevice<'_>,
        extent: UpscaleExtent,
        preset: DlssPreset,
    ) -> RenderResult<bool> {
        if !self.feature.is_null() {
            // SAFETY: the feature is the live one this session created, idle since its creating
            // submit was waited on, and released exactly once here.
            unsafe { NVSDK_NGX_D3D12_ReleaseFeature(self.feature) };
            self.feature = ptr::null_mut();
        }
        // SAFETY: `self.params` is the live bag NGX returned.
        unsafe { set_create_parameters(self.params, extent, preset) };
        let mut create_rc = NVSDK_NGX_RESULT_FAIL;
        crate::directx::texture::one_shot_submit(gpu.device, gpu.command_queue, |cmd| {
            // SAFETY: `cmd` is the one-shot list in the recording state, `self.params` the live
            // bag, and `self.feature` a live field the SDK fills.
            create_rc = unsafe {
                NVSDK_NGX_D3D12_CreateFeature(
                    cmd.as_raw(),
                    NVSDK_NGX_FEATURE_SUPERSAMPLING,
                    self.params,
                    &mut self.feature,
                )
            };
        })?;
        if !ngx_succeeded(create_rc) || self.feature.is_null() {
            // A refused create leaves no feature to release.
            self.feature = ptr::null_mut();
            tracing::warn!(
                "DLSS: CreateFeature (render preset {}) returned {create_rc:#x}",
                preset.label()
            );
            return Ok(false);
        }
        Ok(true)
    }
}

// The DLSS feature and the output texture it writes.
pub(super) struct DlssUpscaler {
    ngx: NgxSession,
    extent: UpscaleExtent,
    output: UpscaleOutput,
    reset: UpscalerResetLatch,
    // The render preset the feature runs, after any fallback.
    preset: DlssPreset,
}

// SAFETY: `DlssUpscaler` owns an NGX feature, its parameter bag and a COM device reference, none of
// which are shared: every entry point runs on the render thread that built them, under the same
// main-thread guard as the rest of `DxContext`. Moving the whole upscaler hands over exclusive
// ownership, so it is `Send` without being `Sync`.
unsafe impl Send for DlssUpscaler {}

impl DlssUpscaler {
    // `Ok(None)` when DLSS is unavailable: NGX failed to initialize, the GPU
    // or driver lacks DLSS, or the feature could not be created. A feature
    // library older than the requested render preset runs the default one.
    pub(super) fn try_new(target: UpscalerTarget<'_>) -> RenderResult<Option<Self>> {
        let UpscalerTarget {
            gpu,
            extent,
            descriptors,
            dlss_preset,
        } = target;
        let app_path = app_data_path();
        // SAFETY: an NGX entry point from the linked SDK. `PROJECT_ID`, `ENGINE_VERSION` and
        // `app_path` are NUL-terminated buffers live for the call, and the device is live.
        let rc = unsafe {
            NVSDK_NGX_D3D12_Init_with_ProjectID(
                PROJECT_ID.as_ptr(),
                NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                ENGINE_VERSION.as_ptr(),
                app_path.as_ptr(),
                gpu.device.as_raw(),
                ptr::null(),
                NVSDK_NGX_VERSION_API,
            )
        };
        if !ngx_succeeded(rc) {
            tracing::warn!(
                "DLSS: NVSDK_NGX_D3D12_Init returned {rc:#x} (NGX unavailable / not RTX). \
                 Trying the next backend."
            );
            return Ok(None);
        }
        let mut ngx = NgxSession {
            device: gpu.device.clone(),
            params: ptr::null_mut(),
            feature: ptr::null_mut(),
        };

        // SAFETY: NGX is initialized, and `ngx.params` is a live field that receives the bag.
        let rc = unsafe { NVSDK_NGX_D3D12_GetCapabilityParameters(&mut ngx.params) };
        if !ngx_succeeded(rc) || ngx.params.is_null() {
            tracing::warn!(
                "DLSS: GetCapabilityParameters returned {rc:#x}; trying the next backend"
            );
            return Ok(None);
        }
        // SAFETY: `ngx.params` is the non-null bag NGX just returned.
        if let Some(reason) = unsafe { supersampling_unavailable(ngx.params) } {
            tracing::warn!("DLSS: {reason}; trying the next backend");
            return Ok(None);
        }
        let Some(created) = create_with_preset_fallback(
            "DLSS",
            dlss_preset,
            |preset| ngx.create_feature(gpu, extent, preset),
            loaded_feature,
        )?
        else {
            return Ok(None);
        };

        let output = UpscaleOutput::create(gpu.device, extent.output, descriptors)?;
        tracing::info!(
            "DLSS: feature created: {extent}, {}, render preset {}",
            created.library,
            created.preset.label()
        );
        Ok(Some(Self {
            ngx,
            extent,
            output,
            reset: UpscalerResetLatch::default(),
            preset: created.preset,
        }))
    }
}

impl UpscaleBackend for DlssUpscaler {
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

    fn dispatch(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        inputs: UpscaleInputs<'_>,
        camera: UpscaleCamera,
    ) -> RenderResult<()> {
        let params = self.ngx.params;
        let resources = [
            (P_COLOR, inputs.color),
            (P_OUTPUT, self.output.resource()),
            (P_DEPTH, inputs.depth),
            (P_MOTION_VECTORS, inputs.motion_vectors),
        ];
        // SAFETY: `params` is the live bag, every name is a NUL-terminated constant, and each
        // resource is a COM object the frame keeps alive until the list executes.
        unsafe {
            for (name, resource) in resources {
                NVSDK_NGX_Parameter_SetD3d12Resource(params, name.as_ptr(), resource.as_raw());
            }
            set_evaluate_parameters(
                params,
                camera.jitter_offset,
                crate::upscale_reset::consume(&self.reset),
                self.extent,
            );
        }
        // SAFETY: `cmd` is recording, and the feature and bag are the live ones created in
        // `try_new`.
        let rc = unsafe {
            NVSDK_NGX_D3D12_EvaluateFeature_C(cmd.as_raw(), self.ngx.feature, params, ptr::null())
        };
        if !ngx_succeeded(rc) {
            return Err(RenderError::Other(format!(
                "NVSDK_NGX_D3D12_EvaluateFeature returned {rc:#x}"
            )));
        }
        Ok(())
    }

    fn request_history_reset(&self) {
        self.reset.request();
    }

    fn dlss_preset(&self) -> Option<DlssPreset> {
        Some(self.preset)
    }
}
