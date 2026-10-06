//! NVIDIA DLSS through the raw NGX API. Both backends drive the same feature
//! through the same parameter bag: its result codes, names and values, the
//! engine identity, and the API-independent `NVSDK_NGX_Parameter_*` setters,
//! which the NGX static library exports once for every API. A backend adds its
//! own init, feature and evaluate entry points and binds its resources.
//!
//! Values match `nvsdk_ngx_defs.h` of the NGX SDK at API 1.5.0.

use std::ffi::c_void;

use concinnity_core::components::UpscaleQuality;
use concinnity_core::render::depth::{CAMERA_DEPTH, DepthMapping};

use super::UpscaleExtent;

/// `NVSDK_NGX_Result_Fail`: the high bits every failure code carries.
pub(crate) const NVSDK_NGX_RESULT_FAIL: u32 = 0xBAD0_0000;
/// `NVSDK_NGX_Version_API`, 1.5.0.
pub(crate) const NVSDK_NGX_VERSION_API: i32 = 0x0000_0015;
pub(crate) const NVSDK_NGX_ENGINE_TYPE_CUSTOM: i32 = 0;
pub(crate) const NVSDK_NGX_FEATURE_SUPERSAMPLING: i32 = 1;

// NVSDK_NGX_PerfQuality_Value.
const PERF_MAX_PERF: i32 = 0;
const PERF_BALANCED: i32 = 1;
const PERF_MAX_QUALITY: i32 = 2;
const PERF_ULTRA_PERFORMANCE: i32 = 3;
const PERF_DLAA: i32 = 5;

// NVSDK_NGX_DLSS_Feature_Flags.
const DLSS_FLAG_IS_HDR: i32 = 1 << 0;
const DLSS_FLAG_DEPTH_INVERTED: i32 = 1 << 3;
const DLSS_FLAG_AUTO_EXPOSURE: i32 = 1 << 6;

// NVSDK_NGX_Parameter names (NUL-terminated).
const P_WIDTH: &[u8] = b"Width\0";
const P_HEIGHT: &[u8] = b"Height\0";
const P_OUT_WIDTH: &[u8] = b"OutWidth\0";
const P_OUT_HEIGHT: &[u8] = b"OutHeight\0";
const P_PERF_QUALITY: &[u8] = b"PerfQualityValue\0";
const P_CREATE_FLAGS: &[u8] = b"DLSS.Feature.Create.Flags\0";
const P_ENABLE_OUTPUT_SUBRECTS: &[u8] = b"DLSS.Enable.Output.Subrects\0";
const P_CREATION_NODE_MASK: &[u8] = b"CreationNodeMask\0";
const P_VISIBILITY_NODE_MASK: &[u8] = b"VisibilityNodeMask\0";
const P_SUPERSAMPLING_AVAILABLE: &[u8] = b"SuperSampling.Available\0";
pub(crate) const P_COLOR: &[u8] = b"Color\0";
pub(crate) const P_OUTPUT: &[u8] = b"Output\0";
pub(crate) const P_DEPTH: &[u8] = b"Depth\0";
pub(crate) const P_MOTION_VECTORS: &[u8] = b"MotionVectors\0";
#[cfg(backend_vk)]
pub(crate) const P_EXPOSURE_TEXTURE: &[u8] = b"ExposureTexture\0";
const P_JITTER_X: &[u8] = b"Jitter.Offset.X\0";
const P_JITTER_Y: &[u8] = b"Jitter.Offset.Y\0";
const P_MV_SCALE_X: &[u8] = b"MV.Scale.X\0";
const P_MV_SCALE_Y: &[u8] = b"MV.Scale.Y\0";
const P_RESET: &[u8] = b"Reset\0";
const P_SUBRECT_WIDTH: &[u8] = b"DLSS.Render.Subrect.Dimensions.Width\0";
const P_SUBRECT_HEIGHT: &[u8] = b"DLSS.Render.Subrect.Dimensions.Height\0";
const P_SHARPNESS: &[u8] = b"Sharpness\0";

/// The engine's identity for NGX. A GUID-like project id avoids needing an
/// NVIDIA-assigned application id.
pub(crate) const PROJECT_ID: &[u8] = b"5f2e1a64-9c3b-4d7e-8a1f-2b6c0d9e7f30\0";
pub(crate) const ENGINE_VERSION: &[u8] = b"1.0.0\0";

unsafe extern "C" {
    fn NVSDK_NGX_Parameter_SetUI(params: *mut c_void, name: *const u8, value: u32);
    fn NVSDK_NGX_Parameter_SetI(params: *mut c_void, name: *const u8, value: i32);
    fn NVSDK_NGX_Parameter_SetF(params: *mut c_void, name: *const u8, value: f32);
    fn NVSDK_NGX_Parameter_GetUI(params: *mut c_void, name: *const u8, out: *mut u32) -> u32;
}

/// `NVSDK_NGX_SUCCEED`: any code without the failure bits.
pub(crate) fn ngx_succeeded(result: u32) -> bool {
    (result & 0xFFF0_0000) != NVSDK_NGX_RESULT_FAIL
}

/// The NUL-terminated UTF-16 path NGX writes its logs and data under: the
/// working directory.
pub(crate) fn app_data_path() -> Vec<u16> {
    ".".encode_utf16().chain(std::iter::once(0)).collect()
}

// The DLSS preset nearest the engine preset: DLAA at native resolution.
fn dlss_perf_quality(quality: Option<UpscaleQuality>) -> i32 {
    match quality {
        None => PERF_DLAA,
        Some(UpscaleQuality::Quality) => PERF_MAX_QUALITY,
        Some(UpscaleQuality::Balanced) => PERF_BALANCED,
        Some(UpscaleQuality::Performance) => PERF_MAX_PERF,
        Some(UpscaleQuality::UltraPerformance) => PERF_ULTRA_PERFORMANCE,
    }
}

// HDR linear input with render-resolution motion vectors, DLSS's own
// auto-exposure unless an exposure texture is bound, and inverted depth when
// the near plane is device depth 1.
const fn dlss_create_flags(depth: DepthMapping, exposure_texture: bool) -> i32 {
    let mut flags = DLSS_FLAG_IS_HDR;
    if !exposure_texture {
        flags |= DLSS_FLAG_AUTO_EXPOSURE;
    }
    if depth.reversed {
        flags |= DLSS_FLAG_DEPTH_INVERTED;
    }
    flags
}

/// Whether the capability bag reports DLSS super sampling on this GPU and
/// driver.
///
/// # Safety
///
/// `params` must be the live parameter bag NGX returned.
pub(crate) unsafe fn supersampling_available(params: *mut c_void) -> bool {
    let mut available: u32 = 0;
    // SAFETY: the caller guarantees the bag, the name is a NUL-terminated constant, and
    // `available` is a live local.
    let rc = unsafe {
        NVSDK_NGX_Parameter_GetUI(params, P_SUPERSAMPLING_AVAILABLE.as_ptr(), &mut available)
    };
    ngx_succeeded(rc) && available != 0
}

/// Fill the parameters the super-sampling feature is created from.
/// `exposure_texture` is whether the backend binds an exposure texture with
/// every evaluate; without one DLSS computes its own exposure.
///
/// # Safety
///
/// `params` must be the live parameter bag NGX returned.
pub(crate) unsafe fn set_create_parameters(
    params: *mut c_void,
    extent: UpscaleExtent,
    exposure_texture: bool,
) {
    let ((rw, rh), (ow, oh)) = (extent.render, extent.output);
    let quality = dlss_perf_quality(UpscaleQuality::nearest(extent.scale));
    let flags = dlss_create_flags(CAMERA_DEPTH, exposure_texture);
    // SAFETY: the caller guarantees the bag, and every name is a NUL-terminated constant.
    unsafe {
        NVSDK_NGX_Parameter_SetUI(params, P_WIDTH.as_ptr(), rw);
        NVSDK_NGX_Parameter_SetUI(params, P_HEIGHT.as_ptr(), rh);
        NVSDK_NGX_Parameter_SetUI(params, P_OUT_WIDTH.as_ptr(), ow);
        NVSDK_NGX_Parameter_SetUI(params, P_OUT_HEIGHT.as_ptr(), oh);
        NVSDK_NGX_Parameter_SetI(params, P_PERF_QUALITY.as_ptr(), quality);
        NVSDK_NGX_Parameter_SetI(params, P_CREATE_FLAGS.as_ptr(), flags);
        NVSDK_NGX_Parameter_SetI(params, P_ENABLE_OUTPUT_SUBRECTS.as_ptr(), 0);
        NVSDK_NGX_Parameter_SetUI(params, P_CREATION_NODE_MASK.as_ptr(), 1);
        NVSDK_NGX_Parameter_SetUI(params, P_VISIBILITY_NODE_MASK.as_ptr(), 1);
    }
}

/// Fill the per-frame evaluate parameters other than the resources: the
/// jitter (render pixels), the motion-vector scale, the history reset and the
/// render subrect.
///
/// # Safety
///
/// `params` must be the live parameter bag the feature was created from.
pub(crate) unsafe fn set_evaluate_parameters(
    params: *mut c_void,
    jitter: [f32; 2],
    reset: bool,
    extent: UpscaleExtent,
) {
    let (rw, rh) = extent.render;
    // SAFETY: the caller guarantees the bag, and every name is a NUL-terminated constant.
    unsafe {
        NVSDK_NGX_Parameter_SetF(params, P_JITTER_X.as_ptr(), jitter[0]);
        NVSDK_NGX_Parameter_SetF(params, P_JITTER_Y.as_ptr(), jitter[1]);
        // Motion vectors are `prev_uv - cur_uv` in UV space; DLSS takes them
        // in render pixels.
        NVSDK_NGX_Parameter_SetF(params, P_MV_SCALE_X.as_ptr(), rw as f32);
        NVSDK_NGX_Parameter_SetF(params, P_MV_SCALE_Y.as_ptr(), rh as f32);
        NVSDK_NGX_Parameter_SetI(params, P_RESET.as_ptr(), i32::from(reset));
        NVSDK_NGX_Parameter_SetUI(params, P_SUBRECT_WIDTH.as_ptr(), rw);
        NVSDK_NGX_Parameter_SetUI(params, P_SUBRECT_HEIGHT.as_ptr(), rh);
        NVSDK_NGX_Parameter_SetF(params, P_SHARPNESS.as_ptr(), 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngx_constants_match_sdk() {
        assert!(ngx_succeeded(0x1));
        assert!(!ngx_succeeded(0xBAD0_0005));
        assert_eq!(NVSDK_NGX_VERSION_API, 0x0000_0015);
        assert_eq!(NVSDK_NGX_FEATURE_SUPERSAMPLING, 1);
        assert_eq!(PERF_MAX_PERF, 0);
        assert_eq!(PERF_BALANCED, 1);
        assert_eq!(PERF_MAX_QUALITY, 2);
        assert_eq!(PERF_ULTRA_PERFORMANCE, 3);
        assert_eq!(PERF_DLAA, 5);
        assert_eq!(DLSS_FLAG_IS_HDR, 1);
        assert_eq!(DLSS_FLAG_DEPTH_INVERTED, 8);
        assert_eq!(DLSS_FLAG_AUTO_EXPOSURE, 64);
    }

    #[test]
    fn perf_quality_follows_the_nearest_engine_preset() {
        assert_eq!(dlss_perf_quality(None), PERF_DLAA);
        assert_eq!(
            dlss_perf_quality(Some(UpscaleQuality::Quality)),
            PERF_MAX_QUALITY
        );
        assert_eq!(
            dlss_perf_quality(Some(UpscaleQuality::Balanced)),
            PERF_BALANCED
        );
        assert_eq!(
            dlss_perf_quality(Some(UpscaleQuality::Performance)),
            PERF_MAX_PERF
        );
        assert_eq!(
            dlss_perf_quality(Some(UpscaleQuality::UltraPerformance)),
            PERF_ULTRA_PERFORMANCE
        );
    }

    // DepthInverted is set exactly when the depth is reversed, which the
    // camera's is, and AutoExposure exactly when no exposure texture is bound.
    #[test]
    fn the_create_flags_follow_the_depth_and_exposure() {
        for exposure_texture in [false, true] {
            for reversed in [false, true] {
                for infinite in [false, true] {
                    let depth = DepthMapping { reversed, infinite };
                    let flags = dlss_create_flags(depth, exposure_texture);
                    assert_eq!((flags & DLSS_FLAG_DEPTH_INVERTED) != 0, reversed);
                    assert_eq!((flags & DLSS_FLAG_AUTO_EXPOSURE) != 0, !exposure_texture);
                    assert_ne!(flags & DLSS_FLAG_IS_HDR, 0);
                }
            }
        }
        assert_ne!(
            dlss_create_flags(CAMERA_DEPTH, true) & DLSS_FLAG_DEPTH_INVERTED,
            0
        );
    }

    #[test]
    fn the_app_data_path_is_nul_terminated() {
        assert_eq!(app_data_path(), [u16::from(b'.'), 0]);
    }
}
