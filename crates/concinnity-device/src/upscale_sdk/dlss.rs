//! NVIDIA DLSS through the raw NGX API. Both backends drive the same feature
//! through the same parameter bag: its result codes, names and values, the
//! engine identity, and the API-independent `NVSDK_NGX_Parameter_*` setters,
//! which the NGX static library exports once for every API. A backend adds its
//! own init, feature and evaluate entry points and binds its resources.
//!
//! Values match `nvsdk_ngx_defs.h` of the NGX SDK at API 1.5.0, as shipped in
//! Streamline 2.14.1 (DLSS 310.9.1). The build links the SDK only when its
//! headers declare that API version. At load, a driver NGX reports as too old
//! fails over to the next upscaler, and a feature library older than the
//! requested render preset runs the default preset instead.

use std::ffi::c_void;
use std::fmt;

use concinnity_core::components::UpscaleQuality;
use concinnity_core::render::depth::{CAMERA_DEPTH, DepthMapping};
use concinnity_core::render::dlss::{DlssPreset, NGX_API_VERSION};
use concinnity_core::render::error::RenderResult;

use super::UpscaleExtent;

/// `NVSDK_NGX_Result_Fail`: the high bits every failure code carries.
pub(crate) const NVSDK_NGX_RESULT_FAIL: u32 = 0xBAD0_0000;
/// `NVSDK_NGX_Version_API`, the version the build checked the SDK headers for.
pub(crate) const NVSDK_NGX_VERSION_API: i32 = NGX_API_VERSION as i32;
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

// NVSDK_NGX_DLSS_Hint_Render_Preset.
const PRESET_DEFAULT: u32 = 0;
const PRESET_K: u32 = 11;
const PRESET_L: u32 = 12;
const PRESET_M: u32 = 13;

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
const P_NEEDS_UPDATED_DRIVER: &[u8] = b"SuperSampling.NeedsUpdatedDriver\0";
const P_MIN_DRIVER_MAJOR: &[u8] = b"SuperSampling.MinDriverVersionMajor\0";
const P_MIN_DRIVER_MINOR: &[u8] = b"SuperSampling.MinDriverVersionMinor\0";
// One render-preset hint per quality mode.
const P_PRESETS: [&[u8]; 6] = [
    b"DLSS.Hint.Render.Preset.DLAA\0",
    b"DLSS.Hint.Render.Preset.Quality\0",
    b"DLSS.Hint.Render.Preset.Balanced\0",
    b"DLSS.Hint.Render.Preset.Performance\0",
    b"DLSS.Hint.Render.Preset.UltraPerformance\0",
    b"DLSS.Hint.Render.Preset.UltraQuality\0",
];
pub(crate) const P_COLOR: &[u8] = b"Color\0";
pub(crate) const P_OUTPUT: &[u8] = b"Output\0";
pub(crate) const P_DEPTH: &[u8] = b"Depth\0";
pub(crate) const P_MOTION_VECTORS: &[u8] = b"MotionVectors\0";
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

// The feature library NGX loads for super sampling.
const FEATURE_LIBRARY: &str = "nvngx_dlss.dll";

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

// The render preset every quality mode runs for `preset`.
fn render_preset(preset: DlssPreset) -> u32 {
    match preset {
        DlssPreset::Default => PRESET_DEFAULT,
        DlssPreset::K => PRESET_K,
        DlssPreset::L => PRESET_L,
        DlssPreset::M => PRESET_M,
    }
}

// The oldest feature library that runs `preset`, per the revision history of
// the DLSS Programming Guide: K arrived in 310.2.0, L and M in 310.5.0.
const fn minimum_version(preset: DlssPreset) -> Option<DlssVersion> {
    match preset {
        DlssPreset::Default => None,
        DlssPreset::K => Some(DlssVersion::new(310, 2, 0, 0)),
        DlssPreset::L | DlssPreset::M => Some(DlssVersion::new(310, 5, 0, 0)),
    }
}

// HDR linear input with render-resolution motion vectors, DLSS's own
// auto-exposure (the scene is un-exposed until after the upscale), and
// inverted depth when the near plane is device depth 1.
const fn dlss_create_flags(depth: DepthMapping) -> i32 {
    let mut flags = DLSS_FLAG_IS_HDR | DLSS_FLAG_AUTO_EXPOSURE;
    if depth.reversed {
        flags |= DLSS_FLAG_DEPTH_INVERTED;
    }
    flags
}

// Read one unsigned value from the bag, `None` when NGX does not report it.
unsafe fn get_ui(params: *mut c_void, name: &[u8]) -> Option<u32> {
    let mut value: u32 = 0;
    // SAFETY: the caller guarantees the bag, the name is a NUL-terminated constant, and `value`
    // is a live local.
    let rc = unsafe { NVSDK_NGX_Parameter_GetUI(params, name.as_ptr(), &mut value) };
    ngx_succeeded(rc).then_some(value)
}

/// Why DLSS super sampling cannot run on this GPU and driver, or `None` when
/// it can.
///
/// # Safety
///
/// `params` must be the live capability bag NGX returned.
pub(crate) unsafe fn supersampling_unavailable(params: *mut c_void) -> Option<String> {
    // SAFETY: the caller guarantees the bag.
    let (needs_driver, major, minor, available) = unsafe {
        (
            get_ui(params, P_NEEDS_UPDATED_DRIVER),
            get_ui(params, P_MIN_DRIVER_MAJOR),
            get_ui(params, P_MIN_DRIVER_MINOR),
            get_ui(params, P_SUPERSAMPLING_AVAILABLE),
        )
    };
    unavailable_reason(
        needs_driver.unwrap_or(0) != 0,
        major.zip(minor),
        available.unwrap_or(0) != 0,
    )
}

fn unavailable_reason(
    needs_driver: bool,
    min_driver: Option<(u32, u32)>,
    available: bool,
) -> Option<String> {
    if needs_driver {
        return Some(match min_driver {
            Some((major, minor)) => format!("super sampling needs driver {major}.{minor} or newer"),
            None => "super sampling needs a newer driver".to_string(),
        });
    }
    (!available).then(|| "super sampling is not available on this GPU".to_string())
}

/// Fill the parameters the super-sampling feature is created from, including
/// `preset` for every quality mode.
///
/// # Safety
///
/// `params` must be the live parameter bag NGX returned.
pub(crate) unsafe fn set_create_parameters(
    params: *mut c_void,
    extent: UpscaleExtent,
    preset: DlssPreset,
) {
    let ((rw, rh), (ow, oh)) = (extent.render, extent.output);
    let quality = dlss_perf_quality(UpscaleQuality::nearest(extent.scale));
    let flags = dlss_create_flags(CAMERA_DEPTH);
    let preset = render_preset(preset);
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
        for name in P_PRESETS {
            NVSDK_NGX_Parameter_SetUI(params, name.as_ptr(), preset);
        }
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

/// A Windows file version, `major.minor.patch.build`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DlssVersion([u16; 4]);

impl DlssVersion {
    const fn new(major: u16, minor: u16, patch: u16, build: u16) -> Self {
        Self([major, minor, patch, build])
    }

    // From `VS_FIXEDFILEINFO`'s most and least significant halves.
    const fn from_file_version(ms: u32, ls: u32) -> Self {
        Self::new((ms >> 16) as u16, ms as u16, (ls >> 16) as u16, ls as u16)
    }
}

impl fmt::Display for DlssVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [major, minor, patch, build] = self.0;
        write!(f, "{major}.{minor}.{patch}.{build}")
    }
}

/// The super-sampling feature library NGX loaded, for the creation log.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LoadedFeature {
    version: Option<DlssVersion>,
}

impl fmt::Display for LoadedFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.version {
            Some(version) => write!(f, "{FEATURE_LIBRARY} {version}"),
            // The driver loaded its own update under another file name.
            None => write!(f, "{FEATURE_LIBRARY} version unknown"),
        }
    }
}

/// The feature library NGX loaded for the feature just created. Call after
/// feature creation, which is when NGX has loaded the library.
pub(crate) fn loaded_feature() -> LoadedFeature {
    LoadedFeature {
        version: loaded_feature_version(),
    }
}

impl LoadedFeature {
    /// Why this library cannot run `preset`, or `None` when it can. A library
    /// older than a preset reads its hint as another preset, or none; an
    /// unknown version is the driver's own update, which is newer than any
    /// preset here.
    pub(crate) fn preset_unsupported(&self, preset: DlssPreset) -> Option<String> {
        let (version, minimum) = (self.version?, minimum_version(preset)?);
        (version < minimum).then(|| {
            format!(
                "{FEATURE_LIBRARY} {version} predates render preset {} ({minimum} or newer)",
                preset.label()
            )
        })
    }
}

/// The preset a created feature runs, and the library NGX loaded for it.
pub(crate) struct CreatedFeature {
    pub(crate) preset: DlssPreset,
    pub(crate) library: LoadedFeature,
}

/// Create the feature for `preset` through `create`, which reports whether
/// NGX accepted it, falling back once to the default preset when NGX refuses
/// the requested one or the library it loaded (known only after a creation)
/// predates it. `None` when NGX refuses the default too. `label` names the
/// backend in the warnings.
pub(crate) fn create_with_preset_fallback(
    label: &str,
    preset: DlssPreset,
    mut create: impl FnMut(DlssPreset) -> RenderResult<bool>,
    loaded: impl FnOnce() -> LoadedFeature,
) -> RenderResult<Option<CreatedFeature>> {
    if !create(preset)? {
        if preset == DlssPreset::Default {
            return Ok(None);
        }
        tracing::warn!(
            "{label}: render preset {} refused; retrying with the default preset",
            preset.label()
        );
        if !create(DlssPreset::Default)? {
            return Ok(None);
        }
        return Ok(Some(CreatedFeature {
            preset: DlssPreset::Default,
            library: loaded(),
        }));
    }
    let library = loaded();
    let Some(reason) = library.preset_unsupported(preset) else {
        return Ok(Some(CreatedFeature { preset, library }));
    };
    tracing::warn!("{label}: {reason}; recreating with the default preset");
    if !create(DlssPreset::Default)? {
        return Ok(None);
    }
    Ok(Some(CreatedFeature {
        preset: DlssPreset::Default,
        library,
    }))
}

// The file version of the loaded feature library, read from its version
// resource. `None` when no module of that name is loaded, which is the case
// when NGX runs the driver's own update of the library instead.
fn loaded_feature_version() -> Option<DlssVersion> {
    use windows::Win32::Storage::FileSystem::{
        GetFileVersionInfoSizeW, GetFileVersionInfoW, VS_FIXEDFILEINFO, VerQueryValueW,
    };
    use windows::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
    use windows::core::{PCWSTR, w};

    let name: Vec<u16> = FEATURE_LIBRARY.encode_utf16().chain([0]).collect();
    // SAFETY: `name` is NUL-terminated and live for the call; the handle is only looked up, not
    // reference counted, and NGX keeps the module loaded while its feature lives.
    let module = unsafe { GetModuleHandleW(PCWSTR(name.as_ptr())) }.ok()?;
    let mut path = [0u16; 1024];
    // SAFETY: `module` is a loaded module and `path` a live buffer of the length passed.
    let len = unsafe { GetModuleFileNameW(Some(module), &mut path) } as usize;
    if len == 0 || len >= path.len() {
        return None;
    }
    let path = PCWSTR(path.as_ptr());
    // SAFETY: `path` is the NUL-terminated path just written.
    let size = unsafe { GetFileVersionInfoSizeW(path, None) };
    if size == 0 {
        return None;
    }
    let mut block = vec![0u8; size as usize];
    // SAFETY: `block` is a live buffer of exactly `size` bytes.
    unsafe { GetFileVersionInfoW(path, None, size, block.as_mut_ptr().cast()) }.ok()?;
    let mut fixed: *mut c_void = std::ptr::null_mut();
    let mut fixed_len: u32 = 0;
    // SAFETY: `block` holds the version resource just read, and `\` names its root, whose value
    // is a pointer into `block`.
    let found =
        unsafe { VerQueryValueW(block.as_ptr().cast(), w!("\\"), &mut fixed, &mut fixed_len) };
    if !found.as_bool() || fixed.is_null() || (fixed_len as usize) < size_of::<VS_FIXEDFILEINFO>() {
        return None;
    }
    // SAFETY: `fixed` points at a `VS_FIXEDFILEINFO` of the length checked above, inside
    // `block`, which is still alive; the byte buffer promises no alignment, so it is read
    // unaligned.
    let info = unsafe { std::ptr::read_unaligned(fixed.cast::<VS_FIXEDFILEINFO>()) };
    Some(DlssVersion::from_file_version(
        info.dwFileVersionMS,
        info.dwFileVersionLS,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngx_constants_match_sdk() {
        assert!(ngx_succeeded(0x1));
        assert!(!ngx_succeeded(0xBAD0_0005));
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
    fn render_preset_values_match_sdk() {
        assert_eq!(PRESET_DEFAULT, 0);
        assert_eq!(PRESET_K, 11);
        assert_eq!(PRESET_L, 12);
        assert_eq!(PRESET_M, 13);
    }

    #[test]
    fn every_preset_maps_to_its_value_and_the_default_to_ngx_default() {
        assert_eq!(render_preset(DlssPreset::Default), PRESET_DEFAULT);
        assert_eq!(render_preset(DlssPreset::K), PRESET_K);
        assert_eq!(render_preset(DlssPreset::L), PRESET_L);
        assert_eq!(render_preset(DlssPreset::M), PRESET_M);
    }

    // The hint is set for every quality mode, so the preset holds whichever
    // mode the render scale lands on.
    #[test]
    fn a_preset_hint_is_named_for_every_quality_mode() {
        let names: Vec<&str> = P_PRESETS
            .iter()
            .map(|n| std::str::from_utf8(&n[..n.len() - 1]).unwrap())
            .collect();
        for mode in [
            "DLAA",
            "Quality",
            "Balanced",
            "Performance",
            "UltraPerformance",
            "UltraQuality",
        ] {
            assert!(
                names.contains(&format!("DLSS.Hint.Render.Preset.{mode}").as_str()),
                "{mode}"
            );
        }
        assert!(P_PRESETS.iter().all(|n| n.ends_with(b"\0")));
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
    // camera's is; DLSS always computes its own exposure.
    #[test]
    fn the_create_flags_follow_the_depth_and_always_auto_expose() {
        for reversed in [false, true] {
            for infinite in [false, true] {
                let flags = dlss_create_flags(DepthMapping { reversed, infinite });
                assert_eq!((flags & DLSS_FLAG_DEPTH_INVERTED) != 0, reversed);
                assert_ne!(flags & DLSS_FLAG_AUTO_EXPOSURE, 0);
                assert_ne!(flags & DLSS_FLAG_IS_HDR, 0);
            }
        }
        assert_ne!(
            dlss_create_flags(CAMERA_DEPTH) & DLSS_FLAG_DEPTH_INVERTED,
            0
        );
    }

    #[test]
    fn an_outdated_driver_is_named_before_availability() {
        assert_eq!(unavailable_reason(false, None, true), None);
        assert_eq!(
            unavailable_reason(true, Some((580, 88)), true).as_deref(),
            Some("super sampling needs driver 580.88 or newer")
        );
        assert_eq!(
            unavailable_reason(true, None, false).as_deref(),
            Some("super sampling needs a newer driver")
        );
        assert_eq!(
            unavailable_reason(false, Some((1, 0)), false).as_deref(),
            Some("super sampling is not available on this GPU")
        );
    }

    #[test]
    fn file_versions_read_and_order_like_their_fields() {
        let bundled = DlssVersion::from_file_version(310 << 16 | 9, 1 << 16);
        assert_eq!(bundled, DlssVersion::new(310, 9, 1, 0));
        assert_eq!(bundled.to_string(), "310.9.1.0");
        assert!(DlssVersion::new(310, 4, 9, 9) < DlssVersion::new(310, 5, 0, 0));
        assert!(DlssVersion::new(311, 0, 0, 0) > bundled);
    }

    // Each preset has its own floor, and a library that is too old names the
    // floor it missed; the default runs anywhere.
    #[test]
    fn a_library_runs_each_preset_from_that_presets_first_version() {
        let at = |major, minor, patch| LoadedFeature {
            version: Some(DlssVersion::new(major, minor, patch, 0)),
        };
        assert_eq!(at(310, 1, 0).preset_unsupported(DlssPreset::Default), None);
        for preset in DlssPreset::ALL {
            assert_eq!(at(310, 9, 1).preset_unsupported(preset), None, "{preset:?}");
            let unknown = LoadedFeature { version: None };
            assert_eq!(unknown.preset_unsupported(preset), None, "{preset:?}");
        }
        assert_eq!(at(310, 2, 0).preset_unsupported(DlssPreset::K), None);
        assert!(at(310, 1, 9).preset_unsupported(DlssPreset::K).is_some());
        assert_eq!(at(310, 5, 0).preset_unsupported(DlssPreset::M), None);
        let err = at(310, 4, 0).preset_unsupported(DlssPreset::L).unwrap();
        assert_eq!(
            err,
            "nvngx_dlss.dll 310.4.0.0 predates render preset L (310.5.0.0 or newer)"
        );
    }

    // Runs `create_with_preset_fallback` against a fake NGX that accepts the
    // presets in `accepts` and loads a library of `version`, returning the
    // presets it was asked to create and what ran.
    fn fallback(
        preset: DlssPreset,
        accepts: &[DlssPreset],
        version: Option<DlssVersion>,
    ) -> (Vec<DlssPreset>, Option<DlssPreset>) {
        let mut tried = Vec::new();
        let created = create_with_preset_fallback(
            "test",
            preset,
            |p| {
                tried.push(p);
                Ok(accepts.contains(&p))
            },
            || LoadedFeature { version },
        )
        .unwrap();
        (tried, created.map(|c| c.preset))
    }

    #[test]
    fn an_accepted_supported_preset_is_created_once() {
        let current = Some(DlssVersion::new(310, 9, 1, 0));
        assert_eq!(
            fallback(DlssPreset::M, &DlssPreset::ALL, current),
            (vec![DlssPreset::M], Some(DlssPreset::M))
        );
        assert_eq!(
            fallback(DlssPreset::Default, &DlssPreset::ALL, current),
            (vec![DlssPreset::Default], Some(DlssPreset::Default))
        );
    }

    #[test]
    fn a_refused_preset_retries_once_with_the_default() {
        let current = Some(DlssVersion::new(310, 9, 1, 0));
        assert_eq!(
            fallback(DlssPreset::L, &[DlssPreset::Default], current),
            (
                vec![DlssPreset::L, DlssPreset::Default],
                Some(DlssPreset::Default)
            )
        );
        assert_eq!(
            fallback(DlssPreset::L, &[], current),
            (vec![DlssPreset::L, DlssPreset::Default], None)
        );
        assert_eq!(
            fallback(DlssPreset::Default, &[], current),
            (vec![DlssPreset::Default], None)
        );
    }

    #[test]
    fn a_library_older_than_the_preset_recreates_with_the_default() {
        let old = Some(DlssVersion::new(310, 3, 0, 0));
        assert_eq!(
            fallback(DlssPreset::M, &DlssPreset::ALL, old),
            (
                vec![DlssPreset::M, DlssPreset::Default],
                Some(DlssPreset::Default)
            )
        );
        assert_eq!(
            fallback(DlssPreset::K, &DlssPreset::ALL, old),
            (vec![DlssPreset::K], Some(DlssPreset::K))
        );
        assert_eq!(
            fallback(DlssPreset::M, &[DlssPreset::M], old),
            (vec![DlssPreset::M, DlssPreset::Default], None)
        );
    }

    #[test]
    fn the_api_version_is_the_one_the_build_checks_for() {
        assert_eq!(NVSDK_NGX_VERSION_API, 0x15);
        assert_eq!(NVSDK_NGX_VERSION_API as u32, NGX_API_VERSION);
    }

    #[test]
    fn the_loaded_feature_names_its_version_or_says_it_is_unknown() {
        let known = LoadedFeature {
            version: Some(DlssVersion::new(310, 9, 1, 0)),
        };
        assert_eq!(known.to_string(), "nvngx_dlss.dll 310.9.1.0");
        let unknown = LoadedFeature { version: None };
        assert_eq!(unknown.to_string(), "nvngx_dlss.dll version unknown");
    }

    #[test]
    fn the_app_data_path_is_nul_terminated() {
        assert_eq!(app_data_path(), [u16::from(b'.'), 0]);
    }
}
