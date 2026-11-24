//! NVIDIA DLSS super resolution through Streamline's `sl.dlss` plugin. Both
//! backends drive it the same way: one viewport per upscaler, its options and
//! the frame's camera constants set before each evaluate, and the four inputs
//! tagged on the evaluate itself. A backend only describes its resources and
//! supplies the command list.
//!
//! The plugin creates the NGX feature on the first evaluate and again whenever
//! the mode, output size or preset changes. A feature library older than the
//! requested render preset runs the default preset instead, and a preset NGX
//! refuses falls back to the default once.

use std::cell::Cell;
use std::ffi::c_void;
use std::fmt;

use concinnity_engine::components::UpscaleQuality;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::depth::{CAMERA_DEPTH, camera_projection};
use concinnity_engine::render::output::dlss::DlssPreset;
use concinnity_engine::transform::{
    IDENTITY, Mat4, mat4_affine_inverse, mat4_inverse, mat4_mul, view_basis,
};

use super::UpscaleCamera;
use super::UpscaleExtent;
use super::streamline::abi::{
    self, AdapterInfo, BUFFER_DEPTH, BUFFER_MOTION_VECTORS, BUFFER_SCALING_INPUT_COLOR,
    BUFFER_SCALING_OUTPUT_COLOR, BaseStructure, Constants, DlssOptions, Extent, FEATURE_DLSS,
    Float2, Float4x4, FrameToken, PfnSlDlssSetOptions, Resource, ResourceTag, SlResult, Version,
    ViewportHandle,
};
use super::streamline::{Streamline, next_viewport};

/// The feature library NGX loads for super resolution.
const FEATURE_LIBRARY: &str = "nvngx_dlss.dll";

/// The far distance reported for the camera, whose projection has no far
/// plane.
const INFINITE_FAR: f32 = 1.0e30;

// The DLSS mode nearest the engine preset: DLAA at native resolution.
fn dlss_mode(quality: UpscaleQuality) -> u32 {
    match quality {
        UpscaleQuality::Native => abi::DLSS_MODE_DLAA,
        UpscaleQuality::Quality => abi::DLSS_MODE_MAX_QUALITY,
        UpscaleQuality::Balanced => abi::DLSS_MODE_BALANCED,
        UpscaleQuality::Performance => abi::DLSS_MODE_MAX_PERFORMANCE,
        UpscaleQuality::UltraPerformance => abi::DLSS_MODE_ULTRA_PERFORMANCE,
    }
}

// The render preset every quality mode runs for `preset`.
fn render_preset(preset: DlssPreset) -> u32 {
    match preset {
        DlssPreset::Default => abi::DLSS_PRESET_DEFAULT,
        DlssPreset::K => abi::DLSS_PRESET_K,
        DlssPreset::L => abi::DLSS_PRESET_L,
        DlssPreset::M => abi::DLSS_PRESET_M,
    }
}

// The oldest feature library that runs `preset`, per the revision history of
// the DLSS Programming Guide: K arrived in 310.2.0, L and M in 310.5.0.
const fn minimum_version(preset: DlssPreset) -> Option<DlssVersion> {
    match preset {
        DlssPreset::Default => None,
        DlssPreset::K => Some(DlssVersion::new(310, 2, 0)),
        DlssPreset::L | DlssPreset::M => Some(DlssVersion::new(310, 5, 0)),
    }
}

// The options a viewport of `extent` runs `preset` with: HDR color divided by
// `pre_exposure`, and DLSS's own exposure, since the scene is un-exposed until
// after the upscale.
fn dlss_options(extent: UpscaleExtent, preset: DlssPreset, pre_exposure: f32) -> DlssOptions {
    let preset = render_preset(preset);
    let (output_width, output_height) = extent.output;
    DlssOptions {
        mode: dlss_mode(UpscaleQuality::nearest(extent.scale)),
        output_width,
        output_height,
        pre_exposure,
        color_buffers_hdr: abi::TRUE,
        dlaa_preset: preset,
        quality_preset: preset,
        balanced_preset: preset,
        performance_preset: preset,
        ultra_performance_preset: preset,
        ultra_quality_preset: preset,
        use_auto_exposure: abi::TRUE,
        ..DlssOptions::default()
    }
}

/// The camera transforms one frame drew with, unjittered.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ViewProjection {
    view: Mat4,
    projection: Mat4,
}

impl ViewProjection {
    fn of(camera: &UpscaleCamera, extent: UpscaleExtent) -> Self {
        let (width, height) = extent.render;
        let aspect = width as f32 / height.max(1) as f32;
        Self {
            view: camera.view,
            projection: camera_projection(camera.fov_y_radians, aspect, camera.near),
        }
    }
}

// The frame's camera constants. Motion vectors are `prev_uv - cur_uv` in UV
// space and already carry the camera's motion, so they scale by one and the
// plugin reads them as they are; `previous` (none across a reset) yields the
// clip-to-previous-clip transform features beyond super resolution read.
fn dlss_constants(
    camera: &UpscaleCamera,
    current: ViewProjection,
    previous: Option<ViewProjection>,
    reset: bool,
) -> Constants {
    let clip_to_view = mat4_inverse(current.projection);
    let view_to_world = mat4_affine_inverse(current.view);
    let clip_to_prev_clip = previous.map_or(IDENTITY, |prev| {
        mat4_mul(
            mat4_mul(prev.projection, prev.view),
            mat4_mul(view_to_world, clip_to_view),
        )
    });
    let [right, up, forward] = view_basis(&current.view);
    let [x, y, z, _] = view_to_world[3];
    let flag = |on: bool| if on { abi::TRUE } else { abi::FALSE };
    Constants {
        camera_view_to_clip: Float4x4(current.projection),
        clip_to_camera_view: Float4x4(clip_to_view),
        clip_to_prev_clip: Float4x4(clip_to_prev_clip),
        prev_clip_to_clip: Float4x4(mat4_inverse(clip_to_prev_clip)),
        jitter_offset: Float2 {
            x: camera.jitter_offset[0],
            y: camera.jitter_offset[1],
        },
        mvec_scale: Float2 { x: 1.0, y: 1.0 },
        camera_pos: [x, y, z].into(),
        camera_up: up.into(),
        camera_right: right.into(),
        camera_fwd: forward.into(),
        camera_near: camera.near,
        camera_far: INFINITE_FAR,
        camera_fov: camera.fov_y_radians,
        camera_aspect_ratio: current.projection[1][1] / current.projection[0][0],
        depth_inverted: flag(CAMERA_DEPTH.reversed),
        camera_motion_included: abi::TRUE,
        reset: flag(reset),
        ..Constants::default()
    }
}

/// A file version's `major.minor.patch`, as Streamline reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DlssVersion([u32; 3]);

impl DlssVersion {
    const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self([major, minor, patch])
    }

    // `None` for the all-zero version a plugin reports when it found no
    // library.
    fn reported(version: Version) -> Option<Self> {
        let v = Self::new(version.major, version.minor, version.build);
        (v != Self::new(0, 0, 0)).then_some(v)
    }
}

impl fmt::Display for DlssVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [major, minor, patch] = self.0;
        write!(f, "{major}.{minor}.{patch}")
    }
}

/// The super-resolution feature library the plugin loaded, for the creation
/// log and the preset check.
#[derive(Clone, Copy, Debug, PartialEq)]
struct FeatureLibrary {
    version: Option<DlssVersion>,
}

impl fmt::Display for FeatureLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.version {
            Some(version) => write!(f, "{FEATURE_LIBRARY} {version}"),
            None => write!(f, "{FEATURE_LIBRARY} version unknown"),
        }
    }
}

impl FeatureLibrary {
    // Why this library cannot run `preset`, or `None` when it can. A library
    // older than a preset reads its hint as another preset, or none; an
    // unknown version is the driver's own update, newer than any preset here.
    fn preset_unsupported(self, preset: DlssPreset) -> Option<String> {
        let (version, minimum) = (self.version?, minimum_version(preset)?);
        (version < minimum).then(|| {
            format!(
                "{FEATURE_LIBRARY} {version} predates render preset {} ({minimum} or newer)",
                preset.label()
            )
        })
    }

    // The preset to run for a request of `requested`.
    fn runnable_preset(self, label: &str, requested: DlssPreset) -> DlssPreset {
        match self.preset_unsupported(requested) {
            Some(reason) => {
                tracing::warn!("{label}: {reason}; running the default preset");
                DlssPreset::Default
            }
            None => requested,
        }
    }
}

/// The four images one evaluate reads and writes, described for Streamline in
/// the states they are in when the evaluate's commands run.
pub(crate) struct DlssResources<'a> {
    pub(crate) color: &'a Resource,
    pub(crate) depth: &'a Resource,
    pub(crate) motion: &'a Resource,
    pub(crate) output: &'a Resource,
}

/// One upscaler's DLSS viewport. Dropping it frees the viewport's NGX
/// feature, so the GPU must be done with its last evaluate by then.
pub(crate) struct DlssFeature {
    sl: Streamline,
    label: &'static str,
    viewport: ViewportHandle,
    set_options: PfnSlDlssSetOptions,
    extent: UpscaleExtent,
    library: FeatureLibrary,
    preset: Cell<DlssPreset>,
    previous: Cell<Option<ViewProjection>>,
}

impl DlssFeature {
    /// A viewport upscaling to `extent` with `requested`, or `None` (logged)
    /// when DLSS does not run on `adapter`. `label` names the backend in logs.
    pub(crate) fn new(
        sl: &Streamline,
        adapter: &AdapterInfo,
        extent: UpscaleExtent,
        requested: DlssPreset,
        label: &'static str,
    ) -> Option<Self> {
        if let Err(result) = sl.feature_supported(FEATURE_DLSS, adapter) {
            tracing::warn!("{label}: not supported on this adapter ({})", result.name());
            return None;
        }
        // SAFETY: `PfnSlDlssSetOptions` is the prototype `sl_dlss.h` declares for the name.
        let set_options = unsafe {
            sl.feature_function::<PfnSlDlssSetOptions>(FEATURE_DLSS, c"slDLSSSetOptions")
        }?;
        let library = FeatureLibrary {
            version: sl
                .feature_version(FEATURE_DLSS)
                .and_then(|v| DlssVersion::reported(v.version_ngx)),
        };
        let preset = library.runnable_preset(label, requested);
        tracing::info!(
            "{label}: viewport for {extent}, {library}, render preset {}",
            preset.label()
        );
        Some(Self {
            sl: sl.clone(),
            label,
            viewport: ViewportHandle::new(next_viewport()),
            set_options,
            extent,
            library,
            preset: Cell::new(preset),
            previous: Cell::new(None),
        })
    }

    pub(crate) fn extent(&self) -> UpscaleExtent {
        self.extent
    }

    /// The render preset the viewport runs, after any fallback.
    pub(crate) fn preset(&self) -> DlssPreset {
        self.preset.get()
    }

    /// Record one upscale onto `cmd` (an `ID3D12GraphicsCommandList` or a
    /// `VkCommandBuffer`), discarding the history when `reset`.
    pub(crate) fn evaluate(
        &self,
        cmd: *mut c_void,
        resources: &DlssResources<'_>,
        camera: &UpscaleCamera,
        reset: bool,
    ) -> RenderResult<()> {
        let api = self.sl.api();
        let mut token: *mut FrameToken = std::ptr::null_mut();
        // SAFETY: `token` is a live local the call fills; Streamline counts the frames itself.
        let result = unsafe { (api.get_new_frame_token)(&raw mut token, std::ptr::null()) };
        self.check(result, "slGetNewFrameToken")?;

        let current = ViewProjection::of(camera, self.extent);
        let previous = self.previous.replace(Some(current)).filter(|_| !reset);
        let constants = dlss_constants(camera, current, previous, reset);
        // SAFETY: the constants, token and viewport are live for the call, which copies them.
        let result =
            unsafe { (api.set_constants)(&raw const constants, token, &raw const self.viewport) };
        self.check(result, "slSetConstants")?;

        let render = Extent::sized(self.extent.render);
        let tags = [
            ResourceTag::new(resources.color, BUFFER_SCALING_INPUT_COLOR, render),
            ResourceTag::new(resources.depth, BUFFER_DEPTH, render),
            ResourceTag::new(resources.motion, BUFFER_MOTION_VECTORS, render),
            ResourceTag::new(
                resources.output,
                BUFFER_SCALING_OUTPUT_COLOR,
                Extent::sized(self.extent.output),
            ),
        ];
        let inputs: [*const BaseStructure; 5] = [
            self.viewport.as_base(),
            tags[0].as_base(),
            tags[1].as_base(),
            tags[2].as_base(),
            tags[3].as_base(),
        ];
        let result = self.evaluate_with(cmd, token, &inputs, camera.pre_exposure)?;
        if result.is_ok() {
            return Ok(());
        }
        if self.preset.get() == DlssPreset::Default {
            return self.check(result, "slEvaluateFeature");
        }
        tracing::warn!(
            "{}: render preset {} refused ({}); retrying with the default preset",
            self.label,
            self.preset.get().label(),
            result.name()
        );
        self.preset.set(DlssPreset::Default);
        let result = self.evaluate_with(cmd, token, &inputs, camera.pre_exposure)?;
        self.check(result, "slEvaluateFeature")
    }

    // Set the viewport's options for this frame, then evaluate. The options
    // carry the frame's pre-exposure; the plugin recreates its feature only
    // when the mode, size or preset changes.
    fn evaluate_with(
        &self,
        cmd: *mut c_void,
        token: *mut FrameToken,
        inputs: &[*const BaseStructure; 5],
        pre_exposure: f32,
    ) -> RenderResult<SlResult> {
        let options = dlss_options(self.extent, self.preset.get(), pre_exposure);
        // SAFETY: the viewport and options are live for the call, which copies them.
        let result = unsafe { (self.set_options)(&raw const self.viewport, &raw const options) };
        self.check(result, "slDLSSSetOptions")?;
        let count = u32::try_from(inputs.len()).unwrap_or(u32::MAX);
        // SAFETY: `cmd` is recording on the device the session is bound to, the token is this
        // frame's, and every input (and each resource a tag points at) is live for the call.
        Ok(unsafe {
            (self.sl.api().evaluate_feature)(FEATURE_DLSS, token, inputs.as_ptr(), count, cmd)
        })
    }

    fn check(&self, result: SlResult, call: &str) -> RenderResult<()> {
        if result.is_ok() {
            Ok(())
        } else {
            Err(RenderError::Other(format!(
                "{}: {call} returned {}",
                self.label,
                result.name()
            )))
        }
    }
}

impl Drop for DlssFeature {
    fn drop(&mut self) {
        // SAFETY: the viewport is this feature's own, and its owner idled the GPU before dropping
        // it.
        let result =
            unsafe { (self.sl.api().free_resources)(FEATURE_DLSS, &raw const self.viewport) };
        // A viewport that never evaluated has nothing to free.
        if !result.is_ok() && result != SlResult::ERROR_INVALID_PARAMETER {
            tracing::warn!("{}: slFreeResources returned {}", self.label, result.name());
        }
        tracing::debug!("{}: released {}", self.label, self.library);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::transform::mat4_transform;

    fn camera(view: Mat4) -> UpscaleCamera {
        UpscaleCamera::new([0.25, -0.125], 0.0, 0.1, 1.0, 2.0, view)
    }

    fn extent() -> UpscaleExtent {
        UpscaleExtent::resolve((1920, 1080), 0.5)
    }

    #[test]
    fn modes_follow_the_nearest_engine_quality() {
        assert_eq!(dlss_mode(UpscaleQuality::Native), abi::DLSS_MODE_DLAA);
        assert_eq!(
            dlss_mode(UpscaleQuality::Quality),
            abi::DLSS_MODE_MAX_QUALITY
        );
        assert_eq!(dlss_mode(UpscaleQuality::Balanced), abi::DLSS_MODE_BALANCED);
        assert_eq!(
            dlss_mode(UpscaleQuality::Performance),
            abi::DLSS_MODE_MAX_PERFORMANCE
        );
        assert_eq!(
            dlss_mode(UpscaleQuality::UltraPerformance),
            abi::DLSS_MODE_ULTRA_PERFORMANCE
        );
    }

    #[test]
    fn every_preset_maps_to_its_sdk_value() {
        assert_eq!(render_preset(DlssPreset::Default), 0);
        assert_eq!(render_preset(DlssPreset::K), 11);
        assert_eq!(render_preset(DlssPreset::L), 12);
        assert_eq!(render_preset(DlssPreset::M), 13);
    }

    // The preset holds whichever mode the render scale lands on, and DLSS
    // computes its own exposure from HDR input.
    #[test]
    fn options_set_the_preset_for_every_mode() {
        let options = dlss_options(extent(), DlssPreset::M, 0.5);
        for preset in [
            options.dlaa_preset,
            options.quality_preset,
            options.balanced_preset,
            options.performance_preset,
            options.ultra_performance_preset,
            options.ultra_quality_preset,
        ] {
            assert_eq!(preset, abi::DLSS_PRESET_M);
        }
        assert_eq!((options.output_width, options.output_height), (1920, 1080));
        assert_eq!(options.mode, abi::DLSS_MODE_MAX_PERFORMANCE);
        assert_eq!(options.pre_exposure, 0.5);
        assert_eq!(options.use_auto_exposure, abi::TRUE);
        assert_eq!(options.color_buffers_hdr, abi::TRUE);
    }

    #[test]
    fn constants_take_the_motion_vectors_as_written() {
        let cam = camera(IDENTITY);
        let current = ViewProjection::of(&cam, extent());
        let c = dlss_constants(&cam, current, None, true);
        assert_eq!(c.mvec_scale, Float2 { x: 1.0, y: 1.0 });
        assert_eq!(c.camera_motion_included, abi::TRUE);
        assert_eq!(c.depth_inverted, abi::TRUE);
        assert_eq!(c.reset, abi::TRUE);
        assert_eq!(c.jitter_offset, Float2 { x: 0.25, y: -0.125 });
        assert!((c.camera_aspect_ratio - 960.0 / 540.0).abs() < 1e-4);
        assert_eq!(c.clip_to_prev_clip, Float4x4(IDENTITY));
    }

    // A camera that moved maps each current clip point back onto where the
    // same world point was in the previous frame's clip space.
    #[test]
    fn clip_to_prev_clip_follows_the_camera_move() {
        let mut moved = IDENTITY;
        moved[3][0] = -1.0;
        let before = camera(IDENTITY);
        let after = camera(moved);
        let prev = ViewProjection::of(&before, extent());
        let cur = ViewProjection::of(&after, extent());
        let c = dlss_constants(&after, cur, Some(prev), false);
        let world = [0.5, 0.25, -4.0, 1.0];
        let clip_now = mat4_transform(&mat4_mul(cur.projection, cur.view), world);
        let clip_then = mat4_transform(&mat4_mul(prev.projection, prev.view), world);
        let mapped = mat4_transform(&c.clip_to_prev_clip.0, clip_now);
        for k in 0..4 {
            assert!(
                (mapped[k] - clip_then[k]).abs() < 1e-3,
                "{mapped:?} vs {clip_then:?}"
            );
        }
        let [x, y, z] = [c.camera_pos.x, c.camera_pos.y, c.camera_pos.z];
        assert!((x - 1.0).abs() < 1e-6 && y.abs() < 1e-6 && z.abs() < 1e-6);
    }

    #[test]
    fn versions_read_and_order_like_their_fields() {
        let bundled = DlssVersion::reported(Version {
            major: 310,
            minor: 9,
            build: 1,
        });
        assert_eq!(bundled, Some(DlssVersion::new(310, 9, 1)));
        assert_eq!(DlssVersion::reported(Version::default()), None);
        assert!(DlssVersion::new(310, 4, 9) < DlssVersion::new(310, 5, 0));
        assert_eq!(DlssVersion::new(310, 9, 1).to_string(), "310.9.1");
    }

    // Each preset has its own floor, and a library that is too old names the
    // floor it missed; the default runs anywhere.
    #[test]
    fn a_library_runs_each_preset_from_that_presets_first_version() {
        let at = |major, minor, patch| FeatureLibrary {
            version: Some(DlssVersion::new(major, minor, patch)),
        };
        let unknown = FeatureLibrary { version: None };
        for preset in DlssPreset::ALL {
            assert_eq!(at(310, 9, 1).preset_unsupported(preset), None, "{preset:?}");
            assert_eq!(unknown.preset_unsupported(preset), None, "{preset:?}");
        }
        assert_eq!(at(310, 1, 0).preset_unsupported(DlssPreset::Default), None);
        assert_eq!(at(310, 2, 0).preset_unsupported(DlssPreset::K), None);
        assert!(at(310, 1, 9).preset_unsupported(DlssPreset::K).is_some());
        assert_eq!(
            at(310, 4, 0).preset_unsupported(DlssPreset::L).as_deref(),
            Some("nvngx_dlss.dll 310.4.0 predates render preset L (310.5.0 or newer)")
        );
        assert_eq!(
            at(310, 4, 0).runnable_preset("test", DlssPreset::M),
            DlssPreset::Default
        );
        assert_eq!(
            at(310, 4, 0).runnable_preset("test", DlssPreset::K),
            DlssPreset::K
        );
    }

    #[test]
    fn the_library_names_its_version_or_says_it_is_unknown() {
        let known = FeatureLibrary {
            version: Some(DlssVersion::new(310, 9, 1)),
        };
        assert_eq!(known.to_string(), "nvngx_dlss.dll 310.9.1");
        let unknown = FeatureLibrary { version: None };
        assert_eq!(unknown.to_string(), "nvngx_dlss.dll version unknown");
    }
}
