// SettingCommand / SceneCommand application + settings snapshot ownership.
// Internal system, constructed alongside GraphicsSystem (same gate) and
// scheduled just before it.
pub(crate) mod system;

pub(crate) mod action;

// The engine-side registry of user-facing settings a cycle row can change. The
// ordered option labels live in `concinnity_core::gfx::settings` (shared with
// the build pipeline, which reads a key's label count to pick a stepper vs a
// dropdown); this module re-exports `options` + `is_quality_toggle` from there
// and holds the client-only half: how a chosen option index maps to the applied
// value (the `*_at` / `*_index` pairs), the `SLIDERS` table, and the cycle
// math. How a chosen option is applied (which backend call, which persisted
// field) lives in SettingsSystem's drain, keyed by the same string.

use concinnity_core::components::{
    AaMode, ControlsCommand, PostProcessConfig, ReflectionBlurResolution, SettingOp, ShadowUpdate,
    SsgiResolution, UpscaleQuality, UpscalerBackend, WindowMode,
};
use concinnity_core::gfx::render_types::PostProcessTunables;
use concinnity_core::render::backend;
use concinnity_core::render::backend::GpuVendor;

use crate::config::{GraphicsSettings, Settings};
// This module presents one settings vocabulary. The option-label registry half
// (labels + classification) lives in core so the cook and the client agree on
// every setting's option count, and is re-exported here alongside the
// client-only half below.
pub(crate) use concinnity_core::gfx::settings::{QUALITY_TOGGLE_KEYS, is_quality_toggle, options};

// Whether setting `key` can be changed on a device with the given capabilities.
// A capability-gated setting (e.g. `ray_traced_reflections`, which needs
// hardware ray tracing) is unavailable when the device lacks that capability;
// every other setting is always available. The settings menu grays out and
// disables an unavailable row. This is the one place to gate a future
// capability-dependent toggle.
pub(crate) fn setting_available(key: &str, caps: &backend::DeviceCapabilities) -> bool {
    match key {
        "ray_traced_reflections" => caps.ray_tracing,
        // The upscaler selector (FSR3 / DLSS / XeSS) grays out on a device whose
        // upscaler is fixed, rather than offering a dead selection.
        "upscale_backend" => caps.selectable_upscaler,
        _ => true,
    }
}

// InputKey-rebind settings (Controls tab) are a third setting category alongside
// cycle rows (`options`) and sliders (`slider`): a rebind key's value is a
// physical `InputKey`, not an option index or a fraction. Their classification +
// per-action data live in `gfx/keymap.rs` (the `Bindable` / `KeyMap` types) and
// the live map is owned by `GraphicsSystem`, so there is nothing to register
// here; this comment just records the third category for the reader.

// The discrete numeric levels each `*_at` / `*_index` mapping recovers from an
// option index (the ordered labels for these live in core's option registry).
// An authored value off a discrete level snaps to the nearest.
const FPS_CAP_VALUES: [u32; 6] = [0, 30, 60, 120, 144, 240];
const SSGI_RAYS_COUNTS: [u32; 4] = [4, 8, 16, 32];
const SSGI_STEPS_COUNTS: [u32; 4] = [8, 12, 24, 48];
const SHADOW_RESOLUTION_SIZES: [u32; 4] = [0, 1024, 2048, 4096];
const SHADOW_DISTANCE_VALUES: [u32; 4] = [40, 80, 160, 320];
const SHADOW_CASCADES_VALUES: [u32; 3] = [2, 3, 4];
const ANISOTROPY_LEVELS: [u32; 5] = [1, 2, 4, 8, 16];
const FRAME_BUFFERING_COUNTS: [u32; 3] = [1, 2, 3];
// Texture quality: one option index drives both the streaming pool cap (how many
// high-resolution textures stay resident) and the per-frame upload budget.
const TEXTURE_QUALITY_CAPS: [u32; 4] = [48, 96, 192, 384];
const TEXTURE_QUALITY_BUDGETS: [u32; 4] = [2, 4, 8, 12];

// The cycle (dropdown) quality knobs governed by the preset ceiling like the
// boolean QUALITY_TOGGLE_KEYS. Each rides the feature's live-reinit rebuild
// (`apply_quality_settings`) -- the sub-tunable travels in its settings payload,
// so no new backend method is needed. `GraphicsSystem` maps each key to the
// `PostProcessConfig` field it cycles.
pub(crate) const QUALITY_CYCLE_KEYS: [&str; 5] = [
    "aa_mode",
    "ssgi_resolution",
    "ssgi_rays",
    "ssgi_steps",
    "reflection_blur_resolution",
];

// Volume gains shared by the master and per-bus rows, one per option index
// (the labels live in core). Indices map to a linear gain via `volume_at` /
// `volume_index`.
const VOLUME_GAINS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
// Effective volume for any stage the user has never chosen (full gain).
pub(crate) const DEFAULT_VOLUME: f32 = 1.0;

// Effective mouse sensitivity (radians per pixel) when the user has never
// chosen one. Matches `CameraController`'s authored default. Mouse sensitivity
// is a slider (1..100 -> radians/pixel), not a cycle row.
pub(crate) const DEFAULT_MOUSE_SENSITIVITY: f32 = 0.0015;

// Effective gamepad look sensitivity (radians per second at full stick
// deflection) when the user has never chosen one. A slider (1..100 -> rate).
pub(crate) const DEFAULT_GAMEPAD_LOOK_SENSITIVITY: f32 = 2.5;
// Effective gamepad stick deadzone (deflection fraction) when the user has
// never chosen one. A slider shown as a percentage.
pub(crate) const DEFAULT_GAMEPAD_DEADZONE: f32 = 0.15;

// WindowMode for an option index, and the index for a WindowMode. Order matches
// WINDOW_MODE_OPTIONS, not the enum's declaration order.
pub(crate) fn window_mode_at(index: usize) -> WindowMode {
    match index {
        1 => WindowMode::Borderless,
        2 => WindowMode::Fullscreen,
        _ => WindowMode::Windowed,
    }
}
pub(crate) fn window_mode_index(mode: WindowMode) -> usize {
    match mode {
        WindowMode::Windowed => 0,
        WindowMode::Borderless => 1,
        WindowMode::Fullscreen => 2,
    }
}

// UpscaleQuality for an option index, and the index for a quality.
pub(crate) fn render_scale_at(index: usize) -> UpscaleQuality {
    match index {
        1 => UpscaleQuality::Balanced,
        2 => UpscaleQuality::Performance,
        3 => UpscaleQuality::UltraPerformance,
        _ => UpscaleQuality::Quality,
    }
}
pub(crate) fn render_scale_index(quality: UpscaleQuality) -> usize {
    match quality {
        UpscaleQuality::Quality => 0,
        UpscaleQuality::Balanced => 1,
        UpscaleQuality::Performance => 2,
        UpscaleQuality::UltraPerformance => 3,
    }
}

// UpscalerBackend for an option index, and the index for a backend. Order matches
// UPSCALE_BACKEND_OPTIONS (Auto / FSR3 / DLSS / XeSS).
pub(crate) fn upscale_backend_at(index: usize) -> UpscalerBackend {
    match index {
        1 => UpscalerBackend::Fsr3,
        2 => UpscalerBackend::Dlss,
        3 => UpscalerBackend::Xess,
        _ => UpscalerBackend::Auto,
    }
}
pub(crate) fn upscale_backend_index(backend: UpscalerBackend) -> usize {
    match backend {
        UpscalerBackend::Auto => 0,
        UpscalerBackend::Fsr3 => 1,
        UpscalerBackend::Dlss => 2,
        UpscalerBackend::Xess => 3,
    }
}

// Whether an upscaler backend is offered on the given GPU vendor. Auto and FSR3
// are vendor-agnostic; DLSS (NVIDIA NGX) is NVIDIA-only and XeSS is offered on
// Intel. The settings-menu cycle skips the unavailable entries so the user only
// lands on an upscaler the GPU can actually drive; the backend's `build_upscaler`
// still resolves + falls back on its own, so an unavailable explicit value is
// always safe even if it somehow gets set.
pub(crate) fn upscale_backend_available(backend: UpscalerBackend, vendor: GpuVendor) -> bool {
    match backend {
        UpscalerBackend::Auto | UpscalerBackend::Fsr3 => true,
        UpscalerBackend::Dlss => vendor == GpuVendor::Nvidia,
        UpscalerBackend::Xess => vendor == GpuVendor::Intel,
    }
}

// Anti-aliasing mode for an option index, and the index for a mode. Order
// matches AA_MODE_OPTIONS (Off, FXAA, TAA), which is also ascending cost so the
// index doubles as the aggressiveness rank the preset ceiling clamps against.
pub(crate) fn aa_mode_at(index: usize) -> AaMode {
    match index {
        0 => AaMode::Off,
        2 => AaMode::Taa,
        _ => AaMode::Fxaa,
    }
}
pub(crate) fn aa_mode_index(mode: AaMode) -> usize {
    match mode {
        AaMode::Off => 0,
        AaMode::Fxaa => 1,
        AaMode::Taa => 2,
    }
}

// SSGI gather resolution for an option index, and the index for a resolution.
// Order matches SSGI_RESOLUTION_OPTIONS (finest first).
pub(crate) fn ssgi_resolution_at(index: usize) -> SsgiResolution {
    match index {
        0 => SsgiResolution::Full,
        2 => SsgiResolution::Quarter,
        _ => SsgiResolution::Half,
    }
}
pub(crate) fn ssgi_resolution_index(res: SsgiResolution) -> usize {
    match res {
        SsgiResolution::Full => 0,
        SsgiResolution::Half => 1,
        SsgiResolution::Quarter => 2,
    }
}

// SSGI ray / step counts for an option index, and the menu index nearest an
// authored count (the world may author a value off the discrete levels; the row
// then shows the closest one).
pub(crate) fn ssgi_rays_at(index: usize) -> u32 {
    *SSGI_RAYS_COUNTS.get(index).unwrap_or(&SSGI_RAYS_COUNTS[1])
}
pub(crate) fn ssgi_rays_index(count: u32) -> usize {
    nearest_count_index(&SSGI_RAYS_COUNTS, count)
}
pub(crate) fn ssgi_steps_at(index: usize) -> u32 {
    *SSGI_STEPS_COUNTS
        .get(index)
        .unwrap_or(&SSGI_STEPS_COUNTS[1])
}
pub(crate) fn ssgi_steps_index(count: u32) -> usize {
    nearest_count_index(&SSGI_STEPS_COUNTS, count)
}
// The index of the level closest to `count` (ties pick the lower level).
fn nearest_count_index(levels: &[u32], count: u32) -> usize {
    levels
        .iter()
        .enumerate()
        .min_by_key(|&(_, &v)| v.abs_diff(count))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// Reflection blur resolution for an option index, and the index for a
// resolution. Order matches REFLECTION_BLUR_OPTIONS (finest first).
pub(crate) fn reflection_blur_at(index: usize) -> ReflectionBlurResolution {
    match index {
        0 => ReflectionBlurResolution::Full,
        2 => ReflectionBlurResolution::Quarter,
        _ => ReflectionBlurResolution::Half,
    }
}
pub(crate) fn reflection_blur_index(res: ReflectionBlurResolution) -> usize {
    match res {
        ReflectionBlurResolution::Full => 0,
        ReflectionBlurResolution::Half => 1,
        ReflectionBlurResolution::Quarter => 2,
    }
}

// Shadow-map resolution (texels) for an option index, and the menu index nearest
// an authored size (the world may author a size off the discrete levels; the row
// then shows the closest one). The default fallback is the world default (2048).
pub(crate) fn shadow_resolution_at(index: usize) -> u32 {
    *SHADOW_RESOLUTION_SIZES
        .get(index)
        .unwrap_or(&SHADOW_RESOLUTION_SIZES[2])
}
pub(crate) fn shadow_resolution_index(size: u32) -> usize {
    nearest_count_index(&SHADOW_RESOLUTION_SIZES, size)
}

// Shadow re-render cadence for an option index, and the index for a cadence.
// Order matches SHADOW_UPDATE_OPTIONS (EveryFrame first).
pub(crate) fn shadow_update_at(index: usize) -> ShadowUpdate {
    match index {
        0 => ShadowUpdate::EveryFrame,
        _ => ShadowUpdate::Hybrid,
    }
}
pub(crate) fn shadow_update_index(update: ShadowUpdate) -> usize {
    match update {
        ShadowUpdate::EveryFrame => 0,
        ShadowUpdate::Hybrid => 1,
    }
}

// Shadow distance (world units) for an option index, and the menu index nearest
// an authored distance (the world may author a distance off the discrete levels;
// the row then shows the closest one). The default fallback is the world default
// (80).
pub(crate) fn shadow_distance_at(index: usize) -> u32 {
    *SHADOW_DISTANCE_VALUES
        .get(index)
        .unwrap_or(&SHADOW_DISTANCE_VALUES[1])
}
pub(crate) fn shadow_distance_index(distance: u32) -> usize {
    nearest_count_index(&SHADOW_DISTANCE_VALUES, distance)
}

// Shadow cascade count for an option index, and the menu index nearest an
// authored count. The default fallback is the world default (4, the last index).
pub(crate) fn shadow_cascades_at(index: usize) -> u32 {
    *SHADOW_CASCADES_VALUES
        .get(index)
        .unwrap_or(&SHADOW_CASCADES_VALUES[2])
}
pub(crate) fn shadow_cascades_index(count: u32) -> usize {
    nearest_count_index(&SHADOW_CASCADES_VALUES, count)
}

// Anisotropic-filtering degree for an option index, and the menu index nearest an
// authored degree (the world may author a degree off the discrete levels; the row
// then shows the closest one). The default fallback is the world default (8x).
pub(crate) fn anisotropy_at(index: usize) -> u32 {
    *ANISOTROPY_LEVELS
        .get(index)
        .unwrap_or(&ANISOTROPY_LEVELS[3])
}
pub(crate) fn anisotropy_index(level: u32) -> usize {
    nearest_count_index(&ANISOTROPY_LEVELS, level)
}

// Frame-rate cap (FPS) for an option index, and the menu index nearest an
// authored cap (the world may author a cap off the discrete levels; the row then
// shows the closest one). The default fallback is "Unlimited" (index 0).
pub(crate) fn fps_cap_at(index: usize) -> u32 {
    *FPS_CAP_VALUES.get(index).unwrap_or(&FPS_CAP_VALUES[0])
}
pub(crate) fn fps_cap_index(cap: u32) -> usize {
    nearest_count_index(&FPS_CAP_VALUES, cap)
}

// Frames-in-flight (ring-buffer depth) for an option index, and the index for a
// count. Order matches FRAME_BUFFERING_OPTIONS (1, 2, 3); an out-of-range count
// snaps to the nearest level.
pub(crate) fn frames_in_flight_at(index: usize) -> u32 {
    *FRAME_BUFFERING_COUNTS
        .get(index)
        .unwrap_or(&FRAME_BUFFERING_COUNTS[1])
}
pub(crate) fn frames_in_flight_index(count: u32) -> usize {
    nearest_count_index(&FRAME_BUFFERING_COUNTS, count)
}

// Texture-quality level for an option index -> the (pool cap, per-frame budget)
// pair it sets, and the index recovered from a pool cap (the quality axis). An
// authored cap off the discrete levels snaps to the nearest level.
pub(crate) fn texture_quality_at(index: usize) -> (u32, u32) {
    let i = index.min(TEXTURE_QUALITY_CAPS.len() - 1);
    (TEXTURE_QUALITY_CAPS[i], TEXTURE_QUALITY_BUDGETS[i])
}
pub(crate) fn texture_quality_index(cap: u32) -> usize {
    nearest_count_index(&TEXTURE_QUALITY_CAPS, cap)
}

// Linear gain for a volume option index, and the index for a gain. A gain
// that is not a preset (an authored value) falls back to the last index
// (full).
pub(crate) fn volume_at(index: usize) -> f32 {
    *VOLUME_GAINS.get(index).unwrap_or(&DEFAULT_VOLUME)
}
pub(crate) fn volume_index(gain: f32) -> usize {
    VOLUME_GAINS
        .iter()
        .position(|g| (g - gain).abs() < 1.0e-4)
        .unwrap_or(VOLUME_GAINS.len() - 1)
}

// Advance an option index one step in the given direction (wrapping at the
// ends), or jump straight to a dropdown pick's absolute index. `len` must be
// non-zero (a known setting always has options). A `SetFraction` op only
// applies to slider settings and never reaches here.
pub(crate) fn cycle(index: usize, len: usize, op: SettingOp) -> usize {
    debug_assert!(len > 0);
    match op {
        SettingOp::Prev => (index + len - 1) % len,
        // A dropdown pick jumps to its option index, clamped to the last option.
        SettingOp::SetIndex(i) => i.min(len.saturating_sub(1)),
        // Next steps forward; the slider and rebind ops never reach a cycle
        // setting, so treating them as Next is harmless.
        SettingOp::Next
        | SettingOp::SetFraction(_)
        | SettingOp::Rebind(_)
        | SettingOp::RebindButton(_) => (index + 1) % len,
    }
}

// Slider (continuous) settings. Unlike the cycle settings above, these map a
// fraction in `[0, 1]` to a value in a fixed range. `SLIDERS` holds every slider
// key with its range, value transforms, label format, and the state it drives,
// so a Slider row can only target a setting the engine knows how to apply.
// `slider` returning `Some` is what marks a key as a slider (vs `options` for a
// cycle row).

// Mouse-sensitivity slider: a 1..100 UI scale mapped linearly to radians per
// pixel. The camera's authored default (`DEFAULT_MOUSE_SENSITIVITY`) sits low on
// the track.
const MOUSE_SENS_MIN: f32 = 0.0003;
const MOUSE_SENS_MAX: f32 = 0.005;

// Gamepad look-sensitivity slider: the same 1..100 UI scale, mapped linearly to
// radians per second at full stick deflection. The engine default sits mid-track.
const GAMEPAD_LOOK_MIN: f32 = 0.5;
const GAMEPAD_LOOK_MAX: f32 = 6.0;

// Effective vertical FOV in degrees when the user has never chosen one. Matches
// Camera3D's authored default.
pub(crate) const DEFAULT_FOV: f32 = 75.0;

// Fraction of a slider's range one focused Left/Right pulse steps.
pub(crate) const SLIDER_STEP_FRACTION: f32 = 0.05;

// A shared and a mutable accessor for the same field of `T`.
pub(crate) struct Lens<T, V> {
    pub(crate) get: fn(&T) -> &V,
    pub(crate) get_mut: fn(&mut T) -> &mut V,
}

macro_rules! lens {
    ($($field:ident).+) => {
        Lens {
            get: |t| &t.$($field).+,
            get_mut: |t| &mut t.$($field).+,
        }
    };
}

// The live state a slider's applied value drives, and where its choice persists.
// The render-side targets persist the user-facing value; `Controls` persists the
// applied value the camera and input sampling read.
pub(crate) enum SliderTarget {
    // A `PostProcessTunables` field, pushed live through `update_post_process`.
    PostProcess {
        field: Lens<PostProcessTunables, f32>,
        persisted: Lens<GraphicsSettings, Option<f32>>,
    },
    // A per-feature sub-quality `PostProcessConfig` field, pushed live through
    // `update_quality_params` (no pass rebuild). Look tuning, so no preset ceiling.
    PostConfig {
        field: Lens<PostProcessConfig, f32>,
        persisted: Lens<GraphicsSettings, Option<f32>>,
    },
    // The ambient (IBL) scale, which rides `LightUniforms` behind its own setter.
    Ambient {
        persisted: Lens<GraphicsSettings, Option<f32>>,
    },
    // A camera / input preference, sent live as a `ControlsCommand`. `default` is
    // the applied value when nothing is persisted.
    Controls {
        persisted: Lens<Settings, Option<f32>>,
        command: fn(f32) -> ControlsCommand,
        default: f32,
    },
}

pub(crate) struct SliderSetting {
    pub(crate) key: &'static str,
    // The (min, max) user-facing value range.
    pub(crate) range: (f32, f32),
    // User-facing value to applied value, clamped to the engine's domain. Shared
    // by the live drag and the persisted re-apply at init.
    pub(crate) apply: fn(f32) -> f32,
    // The inverse of `apply`, so a handle and label re-sync to the live value.
    pub(crate) recover: fn(f32) -> f32,
    // Value-label text for a user-facing value.
    pub(crate) format: fn(f32) -> String,
    pub(crate) target: SliderTarget,
}

impl SliderSetting {
    // The value at a `0.0..=1.0` fraction of the range. The fraction is clamped.
    pub(crate) fn value_at(&self, fraction: f32) -> f32 {
        let (lo, hi) = self.range;
        lo + (hi - lo) * fraction.clamp(0.0, 1.0)
    }

    // The `0.0..=1.0` fraction a value sits at within the range, clamped so an
    // out-of-range authored value pins the handle to an end.
    pub(crate) fn fraction(&self, value: f32) -> f32 {
        let (lo, hi) = self.range;
        let span = hi - lo;
        if span.abs() < f32::EPSILON {
            return 0.0;
        }
        ((value - lo) / span).clamp(0.0, 1.0)
    }

    // The user-facing value the slider shows: the live render state, or the
    // persisted choice (else the default) for a controls slider.
    pub(crate) fn current_value(
        &self,
        post_process: &PostProcessTunables,
        post_config: &PostProcessConfig,
        ambient_intensity: f32,
        persisted: &Settings,
    ) -> f32 {
        let stored = match &self.target {
            SliderTarget::PostProcess { field, .. } => *(field.get)(post_process),
            SliderTarget::PostConfig { field, .. } => *(field.get)(post_config),
            SliderTarget::Ambient { .. } => ambient_intensity,
            SliderTarget::Controls {
                persisted: lens,
                default,
                ..
            } => (lens.get)(persisted).unwrap_or(*default),
        };
        (self.recover)(stored)
    }

    // Record the user-facing `value` in the settings store.
    pub(crate) fn persist(&self, cfg: &mut Settings, value: f32) {
        match &self.target {
            SliderTarget::PostProcess { persisted, .. }
            | SliderTarget::PostConfig { persisted, .. }
            | SliderTarget::Ambient { persisted } => {
                *(persisted.get_mut)(&mut cfg.graphics) = Some(value);
            }
            SliderTarget::Controls { persisted, .. } => {
                *(persisted.get_mut)(cfg) = Some((self.apply)(value));
            }
        }
    }
}

// The slider entry for `key`, or `None` if the key is not a slider setting.
pub(crate) fn slider(key: &str) -> Option<&'static SliderSetting> {
    SLIDERS.iter().find(|s| s.key == key)
}

fn identity(value: f32) -> f32 {
    value
}

fn format_ev(value: f32) -> String {
    format!("{value:+.1} EV")
}

fn format_meters(value: f32) -> String {
    format!("{value:.1} m")
}

fn format_fraction_percent(value: f32) -> String {
    format!("{}%", (value * 100.0).round() as i32)
}

fn format_two_decimals(value: f32) -> String {
    format!("{value:.2}")
}

// A 1..100 UI value mapped linearly onto `[min, max]`.
fn hundred_scale_to(value: f32, min: f32, max: f32) -> f32 {
    min + (max - min) * (value.clamp(1.0, 100.0) - 1.0) / 99.0
}

// The 1..100 UI value for a stored value on `[min, max]`.
fn hundred_scale_from(stored: f32, min: f32, max: f32) -> f32 {
    1.0 + (stored - min) / (max - min) * 99.0
}

// The post-process ranges are practical UI ceilings; `apply` mirrors the clamps
// in `PostProcessConfig::resolve` and each feature's `*Settings::resolve`. The
// 16.0 EV bound mirrors core's `EXPOSURE_EV_LIMIT`.
pub(crate) static SLIDERS: [SliderSetting; 20] = [
    // Authored in EV (centered on neutral), applied as the multiplier 2^ev.
    SliderSetting {
        key: "exposure",
        range: (-3.0, 3.0),
        apply: |v| v.clamp(-16.0, 16.0).exp2(),
        // Guard log2(0); the slider range keeps the multiplier well above this.
        recover: |stored| stored.max(1.0e-6).log2(),
        format: format_ev,
        target: SliderTarget::PostProcess {
            field: lens!(exposure),
            persisted: lens!(exposure_ev),
        },
    },
    SliderSetting {
        key: "bloom_intensity",
        range: (0.0, 2.0),
        apply: |v| v.max(0.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::PostProcess {
            field: lens!(bloom_intensity),
            persisted: lens!(bloom_intensity),
        },
    },
    SliderSetting {
        key: "bloom_threshold",
        range: (0.0, 4.0),
        apply: |v| v.max(0.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::PostProcess {
            field: lens!(bloom_threshold),
            persisted: lens!(bloom_threshold),
        },
    },
    SliderSetting {
        key: "vignette",
        range: (0.0, 1.0),
        apply: |v| v.clamp(0.0, 1.0),
        recover: identity,
        format: format_fraction_percent,
        target: SliderTarget::PostProcess {
            field: lens!(vignette),
            persisted: lens!(vignette),
        },
    },
    SliderSetting {
        key: "lut_strength",
        range: (0.0, 1.0),
        apply: |v| v.clamp(0.0, 1.0),
        recover: identity,
        format: format_fraction_percent,
        target: SliderTarget::PostProcess {
            field: lens!(lut_strength),
            persisted: lens!(lut_strength),
        },
    },
    SliderSetting {
        key: "ambient_intensity",
        range: (0.0, 4.0),
        apply: |v| v.clamp(0.0, 16.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::Ambient {
            persisted: lens!(ambient_intensity),
        },
    },
    // Soft-knee width below the bloom threshold, lower-bounded like the other
    // bloom params.
    SliderSetting {
        key: "bloom_knee",
        range: (0.0, 1.0),
        apply: |v| v.max(0.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::PostProcess {
            field: lens!(bloom_knee),
            persisted: lens!(bloom_knee),
        },
    },
    SliderSetting {
        key: "ssao_radius",
        range: (0.05, 2.0),
        apply: |v| v.max(1.0e-3),
        recover: identity,
        format: format_meters,
        target: SliderTarget::PostConfig {
            field: lens!(ssao_radius),
            persisted: lens!(ssao_radius),
        },
    },
    SliderSetting {
        key: "ssao_intensity",
        range: (0.0, 4.0),
        apply: |v| v.clamp(0.0, 4.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::PostConfig {
            field: lens!(ssao_intensity),
            persisted: lens!(ssao_intensity),
        },
    },
    SliderSetting {
        key: "ssr_intensity",
        range: (0.0, 1.0),
        apply: |v| v.clamp(0.0, 1.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::PostConfig {
            field: lens!(ssr_intensity),
            persisted: lens!(ssr_intensity),
        },
    },
    SliderSetting {
        key: "ssr_max_distance",
        range: (1.0, 200.0),
        apply: |v| v.clamp(1.0, 200.0),
        recover: identity,
        format: format_meters,
        target: SliderTarget::PostConfig {
            field: lens!(ssr_max_distance),
            persisted: lens!(ssr_max_distance),
        },
    },
    SliderSetting {
        key: "ssgi_intensity",
        range: (0.0, 4.0),
        apply: |v| v.clamp(0.0, 4.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::PostConfig {
            field: lens!(ssgi_intensity),
            persisted: lens!(ssgi_intensity),
        },
    },
    SliderSetting {
        key: "ssgi_max_distance",
        range: (0.5, 40.0),
        apply: |v| v.clamp(0.5, 100.0),
        recover: identity,
        format: format_meters,
        target: SliderTarget::PostConfig {
            field: lens!(ssgi_max_distance),
            persisted: lens!(ssgi_max_distance),
        },
    },
    // The resolve also orders the EV bounds (min <= max) when the config is
    // resolved.
    SliderSetting {
        key: "auto_exposure_min_ev",
        range: (-16.0, 16.0),
        apply: |v| v.clamp(-16.0, 16.0),
        recover: identity,
        format: format_ev,
        target: SliderTarget::PostConfig {
            field: lens!(auto_exposure_min_ev),
            persisted: lens!(auto_exposure_min_ev),
        },
    },
    SliderSetting {
        key: "auto_exposure_max_ev",
        range: (-16.0, 16.0),
        apply: |v| v.clamp(-16.0, 16.0),
        recover: identity,
        format: format_ev,
        target: SliderTarget::PostConfig {
            field: lens!(auto_exposure_max_ev),
            persisted: lens!(auto_exposure_max_ev),
        },
    },
    SliderSetting {
        key: "auto_exposure_speed",
        range: (0.1, 6.0),
        apply: |v| v.clamp(1.0e-3, 20.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::PostConfig {
            field: lens!(auto_exposure_speed),
            persisted: lens!(auto_exposure_speed),
        },
    },
    SliderSetting {
        key: "mouse_sensitivity",
        range: (1.0, 100.0),
        apply: |v| hundred_scale_to(v, MOUSE_SENS_MIN, MOUSE_SENS_MAX),
        recover: |stored| hundred_scale_from(stored, MOUSE_SENS_MIN, MOUSE_SENS_MAX),
        format: |v| format!("{}", v.round() as i32),
        target: SliderTarget::Controls {
            persisted: lens!(controls.mouse_sensitivity),
            command: |v| ControlsCommand {
                mouse_sensitivity: Some(v),
                ..ControlsCommand::default()
            },
            default: DEFAULT_MOUSE_SENSITIVITY,
        },
    },
    SliderSetting {
        key: "gamepad_look_sensitivity",
        range: (1.0, 100.0),
        apply: |v| hundred_scale_to(v, GAMEPAD_LOOK_MIN, GAMEPAD_LOOK_MAX),
        recover: |stored| hundred_scale_from(stored, GAMEPAD_LOOK_MIN, GAMEPAD_LOOK_MAX),
        format: |v| format!("{}", v.round() as i32),
        target: SliderTarget::Controls {
            persisted: lens!(controls.gamepad_look_sensitivity),
            command: |v| ControlsCommand {
                gamepad_look_sensitivity: Some(v),
                ..ControlsCommand::default()
            },
            default: DEFAULT_GAMEPAD_LOOK_SENSITIVITY,
        },
    },
    // Shown as a percentage of stick deflection, stored as the fraction the
    // radial deadzone consumes.
    SliderSetting {
        key: "gamepad_deadzone",
        range: (0.0, 40.0),
        apply: |v| v.clamp(0.0, 40.0) / 100.0,
        recover: |stored| stored * 100.0,
        format: |v| format!("{}%", v.round() as i32),
        target: SliderTarget::Controls {
            persisted: lens!(controls.gamepad_deadzone),
            command: |v| ControlsCommand {
                gamepad_deadzone: Some(v),
                ..ControlsCommand::default()
            },
            default: DEFAULT_GAMEPAD_DEADZONE,
        },
    },
    // A vertical FOV in degrees applied to every Camera3D, so apply only clamps.
    // Persisted in the graphics store alongside the look sliders.
    SliderSetting {
        key: "fov",
        range: (50.0, 100.0),
        apply: |v| v.clamp(50.0, 100.0),
        recover: identity,
        format: |v| format!("{}\u{00b0}", v.round() as i32),
        target: SliderTarget::Controls {
            persisted: lens!(graphics.fov),
            command: |v| ControlsCommand {
                fov_y_degrees: Some(v),
                ..ControlsCommand::default()
            },
            default: DEFAULT_FOV,
        },
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vsync_options_are_off_then_on() {
        assert_eq!(options("vsync"), Some(&["Off", "On"][..]));
    }

    #[test]
    fn stats_hud_toggles_are_off_then_on() {
        // The "Display performance stats" master and its per-readout sub-toggles
        // are plain Off/On cycle rows (index 0 = Off, 1 = On, like vsync).
        for key in ["perf_stats", "show_fps", "show_vram"] {
            assert_eq!(options(key), Some(&["Off", "On"][..]), "{key}");
            // Both are available regardless of GPU capability (not gated).
            let caps = backend::DeviceCapabilities {
                ray_tracing: false,
                ..backend::DeviceCapabilities::ALL
            };
            assert!(setting_available(key, &caps), "{key}");
        }
    }

    #[test]
    fn unknown_key_has_no_options() {
        assert!(options("does_not_exist").is_none());
    }

    #[test]
    fn graphics_quality_options_match_preset_order() {
        use crate::gfx::quality_preset::QualityPreset;
        // The master row's labels (in core's registry) must line up 1:1 with the
        // preset cycle order, so an index from `preset_index` selects the right
        // label and vice versa.
        let labels = options("graphics_quality").expect("graphics_quality options");
        assert_eq!(labels.len(), QualityPreset::ALL.len());
        for (i, p) in QualityPreset::ALL.iter().enumerate() {
            assert_eq!(labels[i], p.name(), "label {i}");
        }
    }

    #[test]
    fn quality_toggles_are_off_then_on_and_classified() {
        for key in QUALITY_TOGGLE_KEYS {
            assert!(is_quality_toggle(key), "{key} should classify as a toggle");
            assert_eq!(options(key), Some(&["Off", "On"][..]), "{key} options");
            // A quality toggle is a cycle row, never a slider.
            assert!(slider(key).is_none(), "{key} should not be a slider");
        }
        // Non-toggle keys are not misclassified.
        assert!(!is_quality_toggle("vsync"));
        assert!(!is_quality_toggle("exposure"));
        assert!(!is_quality_toggle("nope"));
    }

    #[test]
    fn rebind_keys_are_a_distinct_category() {
        use concinnity_core::render::keymap::Bindable;
        // A rebind key is neither a cycle row nor a slider, so the three setting
        // categories never collide on one key.
        for b in Bindable::ALL {
            let key = b.setting_key();
            assert!(options(key).is_none(), "{key} should not be a cycle row");
            assert!(slider(key).is_none(), "{key} should not be a slider");
        }
    }

    #[test]
    fn rt_toggle_gated_on_ray_tracing_capability() {
        use concinnity_core::render::backend::DeviceCapabilities;
        let capable = DeviceCapabilities {
            ray_tracing: true,
            ..DeviceCapabilities::ALL
        };
        let incapable = DeviceCapabilities {
            ray_tracing: false,
            ..DeviceCapabilities::ALL
        };
        // RT reflections follow the device's ray-tracing capability.
        assert!(setting_available("ray_traced_reflections", &capable));
        assert!(!setting_available("ray_traced_reflections", &incapable));
        // Every other setting is always available, regardless of capability.
        for key in ["vsync", "aa_mode", "ssao", "ssr", "ssgi", "auto_exposure"] {
            assert!(
                setting_available(key, &incapable),
                "{key} should be available"
            );
        }
        // The default reports all capabilities present (an unwired backend keeps
        // every toggle live).
        assert!(setting_available(
            "ray_traced_reflections",
            &DeviceCapabilities::default()
        ));
    }

    #[test]
    fn aa_mode_round_trips_and_orders_by_cost() {
        // Index order is ascending cost (Off < FXAA < TAA), so it doubles as the
        // aggressiveness rank the preset ceiling clamps against.
        for (i, mode) in [AaMode::Off, AaMode::Fxaa, AaMode::Taa]
            .into_iter()
            .enumerate()
        {
            assert_eq!(aa_mode_index(mode), i);
            assert_eq!(aa_mode_at(i), mode);
        }
        assert_eq!(options("aa_mode").unwrap().len(), 3);
        // An out-of-range index falls back to the FXAA default.
        assert_eq!(aa_mode_at(9), AaMode::Fxaa);
    }

    #[test]
    fn fps_cap_round_trips_and_snaps() {
        assert_eq!(options("fps_cap").unwrap().len(), FPS_CAP_VALUES.len());
        for (i, &cap) in FPS_CAP_VALUES.iter().enumerate() {
            assert_eq!(fps_cap_index(cap), i);
            assert_eq!(fps_cap_at(i), cap);
        }
        // "Unlimited" is 0 and the index-0 fallback.
        assert_eq!(fps_cap_at(0), 0);
        assert_eq!(fps_cap_at(99), 0);
        // An authored cap off the discrete levels snaps to the nearest.
        assert_eq!(fps_cap_index(58), fps_cap_index(60));
        assert_eq!(fps_cap_index(1000), FPS_CAP_VALUES.len() - 1);
    }

    #[test]
    fn cycle_next_wraps() {
        assert_eq!(cycle(0, 2, SettingOp::Next), 1);
        assert_eq!(cycle(1, 2, SettingOp::Next), 0);
    }

    #[test]
    fn cycle_prev_wraps() {
        assert_eq!(cycle(0, 2, SettingOp::Prev), 1);
        assert_eq!(cycle(1, 2, SettingOp::Prev), 0);
    }

    #[test]
    fn cycle_three_options() {
        assert_eq!(cycle(2, 3, SettingOp::Next), 0);
        assert_eq!(cycle(0, 3, SettingOp::Prev), 2);
    }

    #[test]
    fn cycle_set_index_jumps_and_clamps() {
        // A dropdown pick jumps straight to the chosen index, regardless of the
        // current one, and clamps to the last option if it is out of range.
        assert_eq!(cycle(0, 4, SettingOp::SetIndex(2)), 2);
        assert_eq!(cycle(3, 4, SettingOp::SetIndex(0)), 0);
        assert_eq!(cycle(1, 4, SettingOp::SetIndex(9)), 3);
    }

    #[test]
    fn known_settings_have_options() {
        assert_eq!(options("window_mode").unwrap().len(), 3);
        assert_eq!(options("render_scale").unwrap().len(), 4);
        // resolution is a dynamic dropdown: options are enumerated from the
        // display at runtime, so the static registry has none for it.
        assert!(options("resolution").is_none());
        assert!(concinnity_core::gfx::settings::is_dynamic_dropdown(
            "resolution"
        ));
        assert_eq!(options("master_volume").unwrap().len(), 5);
        // mouse_sensitivity is a slider, not a cycle row.
        assert!(options("mouse_sensitivity").is_none());
        assert!(slider("mouse_sensitivity").is_some());
    }

    #[test]
    fn volume_index_and_at_round_trip() {
        for i in 0..VOLUME_GAINS.len() {
            assert_eq!(volume_index(volume_at(i)), i);
        }
        // A non-preset gain falls back to the full (last) index.
        assert_eq!(volume_index(0.33), VOLUME_GAINS.len() - 1);
        // The default reads as the full preset.
        assert_eq!(volume_at(volume_index(DEFAULT_VOLUME)), 1.0);
    }

    #[test]
    fn mouse_sensitivity_is_a_slider_1_to_100() {
        let s = slider("mouse_sensitivity").expect("a slider");
        assert_eq!(s.range, (1.0, 100.0));
        assert!(options("mouse_sensitivity").is_none());
        // The 1..100 UI value maps linearly to radians/pixel and back.
        for &ui in &[1.0_f32, 25.0, 50.0, 100.0] {
            let stored = (s.apply)(ui);
            let back = (s.recover)(stored);
            assert!((back - ui).abs() < 1.0e-2, "ui={ui} -> {stored} -> {back}");
        }
        // Endpoints land on the radians/pixel span; values rise with the UI value.
        assert!(((s.apply)(1.0) - MOUSE_SENS_MIN).abs() < 1.0e-9);
        assert!(((s.apply)(100.0) - MOUSE_SENS_MAX).abs() < 1.0e-9);
        assert!((s.apply)(10.0) < (s.apply)(90.0));
        assert_eq!((s.format)(26.3), "26");
        // The authored default recovers to a position inside the track.
        let def = (s.recover)(DEFAULT_MOUSE_SENSITIVITY);
        assert!(
            (1.0..=100.0).contains(&def),
            "default UI value {def} in range"
        );
    }

    #[test]
    fn fov_is_a_degrees_slider() {
        let s = slider("fov").expect("a slider");
        assert_eq!(s.range, (50.0, 100.0));
        assert!(options("fov").is_none());
        // The slider value IS the degrees: apply only clamps, recover is identity.
        for &deg in &[50.0_f32, 75.0, 100.0] {
            let stored = (s.apply)(deg);
            assert!((stored - deg).abs() < 1.0e-6);
            assert!(((s.recover)(stored) - deg).abs() < 1.0e-6);
        }
        assert_eq!((s.apply)(10.0), 50.0);
        assert_eq!((s.apply)(200.0), 100.0);
        assert_eq!((s.format)(74.6), "75\u{00b0}");
        assert!((50.0..=100.0).contains(&DEFAULT_FOV));
    }

    #[test]
    fn window_mode_index_and_at_round_trip() {
        for m in [
            WindowMode::Windowed,
            WindowMode::Borderless,
            WindowMode::Fullscreen,
        ] {
            assert_eq!(window_mode_at(window_mode_index(m)), m);
        }
    }

    #[test]
    fn ssgi_sub_quality_round_trips_and_snaps() {
        // Resolution round-trips across every option.
        for r in [
            SsgiResolution::Full,
            SsgiResolution::Half,
            SsgiResolution::Quarter,
        ] {
            assert_eq!(ssgi_resolution_at(ssgi_resolution_index(r)), r);
        }
        // Ray / step levels round-trip on their preset values.
        for i in 0..SSGI_RAYS_COUNTS.len() {
            assert_eq!(ssgi_rays_index(ssgi_rays_at(i)), i);
        }
        for i in 0..SSGI_STEPS_COUNTS.len() {
            assert_eq!(ssgi_steps_index(ssgi_steps_at(i)), i);
        }
        // An authored value off the discrete levels snaps to the nearest.
        assert_eq!(ssgi_rays_index(7), 1); // 7 -> 8
        assert_eq!(ssgi_rays_index(20), 2); // 20 -> 16
        assert_eq!(ssgi_steps_index(40), 3); // 40 -> 48
        // The three SSGI sub-quality keys are cycle rows, not sliders.
        for key in ["ssgi_resolution", "ssgi_rays", "ssgi_steps"] {
            assert!(options(key).is_some(), "{key} should be a cycle row");
            assert!(slider(key).is_none(), "{key} should not be a slider");
        }
    }

    #[test]
    fn reflection_blur_round_trips() {
        for r in [
            ReflectionBlurResolution::Full,
            ReflectionBlurResolution::Half,
            ReflectionBlurResolution::Quarter,
        ] {
            assert_eq!(reflection_blur_at(reflection_blur_index(r)), r);
        }
        // It is registered as a cycle row + a governed cycle quality knob.
        assert_eq!(
            options("reflection_blur_resolution").map(|o| o.len()),
            Some(3)
        );
        assert!(QUALITY_CYCLE_KEYS.contains(&"reflection_blur_resolution"));
    }

    #[test]
    fn display_toggles_are_off_on_cycle_rows() {
        // The display-output / upscaling preferences are Off/On cycle rows, and
        // are NOT quality knobs (independent of the preset ceiling).
        for key in ["temporal_upscaling", "hdr_display", "hdr_pq"] {
            assert_eq!(options(key), Some(&["Off", "On"][..]), "{key} options");
            assert!(slider(key).is_none(), "{key} should not be a slider");
            assert!(!is_quality_toggle(key), "{key} is not a quality toggle");
            assert!(
                !QUALITY_CYCLE_KEYS.contains(&key),
                "{key} is not a quality cycle knob"
            );
        }
    }

    #[test]
    fn shadow_resolution_round_trips_and_snaps() {
        // Each discrete level round-trips through its index.
        for i in 0..SHADOW_RESOLUTION_SIZES.len() {
            assert_eq!(shadow_resolution_index(shadow_resolution_at(i)), i);
        }
        // "Off" is size 0 at index 0; the world default 2048 is index 2.
        assert_eq!(shadow_resolution_at(0), 0);
        assert_eq!(shadow_resolution_index(2048), 2);
        // An authored size off the discrete levels snaps to the nearest, and a
        // size above the top level snaps down to it.
        assert_eq!(shadow_resolution_index(1500), 1); // 1500 -> 1024
        assert_eq!(shadow_resolution_index(8192), 3); // 8192 -> 4096
        // It is a cycle row, not a slider.
        assert!(options("shadow_map_size").is_some());
        assert!(slider("shadow_map_size").is_none());
    }

    #[test]
    fn anisotropy_round_trips_and_snaps() {
        // Each discrete level round-trips through its index.
        for i in 0..ANISOTROPY_LEVELS.len() {
            assert_eq!(anisotropy_index(anisotropy_at(i)), i);
        }
        // "Off" is 1x at index 0; the world default 8x is index 3.
        assert_eq!(anisotropy_at(0), 1);
        assert_eq!(anisotropy_index(8), 3);
        // An authored degree off the discrete levels snaps to the nearest, and a
        // degree above the top level snaps down to it.
        assert_eq!(anisotropy_index(3), 1); // 3 -> 2x
        assert_eq!(anisotropy_index(32), 4); // 32 -> 16x
        // It is a cycle row, not a slider.
        assert!(options("anisotropy").is_some());
        assert!(slider("anisotropy").is_none());
    }

    #[test]
    fn shadow_distance_round_trips_and_snaps() {
        // Each discrete level round-trips through its index.
        for i in 0..SHADOW_DISTANCE_VALUES.len() {
            assert_eq!(shadow_distance_index(shadow_distance_at(i)), i);
        }
        // The world default 80 is index 1.
        assert_eq!(shadow_distance_at(1), 80);
        assert_eq!(shadow_distance_index(80), 1);
        // An authored distance off the discrete levels snaps to the nearest, and
        // one above the top level snaps down to it.
        assert_eq!(shadow_distance_index(50), 0); // 50 -> 40
        assert_eq!(shadow_distance_index(1000), 3); // 1000 -> 320
        // It is a cycle row, not a slider.
        assert!(options("shadow_distance").is_some());
        assert!(slider("shadow_distance").is_none());
    }

    #[test]
    fn shadow_cascades_round_trips_and_snaps() {
        for i in 0..SHADOW_CASCADES_VALUES.len() {
            assert_eq!(shadow_cascades_index(shadow_cascades_at(i)), i);
        }
        // The world default 4 is the last index; out-of-range falls back to it.
        assert_eq!(shadow_cascades_at(2), 4);
        assert_eq!(shadow_cascades_index(4), 2);
        assert_eq!(shadow_cascades_at(9), 4);
        // An authored count off the levels snaps to the nearest.
        assert_eq!(shadow_cascades_index(1), 0); // 1 -> 2
        assert!(options("shadow_cascades").is_some());
        assert!(slider("shadow_cascades").is_none());
    }

    #[test]
    fn shadow_update_round_trips() {
        for u in [ShadowUpdate::EveryFrame, ShadowUpdate::Hybrid] {
            assert_eq!(shadow_update_at(shadow_update_index(u)), u);
        }
        // EveryFrame leads the cycle (best / most expensive first).
        assert_eq!(shadow_update_at(0), ShadowUpdate::EveryFrame);
        assert_eq!(options("shadow_update").map(|o| o.len()), Some(2));
    }

    #[test]
    fn frame_buffering_round_trips_and_snaps() {
        for i in 0..FRAME_BUFFERING_COUNTS.len() {
            assert_eq!(frames_in_flight_index(frames_in_flight_at(i)), i);
        }
        assert_eq!(frames_in_flight_at(0), 1);
        // An out-of-range depth snaps to the nearest level.
        assert_eq!(frames_in_flight_index(4), 2); // 4 -> 3
        assert!(options("frames_in_flight").is_some());
    }

    #[test]
    fn texture_quality_pairs_cap_and_budget() {
        // Each level round-trips through its index (recovered from the pool cap),
        // and sets both the pool cap and the per-frame upload budget.
        for i in 0..TEXTURE_QUALITY_CAPS.len() {
            let (cap, budget) = texture_quality_at(i);
            assert_eq!(texture_quality_index(cap), i);
            assert_eq!(cap, TEXTURE_QUALITY_CAPS[i]);
            assert_eq!(budget, TEXTURE_QUALITY_BUDGETS[i]);
        }
        // The default world cap (96) reads as "Medium"; an authored cap off the
        // levels snaps to the nearest.
        assert_eq!(texture_quality_index(96), 1);
        assert_eq!(texture_quality_index(300), 3); // 300 -> 384 (Ultra)
        // occlusion_two_pass is an Off/On row, not a slider or preset knob.
        assert_eq!(options("occlusion_two_pass"), Some(&["Off", "On"][..]));
        assert!(slider("occlusion_two_pass").is_none());
        assert!(!is_quality_toggle("occlusion_two_pass"));
    }

    #[test]
    fn render_scale_index_and_at_round_trip() {
        for q in [
            UpscaleQuality::Quality,
            UpscaleQuality::Balanced,
            UpscaleQuality::Performance,
            UpscaleQuality::UltraPerformance,
        ] {
            assert_eq!(render_scale_at(render_scale_index(q)), q);
        }
    }

    #[test]
    fn upscale_backend_round_trips_and_vendor_gates() {
        // Every variant round-trips through its index, and the option table lines
        // up with the four variants.
        assert_eq!(options("upscale_backend").unwrap().len(), 4);
        for b in [
            UpscalerBackend::Auto,
            UpscalerBackend::Fsr3,
            UpscalerBackend::Dlss,
            UpscalerBackend::Xess,
        ] {
            assert_eq!(upscale_backend_at(upscale_backend_index(b)), b);
        }
        // It is a cycle row, not a slider.
        assert!(options("upscale_backend").is_some());
        assert!(slider("upscale_backend").is_none());
        // Auto / FSR3 are offered on every vendor; DLSS is NVIDIA-only and XeSS
        // is Intel-only, so the menu cycle skips them elsewhere. Auto / FSR3 stay
        // available even on an Unknown (Other) GPU, so the skip loop always
        // terminates.
        for vendor in [
            GpuVendor::Apple,
            GpuVendor::Nvidia,
            GpuVendor::Amd,
            GpuVendor::Intel,
            GpuVendor::Other,
        ] {
            assert!(upscale_backend_available(UpscalerBackend::Auto, vendor));
            assert!(upscale_backend_available(UpscalerBackend::Fsr3, vendor));
        }
        assert!(upscale_backend_available(
            UpscalerBackend::Dlss,
            GpuVendor::Nvidia
        ));
        assert!(!upscale_backend_available(
            UpscalerBackend::Dlss,
            GpuVendor::Amd
        ));
        assert!(upscale_backend_available(
            UpscalerBackend::Xess,
            GpuVendor::Intel
        ));
        assert!(!upscale_backend_available(
            UpscalerBackend::Xess,
            GpuVendor::Nvidia
        ));
        // The whole row is capability-gated: a device that offers a choice of
        // upscaler keeps it, one with a fixed upscaler grays it out.
        assert!(setting_available(
            "upscale_backend",
            &backend::DeviceCapabilities::ALL
        ));
        assert!(!setting_available(
            "upscale_backend",
            &backend::DeviceCapabilities {
                selectable_upscaler: false,
                ..backend::DeviceCapabilities::ALL
            }
        ));
    }

    #[test]
    fn exposure_is_a_slider_not_a_cycle() {
        // A slider key has a table entry and no cycle option list, and vice versa.
        assert!(slider("exposure").is_some());
        assert!(options("exposure").is_none());
        assert!(slider("vsync").is_none());
        assert!(slider("nope").is_none());
    }

    #[test]
    fn slider_value_and_fraction_round_trip() {
        let exposure = slider("exposure").unwrap();
        assert_eq!(exposure.value_at(0.0), -3.0);
        assert_eq!(exposure.value_at(1.0), 3.0);
        assert_eq!(exposure.value_at(0.5), 0.0);
        for &f in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let v = exposure.value_at(f);
            let back = exposure.fraction(v);
            assert!((back - f).abs() < 1.0e-5, "f={f} -> v={v} -> {back}");
        }
    }

    #[test]
    fn slider_fraction_clamps_out_of_range() {
        let exposure = slider("exposure").unwrap();
        // A value past either end pins the handle to that end.
        assert_eq!(exposure.fraction(-100.0), 0.0);
        assert_eq!(exposure.fraction(100.0), 1.0);
        // The neutral default sits at the midpoint.
        assert_eq!(exposure.fraction(0.0), 0.5);
    }

    #[test]
    fn exposure_value_is_formatted_in_stops() {
        let format = slider("exposure").unwrap().format;
        assert_eq!(format(0.0), "+0.0 EV");
        assert_eq!(format(1.5), "+1.5 EV");
        assert_eq!(format(-2.0), "-2.0 EV");
    }

    // Every slider key is listed once, resolves to its own entry, spans a
    // non-empty range its fraction mapping round-trips, and is not a cycle row.
    #[test]
    fn sliders_have_unique_keys_and_valid_ranges() {
        let mut seen = std::collections::HashSet::new();
        for s in &SLIDERS {
            assert!(seen.insert(s.key), "{} is listed twice", s.key);
            assert!(std::ptr::eq(slider(s.key).unwrap(), s), "{}", s.key);
            assert!(s.range.0 < s.range.1, "{} range must be non-empty", s.key);
            assert!(
                options(s.key).is_none(),
                "{} should not be a cycle row",
                s.key
            );
            assert_eq!(s.value_at(0.0), s.range.0);
            assert_eq!(s.value_at(1.0), s.range.1);
            for &f in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                let back = s.fraction(s.value_at(f));
                assert!((back - f).abs() < 1.0e-5, "{}: f={f} -> {back}", s.key);
            }
        }
    }

    // Applying a slider value then recovering it returns the same value, so the
    // handle never jumps when a persisted choice is re-applied at the next launch.
    #[test]
    fn slider_apply_and_recover_round_trip() {
        for s in &SLIDERS {
            let (lo, hi) = s.range;
            for v in [lo, (lo + hi) * 0.5, hi] {
                let recovered = (s.recover)((s.apply)(v));
                assert!(
                    (recovered - v).abs() < 1.0e-3,
                    "{}: v={v} recovered={recovered}",
                    s.key
                );
            }
        }
    }

    // A distinct value written through every entry's accessors reads back
    // unchanged, so no two keys share a live or persisted field.
    #[test]
    fn no_two_sliders_alias_one_field() {
        let mut post_process = PostProcessTunables::DEFAULT;
        let mut post_config = PostProcessConfig::default();
        let mut cfg = Settings::default();
        let marker = |i: usize| 1000.0 + i as f32;
        for (i, s) in SLIDERS.iter().enumerate() {
            match &s.target {
                SliderTarget::PostProcess { field, persisted } => {
                    *(field.get_mut)(&mut post_process) = marker(i);
                    *(persisted.get_mut)(&mut cfg.graphics) = Some(marker(i));
                }
                SliderTarget::PostConfig { field, persisted } => {
                    *(field.get_mut)(&mut post_config) = marker(i);
                    *(persisted.get_mut)(&mut cfg.graphics) = Some(marker(i));
                }
                SliderTarget::Ambient { persisted } => {
                    *(persisted.get_mut)(&mut cfg.graphics) = Some(marker(i));
                }
                SliderTarget::Controls { persisted, .. } => {
                    *(persisted.get_mut)(&mut cfg) = Some(marker(i));
                }
            }
        }
        for (i, s) in SLIDERS.iter().enumerate() {
            let (live, persisted) = match &s.target {
                SliderTarget::PostProcess { field, persisted } => (
                    Some(*(field.get)(&post_process)),
                    *(persisted.get)(&cfg.graphics),
                ),
                SliderTarget::PostConfig { field, persisted } => (
                    Some(*(field.get)(&post_config)),
                    *(persisted.get)(&cfg.graphics),
                ),
                SliderTarget::Ambient { persisted } => (None, *(persisted.get)(&cfg.graphics)),
                SliderTarget::Controls { persisted, .. } => (None, *(persisted.get)(&cfg)),
            };
            if let Some(v) = live {
                assert_eq!(v, marker(i), "{} shares its live field", s.key);
            }
            assert_eq!(
                persisted,
                Some(marker(i)),
                "{} shares its persisted field",
                s.key
            );
        }
    }

    // Render sliders persist the user-facing value; controls sliders persist the
    // applied value the camera and input sampling read.
    #[test]
    fn persist_keeps_ui_values_for_render_sliders_and_applied_values_for_controls() {
        let mut cfg = Settings::default();
        slider("exposure").unwrap().persist(&mut cfg, 2.0);
        assert_eq!(cfg.graphics.exposure_ev, Some(2.0));
        slider("ambient_intensity").unwrap().persist(&mut cfg, 1.5);
        assert_eq!(cfg.graphics.ambient_intensity, Some(1.5));
        let mouse = slider("mouse_sensitivity").unwrap();
        mouse.persist(&mut cfg, 100.0);
        assert_eq!(cfg.controls.mouse_sensitivity, Some((mouse.apply)(100.0)));
        let fov = slider("fov").unwrap();
        fov.persist(&mut cfg, 200.0);
        assert_eq!(cfg.graphics.fov, Some(100.0));
    }

    #[test]
    fn current_value_reads_live_state_or_the_persisted_controls() {
        let post_process = PostProcessTunables {
            exposure: 4.0,
            ..PostProcessTunables::DEFAULT
        };
        let post_config = PostProcessConfig::default();
        let mut cfg = Settings::default();
        cfg.controls.gamepad_deadzone = Some(0.25);
        let read = |key: &str| {
            slider(key)
                .unwrap()
                .current_value(&post_process, &post_config, 1.5, &cfg)
        };
        assert!((read("exposure") - 2.0).abs() < 1.0e-5);
        assert_eq!(read("ambient_intensity"), 1.5);
        assert_eq!(read("ssao_radius"), post_config.ssao_radius);
        assert_eq!(read("gamepad_deadzone"), 25.0);
        assert_eq!(read("fov"), DEFAULT_FOV);
    }

    #[test]
    fn slider_apply_clamps_match_resolve() {
        let apply = |key: &str, v: f32| (slider(key).unwrap().apply)(v);
        // Out-of-range inputs (e.g. a hand-edited settings.bin) clamp to the
        // engine's domain, matching PostProcessConfig::resolve.
        assert_eq!(apply("bloom_intensity", -5.0), 0.0);
        assert_eq!(apply("vignette", 2.0), 1.0);
        assert_eq!(apply("lut_strength", -1.0), 0.0);
        assert_eq!(apply("ambient_intensity", 100.0), 16.0);
        // Exposure stores the linear multiplier 2^ev (clamped EV).
        assert_eq!(apply("exposure", 2.0), 4.0);
        assert!(((slider("exposure").unwrap().recover)(4.0) - 2.0).abs() < 1.0e-5);
        // Per-feature sub-quality sliders clamp to their `*Settings::resolve`
        // domains; bloom_knee is lower-bounded like the other bloom params.
        assert_eq!(apply("bloom_knee", -1.0), 0.0);
        assert_eq!(apply("ssao_intensity", 100.0), 4.0);
        assert_eq!(apply("ssr_intensity", 9.0), 1.0);
        assert_eq!(apply("ssr_max_distance", 1.0e6), 200.0);
        assert_eq!(apply("ssgi_intensity", 99.0), 4.0);
        assert_eq!(apply("ssgi_max_distance", 1.0e6), 100.0);
        assert_eq!(apply("auto_exposure_min_ev", -100.0), -16.0);
        assert_eq!(apply("auto_exposure_max_ev", 100.0), 16.0);
        assert_eq!(apply("auto_exposure_speed", 100.0), 20.0);
    }

    #[test]
    fn quality_param_sliders_are_independent_sliders() {
        // The sub-quality sliders are look tuning, not preset-governed knobs.
        let quality_params: Vec<&str> = SLIDERS
            .iter()
            .filter(|s| matches!(s.target, SliderTarget::PostConfig { .. }))
            .map(|s| s.key)
            .collect();
        assert_eq!(quality_params.len(), 9);
        for key in quality_params {
            assert!(
                !QUALITY_CYCLE_KEYS.contains(&key),
                "{key} should not be preset-governed"
            );
        }
        // bloom_knee is a PostProcessParams field, not a quality param.
        assert!(matches!(
            slider("bloom_knee").unwrap().target,
            SliderTarget::PostProcess { .. }
        ));
    }

    #[test]
    fn strength_sliders_format_as_percent() {
        let format = |key: &str, v: f32| (slider(key).unwrap().format)(v);
        assert_eq!(format("vignette", 0.0), "0%");
        assert_eq!(format("vignette", 0.5), "50%");
        assert_eq!(format("lut_strength", 1.0), "100%");
        // Bloom / ambient use the plain two-decimal format.
        assert_eq!(format("bloom_intensity", 0.6), "0.60");
        assert_eq!(format("ambient_intensity", 1.25), "1.25");
    }
}
