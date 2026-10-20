//! The settings menu's rows beyond their option labels: whether a row is
//! available on a device, how a cycle row steps, the Graphics API row's
//! choices, and the slider table (each slider's range, value transforms, label
//! format, and the state it drives). How a chosen option is applied (which
//! backend call, which persisted field) lives in the settings system's drain,
//! keyed by the same `SettingKey`.
//!
//! Key-rebind rows are a third category beside cycle rows ([`options`]) and
//! sliders ([`slider`]): their value is a physical key or button, and the live
//! map is the settings state's, so nothing about them is registered here.
//!
//! [`options`]: crate::settings::options

use alloc::format;
use alloc::string::{String, ToString};
use concinnity_core::math::{SaturatingFrom, round};

use crate::components::{ControlsCommand, SettingOp, UpscalerBackend};
use crate::gfx::auto_exposure::EV_LIMIT;
use crate::platform::Platform;
use crate::render::backend::{DeviceCapabilities, GpuVendor};
use crate::settings::rows::{GRAPHICS_SLIDERS, GraphicsSlider, Lens};
use crate::settings::{
    DEFAULT_GAMEPAD_LOOK_SENSITIVITY, GraphicsApiTrial, ResolvedGraphics, SettingKey,
    StoredSettings,
};

/// The value label an unavailable row shows: the resolved value, marked as one
/// the device cannot change.
pub fn unsupported_label(value: &str) -> String {
    format!("{value} (not supported)")
}

/// Whether setting `key` can be changed on a device with the given
/// capabilities and GPU vendor, in a build that compiled the graphics APIs in
/// `compiled`. The settings menu grays out and disables an unavailable row.
/// This is the one place to gate a capability- or vendor-dependent row.
pub fn setting_available(
    key: SettingKey,
    caps: &DeviceCapabilities,
    vendor: GpuVendor,
    compiled: &[Platform],
) -> bool {
    match key {
        SettingKey::RayTracedReflections
        | SettingKey::RtReflectionResolution
        | SettingKey::RtReflectionShadows => caps.ray_tracing,
        // The upscaler selector (FSR3 / DLSS / XeSS) grays out on a device whose
        // upscaler is fixed, rather than offering a dead selection.
        SettingKey::UpscaleBackend => caps.selectable_upscaler,
        // Live wherever the upscaler row can pick DLSS, even while another
        // upscaler runs.
        SettingKey::DlssPreset => {
            caps.selectable_upscaler && upscale_backend_available(UpscalerBackend::Dlss, vendor)
        }
        // Shown on every build, but only a build with more than one graphics
        // API has a choice to make.
        SettingKey::GraphicsApi => compiled.len() > 1,
        _ => true,
    }
}

/// The Graphics API row's value at init: the API this launch renders with, and
/// the one it gave up on when the last launch on that never drew a frame.
pub fn graphics_api_label(running: Platform, reverted: Option<Platform>) -> String {
    match reverted {
        Some(failed) => format!("{} ({} failed)", running.label(), failed.label()),
        None => running.label().to_string(),
    }
}

/// A Graphics API row change: the API to persist, and the trial it starts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphicsApiChoice {
    /// The API to persist.
    pub platform: Platform,
    /// The trial the next launch runs it under; `None` for the native API.
    pub trial: Option<GraphicsApiTrial>,
}

/// Where `op` moves the Graphics API row from `chosen` among `compiled` (native
/// first). `None` when there is nothing to choose between or the choice stands.
/// Switching to a non-native API puts it on trial; the native API needs none.
pub fn graphics_api_choice(
    compiled: &[Platform],
    chosen: Platform,
    op: SettingOp,
) -> Option<GraphicsApiChoice> {
    let native = *compiled.first()?;
    if compiled.len() < 2 {
        return None;
    }
    let cur = compiled.iter().position(|p| *p == chosen).unwrap_or(0);
    let platform = *compiled.get(cycle(cur, compiled.len(), op))?;
    (platform != chosen).then(|| GraphicsApiChoice {
        platform,
        trial: (platform != native).then_some(GraphicsApiTrial::Untried),
    })
}

/// Effective volume for any stage the user has never chosen (full gain).
pub const DEFAULT_VOLUME: f32 = 1.0;

/// Effective mouse sensitivity (radians per pixel) when the user has never
/// chosen one. Matches `CameraController`'s authored default.
const DEFAULT_MOUSE_SENSITIVITY: f32 = 0.0015;

/// Effective gamepad stick deadzone (deflection fraction) when the user has
/// never chosen one.
pub const DEFAULT_GAMEPAD_DEADZONE: f32 = 0.15;

/// Effective vertical FOV in degrees when the user has never chosen one.
/// Matches `Camera3D`'s authored default.
const DEFAULT_FOV: f32 = 75.0;

/// Fraction of a slider's range one focused Left/Right pulse steps.
pub const SLIDER_STEP_FRACTION: f32 = 0.05;

/// Whether a settings change alters which render passes run or how, so the
/// temporal history accumulated under the old settings no longer matches the
/// frame. Restart-required rows change nothing until the next launch.
pub fn resets_temporal_history(key: SettingKey) -> bool {
    key == SettingKey::GraphicsQuality || key.is_quality_toggle() || key.is_quality_cycle()
}

/// Whether an upscaler backend is offered on the given GPU vendor. Auto and
/// FSR3 are vendor-agnostic; DLSS (NVIDIA NGX) is NVIDIA-only and XeSS is
/// offered on Intel. The settings-menu cycle skips the unavailable entries; the
/// backend still resolves and falls back on its own, so an unavailable explicit
/// value is always safe.
pub fn upscale_backend_available(backend: UpscalerBackend, vendor: GpuVendor) -> bool {
    match backend {
        UpscalerBackend::Auto | UpscalerBackend::Fsr3 => true,
        UpscalerBackend::Dlss => vendor == GpuVendor::Nvidia,
        UpscalerBackend::Xess => vendor == GpuVendor::Intel,
    }
}

/// Advance an option index one step in the given direction (wrapping at the
/// ends), or jump straight to a dropdown pick's absolute index. A `len` of zero
/// stays at zero. A `SetFraction` op only applies to slider settings and never
/// reaches here.
pub fn cycle(index: usize, len: usize, op: SettingOp) -> usize {
    if len == 0 {
        return 0;
    }
    match op {
        SettingOp::Prev => (index + len - 1) % len,
        // A dropdown pick jumps to its option index, clamped to the last option.
        SettingOp::SetIndex(i) => i.min(len - 1),
        // Next steps forward; the slider and rebind ops never reach a cycle
        // setting, so treating them as Next is harmless.
        SettingOp::Next
        | SettingOp::SetFraction(_)
        | SettingOp::Rebind(_)
        | SettingOp::RebindButton(_) => (index + 1) % len,
    }
}

// Mouse-sensitivity slider: a 1..100 UI scale mapped linearly to radians per
// pixel. The camera's authored default sits low on the track.
const MOUSE_SENS_MIN: f32 = 0.0003;
const MOUSE_SENS_MAX: f32 = 0.005;

// Gamepad look-sensitivity slider: the same 1..100 UI scale, mapped linearly to
// radians per second at full stick deflection. The default sits mid-track.
const GAMEPAD_LOOK_MIN: f32 = 0.5;
const GAMEPAD_LOOK_MAX: f32 = 6.0;

macro_rules! lens {
    ($($field:ident).+) => {
        Lens {
            get: |t| &t.$($field).+,
            get_mut: |t| &mut t.$($field).+,
        }
    };
}

/// The live state a slider's applied value drives, and where its choice
/// persists. The graphics targets persist the user-facing value; `Controls`
/// persists the applied value the camera and input sampling read.
pub enum SliderTarget {
    /// A graphics setting: its row in the settings resolver holds the clamp and
    /// the resolved state it drives.
    Graphics(&'static GraphicsSlider),
    /// A camera / input preference, sent live as a `ControlsCommand`.
    Controls {
        /// Where the applied value persists.
        persisted: Lens<StoredSettings, Option<f32>>,
        /// The live command carrying an applied value.
        command: fn(f32) -> ControlsCommand,
        /// The user-facing value to the applied one.
        apply: fn(f32) -> f32,
        /// The applied value when nothing is persisted.
        default: f32,
    },
}

/// One slider row.
pub struct SliderSetting {
    /// The settings row.
    pub key: SettingKey,
    /// The (min, max) user-facing value range.
    pub range: (f32, f32),
    /// The inverse of `apply`, so a handle and label re-sync to the live value.
    pub recover: fn(f32) -> f32,
    /// Value-label text for a user-facing value.
    pub format: fn(f32) -> String,
    /// What the value drives.
    pub target: SliderTarget,
}

impl SliderSetting {
    /// User-facing value to applied value, clamped to the engine's domain.
    /// Shared by the live drag and the persisted re-apply at init.
    pub fn apply(&self, value: f32) -> f32 {
        match &self.target {
            SliderTarget::Graphics(row) => (row.apply)(value),
            SliderTarget::Controls { apply, .. } => apply(value),
        }
    }

    /// The value at a `0.0..=1.0` fraction of the range. The fraction is
    /// clamped.
    pub fn value_at(&self, fraction: f32) -> f32 {
        let (lo, hi) = self.range;
        lo + (hi - lo) * fraction.clamp(0.0, 1.0)
    }

    /// The `0.0..=1.0` fraction a value sits at within the range, clamped so an
    /// out-of-range authored value pins the handle to an end.
    pub fn fraction(&self, value: f32) -> f32 {
        let (lo, hi) = self.range;
        let span = hi - lo;
        if span.abs() < f32::EPSILON {
            return 0.0;
        }
        ((value - lo) / span).clamp(0.0, 1.0)
    }

    /// The user-facing value the slider shows: the live render state, or the
    /// persisted choice (else the default) for a controls slider.
    pub fn current_value(&self, graphics: &ResolvedGraphics, persisted: &StoredSettings) -> f32 {
        let stored = match &self.target {
            SliderTarget::Graphics(row) => *(row.field.get)(graphics),
            SliderTarget::Controls {
                persisted: lens,
                default,
                ..
            } => (lens.get)(persisted).unwrap_or(*default),
        };
        (self.recover)(stored)
    }

    /// Record the user-facing `value` in the settings store.
    pub fn persist(&self, cfg: &mut StoredSettings, value: f32) {
        match &self.target {
            SliderTarget::Graphics(row) => {
                *(row.persisted.get_mut)(&mut cfg.graphics) = Some(value);
            }
            SliderTarget::Controls { persisted, .. } => {
                *(persisted.get_mut)(cfg) = Some(self.apply(value));
            }
        }
    }
}

/// The slider entry for `key`, or `None` if the key is not a slider setting.
pub fn slider(key: SettingKey) -> Option<&'static SliderSetting> {
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
    format!("{}%", i32::saturating_from(round(value * 100.0)))
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

/// Every slider row. The post-process ranges are practical UI ceilings; the
/// graphics clamps live with their rows in [`crate::settings::rows`].
static SLIDERS: [SliderSetting; 20] = [
    // Authored in EV (centered on neutral); the composite applies 2^ev.
    SliderSetting {
        key: SettingKey::Exposure,
        range: (-3.0, 3.0),
        recover: identity,
        format: format_ev,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[0]),
    },
    SliderSetting {
        key: SettingKey::BloomIntensity,
        range: (0.0, 2.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[1]),
    },
    SliderSetting {
        key: SettingKey::BloomThreshold,
        range: (0.0, 4.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[2]),
    },
    SliderSetting {
        key: SettingKey::Vignette,
        range: (0.0, 1.0),
        recover: identity,
        format: format_fraction_percent,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[3]),
    },
    SliderSetting {
        key: SettingKey::LutStrength,
        range: (0.0, 1.0),
        recover: identity,
        format: format_fraction_percent,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[4]),
    },
    SliderSetting {
        key: SettingKey::AmbientIntensity,
        range: (0.0, 4.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[5]),
    },
    // Soft-knee width below the bloom threshold, lower-bounded like the other
    // bloom params.
    SliderSetting {
        key: SettingKey::BloomKnee,
        range: (0.0, 1.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[6]),
    },
    SliderSetting {
        key: SettingKey::SsaoRadius,
        range: (0.05, 2.0),
        recover: identity,
        format: format_meters,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[7]),
    },
    SliderSetting {
        key: SettingKey::SsaoIntensity,
        range: (0.0, 4.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[8]),
    },
    SliderSetting {
        key: SettingKey::SsrIntensity,
        range: (0.0, 1.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[9]),
    },
    SliderSetting {
        key: SettingKey::SsrMaxDistance,
        range: (1.0, 200.0),
        recover: identity,
        format: format_meters,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[10]),
    },
    SliderSetting {
        key: SettingKey::SsgiIntensity,
        range: (0.0, 4.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[11]),
    },
    SliderSetting {
        key: SettingKey::SsgiMaxDistance,
        range: (0.5, 40.0),
        recover: identity,
        format: format_meters,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[12]),
    },
    // The resolve also orders the EV bounds (min <= max) when the config is
    // resolved.
    SliderSetting {
        key: SettingKey::AutoExposureMinEv,
        range: (-EV_LIMIT, EV_LIMIT),
        recover: identity,
        format: format_ev,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[13]),
    },
    SliderSetting {
        key: SettingKey::AutoExposureMaxEv,
        range: (-EV_LIMIT, EV_LIMIT),
        recover: identity,
        format: format_ev,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[14]),
    },
    SliderSetting {
        key: SettingKey::AutoExposureSpeed,
        range: (0.1, 6.0),
        recover: identity,
        format: format_two_decimals,
        target: SliderTarget::Graphics(&GRAPHICS_SLIDERS[15]),
    },
    SliderSetting {
        key: SettingKey::MouseSensitivity,
        range: (1.0, 100.0),
        recover: |stored| hundred_scale_from(stored, MOUSE_SENS_MIN, MOUSE_SENS_MAX),
        format: |v| format!("{}", i32::saturating_from(round(v))),
        target: SliderTarget::Controls {
            persisted: lens!(controls.mouse_sensitivity),
            command: |v| ControlsCommand {
                mouse_sensitivity: Some(v),
                ..ControlsCommand::default()
            },
            apply: |v| hundred_scale_to(v, MOUSE_SENS_MIN, MOUSE_SENS_MAX),
            default: DEFAULT_MOUSE_SENSITIVITY,
        },
    },
    SliderSetting {
        key: SettingKey::GamepadLookSensitivity,
        range: (1.0, 100.0),
        recover: |stored| hundred_scale_from(stored, GAMEPAD_LOOK_MIN, GAMEPAD_LOOK_MAX),
        format: |v| format!("{}", i32::saturating_from(round(v))),
        target: SliderTarget::Controls {
            persisted: lens!(controls.gamepad_look_sensitivity),
            command: |v| ControlsCommand {
                gamepad_look_sensitivity: Some(v),
                ..ControlsCommand::default()
            },
            apply: |v| hundred_scale_to(v, GAMEPAD_LOOK_MIN, GAMEPAD_LOOK_MAX),
            default: DEFAULT_GAMEPAD_LOOK_SENSITIVITY,
        },
    },
    // Shown as a percentage of stick deflection, stored as the fraction the
    // radial deadzone consumes.
    SliderSetting {
        key: SettingKey::GamepadDeadzone,
        range: (0.0, 40.0),
        recover: |stored| stored * 100.0,
        format: |v| format!("{}%", i32::saturating_from(round(v))),
        target: SliderTarget::Controls {
            persisted: lens!(controls.gamepad_deadzone),
            command: |v| ControlsCommand {
                gamepad_deadzone: Some(v),
                ..ControlsCommand::default()
            },
            apply: |v| v.clamp(0.0, 40.0) / 100.0,
            default: DEFAULT_GAMEPAD_DEADZONE,
        },
    },
    // A vertical FOV in degrees applied to every Camera3D, so apply only clamps.
    // Persisted in the graphics store alongside the look sliders.
    SliderSetting {
        key: SettingKey::Fov,
        range: (50.0, 100.0),
        recover: identity,
        format: |v| format!("{}\u{00b0}", i32::saturating_from(round(v))),
        target: SliderTarget::Controls {
            persisted: lens!(graphics.fov),
            command: |v| ControlsCommand {
                fov_y_degrees: Some(v),
                ..ControlsCommand::default()
            },
            apply: |v| v.clamp(50.0, 100.0),
            default: DEFAULT_FOV,
        },
    },
];

#[cfg(test)]
mod tests;
