//! User-facing settings and how they resolve.
//!
//! Every cycle setting's option labels live in [`levels`], beside the values
//! they stand for, and are shared by the settings menu and the build pipeline:
//! the menu reads a key's labels to display and persist the chosen value; the
//! cook reads the label count to decide whether a settings row expands into a
//! `<`/`>` stepper (two options) or a click-to-open dropdown (more than two).
//!
//! [`resolve`] turns the world's authored settings, the quality preset's
//! ceiling ([`quality`]), and the user's choices ([`user`]) into the snapshot
//! the renderer is built with, reading the settings rows in [`rows`]. Reading
//! the user's choices from disk and detecting the GPU are the host's.

pub mod api_trial;
pub mod apply;
#[cfg(test)]
mod equivalence_tests;
mod key;
pub mod levels;
pub mod quality;
pub mod registry;
pub mod resolve;
pub mod rows;
pub mod stored;
pub mod system;
pub mod user;

pub use key::{SettingKey, SettingKind};
pub use quality::{QualityCeiling, QualityPreset};
pub use resolve::{
    ResolvedGraphics, ResolvedSettings, SceneOverride, SettingsDiff, SettingsLayers, WorldAssets,
    WorldSettings,
};
pub use stored::{SettingsRevision, StoredSettings};
pub use user::{GraphicsApiTrial, UserGraphicsSettings};

/// Shared Off/On labels for the boolean rows. Index 0 is off, 1 on, so
/// `bool as usize` indexes directly.
const OFF_ON_OPTIONS: [&str; 2] = ["Off", "On"];

/// Effective gamepad look sensitivity (radians per second at full stick
/// deflection) when the user has never chosen one.
pub const DEFAULT_GAMEPAD_LOOK_SENSITIVITY: f32 = 2.5;

/// Whether `key`'s options are enumerated at runtime (a dropdown row with no
/// static label table; `options` returns `None` for it). `resolution` lists the
/// display modes the current display supports, and `graphics_api` the graphics
/// APIs the running build compiled in.
pub fn is_dynamic_dropdown(key: SettingKey) -> bool {
    matches!(key, SettingKey::Resolution | SettingKey::GraphicsApi)
}

/// The option labels for a cycle setting, or `None` for a slider, a rebind, or
/// the runtime-enumerated resolution. A key with more than two labels renders as
/// a dropdown; two labels render as a `<`/`>` stepper.
pub fn options(key: SettingKey) -> Option<&'static [&'static str]> {
    use SettingKey as K;
    match key {
        K::GraphicsQuality => Some(levels::GRAPHICS_QUALITY.labels()),
        K::WindowMode => Some(levels::WINDOW_MODE.labels()),
        K::RenderScale => Some(levels::RENDER_SCALE.labels()),
        K::UpscaleBackend => Some(levels::UPSCALE_BACKEND.labels()),
        K::DlssPreset => Some(levels::DLSS_PRESET.labels()),
        K::FpsCap => Some(levels::FPS_CAP.labels()),
        K::MasterVolume | K::MusicVolume | K::SfxVolume | K::VoiceVolume => {
            Some(levels::VOLUME.labels())
        }
        K::AaMode => Some(levels::AA_MODE.labels()),
        K::SsgiResolution | K::RtReflectionResolution | K::ReflectionBlurResolution => {
            Some(levels::PASS_RESOLUTION.labels())
        }
        K::SsgiRays => Some(levels::SSGI_RAYS.labels()),
        K::ShadowMapSize => Some(levels::SHADOW_RESOLUTION.labels()),
        K::ShadowUpdate => Some(levels::SHADOW_UPDATE.labels()),
        K::ShadowDistance => Some(levels::SHADOW_DISTANCE.labels()),
        K::ShadowCascades => Some(levels::SHADOW_CASCADES.labels()),
        K::Anisotropy => Some(levels::ANISOTROPY.labels()),
        K::FramesInFlight => Some(levels::FRAMES_IN_FLIGHT.labels()),
        K::TextureQuality => Some(levels::TEXTURE_QUALITY.labels()),
        // Display-output / upscaling preference + occlusion toggles (Off/On).
        K::Vsync | K::TemporalUpscaling | K::HdrDisplay | K::HdrPq | K::OcclusionTwoPass => {
            Some(&OFF_ON_OPTIONS)
        }
        // Stats-HUD display toggles: a master and one per readout (Off/On).
        K::PerfStats
        | K::ShowFps
        | K::ShowVram
        | K::Ssao
        | K::Ssr
        | K::RayTracedReflections
        | K::RtReflectionShadows
        | K::Ssgi
        | K::AutoExposure => Some(&OFF_ON_OPTIONS),
        K::Resolution
        | K::GraphicsApi
        | K::Exposure
        | K::BloomIntensity
        | K::BloomThreshold
        | K::BloomKnee
        | K::Vignette
        | K::LutStrength
        | K::AmbientIntensity
        | K::SsaoRadius
        | K::SsaoIntensity
        | K::SsrIntensity
        | K::SsrMaxDistance
        | K::SsgiIntensity
        | K::SsgiMaxDistance
        | K::AutoExposureMinEv
        | K::AutoExposureMaxEv
        | K::AutoExposureSpeed
        | K::MouseSensitivity
        | K::GamepadLookSensitivity
        | K::GamepadDeadzone
        | K::Fov
        | K::KeyRebind(_)
        | K::PadRebind(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_is_a_dynamic_dropdown_with_no_static_options() {
        assert!(is_dynamic_dropdown(SettingKey::Resolution));
        assert!(options(SettingKey::Resolution).is_none());
        assert!(!is_dynamic_dropdown(SettingKey::WindowMode));
    }

    #[test]
    fn quality_toggles_are_off_then_on() {
        for key in SettingKey::QUALITY_TOGGLES {
            assert_eq!(options(key), Some(&["Off", "On"][..]), "{key:?} options");
        }
    }

    // Every cycle row but the runtime-enumerated resolution has static labels,
    // and no slider or rebind does.
    #[test]
    fn only_cycle_keys_have_options() {
        for key in SettingKey::ALL {
            let expected = key.kind() == SettingKind::Cycle && !is_dynamic_dropdown(key);
            assert_eq!(options(key).is_some(), expected, "{key:?}");
        }
    }

    #[test]
    fn multi_option_settings_are_dropdowns_and_toggles_are_steppers() {
        // A setting with more than two options is a dropdown; exactly two is a
        // stepper. The cook keys its row expansion off this length.
        use SettingKey as K;
        for key in [
            K::GraphicsQuality,
            K::WindowMode,
            K::RenderScale,
            K::FpsCap,
            K::MasterVolume,
            K::AaMode,
            K::ShadowMapSize,
            K::Anisotropy,
        ] {
            assert!(
                options(key).unwrap().len() > 2,
                "{key:?} should be a dropdown"
            );
        }
        for key in [K::Vsync, K::ShadowUpdate, K::TemporalUpscaling, K::Ssao] {
            assert_eq!(
                options(key).unwrap().len(),
                2,
                "{key:?} should be a stepper"
            );
        }
    }
}
