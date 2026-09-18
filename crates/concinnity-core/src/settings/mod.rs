//! The ordered option labels for every user-facing cycle setting, shared by the
//! client renderer and the build pipeline. The client's settings drain reads a
//! key's labels to display + persist the chosen value; the cook reads the label
//! count to decide whether a settings row expands into a `<`/`>` stepper (two
//! options) or a click-to-open dropdown (more than two). Keeping the labels here
//! (not duplicated per crate) means the two never drift on a setting's option
//! count. How a chosen option is applied stays in the client, keyed by the same
//! [`SettingKey`].

mod key;

pub use key::{SettingKey, SettingKind};

// Ordered option labels for `vsync`: index 0 is off, index 1 is on.
pub(crate) const VSYNC_OPTIONS: [&str; 2] = ["Off", "On"];
/// Shared Off/On labels for the boolean quality toggles. Index 0 is off, 1 on,
/// so `bool as usize` indexes directly.
pub const OFF_ON_OPTIONS: [&str; 2] = ["Off", "On"];

/// Window mode options, in cycle order.
pub const WINDOW_MODE_OPTIONS: [&str; 3] = ["Windowed", "Borderless", "Fullscreen"];
// Render-scale (upscaling quality) options, in cycle order.
pub(crate) const RENDER_SCALE_OPTIONS: [&str; 4] = ["Quality", "Balanced", "Performance", "Ultra"];
/// Upscaler-backend options, in cycle order matching the UpscalerBackend enum
/// (Auto / FSR3 / DLSS / XeSS). DirectX / Vulkan only (Metal uses MetalFX).
pub const UPSCALE_BACKEND_OPTIONS: [&str; 4] = ["Auto", "FSR 3", "DLSS", "XeSS"];

// Frame-rate cap options (Video Display group), in cycle order. "Unlimited" is
// no cap; the rest are target FPS. The client pairs these with the numeric caps.
pub(crate) const FPS_CAP_OPTIONS: [&str; 6] = ["Unlimited", "30", "60", "120", "144", "240"];

/// SSGI gather sub-quality dropdowns, in cycle order. Resolution is finest-first
/// (Full/Half/Quarter), matching the enum.
pub const SSGI_RESOLUTION_OPTIONS: [&str; 3] = ["Full", "Half", "Quarter"];
pub(crate) const SSGI_RAYS_OPTIONS: [&str; 4] = ["4", "8", "16", "32"];
pub(crate) const SSGI_STEPS_OPTIONS: [&str; 4] = ["8", "12", "24", "48"];
/// Ray-traced reflection trace resolution options, finest-first (matches the
/// enum).
pub const RT_REFLECTION_RESOLUTION_OPTIONS: [&str; 3] = ["Full", "Half", "Quarter"];
/// Reflection blur resolution options, finest-first (matches the enum).
pub const REFLECTION_BLUR_OPTIONS: [&str; 3] = ["Full", "Half", "Quarter"];

/// Anti-aliasing mode options, in cycle order matching the AaMode enum: Off,
/// FXAA (cheap composite edge filter), TAA (temporal accumulation). Ascending
/// cost, so the index doubles as the aggressiveness rank the preset clamps.
pub const AA_MODE_OPTIONS: [&str; 3] = ["Off", "FXAA", "TAA"];

// Shadow-map cascade resolution options (texels), in cycle order. "Off"
// disables shadows; the rest are the per-cascade texel dimensions.
pub(crate) const SHADOW_RESOLUTION_OPTIONS: [&str; 4] = ["Off", "1024", "2048", "4096"];
/// Shadow re-render cadence options, best (most expensive) first.
pub const SHADOW_UPDATE_OPTIONS: [&str; 2] = ["Every Frame", "Hybrid"];
// Shadow-distance options (world units the cascades cover), in cycle order.
pub(crate) const SHADOW_DISTANCE_OPTIONS: [&str; 4] = ["40 m", "80 m", "160 m", "320 m"];
// Shadow cascade-count options, in cycle order.
pub(crate) const SHADOW_CASCADES_OPTIONS: [&str; 3] = ["2", "3", "4"];
// Anisotropic-filtering degree options for the scene sampler, in cycle order.
// "Off" is 1x (plain trilinear); the rest are the max anisotropy degree.
pub(crate) const ANISOTROPY_OPTIONS: [&str; 5] = ["Off", "2x", "4x", "8x", "16x"];
/// Frame-buffering (ring-buffer depth / frames-in-flight) options, in cycle
/// order. Lower is less latency, higher is smoother pacing.
pub const FRAME_BUFFERING_OPTIONS: [&str; 3] = ["1", "2", "3"];
// Texture-quality options, in cycle order (drive the streaming pool cap + the
// per-frame upload budget on the client).
pub(crate) const TEXTURE_QUALITY_OPTIONS: [&str; 4] = ["Low", "Medium", "High", "Ultra"];

/// Master "Graphics Quality" preset options, in cycle order. The labels mirror
/// `QualityPreset::ALL`'s order (locked by a client test); the `Auto` row is
/// relabeled with its resolved tier by the client, which this static table
/// cannot express.
pub const GRAPHICS_QUALITY_OPTIONS: [&str; 6] =
    ["Auto", "Low", "Medium", "High", "Ultra", "Custom"];

// Volume options shared by the master and per-bus volume rows, in cycle
// order. The client maps each to a linear gain.
pub(crate) const VOLUME_OPTIONS: [&str; 5] = ["Off", "25%", "50%", "75%", "100%"];

/// Whether `key`'s options are enumerated at runtime (a dropdown row with no
/// static label table; `options` returns `None` for it). `resolution` lists the
/// display modes the current display supports.
pub fn is_dynamic_dropdown(key: SettingKey) -> bool {
    key == SettingKey::Resolution
}

/// The option labels for a cycle setting, or `None` for a slider, a rebind, or
/// the runtime-enumerated resolution. A key with more than two labels renders as
/// a dropdown; two labels render as a `<`/`>` stepper.
pub fn options(key: SettingKey) -> Option<&'static [&'static str]> {
    use SettingKey as K;
    match key {
        K::GraphicsQuality => Some(&GRAPHICS_QUALITY_OPTIONS),
        K::Vsync => Some(&VSYNC_OPTIONS),
        K::WindowMode => Some(&WINDOW_MODE_OPTIONS),
        K::RenderScale => Some(&RENDER_SCALE_OPTIONS),
        K::UpscaleBackend => Some(&UPSCALE_BACKEND_OPTIONS),
        K::FpsCap => Some(&FPS_CAP_OPTIONS),
        K::MasterVolume | K::MusicVolume | K::SfxVolume | K::VoiceVolume => Some(&VOLUME_OPTIONS),
        K::AaMode => Some(&AA_MODE_OPTIONS),
        K::SsgiResolution => Some(&SSGI_RESOLUTION_OPTIONS),
        K::SsgiRays => Some(&SSGI_RAYS_OPTIONS),
        K::SsgiSteps => Some(&SSGI_STEPS_OPTIONS),
        K::RtReflectionResolution => Some(&RT_REFLECTION_RESOLUTION_OPTIONS),
        K::ReflectionBlurResolution => Some(&REFLECTION_BLUR_OPTIONS),
        K::ShadowMapSize => Some(&SHADOW_RESOLUTION_OPTIONS),
        K::ShadowUpdate => Some(&SHADOW_UPDATE_OPTIONS),
        K::ShadowDistance => Some(&SHADOW_DISTANCE_OPTIONS),
        K::ShadowCascades => Some(&SHADOW_CASCADES_OPTIONS),
        K::Anisotropy => Some(&ANISOTROPY_OPTIONS),
        K::FramesInFlight => Some(&FRAME_BUFFERING_OPTIONS),
        K::TextureQuality => Some(&TEXTURE_QUALITY_OPTIONS),
        // Display-output / upscaling preference + occlusion toggles (Off/On).
        K::TemporalUpscaling | K::HdrDisplay | K::HdrPq | K::OcclusionTwoPass => {
            Some(&OFF_ON_OPTIONS)
        }
        // Stats-HUD display toggles: a master and one per readout (Off/On).
        K::PerfStats | K::ShowFps | K::ShowVram => Some(&OFF_ON_OPTIONS),
        K::Ssao
        | K::Ssr
        | K::RayTracedReflections
        | K::RtReflectionShadows
        | K::Ssgi
        | K::AutoExposure => Some(&OFF_ON_OPTIONS),
        K::Resolution
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
