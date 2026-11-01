//! What each [`SettingKey`] stands for beyond its row: the menu group it sits
//! in and the authored asset field its choice overrides. A setting bound to a
//! field takes that field's documented meaning, default, and apply timing.

use super::SettingKey;

/// The settings-menu group a setting belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SettingGroup {
    /// The window, the display output, and frame pacing.
    Display,
    /// The quality preset and the render features and budgets it governs.
    Quality,
    /// The look of the final image: exposure, bloom, grading, and per-effect
    /// strengths.
    Image,
    /// The performance stats overlay.
    Stats,
    /// Volume.
    Audio,
    /// Look sensitivity and input bindings.
    Controls,
}

impl SettingGroup {
    /// Every group, in menu order.
    pub const ALL: [Self; 6] = [
        Self::Display,
        Self::Quality,
        Self::Image,
        Self::Stats,
        Self::Audio,
        Self::Controls,
    ];

    /// The group's display name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Display => "Display",
            Self::Quality => "Quality",
            Self::Image => "Image",
            Self::Stats => "Stats",
            Self::Audio => "Audio",
            Self::Controls => "Controls",
        }
    }

    /// Whether the group's settings change how the scene is rendered or shown.
    pub const fn is_graphics(self) -> bool {
        matches!(self, Self::Display | Self::Quality | Self::Image)
    }
}

/// An authored asset field a setting overrides: the asset's registry name and
/// the field's key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SettingField {
    /// The asset's registry name, e.g. `Reflections`.
    pub asset: &'static str,
    /// The field's authored key, e.g. `ray_traced_resolution`.
    pub key: &'static str,
}

const fn bind(asset: &'static str, key: &'static str) -> SettingField {
    SettingField { asset, key }
}

impl SettingKey {
    /// The settings-menu group this setting belongs to.
    pub const fn group(self) -> SettingGroup {
        match self {
            Self::GraphicsApi
            | Self::Vsync
            | Self::WindowMode
            | Self::Resolution
            | Self::FpsCap
            | Self::FramesInFlight
            | Self::HdrDisplay
            | Self::HdrPq
            | Self::Fov => SettingGroup::Display,
            Self::GraphicsQuality
            | Self::RenderScale
            | Self::UpscaleBackend
            | Self::DlssPreset
            | Self::AaMode
            | Self::SsgiResolution
            | Self::SsgiRays
            | Self::RtReflectionResolution
            | Self::ReflectionBlurResolution
            | Self::ShadowMapSize
            | Self::ShadowUpdate
            | Self::ShadowDistance
            | Self::ShadowCascades
            | Self::Anisotropy
            | Self::TextureQuality
            | Self::TemporalUpscaling
            | Self::OcclusionTwoPass
            | Self::Ssao
            | Self::Ssr
            | Self::RayTracedReflections
            | Self::RtReflectionShadows
            | Self::Ssgi
            | Self::AutoExposure => SettingGroup::Quality,
            Self::Exposure
            | Self::BloomIntensity
            | Self::BloomThreshold
            | Self::BloomKnee
            | Self::Vignette
            | Self::LutStrength
            | Self::AmbientIntensity
            | Self::SsaoRadius
            | Self::SsaoIntensity
            | Self::SsrIntensity
            | Self::SsrMaxDistance
            | Self::SsgiIntensity
            | Self::SsgiMaxDistance
            | Self::AutoExposureMinEv
            | Self::AutoExposureMaxEv
            | Self::AutoExposureSpeed => SettingGroup::Image,
            Self::PerfStats | Self::ShowFps | Self::ShowVram => SettingGroup::Stats,
            Self::MasterVolume | Self::MusicVolume | Self::SfxVolume | Self::VoiceVolume => {
                SettingGroup::Audio
            }
            Self::MouseSensitivity
            | Self::GamepadLookSensitivity
            | Self::GamepadDeadzone
            | Self::KeyRebind(_)
            | Self::PadRebind(_) => SettingGroup::Controls,
        }
    }

    /// The authored asset field this setting overrides, or `None` for a
    /// setting with no authored counterpart (the preset, the graphics API, a
    /// volume, a binding).
    pub const fn field(self) -> Option<SettingField> {
        match self {
            Self::Vsync => Some(bind("FramePacing", "vsync")),
            Self::FpsCap => Some(bind("FramePacing", "fps_cap")),
            Self::FramesInFlight => Some(bind("FramePacing", "frames_in_flight")),
            Self::WindowMode => Some(bind("Window", "mode")),
            Self::HdrDisplay => Some(bind("Display", "hdr")),
            Self::HdrPq => Some(bind("Display", "pq")),
            Self::Fov => Some(bind("Camera3D", "fov_y_degrees")),
            Self::AaMode => Some(bind("AntiAliasing", "mode")),
            Self::TemporalUpscaling => Some(bind("AntiAliasing", "upscaling")),
            Self::RenderScale => Some(bind("AntiAliasing", "upscale_quality")),
            Self::UpscaleBackend => Some(bind("AntiAliasing", "upscaler")),
            Self::Ssao => Some(bind("AmbientOcclusion", "enabled")),
            Self::SsaoRadius => Some(bind("AmbientOcclusion", "radius")),
            Self::SsaoIntensity => Some(bind("AmbientOcclusion", "intensity")),
            Self::Ssr => Some(bind("Reflections", "screen_space")),
            Self::RayTracedReflections => Some(bind("Reflections", "ray_traced")),
            Self::RtReflectionShadows => Some(bind("Reflections", "ray_traced_shadows")),
            Self::RtReflectionResolution => Some(bind("Reflections", "ray_traced_resolution")),
            Self::ReflectionBlurResolution => Some(bind("Reflections", "blur_resolution")),
            Self::SsrIntensity => Some(bind("Reflections", "intensity")),
            Self::SsrMaxDistance => Some(bind("Reflections", "max_distance")),
            Self::Ssgi => Some(bind("IndirectLighting", "technique")),
            Self::SsgiResolution => Some(bind("IndirectLighting", "ssgi_resolution")),
            Self::SsgiRays => Some(bind("IndirectLighting", "ssgi_rays")),
            Self::SsgiIntensity => Some(bind("IndirectLighting", "ssgi_intensity")),
            Self::SsgiMaxDistance => Some(bind("IndirectLighting", "ssgi_max_distance")),
            Self::AmbientIntensity => Some(bind("IndirectLighting", "ambient_intensity")),
            Self::ShadowMapSize => Some(bind("Shadows", "map_size")),
            Self::ShadowUpdate => Some(bind("Shadows", "update")),
            Self::ShadowDistance => Some(bind("Shadows", "distance")),
            Self::ShadowCascades => Some(bind("Shadows", "cascades")),
            Self::Anisotropy => Some(bind("TextureFiltering", "anisotropy")),
            Self::OcclusionTwoPass => Some(bind("Culling", "two_pass_occlusion")),
            Self::AutoExposure => Some(bind("Exposure", "auto")),
            Self::Exposure => Some(bind("Exposure", "ev")),
            Self::AutoExposureMinEv => Some(bind("Exposure", "auto_min_ev")),
            Self::AutoExposureMaxEv => Some(bind("Exposure", "auto_max_ev")),
            Self::AutoExposureSpeed => Some(bind("Exposure", "auto_speed")),
            Self::BloomIntensity => Some(bind("Bloom", "intensity")),
            Self::BloomThreshold => Some(bind("Bloom", "threshold")),
            Self::BloomKnee => Some(bind("Bloom", "knee")),
            Self::Vignette => Some(bind("Vignette", "strength")),
            Self::LutStrength => Some(bind("ColorLut", "strength")),
            Self::GraphicsApi
            | Self::GraphicsQuality
            | Self::Resolution
            | Self::DlssPreset
            | Self::TextureQuality
            | Self::PerfStats
            | Self::ShowFps
            | Self::ShowVram
            | Self::MasterVolume
            | Self::MusicVolume
            | Self::SfxVolume
            | Self::VoiceVolume
            | Self::MouseSensitivity
            | Self::GamepadLookSensitivity
            | Self::GamepadDeadzone
            | Self::KeyRebind(_)
            | Self::PadRebind(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_graphics_slider_and_quality_row_is_bound_to_a_field() {
        for key in SettingKey::QUALITY_TOGGLES
            .into_iter()
            .chain(SettingKey::QUALITY_CYCLES)
        {
            assert!(key.field().is_some(), "{key:?}");
            assert_eq!(key.group(), SettingGroup::Quality, "{key:?}");
        }
        for slider in &crate::settings::rows::GRAPHICS_SLIDERS {
            assert!(slider.key.field().is_some(), "{:?}", slider.key);
            assert_eq!(slider.key.group(), SettingGroup::Image, "{:?}", slider.key);
        }
    }

    #[test]
    fn only_display_quality_and_image_are_graphics() {
        let graphics: alloc::vec::Vec<_> = SettingGroup::ALL
            .into_iter()
            .filter(|g| g.is_graphics())
            .collect();
        assert_eq!(
            graphics,
            [
                SettingGroup::Display,
                SettingGroup::Quality,
                SettingGroup::Image
            ]
        );
    }
}
