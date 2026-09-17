//! The typed vocabulary of every user-facing setting: cycle rows, sliders, and
//! the keyboard and gamepad rebind rows. The snake_case strings a world authors
//! and the `setting:<key>:<verb>` action grammar carries are parsed into a
//! [`SettingKey`] once, at the edge that reads them.

use crate::components::GamepadAction;
use crate::input::keymap::Bindable;

/// Which kind of settings row a [`SettingKey`] drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingKind {
    /// A row that steps through a list of option labels.
    Cycle,
    /// A row whose value is a fraction of a continuous range.
    Slider,
    /// A row that captures a keyboard key for a gameplay action.
    KeyRebind,
    /// A row that captures a gamepad button for a gameplay action.
    PadRebind,
}

macro_rules! setting_keys {
    (
        cycle { $($(#[$cycle_doc:meta])* $cycle:ident => $cycle_str:literal,)* }
        slider { $($(#[$slider_doc:meta])* $slider:ident => $slider_str:literal,)* }
    ) => {
        /// A user-facing setting, keyed by the string a world authors (e.g. `vsync`).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum SettingKey {
            $($(#[$cycle_doc])* $cycle,)*
            $($(#[$slider_doc])* $slider,)*
            /// The keyboard binding for a gameplay action.
            KeyRebind(Bindable),
            /// The gamepad binding for a gameplay action.
            PadRebind(GamepadAction),
        }

        impl SettingKey {
            /// Every setting: the cycle rows, the sliders, then the rebinds.
            pub const ALL: [SettingKey; 65] = [
                $(SettingKey::$cycle,)*
                $(SettingKey::$slider,)*
                SettingKey::KeyRebind(Bindable::Forward),
                SettingKey::KeyRebind(Bindable::Backward),
                SettingKey::KeyRebind(Bindable::Left),
                SettingKey::KeyRebind(Bindable::Right),
                SettingKey::KeyRebind(Bindable::Sprint),
                SettingKey::KeyRebind(Bindable::Jump),
                SettingKey::KeyRebind(Bindable::Interact),
                SettingKey::PadRebind(GamepadAction::Sprint),
                SettingKey::PadRebind(GamepadAction::Jump),
                SettingKey::PadRebind(GamepadAction::Interact),
            ];

            /// The snake_case key string worlds and `setting:*` actions use.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(SettingKey::$cycle => $cycle_str,)*
                    $(SettingKey::$slider => $slider_str,)*
                    SettingKey::KeyRebind(action) => action.setting_key(),
                    SettingKey::PadRebind(action) => action.setting_key(),
                }
            }

            /// Which kind of row this setting drives.
            pub const fn kind(self) -> SettingKind {
                match self {
                    $(SettingKey::$cycle)|* => SettingKind::Cycle,
                    $(SettingKey::$slider)|* => SettingKind::Slider,
                    SettingKey::KeyRebind(_) => SettingKind::KeyRebind,
                    SettingKey::PadRebind(_) => SettingKind::PadRebind,
                }
            }
        }
    };
}

setting_keys! {
    cycle {
        /// The master graphics-quality preset.
        GraphicsQuality => "graphics_quality",
        /// Vertical sync.
        Vsync => "vsync",
        /// Windowed, borderless, or fullscreen.
        WindowMode => "window_mode",
        /// The display mode, enumerated from the display at runtime.
        Resolution => "resolution",
        /// The upscaling quality tier.
        RenderScale => "render_scale",
        /// The upscaler implementation.
        UpscaleBackend => "upscale_backend",
        /// The frame-rate cap.
        FpsCap => "fps_cap",
        /// The master volume.
        MasterVolume => "master_volume",
        /// The music bus volume.
        MusicVolume => "music_volume",
        /// The sound-effects bus volume.
        SfxVolume => "sfx_volume",
        /// The voice bus volume.
        VoiceVolume => "voice_volume",
        /// The anti-aliasing mode.
        AaMode => "aa_mode",
        /// The SSGI gather resolution.
        SsgiResolution => "ssgi_resolution",
        /// The SSGI rays per pixel.
        SsgiRays => "ssgi_rays",
        /// The SSGI march steps per ray.
        SsgiSteps => "ssgi_steps",
        /// The reflection blur resolution.
        ReflectionBlurResolution => "reflection_blur_resolution",
        /// The shadow-map cascade resolution, or shadows off.
        ShadowMapSize => "shadow_map_size",
        /// The shadow re-render cadence.
        ShadowUpdate => "shadow_update",
        /// The distance the shadow cascades cover.
        ShadowDistance => "shadow_distance",
        /// The shadow cascade count.
        ShadowCascades => "shadow_cascades",
        /// The scene sampler's anisotropic filtering degree.
        Anisotropy => "anisotropy",
        /// The frames buffered ahead of presentation.
        FramesInFlight => "frames_in_flight",
        /// The texture streaming pool and upload budget tier.
        TextureQuality => "texture_quality",
        /// Temporal upscaling.
        TemporalUpscaling => "temporal_upscaling",
        /// HDR display output.
        HdrDisplay => "hdr_display",
        /// The PQ transfer function for HDR output.
        HdrPq => "hdr_pq",
        /// Two-pass occlusion culling.
        OcclusionTwoPass => "occlusion_two_pass",
        /// The stats HUD master toggle.
        PerfStats => "perf_stats",
        /// The stats HUD frame-rate readout.
        ShowFps => "show_fps",
        /// The stats HUD video-memory readout.
        ShowVram => "show_vram",
        /// Screen-space ambient occlusion.
        Ssao => "ssao",
        /// Screen-space reflections.
        Ssr => "ssr",
        /// Hardware ray-traced reflections.
        RayTracedReflections => "ray_traced_reflections",
        /// Screen-space global illumination.
        Ssgi => "ssgi",
        /// Automatic exposure adaptation.
        AutoExposure => "auto_exposure",
    }
    slider {
        /// Exposure compensation in EV.
        Exposure => "exposure",
        /// Bloom strength.
        BloomIntensity => "bloom_intensity",
        /// The luminance bloom starts at.
        BloomThreshold => "bloom_threshold",
        /// The softness of the bloom threshold.
        BloomKnee => "bloom_knee",
        /// Vignette strength.
        Vignette => "vignette",
        /// Color-grading LUT strength.
        LutStrength => "lut_strength",
        /// Image-based ambient light scale.
        AmbientIntensity => "ambient_intensity",
        /// The SSAO sample radius.
        SsaoRadius => "ssao_radius",
        /// SSAO strength.
        SsaoIntensity => "ssao_intensity",
        /// SSR strength.
        SsrIntensity => "ssr_intensity",
        /// The SSR ray length.
        SsrMaxDistance => "ssr_max_distance",
        /// SSGI strength.
        SsgiIntensity => "ssgi_intensity",
        /// The SSGI ray length.
        SsgiMaxDistance => "ssgi_max_distance",
        /// The darkest exposure auto exposure adapts to.
        AutoExposureMinEv => "auto_exposure_min_ev",
        /// The brightest exposure auto exposure adapts to.
        AutoExposureMaxEv => "auto_exposure_max_ev",
        /// How fast auto exposure adapts.
        AutoExposureSpeed => "auto_exposure_speed",
        /// Mouse look sensitivity.
        MouseSensitivity => "mouse_sensitivity",
        /// Gamepad stick look sensitivity.
        GamepadLookSensitivity => "gamepad_look_sensitivity",
        /// The gamepad stick deadzone.
        GamepadDeadzone => "gamepad_deadzone",
        /// The camera field of view.
        Fov => "fov",
    }
}

impl SettingKey {
    /// The quality-feature toggles (Video "Quality" group). Each gates a render
    /// pass whose GPU resources are built at init, so a change rebuilds them.
    pub const QUALITY_TOGGLES: [SettingKey; 5] = [
        SettingKey::Ssao,
        SettingKey::Ssr,
        SettingKey::RayTracedReflections,
        SettingKey::Ssgi,
        SettingKey::AutoExposure,
    ];

    /// The setting named by `key`, or `None` if no setting has that name.
    pub fn parse(key: &str) -> Option<SettingKey> {
        SettingKey::ALL.into_iter().find(|k| k.as_str() == key)
    }

    /// The key strings for a list of settings, in order.
    pub const fn names<const N: usize>(keys: [SettingKey; N]) -> [&'static str; N] {
        let mut names = [""; N];
        let mut i = 0;
        while i < N {
            names[i] = keys[i].as_str();
            i += 1;
        }
        names
    }

    /// Whether this is one of the [`QUALITY_TOGGLES`](Self::QUALITY_TOGGLES).
    pub fn is_quality_toggle(self) -> bool {
        Self::QUALITY_TOGGLES.contains(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_key_round_trips_through_its_string() {
        for key in SettingKey::ALL {
            assert_eq!(SettingKey::parse(key.as_str()), Some(key), "{key:?}");
        }
    }

    #[test]
    fn key_strings_are_unique() {
        for (i, a) in SettingKey::ALL.iter().enumerate() {
            for b in &SettingKey::ALL[i + 1..] {
                assert_ne!(a.as_str(), b.as_str(), "{a:?} and {b:?}");
            }
        }
    }

    #[test]
    fn parse_rejects_names_that_are_not_settings() {
        for key in ["taa", "", "key_nope", "pad_nope", "Vsync", "vsync "] {
            assert_eq!(SettingKey::parse(key), None, "{key:?}");
        }
    }

    #[test]
    fn kind_classifies_sliders_rebinds_and_cycles() {
        let count = |kind| SettingKey::ALL.iter().filter(|k| k.kind() == kind).count();
        assert_eq!(count(SettingKind::Cycle), 35);
        assert_eq!(count(SettingKind::Slider), 20);
        assert_eq!(count(SettingKind::KeyRebind), Bindable::ALL.len());
        assert_eq!(count(SettingKind::PadRebind), GamepadAction::ALL.len());
        assert_eq!(SettingKey::BloomIntensity.kind(), SettingKind::Slider);
        assert_eq!(SettingKey::Vsync.kind(), SettingKind::Cycle);
        assert_eq!(
            SettingKey::KeyRebind(Bindable::Jump).kind(),
            SettingKind::KeyRebind
        );
        assert_eq!(
            SettingKey::PadRebind(GamepadAction::Jump).kind(),
            SettingKind::PadRebind
        );
    }

    #[test]
    fn all_holds_every_rebind_action() {
        for action in Bindable::ALL {
            assert!(SettingKey::ALL.contains(&SettingKey::KeyRebind(action)));
        }
        for action in GamepadAction::ALL {
            assert!(SettingKey::ALL.contains(&SettingKey::PadRebind(action)));
        }
    }

    #[test]
    fn names_follow_the_key_order() {
        assert_eq!(
            SettingKey::names(SettingKey::QUALITY_TOGGLES),
            [
                "ssao",
                "ssr",
                "ray_traced_reflections",
                "ssgi",
                "auto_exposure"
            ]
        );
    }
}
