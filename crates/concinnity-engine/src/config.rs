// src/config.rs: the persistent runtime settings store.
//
// `Settings` (runtime choices made in the in-engine settings menu: graphics,
// audio, controls) lives in the project at `.concinnity/settings` (the
// `settings` file under the state directory), the mutable sibling of the
// build-regenerated `.concinnity/data`. It is
// stored as CBOR: binary like the data blobs, but self-describing, so adding
// or removing a setting never invalidates an existing file (a missing field
// falls back to its default, an unknown field is ignored). bincode, which the
// data blobs use, would be wrong here: it is positional, so it is safe only
// because the data blobs are regenerated each build, whereas settings persist.
//
// Unknown fields are ignored on load so future additions are forwards-compatible.

use serde::{Deserialize, Serialize};
use std::path::Path;

// The runtime settings store: choices made in the in-engine settings menu.
// Persisted as CBOR at `.concinnity/settings`. Each field is
// `Option` (via the sub-structs): `None` means "use the world's default" so an
// unchanged setting never overrides the authored value.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Settings {
    #[serde(default)]
    pub graphics: GraphicsSettings,
    #[serde(default)]
    pub audio: AudioSettings,
    #[serde(default)]
    pub controls: ControlsSettings,
}

// Persisted overrides for graphics settings. Missing fields stay `None` and
// fall back to the world's GraphicsConfig / Window / PostProcessConfig defaults.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct GraphicsSettings {
    // Master graphics-quality preset. `None` means never configured: the first
    // launch seeds `Auto` (detect the GPU tier and clamp quality under the
    // world's authored look) and saves once. `Auto` re-resolves from the
    // detected tier each launch; a named tier is a fixed ceiling; `Custom`
    // imposes no ceiling (only the per-field overrides below apply).
    #[serde(default)]
    pub(crate) quality_preset: Option<crate::gfx::quality_preset::QualityPreset>,
    // Display sync (vsync). `None` uses the world's `GraphicsConfig.vsync`.
    #[serde(default)]
    pub vsync: Option<bool>,
    // Frame-rate cap in FPS (0 = unlimited). `None` uses the world's
    // `GraphicsConfig.fps_cap`. Applied live (the render loop's frame pacer reads
    // it each frame); independent of the quality preset.
    #[serde(default)]
    pub(crate) fps_cap: Option<u32>,
    // Stats-HUD display toggles. `perf_stats` is the master "Display performance
    // stats" switch; `show_fps` / `show_vram` gate the individual readouts under
    // it. `None` means shown (the engine default), so an existing settings file
    // keeps the frame-rate / GPU-memory chips visible. Applied live; with the
    // master off the per-readout rows stay visible in the menu but grayed out.
    #[serde(default)]
    pub(crate) perf_stats: Option<bool>,
    #[serde(default)]
    pub(crate) show_fps: Option<bool>,
    #[serde(default)]
    pub(crate) show_vram: Option<bool>,
    // Window mode (windowed / borderless / fullscreen). `None` uses the world's
    // `Window.mode`. Applied live.
    #[serde(default)]
    pub(crate) window_mode: Option<crate::assets::WindowMode>,
    // Chosen fullscreen display mode [width, height, refresh_hz] in pixels
    // (refresh_hz 0 = unknown / keep the display's rate). `None` means never
    // chosen: the display keeps its own mode and the Resolution row shows it.
    // Fullscreen-only (the row is grayed in windowed / borderless, where the
    // window itself defines the size); applied live while fullscreen.
    #[serde(default)]
    pub resolution: Option<[u32; 3]>,
    // Render-scale preset (upscaling quality). `None` uses the world's
    // `PostProcessConfig.upscale_quality`. Applied at next launch (the upscaler
    // and render targets are sized once at init).
    #[serde(default)]
    pub(crate) render_scale: Option<crate::assets::UpscaleQuality>,
    // Upscaler backend (`PostProcessConfig.upscale_backend`: Auto / FSR3 / DLSS /
    // XeSS). `None` uses the world's value. Applied at next launch (the upscaler
    // is selected + built once at init); DirectX / Vulkan only (Metal uses
    // MetalFX). A user/hardware preference, independent of the quality preset.
    #[serde(default)]
    pub(crate) upscale_backend: Option<crate::assets::UpscalerBackend>,
    // Exposure offset in photographic stops. `None` uses the world's
    // `PostProcessConfig.exposure_ev`. Applied live (a pure post-process
    // uniform), and re-applied at init for a persisted choice.
    #[serde(default)]
    pub exposure_ev: Option<f32>,
    // Bloom additive strength. `None` uses the world's
    // `PostProcessConfig.bloom_intensity`. Applied live.
    #[serde(default)]
    pub bloom_intensity: Option<f32>,
    // Bloom luminance threshold. `None` uses the world's
    // `PostProcessConfig.bloom_threshold`. Applied live.
    #[serde(default)]
    pub bloom_threshold: Option<f32>,
    // Bloom soft-knee width. `None` uses the world's `PostProcessConfig.bloom_knee`.
    // Applied live (a `PostProcessParams` field, like the other bloom sliders).
    #[serde(default)]
    pub(crate) bloom_knee: Option<f32>,
    // Vignette strength in [0, 1]. `None` uses the world's
    // `PostProcessConfig.vignette_strength`. Applied live.
    #[serde(default)]
    pub(crate) vignette: Option<f32>,
    // Colour-LUT blend in [0, 1]. `None` uses the world's
    // `PostProcessConfig.lut_strength`. Applied live.
    #[serde(default)]
    pub lut_strength: Option<f32>,
    // Ambient (IBL) light scale. `None` uses the world's
    // `PostProcessConfig.ambient_intensity`. Applied live on Metal (it rides
    // `LightUniforms`, not `PostProcessParams`); re-applied at init.
    #[serde(default)]
    pub ambient_intensity: Option<f32>,
    // Camera vertical field of view in degrees. `None` uses the world's authored
    // `Camera3D.fov_y_degrees`. Applied live (Camera3DSystem updates the camera
    // from a ControlsCommand; the projection rebuilds from it each frame) and
    // re-applied at init. A user preference, independent of the quality preset.
    #[serde(default)]
    pub fov: Option<f32>,
    // Anti-aliasing mode (`PostProcessConfig.aa_mode`: off / FXAA / TAA). `None`
    // uses the world's value. Applied live on Metal (the TAA pass rebuilds and
    // the composite FXAA flag updates in place) and governed by the quality
    // preset ceiling like the toggles below.
    #[serde(default)]
    pub aa_mode: Option<crate::assets::AaMode>,
    // Quality-feature toggles. Each `None` uses the world's
    // `PostProcessConfig` value. They gate render passes whose GPU resources
    // (pipelines, targets, acceleration structures) are built at init, so a
    // change rebuilds those resources: applied live on Metal (the backend
    // rebuilds the affected effects in place); on backends without a live path
    // the choice persists and applies at the next launch.
    #[serde(default)]
    pub ssao: Option<bool>,
    #[serde(default)]
    pub ssr: Option<bool>,
    // Hardware ray-traced reflections (`PostProcessConfig.ray_traced_reflections`).
    #[serde(default)]
    pub ray_traced_reflections: Option<bool>,
    // Screen-space global illumination (`PostProcessConfig.indirect_lighting ==
    // ssgi`).
    #[serde(default)]
    pub ssgi: Option<bool>,
    #[serde(default)]
    pub auto_exposure: Option<bool>,
    // SSGI gather sub-quality: internal resolution, hemisphere rays per pixel,
    // and ray-march steps per ray (`PostProcessConfig.ssgi_resolution`/`_rays`/
    // `_steps`). Each `None` uses the world's value. Applied live on Metal (the
    // backend rebuilds the SSGI pass in place); persisted + applied at the next
    // launch on backends without a live path. Governed by the quality preset
    // ceiling like the toggles above.
    #[serde(default)]
    pub(crate) ssgi_resolution: Option<crate::assets::SsgiResolution>,
    #[serde(default)]
    pub(crate) ssgi_rays: Option<u32>,
    #[serde(default)]
    pub(crate) ssgi_steps: Option<u32>,
    // Roughness-aware reflection blur resolution
    // (`PostProcessConfig.reflection_blur_resolution`). `None` uses the world's
    // value. Applied live on Metal; governed by the quality preset ceiling like
    // the SSGI sub-quality above (only bites when a reflection feature is on).
    #[serde(default)]
    pub(crate) reflection_blur_resolution: Option<crate::assets::ReflectionBlurResolution>,
    // Per-feature sub-quality tunables (SSAO radius / intensity, SSR intensity /
    // distance, SSGI intensity / distance, auto-exposure EV bounds + speed). Each
    // `None` uses the world's `PostProcessConfig` value. Applied live on Metal via
    // `update_quality_params` (the backend re-reads them into a per-frame uniform,
    // no pass rebuild); look-tuning knobs, independent of the quality preset.
    #[serde(default)]
    pub ssao_radius: Option<f32>,
    #[serde(default)]
    pub ssao_intensity: Option<f32>,
    #[serde(default)]
    pub ssr_intensity: Option<f32>,
    #[serde(default)]
    pub ssr_max_distance: Option<f32>,
    #[serde(default)]
    pub ssgi_intensity: Option<f32>,
    #[serde(default)]
    pub(crate) ssgi_max_distance: Option<f32>,
    #[serde(default)]
    pub(crate) auto_exposure_min_ev: Option<f32>,
    #[serde(default)]
    pub(crate) auto_exposure_max_ev: Option<f32>,
    #[serde(default)]
    pub(crate) auto_exposure_speed: Option<f32>,
    // Shadow quality: cascade map resolution in texels (0 disables shadows) and
    // re-render cadence (`GraphicsConfig.shadow_map_size` / `shadow_update`).
    // `None` uses the world's value. Resolution is restart-required (the shadow
    // map array is sized once at backend init); cadence is applied live on Metal.
    // Both are governed by the quality preset ceiling like the toggles above.
    #[serde(default)]
    pub shadow_map_size: Option<u32>,
    #[serde(default)]
    pub(crate) shadow_update: Option<crate::assets::ShadowUpdate>,
    // Shadow distance in world units (`GraphicsConfig.shadow_distance`). `None`
    // uses the world's value. Applied live on Metal (the cascade-split math reads
    // it each frame) and governed by the quality preset ceiling like the shadow
    // knobs above.
    #[serde(default)]
    pub shadow_distance: Option<u32>,
    // Shadow cascade count, 1..4 (`GraphicsConfig.shadow_cascades`). `None` uses
    // the world's value. Applied live on Metal (the per-frame split + schedule
    // read it) and governed by the quality preset ceiling like the shadow knobs
    // above.
    #[serde(default)]
    pub shadow_cascades: Option<u32>,
    // Anisotropic-filtering degree for the scene sampler
    // (`GraphicsConfig.anisotropy`). `None` uses the world's value. Restart-
    // required (the sampler is built once at backend init) and governed by the
    // quality preset ceiling like the shadow knobs above.
    #[serde(default)]
    pub(crate) anisotropy: Option<u32>,
    // Display-output / upscaling preferences. Unlike the quality knobs above,
    // these are independent of the master preset (a user choice, not a tier), and
    // each is restart-required: the swapchain format / render targets are sized
    // once at backend init, so a change persists and applies at the next launch.
    // `None` uses the world's `PostProcessConfig` value.
    #[serde(default)]
    pub(crate) temporal_upscaling: Option<bool>,
    #[serde(default)]
    pub(crate) hdr_display: Option<bool>,
    #[serde(default)]
    pub(crate) hdr_pq: Option<bool>,
    // System / streaming restart preferences, independent of the master preset
    // (like the display rows above) and each restart-required: ring-buffer depth
    // (`GraphicsConfig.frames_in_flight`), two-pass occlusion culling
    // (`PostProcessConfig.occlusion_two_pass`), and the texture-streaming pool /
    // per-frame upload budget (`StreamingConfig.texture_cap` / `texture_budget`,
    // driven together by one "Texture Quality" row). `None` uses the world's value.
    #[serde(default)]
    pub(crate) frames_in_flight: Option<u32>,
    #[serde(default)]
    pub occlusion_two_pass: Option<bool>,
    #[serde(default)]
    pub texture_cap: Option<u32>,
    #[serde(default)]
    pub(crate) texture_budget: Option<u32>,
}

// Persisted overrides for audio settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct AudioSettings {
    // Master output volume as a linear gain (0.0 = silent, 1.0 = full). `None`
    // leaves each emitter at its authored `AudioEmitter.volume`. Applied when a
    // world's audio initializes (the main menu itself has no audio).
    #[serde(default)]
    pub(crate) master_volume: Option<f32>,
    // Per-bus volumes under the master, same semantics (`None` = unity).
    #[serde(default)]
    pub(crate) music_volume: Option<f32>,
    #[serde(default)]
    pub(crate) sfx_volume: Option<f32>,
    #[serde(default)]
    pub(crate) voice_volume: Option<f32>,
}

// Persisted overrides for control settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ControlsSettings {
    // Mouse-look sensitivity in radians per pixel. `None` uses the controlling
    // camera's authored `CameraController.mouse_sensitivity`. Applied when the
    // camera controller initializes.
    #[serde(default)]
    pub mouse_sensitivity: Option<f32>,
    // Gameplay movement key bindings (forward/back/strafe/sprint/jump/interact).
    // `None` uses the engine defaults (W/S/A/D/Shift/Space/E). Applied live: the
    // active backend decodes physical keys through this map.
    #[serde(default)]
    pub(crate) keymap: Option<crate::gfx::keymap::KeyMap>,
    // Gamepad look sensitivity in radians per second at full stick deflection.
    // `None` uses the engine default. Applied when the camera controller
    // initializes and live via ControlsCommand.
    #[serde(default)]
    pub(crate) gamepad_look_sensitivity: Option<f32>,
    // Gamepad stick deadzone as a deflection fraction in [0, 1]. `None` uses
    // the engine default. Applied by the input sampling.
    #[serde(default)]
    pub(crate) gamepad_deadzone: Option<f32>,
    // Gamepad action button bindings (sprint/jump/interact). `None` uses the
    // engine defaults (L3/South/West). Applied by the input sampling.
    #[serde(default)]
    pub(crate) gamepad_map: Option<crate::assets::GamepadMap>,
}

impl Settings {
    // Load from the `settings` file (CBOR). When the file is absent, fall back
    // to migrating any graphics/audio/controls choices from the legacy
    // `config.json` (where they used to live) so an existing user's choices are
    // not silently reset. The migrated values are persisted on the next `save()`
    // (a settings change). Returns defaults when nothing is stored or the file
    // is unreadable.
    pub(crate) fn load() -> Self {
        Self::load_from(&concinnity_store::paths::settings_path())
    }

    // Persist to the `settings` file as CBOR. Creates the state directory as
    // needed.
    pub(crate) fn save(&self) -> std::io::Result<()> {
        self.save_to(&concinnity_store::paths::settings_path())
    }

    // Read settings from `path`. Split from `load` so the serialize-read path
    // can be tested against a sandbox file, never the developer's real one.
    fn load_from(path: &Path) -> Self {
        // No settings file yet, or a truncated / incompatible one: start from
        // defaults rather than wiping silently mid-run.
        crate::cbor_file::read(path, "settings store").unwrap_or_default()
    }

    // Write settings to `path`. Split from `save` so the serialize-write path
    // can be tested against a sandbox file, never the developer's real one.
    fn save_to(&self, path: &Path) -> std::io::Result<()> {
        crate::cbor_file::write(path, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_cbor_roundtrip() {
        let s = Settings {
            graphics: GraphicsSettings {
                quality_preset: Some(crate::gfx::quality_preset::QualityPreset::High),
                vsync: Some(true),
                fps_cap: Some(144),
                resolution: Some([1920, 1080, 120]),
                upscale_backend: Some(crate::assets::UpscalerBackend::Xess),
                exposure_ev: Some(-1.5),
                bloom_intensity: Some(0.8),
                bloom_threshold: Some(1.2),
                vignette: Some(0.3),
                lut_strength: Some(0.75),
                ambient_intensity: Some(1.5),
                fov: Some(90.0),
                aa_mode: Some(crate::assets::AaMode::Taa),
                ssao: Some(false),
                ssr: Some(true),
                ray_traced_reflections: Some(false),
                ssgi: Some(true),
                auto_exposure: Some(false),
                ssgi_resolution: Some(crate::assets::SsgiResolution::Quarter),
                ssgi_rays: Some(16),
                ssgi_steps: Some(24),
                reflection_blur_resolution: Some(crate::assets::ReflectionBlurResolution::Full),
                bloom_knee: Some(0.4),
                ssao_radius: Some(0.6),
                ssao_intensity: Some(1.2),
                ssr_intensity: Some(0.8),
                ssr_max_distance: Some(50.0),
                ssgi_intensity: Some(0.7),
                ssgi_max_distance: Some(10.0),
                auto_exposure_min_ev: Some(-6.0),
                auto_exposure_max_ev: Some(6.0),
                auto_exposure_speed: Some(2.0),
                shadow_map_size: Some(4096),
                shadow_update: Some(crate::assets::ShadowUpdate::EveryFrame),
                shadow_distance: Some(160),
                shadow_cascades: Some(3),
                anisotropy: Some(16),
                temporal_upscaling: Some(true),
                hdr_display: Some(true),
                hdr_pq: Some(false),
                frames_in_flight: Some(3),
                occlusion_two_pass: Some(true),
                texture_cap: Some(192),
                texture_budget: Some(8),
                ..Default::default()
            },
            audio: AudioSettings {
                master_volume: Some(0.5),
                music_volume: Some(0.75),
                sfx_volume: Some(1.0),
                voice_volume: Some(0.25),
            },
            controls: ControlsSettings {
                mouse_sensitivity: Some(0.0025),
                keymap: Some(crate::gfx::keymap::KeyMap {
                    forward: crate::assets::InputKey::Up,
                    ..crate::gfx::keymap::KeyMap::default()
                }),
                gamepad_look_sensitivity: Some(3.0),
                gamepad_deadzone: Some(0.2),
                gamepad_map: Some(crate::assets::GamepadMap {
                    jump: crate::assets::GamepadButton::East,
                    ..crate::assets::GamepadMap::default()
                }),
            },
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&s, &mut bytes).unwrap();
        let loaded: Settings = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(loaded, s);
    }

    #[test]
    fn settings_empty_cbor_map_is_all_defaults() {
        // An empty CBOR map deserializes to all-default (every section's fields
        // are `#[serde(default)]`), i.e. "use the world's values".
        let mut bytes = Vec::new();
        ciborium::into_writer(&std::collections::BTreeMap::<String, u8>::new(), &mut bytes)
            .unwrap();
        let loaded: Settings = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(loaded, Settings::default());
    }

    // Schema evolution: a file written by an older build (fewer fields) and one
    // written by a newer build (an extra field) both still load. This is the
    // whole reason for choosing self-describing CBOR over positional bincode.
    #[test]
    fn settings_tolerate_missing_and_unknown_fields() {
        #[derive(Serialize)]
        struct OtherShape {
            // Only one known section present...
            graphics: GraphicsSettings,
            // ...plus a field this build has never heard of.
            some_future_setting: u32,
        }
        let other = OtherShape {
            graphics: GraphicsSettings {
                vsync: Some(false),
                ..Default::default()
            },
            some_future_setting: 7,
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&other, &mut bytes).unwrap();
        let loaded: Settings = ciborium::from_reader(&bytes[..]).unwrap();
        // Known field carried; missing sections defaulted; unknown field ignored.
        assert_eq!(loaded.graphics.vsync, Some(false));
        assert_eq!(loaded.audio, AudioSettings::default());
        assert_eq!(loaded.controls, ControlsSettings::default());
    }

    // Regression guard: the on-disk `save`/`load` path must stay sandboxable so a
    // test can never clobber the developer's real `.concinnity/settings` file.
    // Drives the real serialize-write-read cycle, but against a temp file. A
    // non-default `render_scale` mirrors a real persisted choice and proves a
    // populated field survives the round trip (it is the field whose loss the
    // original 271 -> 260 byte clobber would have shown).
    #[test]
    fn settings_save_load_roundtrip_is_sandboxed() {
        let s = Settings {
            graphics: GraphicsSettings {
                render_scale: Some(crate::assets::UpscaleQuality::Performance),
                vsync: Some(true),
                exposure_ev: Some(-1.5),
                ..Default::default()
            },
            ..Default::default()
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings");
        // The sandbox is somewhere else entirely, never the real settings file.
        assert_ne!(path, concinnity_store::paths::settings_path());

        s.save_to(&path).unwrap();
        // The write landed in the sandbox under the expected file name.
        assert!(path.exists());

        let loaded = Settings::load_from(&path);
        assert_eq!(loaded, s);
    }

    // Settings resolve to the `settings` file directly under the state dir.
    #[test]
    fn settings_path_is_under_state_dir() {
        let p = concinnity_store::paths::settings_path();
        assert_eq!(p.file_name().unwrap(), "settings");
        assert_eq!(p, concinnity_store::paths::state_dir().join("settings"));
    }
}
