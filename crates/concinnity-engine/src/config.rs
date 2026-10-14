//! The persistent runtime settings store.
//!
//! `Settings` (runtime choices made in the in-engine settings menu: graphics,
//! audio, controls) lives in the project at the state root's `settings` (the
//! `settings` file under the state directory), the mutable sibling of the
//! build-regenerated `data`. It is
//! stored as CBOR: binary like the data blobs, but self-describing, so adding
//! or removing a setting never invalidates an existing file (a missing field
//! falls back to its default, an unknown field is ignored). bincode, which the
//! data blobs use, would be wrong here: it is positional, so it is safe only
//! because the data blobs are regenerated each build, whereas settings persist.
//!
//! Unknown fields are ignored on load so future additions are forwards-compatible.

use concinnity_core::components::GamepadAction;
use concinnity_core::components::GamepadMap;
use concinnity_core::input::keymap;
use concinnity_core::settings::UserGraphicsSettings;
use concinnity_host::store::paths::StateTree;
use serde::{Deserialize, Serialize};
use std::path::Path;

// The runtime settings store: choices made in the in-engine settings menu.
// Persisted as CBOR at the state root's `settings`. Each field is
// `Option` (via the sub-structs): `None` means "use the world's default" so an
// unchanged setting never overrides the authored value.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Settings {
    #[serde(default)]
    pub graphics: UserGraphicsSettings,
    #[serde(default)]
    pub audio: AudioSettings,
    #[serde(default)]
    pub controls: ControlsSettings,
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
    #[serde(default, deserialize_with = "lenient_bindings")]
    pub(crate) keymap: Option<keymap::KeyMap>,
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
    #[serde(default, deserialize_with = "lenient_bindings")]
    pub(crate) gamepad_map: Option<GamepadMap>,
}

// A persisted binding map that must keep each input bound to one action.
trait BindingMap: serde::de::DeserializeOwned {
    const NAME: &'static str;
    fn binds_each_input_once(&self) -> bool;
}

impl BindingMap for keymap::KeyMap {
    const NAME: &'static str = "key";
    fn binds_each_input_once(&self) -> bool {
        let keys = keymap::Bindable::ALL.map(|action| self.get(action));
        keys.iter()
            .enumerate()
            .all(|(i, key)| !keys[..i].contains(key))
    }
}

impl BindingMap for GamepadMap {
    const NAME: &'static str = "gamepad";
    fn binds_each_input_once(&self) -> bool {
        let buttons = GamepadAction::ALL.map(|action| self.get(action));
        buttons
            .iter()
            .enumerate()
            .all(|(i, button)| !buttons[..i].contains(button))
    }
}

// Read a binding map whose every field has a default, dropping only the
// bindings this build cannot read (a renamed key or button): each entry is kept
// when it reads on its own, and the dropped ones fall back to their defaults.
// A repair that would bind one input to two actions drops the whole map.
fn lenient_bindings<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: BindingMap,
{
    let value = ciborium::Value::deserialize(d)?;
    if let Ok(map) = value.deserialized::<Option<T>>() {
        return Ok(map);
    }
    let ciborium::Value::Map(entries) = value else {
        tracing::warn!("{} bindings unreadable, using the defaults", T::NAME);
        return Ok(None);
    };
    let (readable, dropped): (Vec<_>, Vec<_>) = entries.into_iter().partition(|entry| {
        ciborium::Value::Map(vec![entry.clone()])
            .deserialized::<T>()
            .is_ok()
    });
    let names: Vec<_> = dropped
        .iter()
        .map(|(key, _)| key.as_text().unwrap_or("?"))
        .collect();
    let repaired = ciborium::Value::Map(readable)
        .deserialized::<T>()
        .ok()
        .filter(T::binds_each_input_once);
    match repaired {
        Some(_) => tracing::warn!(
            "{} bindings unreadable, using their defaults: {}",
            T::NAME,
            names.join(", ")
        ),
        None => tracing::warn!(
            "{} bindings unreadable ({}) and their defaults collide, using the default map",
            T::NAME,
            names.join(", ")
        ),
    }
    Ok(repaired)
}

impl Settings {
    // Load from `tree`'s `settings` file (CBOR). Returns defaults when nothing
    // is stored or the file is unreadable.
    pub(crate) fn load(tree: Option<&StateTree>) -> Self {
        tree.map(|tree| Self::load_from(&tree.settings_path()))
            .unwrap_or_default()
    }

    // Persist to `tree`'s `settings` file as CBOR. Creates the directory as
    // needed. An app with no state tree has nowhere to persist to, which is not
    // an error: the choices apply for the rest of the run.
    pub(crate) fn save(&self, tree: Option<&StateTree>) -> std::io::Result<()> {
        match tree {
            Some(tree) => self.save_to(&tree.settings_path()),
            None => Ok(()),
        }
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
    use concinnity_core::components::GamepadButton;
    use concinnity_core::components::InputKey;
    use concinnity_core::components::{
        AaMode, PassResolution, ShadowUpdate, UpscaleQuality, UpscalerBackend,
    };
    use concinnity_core::render::dlss::DlssPreset;
    use concinnity_core::settings::QualityPreset;

    // A value this build does not know unsets that one override and leaves the
    // rest of the store, including the other overrides beside it, intact.
    #[test]
    fn an_unknown_enum_value_unsets_only_its_own_override() {
        use ciborium::Value;
        let text = |v: &str| Value::Text(v.to_string());
        let graphics = Value::Map(vec![
            (text("vsync"), Value::Bool(true)),
            (text("dlss_preset"), text("preset_z")),
            (text("aa_mode"), text("taa")),
            (text("upscale_backend"), Value::Integer(7.into())),
            (text("shadow_update"), Value::Null),
            (text("ssgi_resolution"), text("quarter")),
        ]);
        let store = Value::Map(vec![(text("graphics"), graphics)]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&store, &mut bytes).unwrap();

        let settings: Settings = ciborium::from_reader(&bytes[..]).unwrap();
        let g = &settings.graphics;
        assert_eq!(g.vsync, Some(true));
        assert_eq!(g.dlss_preset, None);
        assert_eq!(g.aa_mode, Some(AaMode::Taa));
        assert_eq!(g.upscale_backend, None);
        assert_eq!(g.shadow_update, None);
        assert_eq!(g.ssgi_resolution, Some(PassResolution::Quarter));
        assert_eq!(g.render_scale, None, "an absent override stays unset");
    }

    // A binding this build cannot read falls back to its default; the other
    // bindings in the same map, and the rest of the store, survive.
    #[test]
    fn an_unknown_binding_drops_only_that_binding() {
        use ciborium::Value;
        let text = |v: &str| Value::Text(v.to_string());
        let keymap = keymap::KeyMap {
            jump: InputKey::J,
            ..Default::default()
        };
        let mut keymap_value = Value::serialized(&keymap).unwrap();
        let Value::Map(entries) = &mut keymap_value else {
            panic!("a key map serializes as a map");
        };
        for (key, value) in entries.iter_mut() {
            if *key == text("forward") {
                *value = text("NoSuchKey");
            }
        }
        let gamepad = Value::Map(vec![
            (text("sprint"), text("NoSuchButton")),
            (text("jump"), text("North")),
        ]);
        let controls = Value::Map(vec![
            (text("keymap"), keymap_value),
            (text("gamepad_map"), gamepad),
            (text("gamepad_deadzone"), Value::Float(0.2)),
        ]);
        let graphics = Value::Map(vec![(text("vsync"), Value::Bool(true))]);
        let store = Value::Map(vec![
            (text("controls"), controls),
            (text("graphics"), graphics),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&store, &mut bytes).unwrap();

        let settings: Settings = ciborium::from_reader(&bytes[..]).unwrap();
        let read = settings
            .controls
            .keymap
            .expect("the readable bindings survive");
        assert_eq!(read.forward, keymap::KeyMap::default().forward);
        assert_eq!(read.jump, InputKey::J);
        let pad = settings
            .controls
            .gamepad_map
            .expect("the readable bindings survive");
        assert_eq!(pad.sprint, GamepadMap::default().sprint);
        assert_eq!(pad.jump, GamepadButton::North);
        assert_eq!(settings.controls.gamepad_deadzone, Some(0.2));
        assert_eq!(settings.graphics.vsync, Some(true));
    }

    // A dropped binding whose default another action already holds would bind
    // one input to two actions, so the whole map falls back to the defaults.
    #[test]
    fn a_repair_that_collides_drops_the_whole_map() {
        use ciborium::Value;
        let text = |v: &str| Value::Text(v.to_string());
        let keymap = Value::Map(vec![
            (text("forward"), text("NoSuchKey")),
            (text("jump"), Value::serialized(&InputKey::W).unwrap()),
        ]);
        let gamepad = Value::Map(vec![
            (text("sprint"), text("NoSuchButton")),
            (
                text("jump"),
                Value::serialized(&GamepadButton::LeftStick).unwrap(),
            ),
        ]);
        let controls = Value::Map(vec![
            (text("keymap"), keymap),
            (text("gamepad_map"), gamepad),
            (text("gamepad_deadzone"), Value::Float(0.2)),
        ]);
        let store = Value::Map(vec![(text("controls"), controls)]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&store, &mut bytes).unwrap();

        let settings: Settings = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(settings.controls.keymap, None);
        assert_eq!(settings.controls.gamepad_map, None);
        assert_eq!(settings.controls.gamepad_deadzone, Some(0.2));
    }

    #[test]
    fn settings_cbor_roundtrip() {
        let s = Settings {
            graphics: UserGraphicsSettings {
                quality_preset: Some(QualityPreset::High),
                vsync: Some(true),
                fps_cap: Some(144),
                resolution: Some([1920, 1080, 120]),
                upscale_backend: Some(UpscalerBackend::Xess),
                dlss_preset: Some(DlssPreset::M),
                exposure_ev: Some(-1.5),
                bloom_intensity: Some(0.8),
                bloom_threshold: Some(1.2),
                vignette: Some(0.3),
                lut_strength: Some(0.75),
                ambient_intensity: Some(1.5),
                fov: Some(90.0),
                aa_mode: Some(AaMode::Taa),
                ssao: Some(false),
                ssr: Some(true),
                ray_traced_reflections: Some(false),
                rt_reflection_shadows: Some(false),
                ssgi: Some(true),
                auto_exposure: Some(false),
                ssgi_resolution: Some(PassResolution::Quarter),
                ssgi_rays: Some(2),
                reflection_blur_resolution: Some(PassResolution::Full),
                rt_reflection_resolution: Some(PassResolution::Quarter),
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
                shadow_update: Some(ShadowUpdate::EveryFrame),
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
                keymap: Some(keymap::KeyMap {
                    forward: InputKey::Up,
                    ..keymap::KeyMap::default()
                }),
                gamepad_look_sensitivity: Some(3.0),
                gamepad_deadzone: Some(0.2),
                gamepad_map: Some(GamepadMap {
                    jump: GamepadButton::East,
                    ..GamepadMap::default()
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
            graphics: UserGraphicsSettings,
            // ...plus a field this build has never heard of.
            some_future_setting: u32,
        }
        let other = OtherShape {
            graphics: UserGraphicsSettings {
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
    // test can never clobber the developer's real `settings` file.
    // Drives the real serialize-write-read cycle, but against a temp file. A
    // non-default `render_scale` mirrors a real persisted choice and proves a
    // populated field survives the round trip (it is the field whose loss the
    // original 271 -> 260 byte clobber would have shown).
    #[test]
    fn settings_save_load_roundtrip_is_sandboxed() {
        let s = Settings {
            graphics: UserGraphicsSettings {
                render_scale: Some(UpscaleQuality::Performance),
                vsync: Some(true),
                exposure_ev: Some(-1.5),
                ..Default::default()
            },
            ..Default::default()
        };

        let dir = concinnity_testing::TempTree::new();
        let path = dir.path().join("settings");

        s.save_to(&path).unwrap();
        // The write landed in the sandbox under the expected file name.
        assert!(path.exists());

        let loaded = Settings::load_from(&path);
        assert_eq!(loaded, s);
    }

    // Settings load and save against the tree they are handed, and an app with
    // no tree loads defaults and persists nothing rather than guessing a file.
    #[test]
    fn settings_follow_the_tree_they_are_given_and_no_tree_writes_nothing() {
        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());

        let stored = Settings {
            graphics: UserGraphicsSettings {
                vsync: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        stored.save(Some(&tree)).unwrap();
        assert!(tree.settings_path().exists());
        assert_eq!(Settings::load(Some(&tree)), stored);

        // No tree: defaults in, nothing out, and no file anywhere.
        assert_eq!(Settings::load(None), Settings::default());
        assert!(stored.save(None).is_ok());
    }
}
