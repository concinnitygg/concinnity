//! The choices a player made in the in-engine settings menu, as the host
//! persists them: graphics, audio, controls. The host loads the store once and
//! publishes it as a world resource; the settings menu edits that resource in
//! place and bumps [`SettingsRevision`] so the host writes it back.
//!
//! Every field is an `Option`: `None` means "use the world's default", so an
//! unchanged setting never overrides the authored value. The host stores the
//! struct in a self-describing format, so adding or removing a setting never
//! invalidates an existing store (a missing field falls back to its default,
//! an unknown field is ignored).

use alloc::string::String;
use alloc::vec::Vec;
use serde::de::value::{Error as ValueError, MapDeserializer};
use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::components::{ControlsCommand, GamepadAction, GamepadMap};
use crate::ecs::Resources;
use crate::input::keymap;
use crate::settings::UserGraphicsSettings;

/// The persisted settings-menu choices.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StoredSettings {
    /// Graphics overrides.
    #[serde(default)]
    pub graphics: UserGraphicsSettings,
    /// Audio overrides.
    #[serde(default)]
    pub audio: AudioSettings,
    /// Control overrides.
    #[serde(default)]
    pub controls: ControlsSettings,
}

/// Persisted overrides for audio settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioSettings {
    /// Master output volume as a linear gain (0.0 = silent, 1.0 = full). `None`
    /// leaves each emitter at its authored `AudioEmitter.volume`. Applied when
    /// a world's audio initializes.
    #[serde(default)]
    pub master_volume: Option<f32>,
    /// Music bus volume under the master, same semantics (`None` = unity).
    #[serde(default)]
    pub music_volume: Option<f32>,
    /// Effects bus volume under the master.
    #[serde(default)]
    pub sfx_volume: Option<f32>,
    /// Voice bus volume under the master.
    #[serde(default)]
    pub voice_volume: Option<f32>,
}

/// Persisted overrides for control settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ControlsSettings {
    /// Mouse-look sensitivity in radians per pixel. `None` uses the controlling
    /// camera's authored `CameraController.mouse_sensitivity`.
    #[serde(default)]
    pub mouse_sensitivity: Option<f32>,
    /// Gameplay movement key bindings. `None` uses the defaults
    /// (W/S/A/D/Shift/Space/E). Applied live: the active backend decodes
    /// physical keys through this map.
    #[serde(default, deserialize_with = "lenient_bindings")]
    pub keymap: Option<keymap::KeyMap>,
    /// Gamepad look sensitivity in radians per second at full stick deflection.
    /// `None` uses the default.
    #[serde(default)]
    pub gamepad_look_sensitivity: Option<f32>,
    /// Gamepad stick deadzone as a deflection fraction in [0, 1]. `None` uses
    /// the default.
    #[serde(default)]
    pub gamepad_deadzone: Option<f32>,
    /// Gamepad action button bindings. `None` uses the defaults
    /// (L3/South/West).
    #[serde(default, deserialize_with = "lenient_bindings")]
    pub gamepad_map: Option<GamepadMap>,
}

impl StoredSettings {
    /// The stored controls, in the shape the controllers and input sampling
    /// apply: at init, the way a live command is applied each tick. A `None`
    /// field keeps the authored value.
    pub fn controls_command(&self) -> ControlsCommand {
        ControlsCommand {
            mouse_sensitivity: self.controls.mouse_sensitivity,
            fov_y_degrees: self.graphics.fov,
            gamepad_look_sensitivity: self.controls.gamepad_look_sensitivity,
            gamepad_deadzone: self.controls.gamepad_deadzone,
            gamepad_map: self.controls.gamepad_map,
        }
    }

    /// Publish `self` as the world's store and mark it changed, so the host
    /// writes it.
    pub fn store_changed(self, resources: &mut Resources) {
        resources.insert(self);
        match resources.get_mut::<SettingsRevision>() {
            Some(revision) => revision.bump(),
            None => {
                resources.insert(SettingsRevision(1));
            }
        }
    }
}

/// How many times the [`StoredSettings`] resource has changed since the world
/// started. The settings menu bumps it on every change worth persisting; the
/// host writes the store back when it sees a new value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SettingsRevision(pub u64);

impl SettingsRevision {
    /// Mark the store changed.
    pub fn bump(&mut self) {
        self.0 = self.0.wrapping_add(1);
    }
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

// One stored binding's action or input name, or anything else this build
// cannot read as one.
#[derive(Deserialize)]
#[serde(untagged)]
enum RawName {
    Text(String),
    Other(IgnoredAny),
}

// A stored binding map's entries, each kept whatever it holds so one
// unreadable entry cannot spoil its neighbors.
struct RawEntries(Vec<(RawName, RawName)>);

impl<'de> Deserialize<'de> for RawEntries {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Entries;
        impl<'de> Visitor<'de> for Entries {
            type Value = RawEntries;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a binding map")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<RawEntries, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry()? {
                    entries.push(entry);
                }
                Ok(RawEntries(entries))
            }
        }
        d.deserialize_map(Entries)
    }
}

// A stored binding map, or anything else this build cannot read as one.
#[derive(Deserialize)]
#[serde(untagged)]
enum RawBindings {
    Map(RawEntries),
    Other(IgnoredAny),
}

// Read a binding map whose every field has a default, dropping only the
// bindings this build cannot read (a renamed key or button, a value of the
// wrong type): each entry is kept when it reads on its own, and the dropped
// ones fall back to their defaults. A repair that would bind one input to two
// actions drops the whole map.
fn lenient_bindings<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: BindingMap,
{
    let entries = match Option::<RawBindings>::deserialize(d)? {
        None => return Ok(None),
        Some(RawBindings::Map(entries)) => entries.0,
        Some(RawBindings::Other(_)) => {
            tracing::warn!("{} bindings unreadable, using the defaults", T::NAME);
            return Ok(None);
        }
    };
    let read = |entries: Vec<(String, String)>| {
        T::deserialize(MapDeserializer::<_, ValueError>::new(entries.into_iter()))
    };
    let mut readable = Vec::with_capacity(entries.len());
    let mut dropped = Vec::new();
    for entry in entries {
        match entry {
            (RawName::Text(key), RawName::Text(value))
                if read(alloc::vec![(key.clone(), value.clone())]).is_ok() =>
            {
                readable.push((key, value));
            }
            (RawName::Text(key), _) => dropped.push(key),
            (RawName::Other(_), _) => dropped.push(String::from("?")),
        }
    }
    if dropped.is_empty() {
        return Ok(read(readable).ok());
    }
    let names: Vec<&str> = dropped.iter().map(String::as_str).collect();
    let repaired = read(readable).ok().filter(T::binds_each_input_once);
    if repaired.is_some() {
        tracing::warn!(
            "{} bindings unreadable, using their defaults: {}",
            T::NAME,
            names.join(", ")
        );
    } else {
        tracing::warn!(
            "{} bindings unreadable ({}) and their defaults collide, using the default map",
            T::NAME,
            names.join(", ")
        );
    }
    Ok(repaired)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{
        AaMode, GamepadButton, InputKey, PassResolution, ShadowUpdate, UpscalerBackend,
    };
    use crate::render::output::dlss::DlssPreset;
    use crate::settings::QualityPreset;
    use alloc::string::ToString;
    use alloc::vec;
    use ciborium::Value;

    fn text(v: &str) -> Value {
        Value::Text(v.to_string())
    }

    fn decode(store: &Value) -> StoredSettings {
        let mut bytes = Vec::new();
        ciborium::into_writer(store, &mut bytes).unwrap();
        ciborium::from_reader(&bytes[..]).unwrap()
    }

    // A value this build does not know unsets that one override and leaves the
    // rest of the store, including the other overrides beside it, intact.
    #[test]
    fn an_unknown_enum_value_unsets_only_its_own_override() {
        let graphics = Value::Map(vec![
            (text("vsync"), Value::Bool(true)),
            (text("dlss_preset"), text("preset_z")),
            (text("aa_mode"), text("taa")),
            (text("upscale_backend"), Value::Integer(7.into())),
            (text("shadow_update"), Value::Null),
            (text("ssgi_resolution"), text("quarter")),
        ]);
        let settings = decode(&Value::Map(vec![(text("graphics"), graphics)]));
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
        let settings = decode(&Value::Map(vec![
            (text("controls"), controls),
            (text("graphics"), graphics),
        ]));
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

    // An entry that is not text at all, in either its key or its value, drops
    // only itself; the readable entries beside it survive.
    #[test]
    fn a_non_text_entry_drops_only_that_binding() {
        let keymap = Value::Map(vec![
            (text("forward"), Value::Integer(3.into())),
            (Value::Integer(9.into()), text("Space")),
            (text("jump"), Value::serialized(&InputKey::J).unwrap()),
        ]);
        let controls = Value::Map(vec![(text("keymap"), keymap)]);
        let settings = decode(&Value::Map(vec![(text("controls"), controls)]));
        let read = settings
            .controls
            .keymap
            .expect("the readable bindings survive");
        assert_eq!(read.jump, InputKey::J);
        assert_eq!(read.forward, keymap::KeyMap::default().forward);
    }

    // A dropped binding whose default another action already holds would bind
    // one input to two actions, so the whole map falls back to the defaults.
    #[test]
    fn a_repair_that_collides_drops_the_whole_map() {
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
        let settings = decode(&Value::Map(vec![(text("controls"), controls)]));
        assert_eq!(settings.controls.keymap, None);
        assert_eq!(settings.controls.gamepad_map, None);
        assert_eq!(settings.controls.gamepad_deadzone, Some(0.2));
    }

    // A binding store that is not a map at all is dropped, and the rest of the
    // store survives.
    #[test]
    fn a_binding_store_that_is_not_a_map_is_dropped() {
        let controls = Value::Map(vec![
            (text("keymap"), Value::Integer(3.into())),
            (text("gamepad_deadzone"), Value::Float(0.2)),
        ]);
        let settings = decode(&Value::Map(vec![(text("controls"), controls)]));
        assert_eq!(settings.controls.keymap, None);
        assert_eq!(settings.controls.gamepad_deadzone, Some(0.2));
    }

    #[test]
    fn settings_cbor_roundtrip() {
        let s = StoredSettings {
            graphics: UserGraphicsSettings {
                quality_preset: Some(QualityPreset::High),
                vsync: Some(true),
                fps_cap: Some(144),
                resolution: Some([1920, 1080, 120]),
                upscale_backend: Some(UpscalerBackend::Xess),
                dlss_preset: Some(DlssPreset::M),
                exposure_ev: Some(-1.5),
                shadow_update: Some(ShadowUpdate::EveryFrame),
                ssgi_resolution: Some(PassResolution::Quarter),
                fov: Some(90.0),
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
        let loaded: StoredSettings = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(loaded, s);
    }

    // An empty store, and one written by a build with fewer or more fields,
    // read as the defaults for whatever is absent.
    #[test]
    fn missing_and_unknown_fields_are_tolerated() {
        assert_eq!(decode(&Value::Map(vec![])), StoredSettings::default());
        let graphics = Value::Map(vec![(text("vsync"), Value::Bool(false))]);
        let loaded = decode(&Value::Map(vec![
            (text("graphics"), graphics),
            (text("some_future_setting"), Value::Integer(7.into())),
        ]));
        assert_eq!(loaded.graphics.vsync, Some(false));
        assert_eq!(loaded.audio, AudioSettings::default());
        assert_eq!(loaded.controls, ControlsSettings::default());
    }

    // Each stored change publishes the store and moves the revision, whether
    // or not the host published one.
    #[test]
    fn a_stored_change_publishes_and_bumps_the_revision() {
        let mut resources = Resources::new();
        let mut stored = StoredSettings::default();
        stored.graphics.vsync = Some(true);
        stored.clone().store_changed(&mut resources);
        assert_eq!(
            resources.get::<SettingsRevision>(),
            Some(&SettingsRevision(1))
        );
        stored.clone().store_changed(&mut resources);
        assert_eq!(
            resources.get::<SettingsRevision>(),
            Some(&SettingsRevision(2))
        );
        assert_eq!(resources.get::<StoredSettings>(), Some(&stored));
    }

    #[test]
    fn stored_controls_carry_over() {
        let mut settings = StoredSettings::default();
        settings.controls.mouse_sensitivity = Some(0.002);
        settings.controls.gamepad_deadzone = Some(0.2);
        settings.graphics.fov = Some(75.0);
        let controls = settings.controls_command();
        assert_eq!(controls.mouse_sensitivity, Some(0.002));
        assert_eq!(controls.gamepad_look_sensitivity, None);
        assert_eq!(controls.gamepad_deadzone, Some(0.2));
        assert_eq!(controls.fov_y_degrees, Some(75.0));
    }
}
