//! The verbs that change a setting live, by sending the same `SettingCommand`
//! the settings menu emits: the graphics system applies it on its next step
//! through its real rebuild, and persists it. The reply fires once the command
//! is sent.

use concinnity_core::components::{InputKey, SettingCommand, SettingOp};
use concinnity_core::ecs::World;
use concinnity_core::input::keymap::Bindable;
use concinnity_core::settings::SettingKey;
use serde_json::Value;

use crate::debug::call::Call;
use crate::debug::verb::{Access, Args, Kind, Reply, Verb, optional, queued, required};

const QUALITY_SETTING_COUNT: usize =
    1 + SettingKey::QUALITY_TOGGLES.len() + SettingKey::QUALITY_CYCLES.len();

// The settings `quality-set` steps: the master preset, then the feature toggles
// and the cycle knobs.
const QUALITY_SETTINGS: [&str; QUALITY_SETTING_COUNT] = quality_settings();

const fn quality_settings() -> [&'static str; QUALITY_SETTING_COUNT] {
    let toggles = SettingKey::names(SettingKey::QUALITY_TOGGLES);
    let cycles = SettingKey::names(SettingKey::QUALITY_CYCLES);
    let mut names = [SettingKey::GraphicsQuality.as_str(); QUALITY_SETTING_COUNT];
    let mut i = 0;
    while i < toggles.len() {
        names[1 + i] = toggles[i];
        i += 1;
    }
    let mut i = 0;
    while i < cycles.len() {
        names[1 + toggles.len() + i] = cycles[i];
        i += 1;
    }
    names
}

pub(in crate::debug) const VERBS: &[Verb] = &[
    Verb {
        name: "quality-set",
        description: "Step the quality preset, a quality feature toggle, or a quality knob live, the way the settings menu does.",
        access: Access::Mutating,
        params: &[
            required(
                "setting",
                Kind::Choice(&QUALITY_SETTINGS),
                "Quality setting key.",
            ),
            optional(
                "op",
                Kind::Choice(&["next", "prev"]),
                "Cycle direction. Defaults to next.",
            ),
        ],
        run: quality_set,
    },
    Verb {
        name: "rebind",
        description: "Bind one movement action to a different key, the way a settings menu capture does.",
        access: Access::Mutating,
        params: &[
            required(
                "setting",
                Kind::Name,
                "Action key, such as key_forward or key_jump.",
            ),
            required(
                "key",
                Kind::Text,
                "Input key variant name, such as W, Space, Shift, Num1, or Up.",
            ),
        ],
        run: rebind,
    },
];

#[derive(Clone, Copy, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum Step {
    #[default]
    Next,
    Prev,
}

#[derive(serde::Deserialize)]
struct QualitySet {
    setting: String,
    #[serde(default)]
    op: Step,
}

#[derive(serde::Deserialize)]
struct Rebind {
    setting: String,
    key: String,
}

fn quality_set(call: &Call, args: Args) -> Reply {
    let QualitySet { setting, op } = args.parse()?;
    let setting = SettingKey::parse(&setting)
        .ok_or_else(|| format!("quality-set: '{setting}' is not a quality setting"))?;
    let op = match op {
        Step::Next => SettingOp::Next,
        Step::Prev => SettingOp::Prev,
    };
    call.on_world(move |world, _| {
        send_setting(world, setting, op);
        Ok(())
    })?;
    queued()
}

// `setting` names a movement action (`key_forward`, `key_jump`, ...); `key` is
// a canonical `InputKey` variant name, which is how the key serializes.
fn rebind(call: &Call, args: Args) -> Reply {
    let Rebind { setting, key } = args.parse()?;
    let Some(SettingKey::KeyRebind(action)) = SettingKey::parse(&setting) else {
        return Err(format!(
            "rebind: '{setting}' is not a key rebind (use {})",
            Bindable::ALL.map(Bindable::setting_key).join(" | ")
        ));
    };
    let key: InputKey = serde_json::from_value(Value::String(key.clone())).map_err(|_| {
        format!("rebind: unknown key '{key}' (use a InputKey variant like W / Space / Shift)")
    })?;
    call.on_world(move |world, _| {
        send_setting(world, SettingKey::KeyRebind(action), SettingOp::Rebind(key));
        Ok(())
    })?;
    queued()
}

fn send_setting(world: &mut World, setting: SettingKey, op: SettingOp) {
    world.events_mut::<SettingCommand>().send(SettingCommand {
        setting,
        op,
        value_label: None,
        persist: true,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::verbs::testing::Engine;
    use concinnity_core::ecs::EventCursor;
    use serde_json::json;

    fn sent(engine: &Engine) -> Vec<SettingCommand> {
        engine
            .world
            .events::<SettingCommand>()
            .map(|events| events.read(&mut EventCursor::default()).cloned().collect())
            .unwrap_or_default()
    }

    #[test]
    fn the_choice_list_is_exactly_the_quality_group() {
        for name in QUALITY_SETTINGS {
            assert!(SettingKey::parse(name).is_some(), "{name}");
        }
        let quality = SettingKey::ALL.into_iter().filter(|key| {
            *key == SettingKey::GraphicsQuality || key.is_quality_toggle() || key.is_quality_cycle()
        });
        assert_eq!(quality.count(), QUALITY_SETTINGS.len());
    }

    #[test]
    fn quality_set_sends_the_menus_setting_command() {
        for (name, op, expected) in [
            ("ssao", json!(null), SettingOp::Next),
            ("reflection_blur_resolution", json!("prev"), SettingOp::Prev),
            ("graphics_quality", json!("next"), SettingOp::Next),
        ] {
            let mut engine = Engine::new(World::new());
            let mut call = json!({ "setting": name });
            if !op.is_null() {
                call["op"] = op;
            }
            assert_eq!(
                engine.call("quality-set", call),
                Ok(json!({ "queued": true }))
            );
            let seen = sent(&engine);
            assert_eq!(seen.len(), 1);
            assert_eq!(seen[0].setting.as_str(), name);
            assert_eq!(seen[0].op, expected);
            assert!(seen[0].persist && seen[0].value_label.is_none());
        }
    }

    // A display or slider setting is refused before anything is sent, so it is
    // never persisted.
    #[test]
    fn quality_set_refuses_a_key_outside_the_quality_group() {
        let mut engine = Engine::new(World::new());
        for key in ["taa", "vsync", "exposure", "render_scale"] {
            let error = engine.call("quality-set", json!({ "setting": key }));
            assert!(
                error
                    .unwrap_err()
                    .contains(&format!("unknown setting '{key}'"))
            );
        }
        assert!(sent(&engine).is_empty());
    }

    #[test]
    fn rebind_resolves_the_key_variant() {
        let mut engine = Engine::new(World::new());
        let call = json!({ "setting": "key_forward", "key": "Space" });
        assert_eq!(engine.call("rebind", call), Ok(json!({ "queued": true })));
        let seen = sent(&engine);
        assert_eq!(seen[0].setting, SettingKey::KeyRebind(Bindable::Forward));
        assert_eq!(seen[0].op, SettingOp::Rebind(InputKey::Space));
    }

    // The verb binds keys only: a gamepad rebind or an unknown action is
    // refused, as is a key that names no variant.
    #[test]
    fn rebind_refuses_what_it_cannot_bind() {
        let mut engine = Engine::new(World::new());
        for (setting, key, needle) in [
            ("pad_jump", "W", "is not a key rebind"),
            ("key_nope", "W", "is not a key rebind"),
            ("key_forward", "NotAKey", "unknown key 'NotAKey'"),
        ] {
            let call = json!({ "setting": setting, "key": key });
            assert!(engine.call("rebind", call).unwrap_err().contains(needle));
        }
        assert!(sent(&engine).is_empty());
    }
}
