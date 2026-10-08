//! The verbs that change a setting live, by sending the same `SettingCommand`
//! the settings menu emits: the graphics system applies it on its next step
//! through its real rebuild, and persists it. The reply fires once the command
//! is sent. `view-set` and `show-set` change the view mode and the show flags
//! the same way the editor's View menu does, through the world's
//! `ViewOverrides`.

use concinnity_core::components::{InputKey, SettingCommand, SettingOp};
use concinnity_core::ecs::{ViewOverrides, World};
use concinnity_core::gfx::view_modes::{ShowFlags, ViewMode};
use concinnity_core::input::keymap::Bindable;
use concinnity_core::settings::SettingKey;
use serde_json::{Value, json};

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

// The view modes `view-set` takes, each a `ViewMode` label in lowercase.
const VIEW_MODES: [&str; ViewMode::ALL.len()] = [
    "lit",
    "unlit",
    "wireframe",
    "normals",
    "roughness",
    "occlusion",
    "depth",
    "motion",
    "reactive",
];

// The show flags `show-set` takes.
const SHOW_FLAGS: [&str; ShowFlags::NAMED.len()] = {
    let mut names = [""; ShowFlags::NAMED.len()];
    let mut i = 0;
    while i < names.len() {
        names[i] = ShowFlags::NAMED[i].1;
        i += 1;
    }
    names
};

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
        name: "view-set",
        description: "Show one stage of the frame in place of the lit image, the way the editor's View menu does.",
        access: Access::Mutating,
        params: &[required(
            "mode",
            Kind::Choice(&VIEW_MODES),
            "View mode; lit restores the shipping image.",
        )],
        run: view_set,
    },
    Verb {
        name: "show-set",
        description: "Turn one feature pass on or off for the frame, the way the editor's View menu does.",
        access: Access::Mutating,
        params: &[
            required("flag", Kind::Choice(&SHOW_FLAGS), "Show flag."),
            required("state", Kind::Choice(&["on", "off"]), "Whether it shows."),
        ],
        run: show_set,
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

#[derive(serde::Deserialize)]
struct ViewSet {
    mode: String,
}

fn view_set(call: &Call, args: Args) -> Reply {
    let ViewSet { mode } = args.parse()?;
    let mode = view_mode(&mode).ok_or_else(|| format!("view-set: unknown mode '{mode}'"))?;
    call.on_world(move |world, _| {
        let show = world
            .resource::<ViewOverrides>()
            .map(|view| view.show)
            .unwrap_or_default();
        world.insert_resource(ViewOverrides { mode, show });
        Ok(())
    })?;
    Ok(json!({ "mode": mode.label() }))
}

#[derive(serde::Deserialize)]
struct ShowSet {
    flag: String,
    state: String,
}

fn show_set(call: &Call, args: Args) -> Reply {
    let ShowSet { flag, state } = args.parse()?;
    let flag = show_flag(&flag).ok_or_else(|| format!("show-set: unknown flag '{flag}'"))?;
    let on = match state.as_str() {
        "on" => true,
        "off" => false,
        other => return Err(format!("show-set: unknown state '{other}' (use on | off)")),
    };
    call.on_world(move |world, _| {
        let view = world
            .resource::<ViewOverrides>()
            .copied()
            .unwrap_or_default();
        let show = view.show.with(flag, on);
        world.insert_resource(ViewOverrides { show, ..view });
        Ok(())
    })?;
    Ok(json!({ "flag": flag_label(flag), "on": on }))
}

fn show_flag(name: &str) -> Option<ShowFlags> {
    ShowFlags::NAMED
        .iter()
        .find(|(_, n)| n.eq_ignore_ascii_case(name))
        .map(|&(flag, _)| flag)
}

fn flag_label(flag: ShowFlags) -> &'static str {
    ShowFlags::LABELED
        .iter()
        .find(|(f, _)| *f == flag)
        .map_or("", |(_, label)| label)
}

fn view_mode(name: &str) -> Option<ViewMode> {
    ViewMode::ALL
        .into_iter()
        .find(|mode| mode.label().eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::verbs::testing::Engine;
    use concinnity_core::ecs::EventCursor;

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

    #[test]
    fn every_view_mode_is_settable_by_its_label() {
        for (name, mode) in VIEW_MODES.iter().zip(ViewMode::ALL) {
            assert_eq!(view_mode(name), Some(mode), "{name}");
        }
    }

    #[test]
    fn view_set_publishes_the_mode_and_keeps_the_show_flags() {
        let mut world = World::new();
        world.insert_resource(ViewOverrides {
            mode: ViewMode::Lit,
            show: ShowFlags(0),
        });
        let mut engine = Engine::new(world);
        let reply = engine.call("view-set", json!({ "mode": "motion" }));
        assert_eq!(reply, Ok(json!({ "mode": "Motion" })));
        let view = engine.world.resource::<ViewOverrides>().copied();
        assert_eq!(
            view,
            Some(ViewOverrides {
                mode: ViewMode::Motion,
                show: ShowFlags(0),
            })
        );
        assert!(
            engine
                .call("view-set", json!({ "mode": "sideways" }))
                .is_err()
        );
    }

    #[test]
    fn every_show_flag_is_settable_by_its_name() {
        for (name, (flag, _)) in SHOW_FLAGS.iter().zip(ShowFlags::LABELED) {
            assert_eq!(show_flag(name), Some(flag), "{name}");
        }
    }

    #[test]
    fn show_set_flips_one_flag_and_keeps_the_mode() {
        let mut world = World::new();
        world.insert_resource(ViewOverrides {
            mode: ViewMode::Motion,
            show: ShowFlags::all(),
        });
        let mut engine = Engine::new(world);
        let off = json!({ "flag": "reactive", "state": "off" });
        assert_eq!(
            engine.call("show-set", off.clone()),
            Ok(json!({ "flag": "Reactive mask", "on": false }))
        );
        // Turning a cleared flag off again leaves it cleared.
        assert!(engine.call("show-set", off).is_ok());
        let view = engine.world.resource::<ViewOverrides>().copied();
        assert_eq!(
            view,
            Some(ViewOverrides {
                mode: ViewMode::Motion,
                show: ShowFlags::all().toggled(ShowFlags::REACTIVE),
            })
        );
        let on = json!({ "flag": "reactive", "state": "on" });
        assert!(engine.call("show-set", on).is_ok());
        let view = engine.world.resource::<ViewOverrides>().copied();
        assert_eq!(view.map(|v| v.show), Some(ShowFlags::all()));
        for bad in [
            json!({ "flag": "sideways", "state": "on" }),
            json!({ "flag": "fog", "state": "maybe" }),
        ] {
            assert!(engine.call("show-set", bad).is_err());
        }
    }
}
