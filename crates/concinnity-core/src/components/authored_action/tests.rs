use alloc::string::ToString;
use alloc::vec::Vec;
use serde_json::json;

use super::*;

fn named(json: serde_json::Value) -> NamedAction {
    serde_json::from_value(json.clone()).unwrap_or_else(|e| panic!("{json}: {e}"))
}

// Each action reads from its documented form and writes back to it.
#[test]
fn every_action_round_trips_through_its_authored_form() {
    let cases = [
        (json!("quit"), AuthoredAction::Quit),
        (
            json!({"scene": "level"}),
            AuthoredAction::Scene("level".into()),
        ),
        (
            json!({"show": "pause"}),
            AuthoredAction::Show("pause".into()),
        ),
        (
            json!({"push": "pause"}),
            AuthoredAction::Push("pause".into()),
        ),
        (
            json!({"toggle": "pause"}),
            AuthoredAction::Toggle("pause".into()),
        ),
        (json!("hide"), AuthoredAction::Hide),
        (
            json!({"story": "advance"}),
            AuthoredAction::Story(StoryCommand::Advance),
        ),
        (
            json!({"story": {"choose": 2}}),
            AuthoredAction::Story(StoryCommand::Choose(2)),
        ),
        (json!({"group_toggle": 3}), AuthoredAction::GroupToggle(3)),
        (
            json!({"setting": {"key": "vsync", "verb": "next"}}),
            AuthoredAction::Setting {
                key: SettingKey::Vsync,
                verb: SettingVerb::Next,
            },
        ),
    ];
    for (json, action) in cases {
        assert_eq!(named(json.clone()), action, "{json}");
        assert_eq!(serde_json::to_value(&action).unwrap(), json);
    }
}

#[test]
fn malformed_actions_are_rejected() {
    for json in [
        json!("teleport"),
        json!("show"),
        json!({"show": "a", "push": "b"}),
        json!({"story": "dance"}),
        json!({"story": "choose"}),
        json!({"setting": {"key": "nope", "verb": "next"}}),
        json!({"setting": {"key": "vsync", "verb": "spin"}}),
        json!({"group_toggle": -1}),
        json!(""),
    ] {
        assert!(
            serde_json::from_value::<NamedAction>(json.clone()).is_err(),
            "{json}"
        );
    }
}

// An action field reports itself and the targets its actions name, so a rename
// of a Screen or Scene finds them the way it finds any reference field.
#[test]
fn an_action_field_reports_its_target_references() {
    let mut table = FieldTable::default();
    NamedAction::collect_action_fields("on_click", &["settings"], &mut table);
    assert_eq!(table.actions.len(), 1);
    assert_eq!(table.actions[0].path, "on_click");
    assert_eq!(table.actions[0].extras, ["settings"]);
    let refs: Vec<(&str, &[&str])> = table
        .refs
        .iter()
        .map(|r| (r.path.as_str(), r.targets))
        .collect();
    assert_eq!(
        refs,
        [
            ("on_click.scene", &["Scene"][..]),
            ("on_click.show", &["Screen"][..]),
            ("on_click.push", &["Screen"][..]),
            ("on_click.toggle", &["Screen"][..]),
        ]
    );
}

// The documented variants are exactly the ones that read: each one's example
// deserializes, and every action the vocabulary holds is documented.
#[cfg(feature = "schema")]
#[test]
fn the_schema_documents_every_variant() {
    use crate::ecs::schema::{Body, FieldType, VariantSchema};

    fn variants(ty: FieldType) -> &'static [VariantSchema] {
        match ty {
            FieldType::Enum(schema) => match schema.body {
                Body::Variants(variants) => variants,
                _ => panic!("{} is not a tagged union", schema.name),
            },
            _ => panic!("not a described enum"),
        }
    }
    fn example(v: &VariantSchema) -> serde_json::Value {
        let Some(payload) = v.payload else {
            return json!(v.name);
        };
        let value = match payload() {
            FieldType::Reference(_) => json!("target"),
            FieldType::Integer => json!(1),
            FieldType::Enum(_) => json!("advance"),
            _ => json!({"key": "vsync", "verb": "next"}),
        };
        json!({ v.name: value })
    }

    use crate::ecs::schema::Described;
    let actions = variants(NamedAction::TYPE);
    let story = variants(actions
        .iter()
        .find(|v| v.name == "story")
        .and_then(|v| v.payload)
        .unwrap()());
    for v in actions {
        named(example(v));
    }
    for v in story {
        let value = example(v);
        let cmd: StoryCommand =
            serde_json::from_value(value.clone()).unwrap_or_else(|e| panic!("{value}: {e}"));
        let name = serde_json::to_value(&cmd).unwrap();
        assert_eq!(name, value);
    }
    assert_eq!(story.len(), StoryCommand::VERBS.len());
    let every: [NamedAction; 9] = [
        AuthoredAction::Quit,
        AuthoredAction::Scene("s".into()),
        AuthoredAction::Show("s".into()),
        AuthoredAction::Push("s".into()),
        AuthoredAction::Toggle("s".into()),
        AuthoredAction::Hide,
        AuthoredAction::Story(StoryCommand::Start),
        AuthoredAction::GroupToggle(0),
        AuthoredAction::Setting {
            key: SettingKey::Vsync,
            verb: SettingVerb::Next,
        },
    ];
    for action in every {
        let json = serde_json::to_value(&action).unwrap();
        let name = json
            .as_str()
            .map(str::to_string)
            .or_else(|| json.as_object().and_then(|o| o.keys().next().cloned()))
            .unwrap();
        assert!(
            actions.iter().any(|v| v.name == name),
            "{name} is undocumented"
        );
    }
}
