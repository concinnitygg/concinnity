//! An action field is one picker over what the world's Screens and Scenes let
//! an action do, writing the chosen action back in its authored form.

use serde_json::{Value, json};

use super::{FieldKind, FormField, NONE_LABEL, assemble, fields_for, set_action_options};

fn action_field(ty: &str, seed: Value, key: &str) -> (Vec<FormField>, usize) {
    let seed = seed.as_object().cloned().unwrap_or_default();
    let mut fields = fields_for(ty, Some(&seed));
    for field in &mut fields {
        set_action_options(field, &["pause".to_string()], &["level".to_string()]);
    }
    let at = fields
        .iter()
        .position(|f| f.key == key)
        .unwrap_or_else(|| panic!("{key} is a field"));
    (fields, at)
}

fn saved(ty: &str, fields: &[FormField]) -> serde_json::Map<String, Value> {
    let texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
    assemble(ty, None, fields, &texts)
}

#[test]
fn every_action_the_world_allows_is_offered() {
    let (fields, at) = action_field("HitRegion", json!({}), "action");
    let field = &fields[at];
    assert_eq!(field.kind, FieldKind::Action { extras: &[] });
    assert_eq!(
        field.variants[..9],
        [
            NONE_LABEL,
            "quit",
            "hide",
            "show pause",
            "push pause",
            "toggle pause",
            "scene level",
            "story start",
            "story continue",
        ]
    );
    assert_eq!(field.variant_idx, 0, "an unset action selects (none)");
    assert_eq!(field.choices[3], json!({"show": "pause"}));
    assert_eq!(field.variants.len(), field.choices.len());
}

#[test]
fn the_current_action_is_selected_and_written_back() {
    let (mut fields, at) =
        action_field("HitRegion", json!({"action": {"scene": "level"}}), "action");
    assert_eq!(fields[at].variants[fields[at].variant_idx], "scene level");
    assert_eq!(
        saved("HitRegion", &fields)["action"],
        json!({"scene": "level"})
    );

    fields[at].variant_idx = fields[at]
        .variants
        .iter()
        .position(|v| v == "push pause")
        .unwrap();
    assert_eq!(
        saved("HitRegion", &fields)["action"],
        json!({"push": "pause"})
    );
    fields[at].variant_idx = 0;
    assert_eq!(saved("HitRegion", &fields)["action"], Value::Null);
}

// A current action the list does not offer is kept as an option of its own, so
// editing the entry leaves it as it was.
#[test]
fn an_action_the_list_does_not_offer_is_kept() {
    let choose = json!({"story": {"choose": 2}});
    let (fields, at) = action_field("KeyBinding", json!({"action": choose.clone()}), "action");
    let field = &fields[at];
    assert_eq!(field.choices[field.variant_idx], choose);
    assert_eq!(field.variants[field.variant_idx], choose.to_string());
    assert_eq!(saved("KeyBinding", &fields)["action"], choose);
}

// A menu item also takes `"settings"`, and each item is its own picker.
#[test]
fn a_menu_item_offers_its_extra_names() {
    let seed = json!({"items": [
        {"label": "Options", "action": "settings"},
        {"label": "Go", "action": {"show": "pause"}},
    ]});
    let (fields, at) = action_field("MainMenu", seed, "items.0.action");
    let options = &fields[at];
    assert_eq!(
        options.kind,
        FieldKind::Action {
            extras: &["settings"]
        }
    );
    assert_eq!(options.variants[options.variant_idx], "settings");
    let go = fields.iter().find(|f| f.key == "items.1.action").unwrap();
    assert_eq!(go.variants[go.variant_idx], "show pause");
}
