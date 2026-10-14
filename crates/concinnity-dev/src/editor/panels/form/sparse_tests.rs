//! A sparse type's form lists every leaf at its default, tags the unset ones,
//! and writes back only what the author set.

use serde_json::{Map, Value, json};

use super::{AUTO_LABEL, FieldKind, FormField, assemble, fields_for};

fn args(v: Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

// Each field's control text as the form seeds it.
fn seeded(fields: &[FormField]) -> Vec<String> {
    fields.iter().map(|f| f.initial.clone()).collect()
}

fn field<'a>(fields: &'a [FormField], key: &str) -> &'a FormField {
    fields
        .iter()
        .find(|f| f.key == key)
        .unwrap_or_else(|| panic!("{key} is a field"))
}

#[test]
fn every_section_leaf_is_listed_and_the_unset_ones_are_tagged() {
    let seed = args(json!({ "shadows": { "map_size": 2048 } }));
    let fields = fields_for("GraphicsConfig", Some(&seed));
    let set = field(&fields, "shadows.map_size");
    assert!(set.sparse && !set.unset);
    assert_eq!(set.initial, "2048");
    let unset = field(&fields, "shadows.distance");
    assert!(unset.unset);
    assert_eq!(unset.initial, "80", "an unset leaf shows its default");
    let toggle = field(&fields, "culling.two_pass_occlusion");
    assert_eq!(toggle.kind, FieldKind::Enum);
    assert_eq!(toggle.variants, [AUTO_LABEL, "false", "true"]);
    assert_eq!(toggle.variant_idx, 0);
}

#[test]
fn an_untouched_form_writes_back_only_what_was_set() {
    let seed = args(json!({ "shadows": { "map_size": 2048 } }));
    let fields = fields_for("GraphicsConfig", Some(&seed));
    let out = assemble("GraphicsConfig", Some(&seed), &fields, &seeded(&fields));
    assert_eq!(
        Value::Object(out),
        json!({ "shadows": { "map_size": 2048 } })
    );
}

#[test]
fn a_changed_leaf_is_set_and_a_cleared_one_unset() {
    let seed = args(json!({ "shadows": { "map_size": 2048 } }));
    let mut fields = fields_for("GraphicsConfig", Some(&seed));
    let mut texts = seeded(&fields);
    for (i, f) in fields.iter_mut().enumerate() {
        match f.key.as_str() {
            "shadows.map_size" => texts[i].clear(),
            "bloom.intensity" => texts[i] = "0.3".to_string(),
            "culling.two_pass_occlusion" => f.variant_idx = 1,
            _ => {}
        }
    }
    let out = assemble("GraphicsConfig", Some(&seed), &fields, &texts);
    assert_eq!(
        Value::Object(out),
        json!({
            "bloom": { "intensity": 0.3 },
            "culling": { "two_pass_occlusion": false },
        })
    );
}

#[test]
fn picking_auto_unsets_a_set_choice() {
    let seed = args(json!({ "anti_aliasing": { "mode": "fxaa" } }));
    let mut fields = fields_for("GraphicsConfig", Some(&seed));
    let mode = fields
        .iter()
        .position(|f| f.key == "anti_aliasing.mode")
        .expect("the mode picker");
    assert_eq!(fields[mode].variants[fields[mode].variant_idx], "fxaa");
    fields[mode].variant_idx = 0;
    let out = assemble("GraphicsConfig", Some(&seed), &fields, &seeded(&fields));
    assert!(out.is_empty(), "{out:?}");
}
