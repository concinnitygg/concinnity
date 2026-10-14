//! A shorthand saved through the form with its references left at `(none)`
//! still builds: the form writes such a reference as null.

use serde_json::{Map, Value, json};

use super::{FieldKind, assemble, fields_for, set_action_options};

// `args` as the form saves them with every reference field at `(none)`.
fn saved_with_no_references(ty: &str, seed: Value) -> Map<String, Value> {
    let seed = seed.as_object().cloned().unwrap_or_default();
    let mut fields = fields_for(ty, Some(&seed));
    let mut refs = 0;
    for field in &mut fields {
        if matches!(field.kind, FieldKind::Ref { .. }) {
            field.variant_idx = 0;
            refs += 1;
        }
    }
    assert!(refs > 0, "{ty} shows no reference field");
    for field in &mut fields {
        set_action_options(field, &[], &[]);
    }
    let texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
    assemble(ty, Some(&seed), &fields, &texts)
}

fn line(ty: &str, id: &str, mut args: Map<String, Value>) -> String {
    args.insert("$id".to_string(), id.into());
    format!("{}\n", json!([ty, args]))
}

#[test]
fn a_shorthand_with_references_at_none_still_builds() {
    let saved = [
        (
            "Panel",
            json!({"title": "Paused", "screen": "s", "title_font": "f"}),
        ),
        (
            "OptionSelect",
            json!({"setting": "vsync", "screen": "s", "font": "f"}),
        ),
        (
            "Slider",
            json!({"setting": "exposure", "screen": "s", "font": "f"}),
        ),
        ("MainMenu", json!({"font": "f"})),
        (
            "MaterialPalette",
            json!({"entries": [{"alias": "wall", "albedo": "t"}]}),
        ),
        (
            "Prefab",
            json!({"props": [{"name": "a", "mesh": "box", "material": "m"}]}),
        ),
    ];
    let mut world = String::from("[\"GraphicsConfig\",{\"$id\":\"gfx\"}]\n");
    world += &line(
        "ProceduralMesh",
        "box",
        json!({"generator": "box"}).as_object().cloned().unwrap(),
    );
    for (i, (ty, seed)) in saved.into_iter().enumerate() {
        let args = saved_with_no_references(ty, seed);
        world += &line(ty, &format!("x{i}"), args);
    }
    if let Err(errors) = concinnity_cook::prepare_world(world.as_str(), None) {
        panic!("{world}\n{errors:?}");
    }
}

// The shorthands that need a source file to expand, and a Prop, which draws
// nothing without one of its references, read their args all the same.
#[test]
fn an_unbuildable_entry_with_references_at_none_still_reads() {
    use concinnity_cook::authoring::registry::build_only::{CharacterModel, SceneImport};
    let import = saved_with_no_references("SceneImport", json!({"source": "a.glb", "scene": "s"}));
    serde_json::from_value::<SceneImport>(Value::Object(import)).expect("SceneImport args");
    let model = saved_with_no_references("CharacterModel", json!({"source": "a.glb"}));
    serde_json::from_value::<CharacterModel>(Value::Object(model)).expect("CharacterModel args");
    let prop = saved_with_no_references("Prop", json!({"prefab": "p"}));
    let prop = serde_json::from_value::<concinnity_core::components::Prop>(Value::Object(prop))
        .expect("Prop args");
    assert!(prop.prefab.is_empty());
}
