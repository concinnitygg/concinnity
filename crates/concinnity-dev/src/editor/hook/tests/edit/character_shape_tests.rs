// src/editor/hook/tests/edit/character_shape_tests.rs
//
// The CharacterShape panel's actions (`hook/edit/character_shape.rs`): the
// single commit a reset or a randomize makes, the shape an add row creates for
// the selected mesh, and the schema a character model supplies its presets
// from. The slider drag itself is `tests/drag/shape_tests.rs`.

use concinnity_core::ecs::World;

use crate::editor::hook::entry_name;
use crate::editor::hook::tests::fixtures::{hook, shape_world_entries};

use crate::editor::inject;

use crate::editor::panels::character_shape;
use crate::editor::panels::character_shape_panel;

// A drag released where it started changes nothing and records no step; the
// header buttons commit through the same path.
#[test]
fn shape_reset_and_randomize_commit_once_each() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(shape_world_entries());
    h.shape_open = true;
    h.selection.set(vec!["body".to_string()]);
    let data = h.shape_data(&world);
    h.apply_shape_action(
        character_shape_panel::ShapeAction::Randomize,
        &data,
        [0.0, 0.0],
        &mut world,
    );
    assert!(h.can_undo());
    let sliders = h.entries[1]["args"]["sliders"].as_array().unwrap();
    for s in sliders {
        let v = s["value"].as_f64().unwrap();
        assert!(
            v.abs() <= f64::from(character_shape::RANDOM_BAND) + 1e-6,
            "{v}"
        );
    }
    let legs = h.entries[1]["args"]["proportions"].as_array().unwrap();
    assert!(
        legs.iter().all(|p| p["joint"] == "thigh_l"),
        "only the skeleton's joints are written: {legs:?}"
    );
    h.undo(&mut world);
    assert!(!h.can_undo(), "randomize was one step");

    // A history jump drops the selection; pick the mesh again.
    h.selection.set(vec!["body".to_string()]);
    let data = h.shape_data(&world);
    h.apply_shape_action(
        character_shape_panel::ShapeAction::Reset,
        &data,
        [0.0, 0.0],
        &mut world,
    );
    assert!(
        h.entries[1]["args"]["sliders"]
            .as_array()
            .unwrap()
            .is_empty(),
        "reset drops every slider"
    );
    h.undo(&mut world);
    assert!(!h.can_undo(), "reset was one step");
}

// Selecting a mesh with no shape offers the add row, which creates one
// targeting it; selecting the shape entry itself binds the same pair.
#[test]
fn shape_add_row_creates_a_shape_for_the_selected_mesh() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut entries = shape_world_entries();
    entries.pop();
    let mut h = hook(entries);
    h.shape_open = true;
    h.selection.set(vec!["body".to_string()]);
    let data = h.shape_data(&world);
    assert_eq!(data.rows, [character_shape::Row::Add]);
    h.apply_shape_action(
        character_shape_panel::ShapeAction::Add,
        &data,
        [0.0, 0.0],
        &mut world,
    );
    assert_eq!(h.entries.len(), 2);
    assert_eq!(h.entries[1]["args"]["target"], "body");
    let name = entry_name(&h.entries[1]).unwrap().to_string();
    h.selection.set(vec![name]);
    let data = h.shape_data(&world);
    let b = data.binding.expect("a selected shape binds");
    assert_eq!((b.mesh.as_str(), b.shape_idx), ("body", Some(1)));
    // An unrelated selection shows the prompt.
    h.selection.clear();
    let data = h.shape_data(&world);
    assert!(data.binding.is_none());
    assert_eq!(data.status.as_deref(), Some("Select a SkinnedMesh"));
}

// A CharacterModel binds like a mesh, its panel follows the schema it names
// (a CharacterSchema entry here, with its own sections, captions and
// presets), and a preset button commits the preset's vector as one step.
#[test]
fn shape_panel_reads_a_character_models_schema_and_applies_presets() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let entries = vec![
        serde_json::json!({"name": "sk", "type": "CharacterSchema", "args": {
            "joints": [{"name": "root"}, {"name": "tail", "parent": "root"}],
            "keys": [{"name": "fluff", "caption": "Fluffiness", "region": "tail"}],
            "regions": [{"name": "tail", "joints": ["tail"]}],
            "proportion_groups": [{"name": "tail_length", "caption": "tail length",
                "region": "tail", "joints": ["tail"], "length": 0.1}],
            "panel": [{"caption": "Tail", "regions": ["tail"]}],
            "presets": [{"name": "bushy", "sliders": [{"name": "fluff", "value": 0.9}],
                "proportions": [{"joint": "tail", "length": 0.05}]}]
        }}),
        serde_json::json!({"name": "body", "type": "CharacterModel", "args": {
            "schema": "sk", "sources": [{"source": "fox.glb"}]
        }}),
        serde_json::json!({"name": "body_shape", "type": "CharacterShape", "args": {
            "target": "body", "sliders": [{"name": "nose", "value": 0.3}]
        }}),
    ];
    let mut h = hook(entries);
    h.shape_open = true;
    h.selection.set(vec!["body".to_string()]);
    // Nothing is inline on a model entry, so the rows are what the live
    // world exposes; publish a pose-free target through the entry fallback
    // by checking the schema half alone.
    let schema = h.shape_schema("body");
    assert_eq!(schema.panel[0].caption, "Tail");
    assert_eq!(schema.presets[0].name, "bushy");
    let rows = character_shape::derive_rows(
        &schema,
        &["fluff".to_string(), "nose".to_string()],
        &["root".to_string(), "tail".to_string()],
    );
    assert_eq!(rows.sections, ["Tail", character_shape::OTHER_SECTION]);
    assert_eq!(rows.sliders[0].caption, "Fluffiness");
    assert_eq!(rows.sliders[1].name, "tail_length");
    assert_eq!(rows.sliders[2].name, "nose");
    assert_eq!(
        rows.sliders[2].section, 1,
        "an unknown key lands under Other"
    );

    let data = h.shape_data(&world);
    let b = data.binding.as_ref().expect("a model binds");
    assert_eq!((b.mesh.as_str(), b.shape_idx), ("body", Some(2)));
    assert_eq!(data.preset_names, ["bushy"]);
    assert_eq!(data.rows[0], character_shape::Row::PresetHeader);
    assert_eq!(data.rows[1], character_shape::Row::Preset(0));
    h.apply_shape_action(
        character_shape_panel::ShapeAction::Preset(0),
        &data,
        [0.0, 0.0],
        &mut world,
    );
    let args = &h.entries[2]["args"];
    assert_eq!(args["sliders"][0]["name"], "fluff");
    assert!((args["sliders"][0]["value"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(args["sliders"].as_array().unwrap().len(), 1);
    assert_eq!(args["proportions"][0]["joint"], "tail");
    assert!(h.can_undo());
    h.undo(&mut world);
    assert!(!h.can_undo(), "a preset is one step");
    assert_eq!(h.entries[2]["args"]["sliders"][0]["name"], "nose");
}
