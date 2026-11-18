//! The add / edit form's lifecycle (`hook/editing.rs`): the type picker and its
//! search, opening the form from a row or the "+", the name rules a confirm
//! applies, and every field kind the form writes back -- scalars, colors,
//! nested objects, enums, asset references and their dropdowns, arrays, and a
//! disclosed vector. Also what survives a reinjection, a panel toggle, and a
//! delete that shifts the open form's index.

use concinnity_engine::components::{FrameInput, Layout, UiEvent, UiValue};
use concinnity_engine::ecs::World;

use super::fixtures::{
    click_row, entry, entry_with_args, hook, row_of, seed_tree, world_with_fields,
};
use crate::editor::hook::{EditorHook, FormTarget, declared_id};
use crate::editor::ui::BuiltUi;
use crate::frame_hook::FrameHook;

use crate::editor::inject;

use crate::editor::panels::form;
use crate::editor::panels::form_panel::{self, FormAction, FormFocus};

use crate::editor::panels::assets_panel::{self, PanelAction};
use crate::editor::panels::registry::PanelKey;

use crate::editor::hook::tests::fixtures::{active, entry_target, select, selected};

// While the "+" picker is open the field narrows its type options instead of
// the tree, so the two never fight over the same text.
#[test]
fn the_search_field_narrows_the_picker_while_it_is_open() {
    let mut h = hook(Vec::new());
    h.open[PanelKey::Assets] = true;
    h.apply_panel(PanelAction::TogglePicker);
    h.assets.filter = "pointlight".into();
    let opts = h.picker_options().unwrap();
    let names: Vec<&str> = opts.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["PointLight"], "case-insensitive type narrowing");
    assert!(
        h.tree_rows().is_empty(),
        "the tree is not filtered by the picker's text"
    );
}

#[test]
fn plus_picker_then_name_form_adds_the_entry() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    // "+" opens the picker, which types into the search field (the draw
    // asserts that focus onto the control each frame).
    h.apply_panel(PanelAction::TogglePicker);
    assert!(h.assets.picker_open && h.assets.search_focus);
    // Pick the first offered type -> AddForm, name field prefilled + focused.
    let ty = h.picker_options().unwrap()[0].name.clone();
    h.apply_panel(PanelAction::PickOption(0));
    assert!(h.form_open());
    assert!(!h.assets.picker_open);
    assert_eq!(h.form.selected_type.as_deref(), Some(ty.as_str()));
    assert_eq!(h.form.target, FormTarget::New);
    assert_eq!(h.form.focus, FormFocus::Name);
    assert!(!h.form.name.is_empty());
    // Edit the name, then confirm.
    h.form.name = "my_light".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert!(!h.form_open());
    assert!(h.dirty);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["args"]["$id"], "my_light");
    assert_eq!(h.entries[0]["type"], ty.as_str());
}

#[test]
fn row_click_opens_the_edit_form_for_a_rename() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    seed_tree(&mut h, Vec::new());
    // Clicking the name row opens the edit form prefilled for a rename.
    click_row(&mut h, "lamp");
    assert!(h.form_open());
    assert_eq!(h.form.target, entry_target(&h, 0));
    assert_eq!(h.form.selected_type.as_deref(), Some("PointLight"));
    assert!(h.assets.row_menu.is_none());
    assert_eq!(h.form.name, "lamp", "name prefilled from the entry");
    // Rename and confirm: same entry, no new one.
    h.form.name = "streetlamp".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(h.entries.len(), 1, "edited in place, not appended");
    assert_eq!(h.entries[0]["args"]["$id"], "streetlamp");
    assert_eq!(h.entries[0]["type"], "PointLight");
    assert!(h.dirty);
}

// The selection holds each member's session key, not its name, so renaming an
// asset through the form leaves it selected (it used to drop out of the set).
#[test]
fn renaming_an_asset_through_the_form_keeps_it_selected() {
    let mut h = hook(vec![entry("floor", "Decal"), entry("lamp", "PointLight")]);
    let world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    seed_tree(&mut h, Vec::new());
    click_row(&mut h, "lamp");
    assert_eq!(active(&h).as_deref(), Some("lamp"));

    h.form.name = "streetlamp".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(h.entries[1]["args"]["$id"], "streetlamp");
    assert_eq!(
        active(&h).as_deref(),
        Some("streetlamp"),
        "the member follows its entry through the rename"
    );
    assert_eq!(selected(&h).len(), 1, "and it is still the only member");
}

// The other way a name-keyed set went wrong: a member sitting after a deleted
// line. Keys do not shift, so the surviving member stays selected and no
// selection lands on the entry that took its place.
#[test]
fn deleting_an_earlier_entry_leaves_the_selection_on_its_own_member() {
    let mut h = hook(vec![entry("a", "Decal"), entry("b", "Decal")]);
    h.open[PanelKey::Assets] = true;
    seed_tree(&mut h, Vec::new());
    select(&mut h, &["b"]);

    let (g, i) = row_of(&h, "a");
    h.apply_panel(PanelAction::OpenRowMenu(g, i));
    h.apply_panel(PanelAction::RowDelete);
    assert_eq!(selected(&h), vec!["b"]);
    assert_eq!(h.handle_index(h.selection.active().unwrap()), Some(0));
}

#[test]
fn row_menu_delete_removes_the_entry() {
    let mut h = hook(vec![entry("a", "Decal"), entry("b", "Decal")]);
    h.open[PanelKey::Assets] = true;
    seed_tree(&mut h, Vec::new());
    let (g, i) = row_of(&h, "a");
    h.apply_panel(PanelAction::OpenRowMenu(g, i));
    h.apply_panel(PanelAction::RowDelete);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["args"]["$id"], "b");
    assert!(h.dirty && h.assets.row_menu.is_none());
}

#[test]
fn edit_rename_to_a_duplicate_is_suffixed() {
    let mut h = hook(vec![entry("a", "Decal"), entry("b", "Decal")]);
    let world = world_with_fields();
    let target = entry_target(&h, 1);
    h.form.target = target;
    h.form.selected_type = Some("Decal".to_string());
    // Rename "b" to "a": collides with the other entry -> suffixed.
    h.form.name = "a".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(h.entries[1]["args"]["$id"], "a_1");
}

fn notices(h: &EditorHook) -> Vec<String> {
    h.notifier
        .stack()
        .cards
        .into_iter()
        .map(|c| c.message)
        .collect()
}

// A form rename moves what a build-only shorthand names: a Panel's screen
// follows the Screen, and its title, which only reads the name, stays.
#[test]
fn edit_rename_moves_a_build_only_reference() {
    let mut h = hook(vec![
        entry("pause", "Screen"),
        entry_with_args(
            "card",
            "Panel",
            serde_json::json!({"screen": "pause", "title": "pause"}),
        ),
    ]);
    let world = world_with_fields();
    h.form.target = entry_target(&h, 0);
    h.form.selected_type = Some("Screen".to_string());
    h.form.name = "paused".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(h.entries[0]["args"]["$id"], "paused");
    assert_eq!(h.entries[1]["args"]["screen"], "paused");
    assert_eq!(h.entries[1]["args"]["title"], "pause");
}

// Blanking the `$id` of an entry another names keeps the id and says why,
// since nothing can name an anonymous entry; an unreferenced one goes
// anonymous.
#[test]
fn blanking_a_referenced_id_keeps_it() {
    let mut h = hook(vec![
        entry("pause", "Screen"),
        entry_with_args("card", "Panel", serde_json::json!({"screen": "pause"})),
        entry("lone", "Screen"),
    ]);
    let world = world_with_fields();
    for at in [0, 2] {
        h.form.target = entry_target(&h, at);
        h.form.selected_type = Some("Screen".to_string());
        h.form.name = String::new();
        h.apply_form(FormAction::Confirm, &world);
    }
    assert_eq!(h.entries[0]["args"]["$id"], "pause");
    assert_eq!(h.entries[1]["args"]["screen"], "pause");
    assert!(
        notices(&h)
            .iter()
            .any(|n| n == "'pause' keeps its `$id`: 1 reference names it")
    );
    assert_eq!(declared_id(&h.entries[2]), None);
}

#[test]
fn confirm_add_with_a_blank_id_adds_it_anonymous() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    h.form.selected_type = Some("PointLight".to_string());
    // Field left blank.
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(declared_id(&h.entries[0]), None);
}

#[test]
fn confirm_add_makes_a_duplicate_name_unique() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let world = world_with_fields();
    h.form.selected_type = Some("PointLight".to_string());
    h.form.name = "lamp".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(
        h.entries[1]["args"]["$id"], "lamp_1",
        "collision is suffixed"
    );
}

// Picking a config singleton from the "+" picker edits the world's existing
// instance if it has one (no second append), and adds one if it does not.
#[test]
fn config_singleton_picker_edits_existing_else_adds() {
    // A world that already has a GraphicsConfig: picking it opens an EDIT.
    let mut h = hook(vec![serde_json::json!({
        "type": "GraphicsConfig", "args": {"$id": "gfx"}
    })]);
    let world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    h.apply_panel(PanelAction::TogglePicker);
    let gi = h
        .picker_options()
        .unwrap()
        .iter()
        .position(|o| o.name == "GraphicsConfig")
        .expect("GraphicsConfig is offered in the picker");
    h.apply_panel(PanelAction::PickOption(gi));
    assert!(h.form_open());
    assert_eq!(
        h.form.target,
        entry_target(&h, 0),
        "picking a present singleton edits it, not a new add"
    );
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(
        h.entries
            .iter()
            .filter(|e| e["type"] == "GraphicsConfig")
            .count(),
        1,
        "the singleton was edited in place, never duplicated"
    );

    // A world WITHOUT the singleton: picking it opens a fresh add.
    let mut h2 = hook(Vec::new());
    let world2 = world_with_fields();
    h2.open[PanelKey::Assets] = true;
    h2.apply_panel(PanelAction::TogglePicker);
    let wi = h2
        .picker_options()
        .unwrap()
        .iter()
        .position(|o| o.name == "Window")
        .expect("Window is offered in the picker");
    h2.apply_panel(PanelAction::PickOption(wi));
    assert!(h2.form_open());
    assert!(
        (h2.form.target == FormTarget::New),
        "no existing Window -> an add form"
    );
    h2.apply_form(FormAction::Confirm, &world2);
    assert_eq!(
        h2.entries.iter().filter(|e| e["type"] == "Window").count(),
        1,
        "the missing singleton was added"
    );
}

#[test]
fn cancel_form_returns_to_the_list_without_adding() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    h.form.selected_type = Some("Decal".to_string());
    h.apply_form(FormAction::Close, &world);
    assert!(!h.form_open());
    assert!(h.form.selected_type.is_none() && (h.form.target == FormTarget::New));
    assert!(h.entries.is_empty() && !h.dirty);
}

#[test]
fn picker_lists_types_alphabetically() {
    let mut h = hook(Vec::new());
    h.assets.picker_open = true;
    let opts = h.picker_options().unwrap();
    let names: Vec<&str> = opts.iter().map(|o| o.name.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted, "the picker is alphabetized ascending");
    assert_eq!(
        opts.len(),
        assets_panel::picker_types().count(),
        "every offered type shown (addables + config singletons)"
    );
    // Concretely: AudioCue sorts before Sprite, and a config singleton is mixed
    // in alphabetically (AppConfig sorts before AudioCue).
    let pos = |t: &str| opts.iter().position(|o| o.name == t).unwrap();
    assert!(pos("AudioCue") < pos("Sprite"));
    assert!(pos("AppConfig") < pos("AudioCue"));
}

// Clicking an asset's name row in the browse list (not just its row menu)
// opens the edit-form panel for that entry, and the row stays selected.
#[test]
fn clicking_a_list_row_opens_its_edit_form() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    h.open[PanelKey::Assets] = true;
    seed_tree(&mut h, Vec::new());
    let mut world = World::new();
    inject::editor_hud(&mut world);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.tick(&mut world);
    // Row 0 is the World group header; row 1 is the asset.
    world.events_mut::<UiEvent>().send(UiEvent {
        name: assets_panel::ROW.into(),
        element: None,
        value: Some(UiValue::Int(1)),
    });
    h.tick(&mut world);
    assert!(h.form_open(), "the row click opened the form");
    assert_eq!(h.form.target, entry_target(&h, 0));
    assert_eq!(h.form.selected_type.as_deref(), Some("PointLight"));
    assert_eq!(
        h.form.name, "lamp",
        "the name heading is seeded from the entry"
    );
}

// The open form holds its entry's session key, so an unrelated delete leaves it
// on the same entry however far that entry moves; deleting the edited entry
// itself is what closes the form.
#[test]
fn deleting_an_unrelated_entry_leaves_the_open_form_on_its_own() {
    let mut h = hook(vec![entry("a", "Decal"), entry("b", "Decal")]);
    let world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    // Edit "b" (index 1), then delete "a" (index 0): "b" slides to 0 and the
    // form is still on it.
    seed_tree(&mut h, Vec::new());
    let target = entry_target(&h, 1);
    h.open_form("Decal".to_string(), target);
    let (g, i) = row_of(&h, "a");
    h.apply_panel(PanelAction::OpenRowMenu(g, i));
    h.apply_panel(PanelAction::RowDelete);
    assert!(h.form_open(), "the form survives an unrelated delete");
    assert_eq!(
        h.form.target,
        entry_target(&h, 0),
        "the same entry, now at 0"
    );
    // Confirm still updates that entry and no other.
    h.form.name = "b2".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["args"]["$id"], "b2");

    // Deleting the edited entry itself closes the form.
    let mut h2 = hook(vec![entry("a", "Decal")]);
    h2.open[PanelKey::Assets] = true;
    seed_tree(&mut h2, Vec::new());
    let target = entry_target(&h2, 0);
    h2.open_form("Decal".to_string(), target);
    let (g2, i2) = row_of(&h2, "a");
    h2.apply_panel(PanelAction::OpenRowMenu(g2, i2));
    h2.apply_panel(PanelAction::RowDelete);
    assert!(!h2.form_open(), "deleting the edited entry closes its form");
}

#[test]
fn add_form_writes_edited_arg_values() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    h.apply_panel(PanelAction::TogglePicker);
    // Pick a type with a float arg through the real picker->pick path.
    let ty = "PointLight".to_string();
    let idx = h
        .picker_options()
        .unwrap()
        .iter()
        .position(|o| o.name == ty)
        .expect("PointLight is offered");
    h.apply_panel(PanelAction::PickOption(idx));
    assert!(h.form_open());
    assert!(!h.form.fields.is_empty(), "the type exposes arg fields");
    // Edit a float field via its input.
    let (j, key) = h
        .form
        .fields
        .iter()
        .enumerate()
        .find(|(_, f)| matches!(f.kind, form::FieldKind::Float))
        .map(|(j, f)| (j, f.key.clone()))
        .expect("a float arg field");
    h.form.texts[j] = "3.5".to_string();
    h.form.name = "lamp".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert!(!h.form_open());
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["args"]["$id"], "lamp");
    assert_eq!(h.entries[0]["type"], ty.as_str());
    assert_eq!(
        h.entries[0]["args"][&key].as_f64(),
        Some(3.5),
        "the edited float persisted into args"
    );
}

#[test]
fn add_form_writes_an_edited_color_vector() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    // VolumetricFog (a newly offered type) has a `color` RGB vector field.
    h.open_form("VolumetricFog".to_string(), FormTarget::New);
    let (j, key) = h
        .form
        .fields
        .iter()
        .enumerate()
        .find(|(_, f)| matches!(f.kind, form::FieldKind::Vec { color: true, .. }))
        .map(|(j, f)| (j, f.key.clone()))
        .expect("a color vector field");
    h.form.texts[j] = "0.1, 0.2, 0.3".to_string();
    h.form.name = "fog".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert!(!h.form_open());
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "VolumetricFog");
    assert_eq!(
        h.entries[0]["args"][&key],
        serde_json::json!([0.1, 0.2, 0.3]),
        "the edited color persisted as a numeric array"
    );
}

// Editing a nested (dotted-path) field through the form persists into the
// sub-object: Camera3D's `controller.move_speed`.
#[test]
fn add_form_writes_a_nested_object_field() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    h.open_form("Camera3D".to_string(), FormTarget::New);
    let j = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "controller.move_speed")
        .expect("the nested controller.move_speed field is offered");
    assert!(matches!(h.form.fields[j].kind, form::FieldKind::Float));
    h.form.texts[j] = "12.5".to_string();
    h.form.name = "cam".to_string();
    h.apply_form(FormAction::Confirm, &world);
    let cam = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == "cam")
        .expect("the camera was added");
    assert_eq!(cam["type"], "Camera3D");
    assert_eq!(
        cam["args"]["controller"]["move_speed"].as_f64(),
        Some(12.5),
        "the nested edit persisted into args.controller.move_speed"
    );
}

// A KeyBinding is a key typed in and an action picked from what the world's
// Screens let one do.
#[test]
fn add_form_writes_a_key_and_a_picked_action_for_a_new_type() {
    let mut h = hook(vec![entry("pause", "Screen")]);
    let world = world_with_fields();
    h.open_form("KeyBinding".to_string(), FormTarget::New);
    let field_pos = |k: &str| {
        h.form
            .fields
            .iter()
            .position(|f| f.key == k)
            .unwrap_or_else(|| panic!("{k} field present"))
    };
    let (key_j, action_j) = (field_pos("key"), field_pos("action"));
    assert!(matches!(h.form.fields[key_j].kind, form::FieldKind::Str));
    let action = &mut h.form.fields[action_j];
    assert_eq!(action.variants[action.variant_idx], "(none)");
    action.variant_idx = action
        .variants
        .iter()
        .position(|v| v == "toggle pause")
        .expect("the Screen is offered");
    h.form.texts[key_j] = "Escape".to_string();
    h.form.name = "pause_key".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert!(!h.form_open());
    assert_eq!(h.entries.len(), 2);
    assert_eq!(h.entries[1]["type"], "KeyBinding");
    assert_eq!(h.entries[1]["args"]["key"], "Escape");
    assert_eq!(
        h.entries[1]["args"]["action"],
        serde_json::json!({"toggle": "pause"})
    );
}

#[test]
fn add_form_cycles_and_persists_an_enum_field() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    // Sprite's `fit` is a string enum -> a cycling picker.
    h.open_form("Sprite".to_string(), FormTarget::New);
    let idx = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "fit")
        .expect("fit enum field");
    assert!(matches!(h.form.fields[idx].kind, form::FieldKind::Enum));
    let n = h.form.fields[idx].variants.len();
    let start = h.form.fields[idx].variant_idx;
    // Cycle once, then confirm.
    h.apply_form(FormAction::CycleField(idx), &world);
    let picked = h.form.fields[idx].variants[(start + 1) % n].clone();
    assert_ne!(
        picked, h.form.fields[idx].variants[start],
        "cycled to a new value"
    );
    h.form.name = "spr".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "Sprite");
    assert_eq!(
        h.entries[0]["args"]["fit"], picked,
        "the cycled enum variant persisted into args"
    );
}

// A field naming two target types offers the assets of both, and nothing else.
#[test]
fn add_form_ref_field_offers_every_target_type() {
    let mut h = hook(vec![
        entry("door", "Prop"),
        entry("pivot", "SkyRotation"),
        entry("grass_tex", "Texture"),
    ]);
    h.open[PanelKey::Assets] = true;
    h.open_form("Prop".to_string(), FormTarget::New);
    let parent = h
        .form
        .fields
        .iter()
        .find(|f| f.key == "parent")
        .expect("parent ref field");
    assert_eq!(parent.variants, vec![form::NONE_LABEL, "door", "pivot"]);
}

#[test]
fn add_form_ref_field_offers_and_persists_an_existing_asset() {
    let mut h = hook(vec![
        entry("grass_tex", "Texture"),
        entry("stone_tex", "Texture"),
    ]);
    let world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    // Add a Decal: its `texture` reference offers the two existing Textures.
    h.open_form("Decal".to_string(), FormTarget::New);
    let idx = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "texture")
        .expect("texture ref field");
    assert!(
        matches!(h.form.fields[idx].kind, form::FieldKind::Ref { targets } if targets == ["Texture"])
    );
    assert_eq!(
        h.form.fields[idx].variants,
        vec![form::NONE_LABEL, "grass_tex", "stone_tex"],
        "options are (none) + the world's Textures"
    );
    assert_eq!(h.form.fields[idx].variant_idx, 0, "starts at (none)");
    // Cycle to the first Texture and confirm.
    h.apply_form(FormAction::CycleField(idx), &world);
    assert_eq!(
        h.form.fields[idx].variants[h.form.fields[idx].variant_idx],
        "grass_tex"
    );
    h.form.name = "splat".to_string();
    h.apply_form(FormAction::Confirm, &world);
    let decal = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == "splat")
        .expect("the decal was added");
    assert_eq!(decal["type"], "Decal");
    assert_eq!(
        decal["args"]["texture"], "grass_tex",
        "the reference persisted as the asset's name"
    );
}

// A ref field with many candidate assets opens a value dropdown (not a cycle):
// the dropdown picks an option, which persists as that asset's name.
#[test]
fn add_form_ref_field_dropdown_picks_and_persists() {
    // More Textures than the cycle cap, so the picker is a dropdown.
    let mut entries = Vec::new();
    for i in 0..(form_panel::CYCLE_MAX + 3) {
        entries.push(entry(&format!("tex_{i}"), "Texture"));
    }
    let mut h = hook(entries);
    let world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    h.open_form("Decal".to_string(), FormTarget::New);
    let idx = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "texture")
        .expect("texture ref field");
    // (none) + the textures exceeds CYCLE_MAX, so a click opens a dropdown.
    assert!(h.form.fields[idx].variants.len() > form_panel::CYCLE_MAX);
    h.apply_form(FormAction::OpenFieldDropdown(idx), &world);
    assert_eq!(h.form.field_dropdown, Some(idx), "the dropdown opened");
    // Pick option 3 (a real texture, past (none) at 0).
    let picked = h.form.fields[idx].variants[3].clone();
    h.apply_form(FormAction::PickFieldOption(3), &world);
    assert!(
        h.form.field_dropdown.is_none(),
        "picking closes the dropdown"
    );
    assert_eq!(h.form.fields[idx].variant_idx, 3, "the option was selected");
    h.form.name = "splat".to_string();
    h.apply_form(FormAction::Confirm, &world);
    let decal = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == "splat")
        .unwrap();
    assert_eq!(
        decal["args"]["texture"], picked,
        "the dropdown-picked reference persisted as the asset's name"
    );
}

// A second click on an open dropdown's field toggles it closed; CloseOverlays
// also dismisses it.
#[test]
fn field_dropdown_toggles_and_close_overlays_dismisses_it() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    h.form.selected_type = Some("Decal".to_string());
    h.apply_form(FormAction::OpenFieldDropdown(0), &world);
    assert_eq!(h.form.field_dropdown, Some(0));
    // Same field again -> closed.
    h.apply_form(FormAction::OpenFieldDropdown(0), &world);
    assert!(h.form.field_dropdown.is_none(), "a second click closes it");
    // Reopen, then the form's CloseOverlays dismisses it.
    h.apply_form(FormAction::OpenFieldDropdown(0), &world);
    h.close_form_menus();
    assert!(
        h.form.field_dropdown.is_none(),
        "CloseOverlays dismisses it"
    );
}

// A WaterSurface entry with one authored swell wave. Its `waves` list is the
// form's array of objects; a new surface has none, and an empty list has no
// element to template a new one from.
fn pond_with_swell() -> serde_json::Value {
    entry_with_args(
        "pond",
        "WaterSurface",
        serde_json::json!({"waves": [{"amplitude": 0.15, "wavelength": 4.0, "direction": [1.0, 0.0], "steepness": 0.4}]}),
    )
}

// Growing an array through the form's [+] and editing the new element persists:
// the pond has one wave; add a second and set its amplitude.
#[test]
fn edit_form_grows_an_array_and_edits_the_new_element() {
    let mut h = hook(vec![pond_with_swell()]);
    let world = world_with_fields();
    let target = entry_target(&h, 0);
    h.open_form("WaterSurface".to_string(), target);
    let header = |h: &EditorHook| {
        h.form
            .fields
            .iter()
            .position(|f| f.key == "waves")
            .expect("waves array header")
    };
    let hj = header(&h);
    assert!(matches!(h.form.fields[hj].kind, form::FieldKind::Array));
    assert_eq!(h.form.fields[hj].variant_idx, 1, "one authored wave");
    // [+] grows the array to two waves (fields re-derive).
    h.apply_form(FormAction::AddArrayElement(hj), &world);
    assert_eq!(
        h.form.fields[header(&h)].variant_idx,
        2,
        "grew to two waves"
    );
    // Edit the second wave's amplitude, then confirm.
    let ej = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "waves.1.amplitude")
        .expect("the second wave's amplitude field");
    h.form.texts[ej] = "4.5".to_string();
    h.apply_form(FormAction::Confirm, &world);
    let ws = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == "pond")
        .expect("the water surface was kept");
    assert_eq!(ws["type"], "WaterSurface");
    assert_eq!(
        ws["args"]["waves"].as_array().map(Vec::len),
        Some(2),
        "the grown array persisted with two waves"
    );
    assert_eq!(
        ws["args"]["waves"][1]["amplitude"].as_f64(),
        Some(4.5),
        "the edited new-element value persisted"
    );
}

// Removing an array element through the form's [-] shrinks it and persists.
#[test]
fn edit_form_removes_an_array_element() {
    let mut h = hook(vec![pond_with_swell()]);
    let world = world_with_fields();
    let target = entry_target(&h, 0);
    h.open_form("WaterSurface".to_string(), target);
    let hj = h.form.fields.iter().position(|f| f.key == "waves").unwrap();
    // Grow to two, then remove one back to one.
    h.apply_form(FormAction::AddArrayElement(hj), &world);
    let hj = h.form.fields.iter().position(|f| f.key == "waves").unwrap();
    assert_eq!(h.form.fields[hj].variant_idx, 2);
    h.apply_form(FormAction::RemoveArrayElement(hj), &world);
    let hj = h.form.fields.iter().position(|f| f.key == "waves").unwrap();
    assert_eq!(h.form.fields[hj].variant_idx, 1, "shrank back to one wave");
    h.apply_form(FormAction::Confirm, &world);
    let ws = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == "pond")
        .unwrap();
    assert_eq!(ws["args"]["waves"].as_array().map(Vec::len), Some(1));
}

// A plain vector opens collapsed; disclosing it exposes per-element leaves whose
// edits write back into the vector (keeping its length) and persist.
#[test]
fn form_discloses_a_vector_and_edits_one_element() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    h.open_form("PointLight".to_string(), FormTarget::New);
    let pos = |h: &EditorHook| {
        h.form
            .fields
            .iter()
            .position(|f| f.key == "position")
            .expect("a position vector field")
    };
    // Collapsed: no element leaves yet.
    assert!(
        h.form
            .fields
            .iter()
            .all(|f| !f.key.starts_with("position."))
    );
    // Disclose it: the element leaves appear and the path is tracked expanded.
    h.apply_form(FormAction::ToggleVecExpand(pos(&h)), &world);
    assert!(h.form.vec_expanded.contains("position"));
    let yj = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "position.1")
        .expect("the y element leaf");
    // Edit y through its control, then confirm.
    h.form.texts[yj] = "4.5".to_string();
    h.form.name = "lamp".to_string();
    h.apply_form(FormAction::Confirm, &world);
    let lamp = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == "lamp")
        .unwrap();
    assert_eq!(
        lamp["args"]["position"].as_array().map(Vec::len),
        Some(3),
        "the vector kept its length"
    );
    assert_eq!(lamp["args"]["position"][1].as_f64(), Some(4.5));
}

// Collapsing a disclosed vector after editing an element keeps the edit (capture
// runs before the field list re-derives).
#[test]
fn collapsing_a_vector_keeps_its_element_edits() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    h.open_form("PointLight".to_string(), FormTarget::New);
    let pj = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "position")
        .unwrap();
    h.apply_form(FormAction::ToggleVecExpand(pj), &world);
    let xj = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "position.0")
        .unwrap();
    h.form.texts[xj] = "2.0".to_string();
    // Collapse again: the element leaves go away but the edit is folded in.
    let pj = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "position")
        .unwrap();
    h.apply_form(FormAction::ToggleVecExpand(pj), &world);
    assert!(!h.form.vec_expanded.contains("position"));
    assert!(
        h.form
            .fields
            .iter()
            .all(|f| !f.key.starts_with("position."))
    );
    h.form.name = "lamp".to_string();
    h.apply_form(FormAction::Confirm, &world);
    let lamp = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == "lamp")
        .unwrap();
    assert_eq!(lamp["args"]["position"][0].as_f64(), Some(2.0));
}

// A form wider than the control pool scrolls: a field past the window is edited
// by wheeling down to it. WaterSurface exposes more than a pool's worth of
// fields, so `roughness` is only reachable after scrolling; its edit must still
// persist (and the untouched off-window fields keep their defaults).
#[test]
fn edit_form_scrolls_to_and_edits_an_off_window_field() {
    let mut h = hook(vec![pond_with_swell()]);
    let world = world_with_fields();
    let target = entry_target(&h, 0);
    h.open_form("WaterSurface".to_string(), target);
    assert!(
        h.form.fields.len() > form::FIELD_POOL,
        "a WaterSurface with a swell wave overflows the control pool"
    );
    let rj = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "roughness")
        .expect("a roughness field");
    h.form.texts[rj] = "0.9".to_string();
    h.apply_form(FormAction::Confirm, &world);
    let ws = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == "pond")
        .expect("the water surface was kept");
    assert_eq!(
        ws["args"]["roughness"].as_f64(),
        Some(0.9),
        "the off-window field's edit persisted after scrolling to it"
    );
    // An untouched off-window top field kept its default (not blanked on capture).
    assert_eq!(
        ws["args"]["extent"],
        form::base_args("WaterSurface")["extent"],
        "a scrolled-away field kept its value"
    );
}

// A reference left at (none) persists as null, not a dangling name.
#[test]
fn add_form_ref_field_defaults_to_none() {
    let mut h = hook(vec![entry("grass_tex", "Texture")]);
    let world = world_with_fields();
    h.open_form("Decal".to_string(), FormTarget::New);
    h.form.name = "bare".to_string();
    h.apply_form(FormAction::Confirm, &world);
    let decal = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == "bare")
        .unwrap();
    assert_eq!(decal["args"]["texture"], serde_json::Value::Null);
}

#[test]
fn invalid_arg_keeps_the_form_open_with_an_error() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    // Font has a u32 `size_px` field; a negative value cannot re-serialize.
    h.open_form("Font".to_string(), FormTarget::New);
    let j = h
        .form
        .fields
        .iter()
        .position(|f| f.key == "size_px")
        .expect("size_px field present");
    assert!(matches!(h.form.fields[j].kind, form::FieldKind::Int));
    h.form.texts[j] = "-5".to_string();
    h.form.name = "myfont".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert!(h.form_open(), "the form stays open on invalid input");
    assert!(h.form.error.is_some(), "an error message is shown");
    assert!(h.entries.is_empty(), "nothing invalid was committed");
}

// Toggling the Assets panel off then on (via the View panel) keeps the open
// form + its browse selection (the state is retained, only hidden), so the same
// view returns.
#[test]
fn toggling_the_assets_panel_keeps_the_open_form_state() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    let target = entry_target(&h, 0);
    h.open_form("PointLight".to_string(), target);
    assert!(h.form_open() && h.form.target == entry_target(&h, 0));
    // Toggle the assets UI off: the form + selection are kept, not discarded.
    h.toggle_view_row(0, &mut world);
    assert!(!h.open[PanelKey::Assets]);
    assert!(
        h.form_open(),
        "the form is kept when the panel is toggled off"
    );
    assert_eq!(
        h.form.target,
        entry_target(&h, 0),
        "the browse selection is kept"
    );
    // Toggle back on: the same form and selection are restored.
    h.toggle_view_row(0, &mut world);
    assert!(h.open[PanelKey::Assets] && h.form_open());
    assert_eq!(h.form.target, entry_target(&h, 0));
}

// Hiding the assets UI hides the form's elements (but keeps its state); showing
// it again re-renders the form.
#[test]
fn a_hidden_assets_panel_hides_the_form_elements() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    h.open[PanelKey::Assets] = true;
    let target = entry_target(&h, 0);
    h.open_form("PointLight".to_string(), target);
    let form_shown = |w: &World| {
        let built = w
            .resource::<BuiltUi>()
            .unwrap()
            .panel(PanelKey::Edit)
            .unwrap();
        w.get_by_id::<Layout>(built.root).unwrap().visible
    };
    h.tick(&mut world);
    assert!(form_shown(&world), "form shown while the panel is open");
    // Toggle off: the form elements hide, but its state is retained.
    h.toggle_view_row(0, &mut world);
    h.tick(&mut world);
    assert!(!form_shown(&world), "form elements hidden when toggled off");
    assert!(h.form_open(), "but the form state is retained");
    // Toggle on: the form re-renders.
    h.toggle_view_row(0, &mut world);
    h.tick(&mut world);
    assert!(form_shown(&world), "form shown again on toggle-on");
}

#[test]
fn edit_form_seeds_and_updates_existing_args() {
    let mut h = hook(vec![serde_json::json!({
        "type": "PointLight", "args": {"$id": "lamp"}
    })]);
    let world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    seed_tree(&mut h, Vec::new());
    click_row(&mut h, "lamp");
    assert_eq!(h.form.target, entry_target(&h, 0));
    assert!(!h.form.fields.is_empty());
    // The name field was seeded from the entry.
    assert_eq!(h.form.name, "lamp");
    // Edit a float and confirm; the same entry gains a full args object.
    let (j, key) = h
        .form
        .fields
        .iter()
        .enumerate()
        .find(|(_, f)| matches!(f.kind, form::FieldKind::Float))
        .map(|(j, f)| (j, f.key.clone()))
        .expect("a float arg field");
    h.form.texts[j] = "9.0".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(h.entries.len(), 1, "edited in place");
    assert_eq!(h.entries[0]["args"][&key].as_f64(), Some(9.0));
}

// Renaming through the form moves every reference to the old name with it, in
// the same undo step: a Prop naming the Material follows, and a name that
// collides is made unique and the references follow THAT name.
#[test]
fn renaming_through_the_form_moves_the_references_with_it() {
    let mut h = hook(vec![
        entry("mat", "Material"),
        entry("stone", "Material"),
        entry_with_args(
            "box",
            "Prop",
            serde_json::json!({"mesh": "cube", "material": "mat"}),
        ),
        entry_with_args(
            "cube",
            "ProceduralMesh",
            serde_json::json!({"generator": "box"}),
        ),
    ]);
    let world = world_with_fields();
    let target = entry_target(&h, 0);
    h.open_form("Material".to_string(), target);
    h.form.name = "stone".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(h.entries[0]["args"]["$id"], "stone_1");
    assert_eq!(h.entries[2]["args"]["material"], "stone_1");

    let target = entry_target(&h, 3);
    h.open_form("ProceduralMesh".to_string(), target);
    h.form.name = "crate".to_string();
    h.apply_form(FormAction::Confirm, &world);
    assert_eq!(
        h.entries[2]["args"]["mesh"], "crate",
        "a structured reference"
    );

    h.undo();
    assert_eq!(h.entries[3]["args"]["$id"], "cube");
    assert_eq!(h.entries[2]["args"]["mesh"], "cube");
    h.undo();
    assert_eq!(h.entries[0]["args"]["$id"], "mat");
    assert_eq!(h.entries[2]["args"]["material"], "mat");
    assert!(!h.can_undo(), "each rename was one step");
}

// The form's rows are rebuilt only when its fields change shape, not as its
// texts change; its derived data follows the typed name and the
// unapplied-edit marker.
#[test]
fn the_form_rebuilds_its_rows_only_when_its_fields_change_shape() {
    let mut h = hook(vec![pond_with_swell()]);
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let target = entry_target(&h, 0);
    h.open_form("WaterSurface".to_string(), target);
    let mut built = world.remove_resource::<BuiltUi>().unwrap();
    let panel = built.panels[PanelKey::Edit].as_mut().unwrap();
    h.prepare_form(&mut world, panel);
    let elements = panel.elements.clone();

    h.form.texts[0] = "typed".to_string();
    h.prepare_form(&mut world, panel);
    assert_eq!(panel.elements, elements, "typing rebuilds nothing");

    h.form.touched = true;
    h.prepare_form(&mut world, panel);
    let title = &h.form.data.as_ref().unwrap().2.title;
    assert!(
        title.ends_with(" *"),
        "the marker reaches the title: {title}"
    );

    let waves = h.form.fields.iter().position(|f| f.key == "waves").unwrap();
    h.apply_form(FormAction::AddArrayElement(waves), &world);
    h.prepare_form(&mut world, panel);
    assert_ne!(panel.elements, elements, "a grown array rebuilds the rows");
}
