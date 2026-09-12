// src/editor/hook/tests/editing_tests.rs
//
// The add / edit form's lifecycle (`hook/editing.rs`): the type picker and its
// search, opening the form from a row or the "+", the name rules a confirm
// applies, and every field kind the form writes back -- scalars, colors,
// nested objects, enums, asset references and their dropdowns, arrays, and a
// disclosed vector. Also what survives a reinjection, a panel toggle, and a
// delete that shifts the open form's index.

use concinnity_core::components::FrameInput;
use concinnity_core::components::Sprite;
use concinnity_core::components::TextInput;
use concinnity_core::ecs::World;

use super::fixtures::{click_row, entry, hook, row_of, seed_tree, set_field, world_with_fields};
use crate::debug_hook::DebugHook;
use crate::editor::hook::{EditorHook, FormTarget, visible_slot};

use crate::editor::inject;

use crate::editor::panels::form;
use crate::editor::panels::form_panel::{self, FormAction};

use crate::editor::panels::panel::{self, PanelAction};
use crate::editor::panels::registry::PanelKey;

use crate::editor::widget;

// A live rebuild re-injects a fresh (blank) HUD; the field snapshot carries the
// editor's typed text (an open form's name, the combo filter) across it so a
// form open during the swap is not blanked.
#[test]
fn field_snapshot_carries_typed_text_across_a_reinjection() {
    let mut old = World::new();
    inject::editor_hud(&mut old);
    widget::seed_field(&mut old, form_panel::NAME_INPUT, "my_light");
    widget::seed_field(&mut old, panel::SEARCH_INPUT, "Point");
    let snapshot = EditorHook::field_snapshot(&old);

    // A fresh HUD injection starts every field blank.
    let mut new = World::new();
    inject::editor_hud(&mut new);
    assert_eq!(widget::field_text(&new, form_panel::NAME_INPUT), "");

    EditorHook::restore_fields(&mut new, &snapshot);
    assert_eq!(widget::field_text(&new, form_panel::NAME_INPUT), "my_light");
    assert_eq!(widget::field_text(&new, panel::SEARCH_INPUT), "Point");
}

// While the "+" picker is open the field narrows its type options instead of
// the tree, so the two never fight over the same text.
#[test]
fn the_search_field_narrows_the_picker_while_it_is_open() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    h.panel_open = true;
    h.apply_panel(PanelAction::TogglePicker, &mut world);
    set_field(&mut world, panel::SEARCH_INPUT, "pointlight");
    let opts = h.picker_options(&world).unwrap();
    assert_eq!(opts, ["PointLight"], "case-insensitive type narrowing");
    assert!(
        h.tree_rows(&world).is_empty(),
        "the tree is not filtered by the picker's text"
    );
}

#[test]
fn plus_picker_then_name_form_adds_the_entry() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    h.panel_open = true;
    // "+" opens the picker, which types into the search field (the draw
    // asserts that focus onto the control each frame).
    h.apply_panel(PanelAction::TogglePicker, &mut world);
    assert!(h.picker_open && h.search_focus);
    // Pick the first offered type -> AddForm, name field prefilled + focused.
    let ty = h.picker_options(&world).unwrap()[0].clone();
    h.apply_panel(PanelAction::PickOption(0), &mut world);
    assert!(h.form_open());
    assert!(!h.picker_open);
    assert_eq!(h.selected_type.as_deref(), Some(ty.as_str()));
    assert!((h.form_target == FormTarget::New));
    let name_field = world
        .query::<TextInput>()
        .find(|t| t.asset_id == form_panel::NAME_INPUT)
        .unwrap();
    assert!(name_field.focused && !name_field.content.is_empty());
    // Edit the name, then confirm.
    set_field(&mut world, form_panel::NAME_INPUT, "my_light");
    h.apply_form(FormAction::Confirm, &mut world);
    assert!(!h.form_open());
    assert!(h.dirty);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["name"], "my_light");
    assert_eq!(h.entries[0]["type"], ty.as_str());
}

#[test]
fn row_click_opens_the_edit_form_for_a_rename() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = world_with_fields();
    h.panel_open = true;
    seed_tree(&mut h, Vec::new());
    // Clicking the name row opens the edit form prefilled for a rename.
    click_row(&mut h, "lamp", &mut world);
    assert!(h.form_open());
    assert_eq!(h.form_target, FormTarget::Entry(0));
    assert_eq!(h.selected_type.as_deref(), Some("PointLight"));
    assert!(h.row_menu.is_none());
    let name_field = world
        .query::<TextInput>()
        .find(|t| t.asset_id == form_panel::NAME_INPUT)
        .unwrap();
    assert_eq!(name_field.content, "lamp", "name prefilled from the entry");
    // Rename and confirm: same entry, no new one.
    set_field(&mut world, form_panel::NAME_INPUT, "streetlamp");
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), 1, "edited in place, not appended");
    assert_eq!(h.entries[0]["name"], "streetlamp");
    assert_eq!(h.entries[0]["type"], "PointLight");
    assert!(h.dirty);
}

#[test]
fn row_menu_delete_removes_the_entry() {
    let mut h = hook(vec![entry("a", "Decal"), entry("b", "Decal")]);
    let mut world = world_with_fields();
    h.panel_open = true;
    seed_tree(&mut h, Vec::new());
    let (g, i) = row_of(&h, "a");
    h.apply_panel(PanelAction::OpenRowMenu(g, i), &mut world);
    h.apply_panel(PanelAction::RowDelete, &mut world);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["name"], "b");
    assert!(h.dirty && h.row_menu.is_none());
}

#[test]
fn edit_rename_to_a_duplicate_is_suffixed() {
    let mut h = hook(vec![entry("a", "Decal"), entry("b", "Decal")]);
    let mut world = world_with_fields();
    h.form_target = FormTarget::Entry(1);
    h.selected_type = Some("Decal".to_string());
    // Rename "b" to "a": collides with the other entry -> suffixed.
    set_field(&mut world, form_panel::NAME_INPUT, "a");
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries[1]["name"], "a_1");
}

#[test]
fn confirm_add_with_blank_name_uses_a_generated_one() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    h.selected_type = Some("PointLight".to_string());
    // Field left blank.
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["name"], "editor_pointlight");
}

#[test]
fn confirm_add_makes_a_duplicate_name_unique() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = world_with_fields();
    h.selected_type = Some("PointLight".to_string());
    set_field(&mut world, form_panel::NAME_INPUT, "lamp");
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries[1]["name"], "lamp_1", "collision is suffixed");
}

// Picking a config singleton from the "+" picker edits the world's existing
// instance if it has one (no second append), and adds one if it does not.
#[test]
fn config_singleton_picker_edits_existing_else_adds() {
    // A world that already has a GraphicsConfig: picking it opens an EDIT.
    let mut h = hook(vec![serde_json::json!({
        "name": "gfx", "type": "GraphicsConfig", "args": {}
    })]);
    let mut world = world_with_fields();
    h.panel_open = true;
    h.apply_panel(PanelAction::TogglePicker, &mut world);
    let gi = h
        .picker_options(&world)
        .unwrap()
        .iter()
        .position(|o| o == "GraphicsConfig")
        .expect("GraphicsConfig is offered in the picker");
    h.apply_panel(PanelAction::PickOption(gi), &mut world);
    assert!(h.form_open());
    assert_eq!(
        h.form_target,
        FormTarget::Entry(0),
        "picking a present singleton edits it, not a new add"
    );
    h.apply_form(FormAction::Confirm, &mut world);
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
    let mut world2 = world_with_fields();
    h2.panel_open = true;
    h2.apply_panel(PanelAction::TogglePicker, &mut world2);
    let wi = h2
        .picker_options(&world2)
        .unwrap()
        .iter()
        .position(|o| o == "Window")
        .expect("Window is offered in the picker");
    h2.apply_panel(PanelAction::PickOption(wi), &mut world2);
    assert!(h2.form_open());
    assert!(
        (h2.form_target == FormTarget::New),
        "no existing Window -> an add form"
    );
    h2.apply_form(FormAction::Confirm, &mut world2);
    assert_eq!(
        h2.entries.iter().filter(|e| e["type"] == "Window").count(),
        1,
        "the missing singleton was added"
    );
}

#[test]
fn cancel_form_returns_to_the_list_without_adding() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    h.selected_type = Some("Decal".to_string());
    h.apply_form(FormAction::Close, &mut world);
    assert!(!h.form_open());
    assert!(h.selected_type.is_none() && (h.form_target == FormTarget::New));
    assert!(h.entries.is_empty() && !h.dirty);
}

#[test]
fn picker_lists_types_alphabetically() {
    let mut h = hook(Vec::new());
    let world = world_with_fields();
    h.picker_open = true;
    let opts = h.picker_options(&world).unwrap();
    let mut sorted = opts.clone();
    sorted.sort();
    assert_eq!(opts, sorted, "the picker is alphabetized ascending");
    assert_eq!(
        opts.len(),
        panel::picker_types().count(),
        "every offered type shown (addables + config singletons)"
    );
    // Concretely: AudioCue sorts before Sprite, and a config singleton is mixed
    // in alphabetically (AppConfig sorts before AudioCue).
    let pos = |t: &str| opts.iter().position(|o| o == t).unwrap();
    assert!(pos("AudioCue") < pos("Sprite"));
    assert!(pos("AppConfig") < pos("AudioCue"));
}

#[test]
fn close_overlays_dismisses_the_picker_and_row_menu() {
    let mut h = hook(vec![entry("a", "Decal")]);
    let mut world = world_with_fields();
    h.picker_open = true;
    h.row_menu = Some("a".to_string());
    h.apply_panel(PanelAction::CloseOverlays, &mut world);
    assert!(!h.picker_open);
    assert!(h.row_menu.is_none());
}

// Clicking an asset's name row in the browse list (not just its row menu)
// opens the edit-form panel for that entry, and the row stays selected.
#[test]
fn clicking_a_list_row_opens_its_edit_form() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    h.panel_open = true;
    seed_tree(&mut h, Vec::new());
    let vp = [1280.0, 720.0];
    let po = h.origin(PanelKey::Assets, vp);
    // Row 0 is the World group header; row 1 is the asset. Aim at its name,
    // clear of the hide toggle now heading the row.
    let row = panel::row_rect(po, panel::PANEL_W, 1);
    let mut world = World::new();
    inject::editor_hud(&mut world);
    world.add_component(FrameInput {
        viewport: vp,
        mouse_x: row[0] + 60.0,
        mouse_y: row[1] + 10.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(h.form_open(), "the row click opened the form");
    assert_eq!(h.form_target, FormTarget::Entry(0));
    assert_eq!(h.selected_type.as_deref(), Some("PointLight"));
    assert_eq!(
        widget::field_text(&world, form_panel::NAME_INPUT),
        "lamp",
        "the name heading is seeded from the entry"
    );
}

// Deleting an entry while a form is open keeps the form's entry index valid:
// deleting the edited entry closes it; deleting an earlier one shifts it.
#[test]
fn deleting_entries_fixes_up_the_open_form_index() {
    let mut h = hook(vec![entry("a", "Decal"), entry("b", "Decal")]);
    let mut world = world_with_fields();
    h.panel_open = true;
    // Edit "b" (index 1), then delete "a" (index 0): the form now edits 0.
    seed_tree(&mut h, Vec::new());
    h.open_form(&mut world, "Decal".to_string(), FormTarget::Entry(1));
    let (g, i) = row_of(&h, "a");
    h.apply_panel(PanelAction::OpenRowMenu(g, i), &mut world);
    h.apply_panel(PanelAction::RowDelete, &mut world);
    assert!(h.form_open(), "the form survives an unrelated delete");
    assert_eq!(
        h.form_target,
        FormTarget::Entry(0),
        "the edited index shifted down"
    );
    // Confirm still updates the right (renamed-index) entry.
    set_field(&mut world, form_panel::NAME_INPUT, "b2");
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["name"], "b2");

    // Deleting the edited entry itself closes the form.
    let mut h2 = hook(vec![entry("a", "Decal")]);
    let mut world2 = world_with_fields();
    h2.panel_open = true;
    seed_tree(&mut h2, Vec::new());
    h2.open_form(&mut world2, "Decal".to_string(), FormTarget::Entry(0));
    let (g2, i2) = row_of(&h2, "a");
    h2.apply_panel(PanelAction::OpenRowMenu(g2, i2), &mut world2);
    h2.apply_panel(PanelAction::RowDelete, &mut world2);
    assert!(!h2.form_open(), "deleting the edited entry closes its form");
}

#[test]
fn add_form_writes_edited_arg_values() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    h.panel_open = true;
    h.apply_panel(PanelAction::TogglePicker, &mut world);
    // Pick a type with a float arg through the real picker->pick path.
    let ty = "PointLight".to_string();
    let idx = h
        .picker_options(&world)
        .unwrap()
        .iter()
        .position(|o| o == &ty)
        .expect("PointLight is offered");
    h.apply_panel(PanelAction::PickOption(idx), &mut world);
    assert!(h.form_open());
    assert!(!h.form_fields.is_empty(), "the type exposes arg fields");
    // Edit a float field via its input.
    let (j, key) = h
        .form_fields
        .iter()
        .enumerate()
        .find(|(_, f)| matches!(f.kind, form::FieldKind::Float))
        .map(|(j, f)| (j, f.key.clone()))
        .expect("a float arg field");
    set_field(&mut world, form_panel::form_input(j), "3.5");
    set_field(&mut world, form_panel::NAME_INPUT, "lamp");
    h.apply_form(FormAction::Confirm, &mut world);
    assert!(!h.form_open());
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["name"], "lamp");
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
    let mut world = world_with_fields();
    // VolumetricFog (a newly offered type) has a `color` RGB vector field.
    h.open_form(&mut world, "VolumetricFog".to_string(), FormTarget::New);
    let (j, key) = h
        .form_fields
        .iter()
        .enumerate()
        .find(|(_, f)| matches!(f.kind, form::FieldKind::Vec { color: true, .. }))
        .map(|(j, f)| (j, f.key.clone()))
        .expect("a color vector field");
    set_field(&mut world, form_panel::form_input(j), "0.1, 0.2, 0.3");
    set_field(&mut world, form_panel::NAME_INPUT, "fog");
    h.apply_form(FormAction::Confirm, &mut world);
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
    let mut world = world_with_fields();
    h.open_form(&mut world, "Camera3D".to_string(), FormTarget::New);
    let j = h
        .form_fields
        .iter()
        .position(|f| f.key == "controller.move_speed")
        .expect("the nested controller.move_speed field is offered");
    assert!(matches!(h.form_fields[j].kind, form::FieldKind::Float));
    set_field(&mut world, form_panel::form_input(j), "12.5");
    set_field(&mut world, form_panel::NAME_INPUT, "cam");
    h.apply_form(FormAction::Confirm, &mut world);
    let cam = h
        .entries
        .iter()
        .find(|e| e["name"] == "cam")
        .expect("the camera was added");
    assert_eq!(cam["type"], "Camera3D");
    assert_eq!(
        cam["args"]["controller"]["move_speed"].as_f64(),
        Some(12.5),
        "the nested edit persisted into args.controller.move_speed"
    );
}

#[test]
fn add_form_writes_string_fields_for_a_new_type() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    // KeyBinding (a newly offered type) is a pair of string fields.
    h.open_form(&mut world, "KeyBinding".to_string(), FormTarget::New);
    let field_pos = |k: &str| {
        h.form_fields
            .iter()
            .position(|f| f.key == k)
            .unwrap_or_else(|| panic!("{k} field present"))
    };
    let (key_j, action_j) = (field_pos("key"), field_pos("action"));
    assert!(matches!(h.form_fields[key_j].kind, form::FieldKind::Str));
    set_field(&mut world, form_panel::form_input(key_j), "Space");
    set_field(&mut world, form_panel::form_input(action_j), "jump");
    set_field(&mut world, form_panel::NAME_INPUT, "jump_key");
    h.apply_form(FormAction::Confirm, &mut world);
    assert!(!h.form_open());
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "KeyBinding");
    assert_eq!(h.entries[0]["args"]["key"], "Space");
    assert_eq!(h.entries[0]["args"]["action"], "jump");
}

#[test]
fn add_form_cycles_and_persists_an_enum_field() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    // Sprite's `fit` is a string enum -> a cycling picker.
    h.open_form(&mut world, "Sprite".to_string(), FormTarget::New);
    let idx = h
        .form_fields
        .iter()
        .position(|f| f.key == "fit")
        .expect("fit enum field");
    assert!(matches!(h.form_fields[idx].kind, form::FieldKind::Enum));
    let n = h.form_fields[idx].variants.len();
    let start = h.form_fields[idx].variant_idx;
    // Cycle once, then confirm.
    h.apply_form(FormAction::CycleField(idx), &mut world);
    let picked = h.form_fields[idx].variants[(start + 1) % n].clone();
    assert_ne!(
        picked, h.form_fields[idx].variants[start],
        "cycled to a new value"
    );
    set_field(&mut world, form_panel::NAME_INPUT, "spr");
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "Sprite");
    assert_eq!(
        h.entries[0]["args"]["fit"], picked,
        "the cycled enum variant persisted into args"
    );
}

#[test]
fn add_form_ref_field_offers_and_persists_an_existing_asset() {
    let mut h = hook(vec![
        entry("grass_tex", "Texture"),
        entry("stone_tex", "Texture"),
    ]);
    let mut world = world_with_fields();
    h.panel_open = true;
    // Add a Decal: its `texture` reference offers the two existing Textures.
    h.open_form(&mut world, "Decal".to_string(), FormTarget::New);
    let idx = h
        .form_fields
        .iter()
        .position(|f| f.key == "texture")
        .expect("texture ref field");
    assert!(
        matches!(h.form_fields[idx].kind, form::FieldKind::Ref { target } if target == "Texture")
    );
    assert_eq!(
        h.form_fields[idx].variants,
        vec![form::NONE_LABEL, "grass_tex", "stone_tex"],
        "options are (none) + the world's Textures"
    );
    assert_eq!(h.form_fields[idx].variant_idx, 0, "starts at (none)");
    // Cycle to the first Texture and confirm.
    h.apply_form(FormAction::CycleField(idx), &mut world);
    assert_eq!(
        h.form_fields[idx].variants[h.form_fields[idx].variant_idx],
        "grass_tex"
    );
    set_field(&mut world, form_panel::NAME_INPUT, "splat");
    h.apply_form(FormAction::Confirm, &mut world);
    let decal = h
        .entries
        .iter()
        .find(|e| e["name"] == "splat")
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
    let mut world = world_with_fields();
    h.panel_open = true;
    h.open_form(&mut world, "Decal".to_string(), FormTarget::New);
    let idx = h
        .form_fields
        .iter()
        .position(|f| f.key == "texture")
        .expect("texture ref field");
    // (none) + the textures exceeds CYCLE_MAX, so a click opens a dropdown.
    assert!(h.form_fields[idx].variants.len() > form_panel::CYCLE_MAX);
    h.apply_form(FormAction::OpenFieldDropdown(idx), &mut world);
    assert_eq!(h.field_dropdown, Some(idx), "the dropdown opened");
    // Pick option 3 (a real texture, past (none) at 0).
    let picked = h.form_fields[idx].variants[3].clone();
    h.apply_form(FormAction::PickFieldOption(3), &mut world);
    assert!(h.field_dropdown.is_none(), "picking closes the dropdown");
    assert_eq!(h.form_fields[idx].variant_idx, 3, "the option was selected");
    set_field(&mut world, form_panel::NAME_INPUT, "splat");
    h.apply_form(FormAction::Confirm, &mut world);
    let decal = h.entries.iter().find(|e| e["name"] == "splat").unwrap();
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
    let mut world = world_with_fields();
    h.selected_type = Some("Decal".to_string());
    h.apply_form(FormAction::OpenFieldDropdown(0), &mut world);
    assert_eq!(h.field_dropdown, Some(0));
    // Same field again -> closed.
    h.apply_form(FormAction::OpenFieldDropdown(0), &mut world);
    assert!(h.field_dropdown.is_none(), "a second click closes it");
    // Reopen, then the form's CloseOverlays dismisses it.
    h.apply_form(FormAction::OpenFieldDropdown(0), &mut world);
    h.apply_form(FormAction::CloseOverlays, &mut world);
    assert!(h.field_dropdown.is_none(), "CloseOverlays dismisses it");
}

// Wheeling scrolls an open value dropdown (which can extend past the fixed
// panel body), independent of the cursor-over-body gate.
#[test]
fn scrolling_advances_an_open_field_dropdown() {
    let mut entries = Vec::new();
    for i in 0..(form_panel::MAX_DROP_ROWS + 4) {
        entries.push(entry(&format!("tex_{i}"), "Texture"));
    }
    let mut h = hook(entries);
    let mut world = world_with_fields();
    h.panel_open = true;
    h.open_form(&mut world, "Decal".to_string(), FormTarget::New);
    let idx = h
        .form_fields
        .iter()
        .position(|f| f.key == "texture")
        .expect("texture ref field");
    h.apply_form(FormAction::OpenFieldDropdown(idx), &mut world);
    assert_eq!(h.field_dropdown_scroll, 0);
    h.scroll_form(1.0, &mut world);
    assert_eq!(
        h.field_dropdown_scroll, 1,
        "wheel down advances the dropdown"
    );
    h.scroll_form(-1.0, &mut world);
    assert_eq!(h.field_dropdown_scroll, 0, "wheel up rewinds it");
    // It cannot scroll past the last page.
    for _ in 0..50 {
        h.scroll_form(1.0, &mut world);
    }
    let total = h.form_fields[idx].variants.len();
    assert_eq!(
        h.field_dropdown_scroll,
        total - form_panel::MAX_DROP_ROWS,
        "scroll clamps to the last full page"
    );
}

// Growing an array through the form's [+] and editing the new element persists:
// WaterSurface starts with one wave; add a second and set its amplitude.
#[test]
fn add_form_grows_an_array_and_edits_the_new_element() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    h.open_form(&mut world, "WaterSurface".to_string(), FormTarget::New);
    let header = |h: &EditorHook| {
        h.form_fields
            .iter()
            .position(|f| f.key == "waves")
            .expect("waves array header")
    };
    let hj = header(&h);
    assert!(matches!(h.form_fields[hj].kind, form::FieldKind::Array));
    assert_eq!(h.form_fields[hj].variant_idx, 1, "one default wave");
    // [+] grows the array to two waves (fields re-derive).
    h.apply_form(FormAction::AddArrayElement(hj), &mut world);
    assert_eq!(
        h.form_fields[header(&h)].variant_idx,
        2,
        "grew to two waves"
    );
    // Edit the second wave's amplitude, then confirm.
    let ej = h
        .form_fields
        .iter()
        .position(|f| f.key == "waves.1.amplitude")
        .expect("the second wave's amplitude field");
    set_field(&mut world, form_panel::form_input(ej), "4.5");
    set_field(&mut world, form_panel::NAME_INPUT, "sea");
    h.apply_form(FormAction::Confirm, &mut world);
    let ws = h
        .entries
        .iter()
        .find(|e| e["name"] == "sea")
        .expect("the water surface was added");
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
fn add_form_removes_an_array_element() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    h.open_form(&mut world, "WaterSurface".to_string(), FormTarget::New);
    let hj = h.form_fields.iter().position(|f| f.key == "waves").unwrap();
    // Grow to two, then remove one back to one.
    h.apply_form(FormAction::AddArrayElement(hj), &mut world);
    let hj = h.form_fields.iter().position(|f| f.key == "waves").unwrap();
    assert_eq!(h.form_fields[hj].variant_idx, 2);
    h.apply_form(FormAction::RemoveArrayElement(hj), &mut world);
    let hj = h.form_fields.iter().position(|f| f.key == "waves").unwrap();
    assert_eq!(h.form_fields[hj].variant_idx, 1, "shrank back to one wave");
    set_field(&mut world, form_panel::NAME_INPUT, "pond");
    h.apply_form(FormAction::Confirm, &mut world);
    let ws = h.entries.iter().find(|e| e["name"] == "pond").unwrap();
    assert_eq!(ws["args"]["waves"].as_array().map(Vec::len), Some(1));
}

// A plain vector opens collapsed; disclosing it exposes per-element leaves whose
// edits write back into the vector (keeping its length) and persist.
#[test]
fn form_discloses_a_vector_and_edits_one_element() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::New);
    let pos = |h: &EditorHook| {
        h.form_fields
            .iter()
            .position(|f| f.key == "position")
            .expect("a position vector field")
    };
    // Collapsed: no element leaves yet.
    assert!(
        h.form_fields
            .iter()
            .all(|f| !f.key.starts_with("position."))
    );
    // Disclose it: the element leaves appear and the path is tracked expanded.
    h.apply_form(FormAction::ToggleVecExpand(pos(&h)), &mut world);
    assert!(h.vec_expanded.contains("position"));
    let yj = h
        .form_fields
        .iter()
        .position(|f| f.key == "position.1")
        .expect("the y element leaf");
    // Edit y through its control, then confirm.
    let slot = visible_slot(yj, h.form_scroll, h.form_window()).expect("y leaf visible");
    set_field(&mut world, form_panel::form_input(slot), "4.5");
    set_field(&mut world, form_panel::NAME_INPUT, "lamp");
    h.apply_form(FormAction::Confirm, &mut world);
    let lamp = h.entries.iter().find(|e| e["name"] == "lamp").unwrap();
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
    let mut world = world_with_fields();
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::New);
    let pj = h
        .form_fields
        .iter()
        .position(|f| f.key == "position")
        .unwrap();
    h.apply_form(FormAction::ToggleVecExpand(pj), &mut world);
    let xj = h
        .form_fields
        .iter()
        .position(|f| f.key == "position.0")
        .unwrap();
    let slot = visible_slot(xj, h.form_scroll, h.form_window()).unwrap();
    set_field(&mut world, form_panel::form_input(slot), "2.0");
    // Collapse again: the element leaves go away but the edit is folded in.
    let pj = h
        .form_fields
        .iter()
        .position(|f| f.key == "position")
        .unwrap();
    h.apply_form(FormAction::ToggleVecExpand(pj), &mut world);
    assert!(!h.vec_expanded.contains("position"));
    assert!(
        h.form_fields
            .iter()
            .all(|f| !f.key.starts_with("position."))
    );
    set_field(&mut world, form_panel::NAME_INPUT, "lamp");
    h.apply_form(FormAction::Confirm, &mut world);
    let lamp = h.entries.iter().find(|e| e["name"] == "lamp").unwrap();
    assert_eq!(lamp["args"]["position"][0].as_f64(), Some(2.0));
}

// A form wider than the control pool scrolls: a field past the window is edited
// by wheeling down to it. WaterSurface exposes more than a pool's worth of
// fields, so `roughness` is only reachable after scrolling; its edit must still
// persist (and the untouched off-window fields keep their defaults).
#[test]
fn add_form_scrolls_to_and_edits_an_off_window_field() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    h.open_form(&mut world, "WaterSurface".to_string(), FormTarget::New);
    assert!(
        h.form_fields.len() > form::FIELD_POOL,
        "WaterSurface overflows the control pool"
    );
    let rj = h
        .form_fields
        .iter()
        .position(|f| f.key == "roughness")
        .expect("a roughness field");
    assert!(
        visible_slot(rj, h.form_scroll, h.form_window()).is_none(),
        "roughness starts past the visible window"
    );
    // Wheel to the bottom; roughness scrolls into the window.
    for _ in 0..h.form_fields.len() {
        h.scroll_form(1.0, &mut world);
    }
    let slot =
        visible_slot(rj, h.form_scroll, h.form_window()).expect("roughness scrolled into view");
    // Edit it through its now-visible control and confirm.
    set_field(&mut world, form_panel::form_input(slot), "0.9");
    set_field(&mut world, form_panel::NAME_INPUT, "sea");
    h.apply_form(FormAction::Confirm, &mut world);
    let ws = h
        .entries
        .iter()
        .find(|e| e["name"] == "sea")
        .expect("the water surface was added");
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
    let mut world = world_with_fields();
    h.open_form(&mut world, "Decal".to_string(), FormTarget::New);
    set_field(&mut world, form_panel::NAME_INPUT, "bare");
    h.apply_form(FormAction::Confirm, &mut world);
    let decal = h.entries.iter().find(|e| e["name"] == "bare").unwrap();
    assert_eq!(decal["args"]["texture"], serde_json::Value::Null);
}

#[test]
fn invalid_arg_keeps_the_form_open_with_an_error() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    // Font has a u32 `size_px` field; a negative value cannot re-serialize.
    h.open_form(&mut world, "Font".to_string(), FormTarget::New);
    let j = h
        .form_fields
        .iter()
        .position(|f| f.key == "size_px")
        .expect("size_px field present");
    assert!(matches!(h.form_fields[j].kind, form::FieldKind::Int));
    set_field(&mut world, form_panel::form_input(j), "-5");
    set_field(&mut world, form_panel::NAME_INPUT, "myfont");
    h.apply_form(FormAction::Confirm, &mut world);
    assert!(h.form_open(), "the form stays open on invalid input");
    assert!(h.form_error.is_some(), "an error message is shown");
    assert!(h.entries.is_empty(), "nothing invalid was committed");
}

// Toggling the Assets panel off then on (via the View panel) keeps the open
// form + its browse selection (the state is retained, only hidden), so the same
// view returns.
#[test]
fn toggling_the_assets_panel_keeps_the_open_form_state() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = world_with_fields();
    h.panel_open = true;
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::Entry(0));
    assert!(h.form_open() && h.form_target == FormTarget::Entry(0));
    // Toggle the assets UI off: the form + selection are kept, not discarded.
    h.toggle_view_row(0, &mut world);
    assert!(!h.panel_open);
    assert!(
        h.form_open(),
        "the form is kept when the panel is toggled off"
    );
    assert_eq!(
        h.form_target,
        FormTarget::Entry(0),
        "the browse selection is kept"
    );
    // Toggle back on: the same form and selection are restored.
    h.toggle_view_row(0, &mut world);
    assert!(h.panel_open && h.form_open());
    assert_eq!(h.form_target, FormTarget::Entry(0));
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
    h.panel_open = true;
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::Entry(0));
    let form_shown = |w: &World| {
        w.query::<Sprite>()
            .find(|s| s.asset_id == form_panel::EDIT_BG)
            .unwrap()
            .visible
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
        "name": "lamp", "type": "PointLight", "args": {}
    })]);
    let mut world = world_with_fields();
    h.panel_open = true;
    seed_tree(&mut h, Vec::new());
    click_row(&mut h, "lamp", &mut world);
    assert_eq!(h.form_target, FormTarget::Entry(0));
    assert!(!h.form_fields.is_empty());
    // The name field was seeded from the entry.
    assert_eq!(widget::field_text(&world, form_panel::NAME_INPUT), "lamp");
    // Edit a float and confirm; the same entry gains a full args object.
    let (j, key) = h
        .form_fields
        .iter()
        .enumerate()
        .find(|(_, f)| matches!(f.kind, form::FieldKind::Float))
        .map(|(j, f)| (j, f.key.clone()))
        .expect("a float arg field");
    set_field(&mut world, form_panel::form_input(j), "9.0");
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), 1, "edited in place");
    assert_eq!(h.entries[0]["args"][&key].as_f64(), Some(9.0));
}
