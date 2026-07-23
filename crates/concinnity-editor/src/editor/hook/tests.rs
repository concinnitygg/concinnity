// src/editor/hook/tests.rs
//
// Unit + tick-level tests for the editor hook.

use super::*;
use crate::assets::{Sprite, TextInput, TextLabel};

fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), entries)
}

// The shared title-bar / close-button rects the routing derives for a panel
// (per-panel geometry fns were retired with the registry).
fn title_rect_of(h: &EditorHook, key: PanelKey, vp: [f32; 2]) -> [f32; 4] {
    let o = h.origin(key, vp);
    [o[0], o[1], registry::panel(key).size(h)[0], widget::TITLE_H]
}
fn close_rect_of(h: &EditorHook, key: PanelKey, vp: [f32; 2]) -> [f32; 4] {
    widget::close_rect(title_rect_of(h, key, vp))
}

// Point the cook's content-addressed cache at a private temp dir for the test
// process, so the in-memory rebuild tests never touch the working directory.
fn isolate_state_dir() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let dir = std::env::temp_dir().join(format!("cn-editor-tests-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        concinnity_core::paths::set_root(dir);
    });
}

// A world holding just a FrameInput, for driving `tick` directly.
fn world_with_input(input: FrameInput) -> World {
    let mut world = World::new_empty();
    world.add_component(input);
    world
}

// A world with the injected typed fields, for the add / edit flow (the
// combo filter, the form's name heading, and its arg-input pool).
fn world_with_fields() -> World {
    let mut world = World::new_empty();
    for id in panel::all_field_ids()
        .into_iter()
        .chain(form_panel::all_field_ids())
    {
        world.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    world
}

fn set_field(world: &mut World, id: crate::ecs::asset_id::AssetId, text: &str) {
    for t in world.query_mut::<TextInput>() {
        if t.asset_id == id {
            t.content = text.to_string();
            break;
        }
    }
}

fn entry(name: &str, ty: &str) -> serde_json::Value {
    serde_json::json!({"name": name, "type": ty, "args": {}})
}

// Seed the cooked tree the panel rows come from, without paying for a real
// world expansion: the working entries under `World`, plus any generated groups
// the test needs. Mirrors what `refresh_tree_if_needed` builds, and clears
// `tree_stale` so the frame drive does not overwrite it.
fn seed_tree(h: &mut EditorHook, extra: Vec<TreeGroup>) {
    let world_group = TreeGroup {
        label: asset_tree::WORLD_GROUP.to_string(),
        assets: h
            .entries
            .iter()
            .map(|e| asset_tree::TreeAsset {
                name: entry_name(e).unwrap_or_default().to_string(),
                asset_type: entry_type(e).unwrap_or_default().to_string(),
                badge: asset_tree::Badge::Authored,
                promote: None,
            })
            .collect(),
    };
    h.tree_groups = std::iter::once(world_group).chain(extra).collect();
    h.tree_unfolded = (0..h.tree_groups.len()).collect();
    h.tree_stale = false;
}

// One generated group, as a scene import's or injection pass's output would
// appear: every asset promotable from the entry the expansion produced.
fn generated_group(label: &str, assets: &[(&str, &str)]) -> TreeGroup {
    TreeGroup {
        label: label.to_string(),
        assets: assets
            .iter()
            .map(|(name, ty)| asset_tree::TreeAsset {
                name: name.to_string(),
                asset_type: ty.to_string(),
                badge: asset_tree::Badge::Imported,
                promote: Some(serde_json::json!({
                    "name": name, "type": ty, "args": {},
                })),
            })
            .collect(),
    }
}

// The (group, index) a row click on `name` resolves to.
fn row_of(h: &EditorHook, name: &str) -> (usize, usize) {
    h.tree_groups
        .iter()
        .enumerate()
        .find_map(|(gi, g)| {
            g.assets
                .iter()
                .position(|a| a.name == name)
                .map(|ai| (gi, ai))
        })
        .unwrap_or_else(|| panic!("{name} is not in the seeded tree"))
}

// Click the tree row for `name` (the panel's plain select-and-edit action).
fn click_row(h: &mut EditorHook, name: &str, world: &mut World) {
    let (g, i) = row_of(h, name);
    h.apply_panel(PanelAction::SelectRow(g, i), world);
}

#[test]
fn starts_in_edit_mode_with_hud_shown() {
    let h = hook(Vec::new());
    assert!(!h.world_capture, "editor holds the cursor at launch");
    assert!(h.hud_visible, "HUD shown at launch");
    // Assets / View / Templates start closed; Preview starts shown.
    assert!(!h.panel_open && !h.view_open && !h.templates_open);
    assert!(h.preview_open, "the Preview panel is shown at launch");
    assert!(!h.picker_open);
}

// The top-bar View button toggles the View panel; the View panel's rows toggle
// the Assets, Preview, and Templates panels independently (no mutual exclusion).
#[test]
fn view_button_and_view_rows_toggle_the_panels() {
    let mut h = hook(Vec::new());
    let mut world = World::new_empty();
    h.apply_top(HudAction::ToggleView, &mut world);
    assert!(h.view_open, "the View button shows the View panel");
    h.apply_top(HudAction::ToggleView, &mut world);
    assert!(!h.view_open, "a second click hides it");
    // Row 0 -> Assets, row 1 -> Preview, row 2 -> Templates.
    h.toggle_view_row(0, &mut world);
    assert!(h.panel_open, "row 0 shows the Assets panel");
    h.toggle_view_row(1, &mut world);
    assert!(
        !h.preview_open,
        "row 1 hides the (default-shown) Preview panel"
    );
    h.toggle_view_row(2, &mut world);
    assert!(h.templates_open, "row 2 shows the Templates panel");
    assert!(
        h.panel_open,
        "Assets stayed shown -- panels are independent"
    );
}

// Picking a template opens its detail panel (nothing is added yet); Apply from
// the detail layers the assets and closes it; re-applying is idempotent.
#[test]
fn template_pick_opens_detail_then_apply_adds_idempotently() {
    let mut h = hook(Vec::new());
    h.open_template_detail(0);
    assert_eq!(h.open_template, Some(0), "the detail panel opens on pick");
    assert!(h.entries.is_empty(), "picking adds nothing on its own");

    h.apply_template_detail(TemplateAction::Apply);
    let first = concinnity_templates::TEMPLATES[0].assets().len();
    assert_eq!(h.entries.len(), first, "Apply adds all template entries");
    assert_eq!(h.open_template, None, "Apply closes the detail panel");

    // Re-open and Apply again: no duplicate entries.
    h.open_template_detail(0);
    h.apply_template_detail(TemplateAction::Apply);
    assert_eq!(h.entries.len(), first, "re-apply is idempotent");
}

// The detail panel's grouped rows come from the shared list model, so they
// match what the template would add (one row per asset plus a type header
// each), and the "X" closes the panel without adding anything.
#[test]
fn template_detail_rows_and_close() {
    let mut h = hook(Vec::new());
    h.open_template_detail(0);
    let rows = h.template_rows(0);
    let names = rows.iter().filter(|r| !r.is_header).count();
    assert_eq!(
        names,
        concinnity_templates::TEMPLATES[0].assets().len(),
        "one name row per template asset"
    );
    assert!(
        rows.iter().any(|r| r.is_header),
        "grouped under type headers"
    );
    h.apply_template_detail(TemplateAction::Close);
    assert_eq!(h.open_template, None);
    assert!(h.entries.is_empty(), "closing adds nothing");
}

// Entry changes drive the live preview: a mutation flags a rebuild AND marks
// the world dirty (unsaved); a plain View toggle does neither.
#[test]
fn entry_changes_request_a_preview_rebuild() {
    let mut h = hook(Vec::new());
    h.apply_top(HudAction::ToggleView, &mut World::new_empty());
    assert!(
        !h.rebuild_preview && !h.dirty,
        "a view toggle is not an entry change"
    );
    // Applying a template layers assets: preview rebuild requested + dirty.
    h.open_template_detail(0);
    h.apply_template_detail(TemplateAction::Apply);
    assert!(
        h.rebuild_preview && h.dirty,
        "applying a template updates the live preview and marks unsaved"
    );
}

// A live rebuild re-injects a fresh (blank) HUD; the field snapshot carries the
// editor's typed text (an open form's name, the combo filter) across it so a
// form open during the swap is not blanked.
#[test]
fn field_snapshot_carries_typed_text_across_a_reinjection() {
    let mut old = World::new_empty();
    super::super::inject::editor_hud(&mut old);
    widget::seed_field(&mut old, form_panel::NAME_INPUT, "my_light");
    widget::seed_field(&mut old, panel::SEARCH_INPUT, "Point");
    let snapshot = EditorHook::field_snapshot(&old);

    // A fresh HUD injection starts every field blank.
    let mut new = World::new_empty();
    super::super::inject::editor_hud(&mut new);
    assert_eq!(widget::field_text(&new, form_panel::NAME_INPUT), "");

    EditorHook::restore_fields(&mut new, &snapshot);
    assert_eq!(widget::field_text(&new, form_panel::NAME_INPUT), "my_light");
    assert_eq!(widget::field_text(&new, panel::SEARCH_INPUT), "Point");
}

// The live preview is rebuilt from the in-memory entries with no disk access:
// authored renderable entries build a rendering world directly, and an empty
// world is seeded so a window still shows. This is the swap's source of truth
// now that SAVE only persists.
#[test]
fn build_preview_world_renders_from_in_memory_entries() {
    isolate_state_dir();
    // Authored renderable entries (a Room + camera) build a rendering world.
    let h = hook(vec![
        serde_json::json!({"name":"cam","type":"Camera3D","args":{}}),
        serde_json::json!({"name":"room","type":"Room","args":{}}),
    ]);
    assert!(
        h.build_preview_world()
            .expect("authored entries build")
            .renders(),
        "authored renderable entries render without disk"
    );
    // Empty entries: the seed keeps the preview window from going blank.
    let h = hook(Vec::new());
    assert!(
        h.build_preview_world()
            .expect("empty world seeds")
            .renders(),
        "an empty world is seeded so it still renders"
    );
}

// The tree lists the world's own lines under `World` and each expansion's
// output under whatever produced it, with the search field narrowing both.
#[test]
fn tree_rows_group_by_origin_and_narrow_by_search() {
    let mut h = hook(vec![entry("lamp", "PointLight"), entry("sign", "Decal")]);
    let mut world = world_with_fields();
    h.panel_open = true;
    seed_tree(
        &mut h,
        vec![generated_group("fox", &[("fox_mat", "Material")])],
    );

    let names = |h: &EditorHook, w: &World| -> Vec<String> {
        h.tree_rows(w)
            .iter()
            .filter_map(|r| match r {
                TreeRow::Asset { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    };
    assert_eq!(names(&h, &world), ["lamp", "sign", "fox_mat"]);
    let headers: Vec<String> = h
        .tree_rows(&world)
        .iter()
        .filter_map(|r| match r {
            TreeRow::Header { label, .. } => Some(label.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(headers, [asset_tree::WORLD_GROUP, "fox"]);

    // The search field matches name or type, across every group.
    set_field(&mut world, panel::SEARCH_INPUT, "mat");
    assert_eq!(names(&h, &world), ["fox_mat"]);
    set_field(&mut world, panel::SEARCH_INPUT, "PointLight");
    assert_eq!(names(&h, &world), ["lamp"], "a type match reaches the row");
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
    // in alphabetically (Application sorts before AudioCue).
    let pos = |t: &str| opts.iter().position(|o| o == t).unwrap();
    assert!(pos("AudioCue") < pos("Sprite"));
    assert!(pos("Application") < pos("AudioCue"));
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

#[test]
fn tick_escape_returns_cursor_to_editor() {
    let mut h = hook(Vec::new());
    h.world_capture = true;
    let mut world = world_with_input(FrameInput {
        escape: true,
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(!h.world_capture, "Escape leaves play mode");
}

#[test]
fn tick_f1_toggles_hud_visibility() {
    let mut h = hook(Vec::new());
    let mut world = world_with_input(FrameInput {
        hud_toggle: true,
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    assert!(h.hud_visible);
    h.tick(&mut world);
    assert!(!h.hud_visible, "first F1 hides the HUD");
    h.tick(&mut world);
    assert!(h.hud_visible, "second F1 shows it again");
}

// Overwrite the world's FrameInput in place (tick reads the live component).
fn set_input(world: &mut World, input: FrameInput) {
    if let Some(i) = world.query_mut::<FrameInput>().last() {
        *i = input;
    } else {
        world.add_component(input);
    }
}

// Clicking the Preview panel's capture row hands the cursor to the world
// (clicking again takes it back); the fly row below toggles the fly camera,
// and the two never hold the cursor together.
#[test]
fn preview_rows_toggle_play_mode_and_fly() {
    let mut h = hook(Vec::new());
    let vp = [1280.0, 720.0];
    let o = h.origin(PanelKey::Preview, vp);
    let row_mid = |i: usize| {
        let r = super::super::list_panel::row_rect(o, 200.0, i);
        [r[0] + 10.0, r[1] + r[3] * 0.5]
    };
    let click = |h: &mut EditorHook, world: &mut World, pos: [f32; 2]| {
        set_input(
            world,
            FrameInput {
                viewport: vp,
                mouse_x: pos[0],
                mouse_y: pos[1],
                left_click: true,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(world);
    };
    let mut world = world_with_input(FrameInput::default());

    click(&mut h, &mut world, row_mid(0));
    assert!(h.world_capture, "the checkbox click enters play mode");
    click(&mut h, &mut world, row_mid(0));
    assert!(!h.world_capture, "a second click leaves it");

    click(&mut h, &mut world, row_mid(1));
    assert!(h.fly, "the fly row starts the fly camera");
    assert!(!h.world_capture);
    click(&mut h, &mut world, row_mid(0));
    assert!(
        h.world_capture && !h.fly,
        "entering play mode ends the fly camera"
    );
}

// Holding a panel's title bar drags it; the origin follows the cursor by the
// grab offset and hard-stops at the window edges. Release ends the drag.
#[test]
fn title_bar_drag_moves_and_clamps_the_assets_panel() {
    let mut h = hook(Vec::new());
    h.panel_open = true;
    let vp = [1280.0, 720.0];
    let start = h.origin(PanelKey::Assets, vp);
    // Press on the title bar, 10 px in from its corner.
    let mut world = world_with_input(FrameInput {
        viewport: vp,
        mouse_x: start[0] + 10.0,
        mouse_y: start[1] + 10.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(h.drag.is_some(), "the title press starts a drag");

    // Hold and move: the origin follows, preserving the grab offset.
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: 400.0,
            mouse_y: 150.0,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(h.origin(PanelKey::Assets, vp), [390.0, 140.0]);

    // Drag far past the top-left corner: the panel hard-stops at the left edge
    // and at the top bar's lower edge, never sliding under the bar.
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: -500.0,
            mouse_y: -500.0,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(
        h.origin(PanelKey::Assets, vp),
        [0.0, hud::BAR_H],
        "never partially off screen or under the top bar"
    );

    // Release ends the drag; the panel stays where it was dropped.
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            left_button_down: false,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(h.drag.is_none(), "release ends the drag");
    assert_eq!(h.origin(PanelKey::Assets, vp), [0.0, hud::BAR_H]);
}

// The Preview panel drags by its own title bar, clamped to the window's far
// corner by its own (smaller) footprint.
#[test]
fn title_bar_drag_moves_and_clamps_the_preview_panel() {
    let mut h = hook(Vec::new());
    let vp = [1280.0, 720.0];
    let start = h.origin(PanelKey::Preview, vp);
    let mut world = world_with_input(FrameInput {
        viewport: vp,
        mouse_x: start[0] + 5.0,
        mouse_y: start[1] + 5.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(h.drag.is_some());
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: 5000.0,
            mouse_y: 5000.0,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    let size = preview::size();
    assert_eq!(
        h.origin(PanelKey::Preview, vp),
        [vp[0] - size[0], vp[1] - size[1]],
        "stops flush with the bottom-right corner"
    );
}

// While a drag is in progress the press's click must not also resolve to a
// control underneath on later frames -- e.g. dragging the Assets panel across
// the Preview checkbox must not toggle play mode.
#[test]
fn dragging_does_not_trigger_controls_it_crosses() {
    let mut h = hook(Vec::new());
    h.panel_open = true;
    let vp = [1280.0, 720.0];
    let start = h.origin(PanelKey::Assets, vp);
    let mut world = world_with_input(FrameInput {
        viewport: vp,
        mouse_x: start[0] + 10.0,
        mouse_y: start[1] + 10.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    h.tick(&mut world);
    // Cross the Preview panel's capture row with the button still held and a
    // stray click edge (e.g. from event coalescing).
    let pv = h.origin(PanelKey::Preview, vp);
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: pv[0] + 10.0,
            mouse_y: pv[1] + preview::size()[1] - 5.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(!h.world_capture, "the drag swallowed the click");
    assert!(h.drag.is_some(), "still dragging");
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
    let row = panel::row_rect(po, 1);
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
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

// The edit-form panel drags by its own title bar, independent of the Assets
// panel.
#[test]
fn edit_panel_drags_by_its_title_bar() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    h.panel_open = true;
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::Entry(0));
    let vp = [1280.0, 720.0];
    let fo = h.origin(PanelKey::Edit, vp);
    world.add_component(FrameInput {
        viewport: vp,
        mouse_x: fo[0] + 12.0,
        mouse_y: fo[1] + 8.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(h.drag.is_some(), "the form title press starts a drag");
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: 112.0,
            mouse_y: 208.0,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(h.origin(PanelKey::Edit, vp), [100.0, 200.0]);
    assert_eq!(
        h.origin(PanelKey::Assets, vp),
        panel::default_origin(vp[0]),
        "the Assets panel did not move"
    );
}

// Focusing a panel moves it to the front of the stack (drawn on top, first
// clicked) without duplicating it.
#[test]
fn focusing_a_panel_moves_it_to_the_front() {
    let mut h = hook(Vec::new());
    let panels = h.panel_order.len();
    // Default order matches the injected draw order: the Template detail panel
    // frontmost (over the Templates list it spawns from).
    assert_eq!(
        h.panel_order.last().copied(),
        Some(PanelKey::TemplateDetail)
    );
    h.focus_panel(PanelKey::Assets);
    assert_eq!(h.panel_order.last().copied(), Some(PanelKey::Assets));
    assert_eq!(h.panel_order.len(), panels, "no duplicates");
    // Re-focusing the frontmost is a no-op.
    h.focus_panel(PanelKey::Assets);
    assert_eq!(h.panel_order.last().copied(), Some(PanelKey::Assets));
    assert_eq!(h.panel_order.len(), panels);
}

// The published HUD layers rank the panels by focus (frontmost highest) and pin
// the top bar above them all, so the renderer occludes overlaps cleanly.
#[test]
fn publish_layers_ranks_panels_below_the_top_bar() {
    let mut h = hook(Vec::new());
    h.focus_panel(PanelKey::Edit); // Edit -> frontmost
    let layers = h.compute_layers();
    let layer = |id| *layers.get(&id).expect("id mapped");
    let edit = layer(form_panel::EDIT_BG);
    let assets = layer(panel::PANEL_BG);
    let preview = layer(preview::PANEL_BG);
    assert!(
        edit > assets && edit > preview,
        "the frontmost panel outranks the others"
    );
    assert!(
        layer(hud::SAVE_BUTTON) > edit,
        "the top bar sits above every panel"
    );
    // A panel's text input shares its panel's layer (it must not sink below it).
    assert_eq!(layer(form_panel::NAME_INPUT), edit);
}

// A press on a shown panel brings it to the front and (on its title bar) starts
// a drag.
#[test]
fn a_panel_press_brings_it_to_the_front() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = world_with_fields();
    h.panel_open = true;
    let vp = [1280.0, 720.0];
    let po = h.origin(PanelKey::Assets, vp);
    let t = panel::title_rect(po);
    let claimed = h.try_panel_press(PanelKey::Assets, t[0] + 5.0, t[1] + 5.0, vp, &mut world);
    assert!(claimed, "the press was claimed by the Assets panel");
    assert_eq!(h.panel_order.last().copied(), Some(PanelKey::Assets));
    assert!(h.drag.is_some(), "a title-bar press starts a drag");
}

// The X in the edit form's title bar closes the form: the hook routes it before
// the title-bar drag, so it closes rather than starting a drag.
#[test]
fn edit_form_title_bar_x_closes_the_form() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = world_with_fields();
    h.panel_open = true;
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::Entry(0));
    assert!(h.form_open());
    let vp = [1280.0, 720.0];
    let x = form_panel::close_rect(h.origin(PanelKey::Edit, vp));
    let claimed = h.try_panel_press(PanelKey::Edit, x[0] + 5.0, x[1] + 5.0, vp, &mut world);
    assert!(claimed, "the X press was claimed");
    assert!(!h.form_open(), "the X closed the form");
    assert!(h.drag.is_none(), "the X did not start a drag");
}

// Every floating panel's title-bar X closes it: the press is checked before the
// title drag, so it closes rather than starting a drag.
#[test]
fn every_panel_title_bar_x_closes_it() {
    let vp = [1280.0, 720.0];
    let mut world = world_with_fields();

    // Preview starts shown; its X hides it.
    let mut h = hook(Vec::new());
    let px = close_rect_of(&h, PanelKey::Preview, vp);
    assert!(h.try_panel_press(PanelKey::Preview, px[0] + 5.0, px[1] + 5.0, vp, &mut world));
    assert!(
        !h.preview_open && h.drag.is_none(),
        "Preview X closed it, no drag"
    );

    // Assets.
    let mut h = hook(Vec::new());
    h.panel_open = true;
    let ax = close_rect_of(&h, PanelKey::Assets, vp);
    assert!(h.try_panel_press(PanelKey::Assets, ax[0] + 5.0, ax[1] + 5.0, vp, &mut world));
    assert!(
        !h.panel_open && h.drag.is_none(),
        "Assets X closed it, no drag"
    );

    // View.
    let mut h = hook(Vec::new());
    h.view_open = true;
    let vx = close_rect_of(&h, PanelKey::View, vp);
    assert!(h.try_panel_press(PanelKey::View, vx[0] + 5.0, vx[1] + 5.0, vp, &mut world));
    assert!(
        !h.view_open && h.drag.is_none(),
        "View X closed it, no drag"
    );

    // Templates.
    let mut h = hook(Vec::new());
    h.templates_open = true;
    let tx = close_rect_of(&h, PanelKey::Templates, vp);
    assert!(h.try_panel_press(
        PanelKey::Templates,
        tx[0] + 5.0,
        tx[1] + 5.0,
        vp,
        &mut world
    ));
    assert!(
        !h.templates_open && h.drag.is_none(),
        "Templates X closed it, no drag"
    );
}

// A panel toggled off (its View checkbox unticked) is not interactive: a press
// where it would be falls through instead of being claimed.
#[test]
fn a_hidden_panel_is_not_interactive() {
    let mut h = hook(Vec::new());
    let vp = [1280.0, 720.0];
    let mut world = world_with_fields();
    // Preview starts shown: a title-bar press is claimed (starts a drag).
    let pt = title_rect_of(&h, PanelKey::Preview, vp);
    assert!(h.try_panel_press(PanelKey::Preview, pt[0] + 5.0, pt[1] + 5.0, vp, &mut world));
    // Hidden: the same press falls through.
    h.drag = None;
    h.preview_open = false;
    assert!(!h.try_panel_press(PanelKey::Preview, pt[0] + 5.0, pt[1] + 5.0, vp, &mut world));
    // The View panel starts hidden: its press falls through until it is opened.
    let vt = title_rect_of(&h, PanelKey::View, vp);
    assert!(!h.try_panel_press(PanelKey::View, vt[0] + 5.0, vt[1] + 5.0, vp, &mut world));
    h.view_open = true;
    assert!(h.try_panel_press(PanelKey::View, vt[0] + 5.0, vt[1] + 5.0, vp, &mut world));
}

// The Templates panel drags by its own title bar and comes to the front on a
// press, like the other floating panels.
#[test]
fn templates_panel_press_drags_and_focuses() {
    let mut h = hook(Vec::new());
    h.templates_open = true;
    let vp = [1280.0, 720.0];
    let mut world = world_with_fields();
    let t = title_rect_of(&h, PanelKey::Templates, vp);
    assert!(h.try_panel_press(PanelKey::Templates, t[0] + 5.0, t[1] + 5.0, vp, &mut world));
    assert!(h.drag.is_some(), "a title-bar press starts a drag");
    assert_eq!(h.panel_order.last().copied(), Some(PanelKey::Templates));
}

// End-to-end through `tick` against a fully injected HUD: the top-bar View
// button opens the View panel, and clicking its "Templates" row opens the
// Templates panel (the same click path a real session drives).
#[test]
fn tick_view_button_opens_view_then_a_row_opens_templates() {
    let vis = |w: &World, id: crate::ecs::asset_id::AssetId| {
        w.query::<Sprite>()
            .find(|s| s.asset_id == id)
            .map(|s| s.visible)
            .unwrap_or(false)
    };
    let rect = |w: &World, id: crate::ecs::asset_id::AssetId| {
        let s = w.query::<Sprite>().find(|s| s.asset_id == id).unwrap();
        [s.x, s.y, s.width, s.height]
    };
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let vp = [1280.0, 720.0];
    let mut h = hook(Vec::new());

    // Frame 1: no interaction. View + Templates start hidden.
    world.add_component(FrameInput {
        viewport: vp,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(!vis(&world, view::PANEL_BG) && !vis(&world, templates::PANEL_BG));

    // Frame 2: click the top-bar View button -> the View panel opens.
    let view_btn = hud::layout(vp[0]).view;
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: view_btn[0] + view_btn[2] * 0.5,
            mouse_y: view_btn[1] + view_btn[3] * 0.5,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(
        h.view_open && vis(&world, view::PANEL_BG),
        "View panel opened"
    );
    // Its "Templates" row (index 2) is laid out; grab its rect to click it.
    let row = rect(&world, view::row_bg(2));

    // Frame 3: click that row -> the Templates panel opens.
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: row[0] + row[2] * 0.5,
            mouse_y: row[1] + row[3] * 0.5,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(h.templates_open, "the Templates row toggled the panel on");
    assert!(vis(&world, templates::PANEL_BG), "Templates panel shown");
}

// Picking a template row spawns the detail panel (title "Template <name>",
// hidden until then); its Apply button layers the template's assets and closes
// the detail. Drives the whole flow through `tick` end to end.
#[test]
fn tick_picking_a_template_spawns_the_detail_panel_then_apply_adds() {
    let vis = |w: &World, id: crate::ecs::asset_id::AssetId| {
        w.query::<Sprite>()
            .find(|s| s.asset_id == id)
            .map(|s| s.visible)
            .unwrap_or(false)
    };
    let rect = |w: &World, id: crate::ecs::asset_id::AssetId| {
        let s = w.query::<Sprite>().find(|s| s.asset_id == id).unwrap();
        [s.x, s.y, s.width, s.height]
    };
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let vp = [1280.0, 720.0];
    let mut h = hook(Vec::new());
    // Start with the Templates list already open.
    h.templates_open = true;
    world.add_component(FrameInput {
        viewport: vp,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(
        vis(&world, templates::PANEL_BG) && !vis(&world, template_panel::PANEL_BG),
        "Templates list shown; detail panel still hidden"
    );

    // Click the first template row -> the detail panel spawns.
    let row = rect(&world, templates::row_bg(0));
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: row[0] + row[2] * 0.5,
            mouse_y: row[1] + row[3] * 0.5,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(h.open_template, Some(0), "the detail panel opened on pick");
    assert!(vis(&world, template_panel::PANEL_BG), "detail panel shown");
    let title = world
        .query::<crate::assets::TextLabel>()
        .find(|l| l.asset_id == template_panel::TITLE_LABEL)
        .unwrap();
    assert!(
        title.content.starts_with("Template "),
        "title bar reads 'Template <name>': {}",
        title.content
    );
    assert!(h.entries.is_empty(), "picking adds nothing yet");

    // Click the detail's Apply button -> the template's assets are added and
    // the detail closes.
    let apply = template_panel::apply_rect(h.origin(PanelKey::TemplateDetail, vp));
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: apply[0] + apply[2] * 0.5,
            mouse_y: apply[1] + apply[3] * 0.5,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(h.open_template, None, "Apply closed the detail panel");
    assert!(
        !vis(&world, template_panel::PANEL_BG),
        "detail panel hidden"
    );
    assert_eq!(
        h.entries.len(),
        concinnity_templates::TEMPLATES[0].assets().len(),
        "Apply layered the template's assets"
    );
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
fn add_form_writes_an_edited_colour_vector() {
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
        .expect("a colour vector field");
    set_field(&mut world, form_panel::form_input(j), "0.1, 0.2, 0.3");
    set_field(&mut world, form_panel::NAME_INPUT, "fog");
    h.apply_form(FormAction::Confirm, &mut world);
    assert!(!h.form_open());
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "VolumetricFog");
    assert_eq!(
        h.entries[0]["args"][&key],
        serde_json::json!([0.1, 0.2, 0.3]),
        "the edited colour persisted as a numeric array"
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
    let slot = visible_slot(yj, h.form_scroll).expect("y leaf visible");
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
    let slot = visible_slot(xj, h.form_scroll).unwrap();
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
        visible_slot(rj, h.form_scroll).is_none(),
        "roughness starts past the visible window"
    );
    // Wheel to the bottom; roughness scrolls into the window.
    for _ in 0..h.form_fields.len() {
        h.scroll_form(1.0, &mut world);
    }
    let slot = visible_slot(rj, h.form_scroll).expect("roughness scrolled into view");
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
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
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

// Drive `tick` against a fully injected HUD world in each panel body state,
// exercising the real `panel::apply` layout path (not just the pure hit-test /
// action logic the other tests cover).
#[test]
fn tick_lays_out_the_open_panel_in_every_state() {
    let sprite_visible = |w: &World, id: crate::ecs::asset_id::AssetId| {
        w.query::<Sprite>()
            .find(|s| s.asset_id == id)
            .unwrap()
            .visible
    };
    let label = |w: &World, id: crate::ecs::asset_id::AssetId| {
        w.query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .unwrap()
            .clone()
    };

    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        mouse_x: 1200.0,
        mouse_y: 300.0,
        ..Default::default()
    });
    let mut h = hook(vec![entry("a", "PointLight"), entry("b", "Decal")]);
    h.panel_open = true;
    seed_tree(&mut h, Vec::new());

    // Tree: panel drawn, first row is the World group header.
    h.tick(&mut world);
    assert!(sprite_visible(&world, panel::PANEL_BG), "panel bg shown");
    let row0 = label(&world, panel::name_label(0));
    assert!(
        row0.visible && row0.content.starts_with("- World"),
        "first row is the World group header, got {:?}",
        row0.content
    );

    // Type picker: the solid backing and the search field show.
    h.picker_open = true;
    h.tick(&mut world);
    assert!(
        sprite_visible(&world, panel::PICKER_BG),
        "picker backing shown"
    );
    assert!(
        world
            .query::<TextInput>()
            .find(|t| t.asset_id == panel::SEARCH_INPUT)
            .unwrap()
            .visible
    );

    // Row menu: the Delete popup shows over the "a" row.
    h.picker_open = false;
    h.row_menu = Some("a".to_string());
    h.tick(&mut world);
    assert!(sprite_visible(&world, panel::MENU_BG), "row menu shown");
    assert_eq!(label(&world, panel::MENU_DELETE_LABEL).content, "Delete");

    // Form open: the edit panel shows alongside the browse list, with its
    // title bar, name heading, and confirm button.
    h.row_menu = None;
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::New);
    h.tick(&mut world);
    assert!(
        sprite_visible(&world, form_panel::APPLY_BG),
        "confirm button shown"
    );
    assert_eq!(
        label(&world, form_panel::TITLE_LABEL).content,
        "New PointLight"
    );
    assert_eq!(label(&world, form_panel::APPLY_LABEL).content, "Add");
    assert!(
        world
            .query::<TextInput>()
            .find(|t| t.asset_id == form_panel::NAME_INPUT)
            .unwrap()
            .visible,
        "the name heading shows"
    );
    assert!(
        label(&world, panel::name_label(0)).visible,
        "the tree stays visible beside the form"
    );

    // Closing the panel + form blanks both.
    h.panel_open = false;
    h.close_form();
    h.tick(&mut world);
    assert!(!sprite_visible(&world, panel::PANEL_BG), "panel bg hidden");
    assert!(
        !sprite_visible(&world, form_panel::EDIT_BG),
        "form panel hidden"
    );
}

#[test]
fn write_jsonl_persists_entries_atomically() {
    let path = std::env::temp_dir().join("cn_editor_write_jsonl_test.jsonl");
    let path_str = path.to_str().unwrap().to_string();
    let _ = std::fs::remove_file(&path);

    let mut h = hook(vec![serde_json::json!({
        "name": "scene", "type": "GraphicsConfig", "args": {}
    })]);
    h.world_path = path_str.clone();
    let mut world = world_with_fields();
    h.selected_type = Some("PointLight".to_string());
    set_field(&mut world, form_panel::NAME_INPUT, "lamp");
    h.apply_form(FormAction::Confirm, &mut world);
    h.write_jsonl().unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    let parsed = crate::world::parse_world_jsonl(&content).unwrap();
    assert_eq!(parsed.len(), 2, "both entries written, one line each");
    assert_eq!(parsed[1]["name"], "lamp");
    assert!(!std::path::Path::new(&format!("{path_str}.tmp")).exists());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn scroll_moves_each_regions_offset() {
    let world = world_with_fields();
    // A tree longer than the row window, so its scroll can advance.
    let mut h = hook(
        (0..20)
            .map(|i| entry(&format!("log{i}"), "Logger"))
            .collect(),
    );
    seed_tree(&mut h, Vec::new());
    h.row_menu = Some("log0".to_string());
    h.scroll_tree(1.0, &world);
    assert!(h.tree_scroll > 0, "a closed picker scrolls the tree");
    assert!(h.row_menu.is_none(), "scrolling dismisses an open row menu");
    h.scroll_tree(-1.0, &world);
    assert_eq!(h.tree_scroll, 0, "scrolling back up clamps at the top");

    // An open picker scrolls its own option list instead of the tree.
    h.picker_open = true;
    let before = h.tree_scroll;
    h.scroll_tree(1.0, &world);
    assert_eq!(
        h.tree_scroll, before,
        "the tree stays put while the picker is open"
    );
    assert!(h.picker_scroll > 0, "the picker's own list scrolled");
    h.picker_open = false;

    // A picked template detail scrolls its own asset list.
    h.open_template = Some(0);
    h.scroll_template_list(1.0);
}

#[test]
fn drive_drag_parks_each_secondary_panel() {
    let vp = [1280.0, 720.0];
    let held = FrameInput {
        left_button_down: true,
        mouse_x: 220.0,
        mouse_y: 160.0,
        ..Default::default()
    };

    let mut h = hook(Vec::new());
    h.drag = Some(Drag {
        key: PanelKey::View,
        grab: [10.0, 10.0],
    });
    h.drive_drag(&held, vp);
    assert!(
        h.positions[PanelKey::View.index()].is_some(),
        "the View panel follows the cursor"
    );

    let mut h = hook(Vec::new());
    h.drag = Some(Drag {
        key: PanelKey::Templates,
        grab: [10.0, 10.0],
    });
    h.drive_drag(&held, vp);
    assert!(
        h.positions[PanelKey::Templates.index()].is_some(),
        "the Templates panel follows"
    );

    let mut h = hook(Vec::new());
    h.open_template = Some(0);
    h.drag = Some(Drag {
        key: PanelKey::TemplateDetail,
        grab: [10.0, 10.0],
    });
    h.drive_drag(&held, vp);
    assert!(
        h.positions[PanelKey::TemplateDetail.index()].is_some(),
        "the Template detail panel follows"
    );
}

#[test]
fn apply_form_focus_toggle_and_consume() {
    let mut world = world_with_fields();
    let mut h = hook(Vec::new());
    h.form_fields = vec![FormField {
        key: "on".into(),
        kind: form::FieldKind::Bool,
        initial: String::new(),
        boolval: false,
        variants: Vec::new(),
        variant_idx: 0,
    }];
    h.apply_form(FormAction::FocusField(0), &mut world);
    assert!(matches!(h.form_focus, FormFocus::Field(0)));
    h.apply_form(FormAction::FocusName, &mut world);
    assert!(matches!(h.form_focus, FormFocus::Name));
    h.apply_form(FormAction::ToggleField(0), &mut world);
    assert!(h.form_fields[0].boolval, "the bool field flipped");
    // A click that hits no control is swallowed without side effects.
    h.apply_form(FormAction::Consume, &mut world);
    assert!(h.form_fields[0].boolval);
}

#[test]
fn apply_panel_toggles_the_picker_and_consumes() {
    let mut world = world_with_fields();
    let mut h = hook(Vec::new());
    h.apply_panel(PanelAction::TogglePicker, &mut world);
    assert!(h.picker_open);
    assert!(h.search_focus, "the picker types into the search field");
    h.apply_panel(PanelAction::TogglePicker, &mut world);
    assert!(!h.picker_open, "a second toggle closes the picker");
    h.apply_panel(PanelAction::Consume, &mut world);
}

#[test]
fn apply_panel_pick_option_opens_the_add_form() {
    let mut world = world_with_fields();
    let mut h = hook(vec![entry("log", "Logger")]);

    h.picker_open = true;
    h.apply_panel(PanelAction::PickOption(0), &mut world);
    assert!(h.form_open(), "a picker pick opens the add form");
    assert!(!h.picker_open, "and closes the picker behind it");

    // A pick with the picker already closed is a no-op: there is no option
    // list to index into.
    h.close_form();
    h.apply_panel(PanelAction::PickOption(0), &mut world);
    assert!(!h.form_open());
}

#[test]
fn confirm_form_without_a_selected_type_just_closes() {
    let mut world = world_with_fields();
    let mut h = hook(Vec::new());
    h.selected_type = None;
    h.confirm_form(&mut world);
    assert!(!h.form_open());
}

// Lighting panel

fn sun_entry() -> serde_json::Value {
    serde_json::json!({"name": "sun", "type": "DirectionalLight", "args": {
        "direction": [-0.35, 0.85, 0.35], "color": [1.0, 0.96, 0.86], "intensity": 2.2
    }})
}

fn fog_entry(enabled: bool) -> serde_json::Value {
    serde_json::json!({"name": "fog", "type": "VolumetricFog", "args": {
        "enabled": enabled, "density": 0.02
    }})
}

// Global binding indices pinned by `lighting::SECTIONS` declaration order.
const SUN_INTENSITY: usize = 4;
fn fog_enabled_binding() -> usize {
    lighting::section_base(1)
}
fn fog_density_binding() -> usize {
    lighting::section_base(1) + 2
}

// The View panel's Lighting row (registry order: Assets, Preview, Templates,
// Lighting) opens the panel through the real tick + click route and seeds its
// controls from the entries.
#[test]
fn lighting_opens_via_the_view_panel_and_seeds() {
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    let mut h = hook(vec![sun_entry()]);
    h.view_open = true;
    let vp = [1280.0, 720.0];
    let vo = h.origin(PanelKey::View, vp);
    let row = super::super::list_panel::row_rect(vo, 200.0, 3);
    assert!(h.try_panel_press(PanelKey::View, row[0] + 5.0, row[1] + 5.0, vp, &mut world));
    assert!(h.lighting_open, "the Lighting row opens the panel");
    assert_eq!(
        widget::field_text(&world, lighting_panel::input(SUN_INTENSITY)),
        "2.2",
        "the sun intensity control seeds from the entry"
    );
    h.tick(&mut world);
    let bg = world
        .query::<Sprite>()
        .find(|s| s.asset_id == lighting_panel::PANEL_BG)
        .unwrap();
    assert!(bg.visible, "the panel renders once open");
}

// Apply captures the text controls and commits through validation; the entry's
// args update and the live preview rebuild is requested.
#[test]
fn lighting_apply_commits_sun_intensity() {
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let mut h = hook(vec![sun_entry()]);
    h.lighting_open = true;
    h.seed_lighting(&mut world);
    widget::seed_field(&mut world, lighting_panel::input(SUN_INTENSITY), "5.5");
    h.apply_lighting(&mut world);
    assert_eq!(h.lighting_status, None);
    assert!(
        h.dirty && h.rebuild_preview,
        "commit marks the world changed"
    );
    let args = h.entries[0]["args"].as_object().unwrap();
    assert_eq!(args["intensity"].as_f64().unwrap() as f32, 5.5);
    assert_eq!(
        args["direction"][1].as_f64().unwrap() as f32,
        0.85,
        "untouched fields keep their authored values"
    );
}

// Unparseable text falls back to the authored value (the same coerce rule the
// edit form uses), so Apply never corrupts an entry.
#[test]
fn lighting_apply_with_unparseable_text_keeps_the_authored_value() {
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let mut h = hook(vec![sun_entry()]);
    h.lighting_open = true;
    h.seed_lighting(&mut world);
    widget::seed_field(&mut world, lighting_panel::input(SUN_INTENSITY), "garbage");
    h.apply_lighting(&mut world);
    assert_eq!(h.lighting_status, None);
    let args = h.entries[0]["args"].as_object().unwrap();
    assert_eq!(args["intensity"].as_f64().unwrap() as f32, 2.2);
}

// A checkbox toggle commits immediately (live preview on the click) without
// capturing the text controls, so an in-progress typed edit stays pending.
#[test]
fn lighting_bool_toggle_commits_immediately_and_keeps_typed_text() {
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let mut h = hook(vec![fog_entry(false)]);
    h.lighting_open = true;
    h.seed_lighting(&mut world);
    // An in-progress density edit, not yet applied.
    widget::seed_field(
        &mut world,
        lighting_panel::input(fog_density_binding()),
        "0.5",
    );
    h.toggle_lighting_bool(fog_enabled_binding());
    let args = h.entries[0]["args"].as_object().unwrap();
    assert_eq!(args["enabled"], serde_json::Value::Bool(true));
    assert_eq!(
        args["density"].as_f64().unwrap() as f32,
        0.02,
        "the pending text edit is not committed by the toggle"
    );
    assert_eq!(
        widget::field_text(&world, lighting_panel::input(fog_density_binding())),
        "0.5",
        "the typed text stays in its control"
    );
    assert!(h.dirty, "the toggle itself is committed");
}

// The add row appends the missing singleton with default args; its fields then
// derive and seed.
#[test]
fn lighting_add_row_appends_the_missing_singleton() {
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let mut h = hook(vec![sun_entry()]);
    h.lighting_open = true;
    assert_eq!(h.lighting_present(), vec![true, false, false, false]);
    h.add_lighting_section(1, &mut world);
    assert_eq!(h.entries.len(), 2);
    assert_eq!(h.entries[1]["type"], "VolumetricFog");
    assert!(h.dirty);
    assert!(h.lighting_present()[1]);
    // A second add is a no-op (the section binds the existing entry).
    h.add_lighting_section(1, &mut world);
    assert_eq!(h.entries.len(), 2);
}

// Lighting fields assert keyboard focus only while the panel is frontmost, so
// its inputs never fight another panel's focused field.
#[test]
fn lighting_focus_yields_when_not_frontmost() {
    let mut h = hook(vec![sun_entry()]);
    h.lighting_open = true;
    h.lighting_focus = Some(SUN_INTENSITY);
    h.focus_panel(PanelKey::Lighting);
    let d = h.lighting_data();
    assert_eq!(
        h.make_lighting_view(&d, [0.0, 0.0]).focus,
        Some(SUN_INTENSITY)
    );
    h.focus_panel(PanelKey::Assets);
    assert_eq!(h.make_lighting_view(&d, [0.0, 0.0]).focus, None);
}

// Story panel

fn story_import(source: &str) -> serde_json::Value {
    serde_json::json!({"name": "tale", "type": "StoryImport", "args": {"source": source}})
}

// A hook + injected world with the story loaded from `lines` (no file IO: the
// line editor operates purely on the loaded lines until Apply).
fn story_session(lines: &[&str]) -> (EditorHook, World) {
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let mut h = hook(vec![story_import("unused.md")]);
    h.story_open = true;
    h.story_lines = lines.iter().map(|s| s.to_string()).collect();
    h.story_focus = true;
    h.focus_panel(PanelKey::Story);
    h.seed_story_line(&mut world);
    (h, world)
}

fn story_key_input(key: crate::assets::Key) -> FrameInput {
    FrameInput {
        captured_key: Some(key),
        viewport: [1280.0, 720.0],
        ..Default::default()
    }
}

fn line_caret(world: &mut World, caret: usize) {
    let t = widget::input_mut(world, story_panel::LINE_INPUT).unwrap();
    t.caret = caret;
}

// Enter splits the current line at the caret; Backspace at column 0 joins it
// back, blurring the control for one frame so the text system does not also
// eat a character.
#[test]
fn story_enter_splits_and_backspace_joins() {
    let (mut h, mut world) = story_session(&["hello world"]);
    line_caret(&mut world, 5);
    h.story_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(h.story_lines, ["hello", " world"]);
    assert_eq!(h.story_line, 1);
    let input = widget::input(&world, story_panel::LINE_INPUT).unwrap();
    assert_eq!(input.content, " world");
    assert_eq!(input.caret, 0, "the caret starts the new line");

    h.story_keys(&mut world, &story_key_input(crate::assets::Key::Backspace));
    assert_eq!(h.story_lines, ["hello world"]);
    assert_eq!(h.story_line, 0);
    let input = widget::input(&world, story_panel::LINE_INPUT).unwrap();
    assert_eq!(input.caret, 5, "the caret sits at the join point");
    assert!(h.story_blur, "the control blurs for the join frame");
    assert!(
        !h.make_story_view([0.0, 0.0]).focus,
        "the view yields focus that frame"
    );
    // The next key frame clears the blur.
    h.story_keys(&mut world, &story_key_input(crate::assets::Key::Left));
    assert!(!h.story_blur);
}

// Up / Down commit the edited line and move; typed text is never lost.
#[test]
fn story_up_down_commit_and_navigate() {
    let (mut h, mut world) = story_session(&["one", "two", "three"]);
    widget::seed_field(&mut world, story_panel::LINE_INPUT, "ONE edited");
    h.story_keys(&mut world, &story_key_input(crate::assets::Key::Down));
    assert_eq!(h.story_lines[0], "ONE edited", "moving commits the edit");
    assert_eq!(h.story_line, 1);
    let input = widget::input(&world, story_panel::LINE_INPUT).unwrap();
    assert_eq!(input.content, "two");
    h.story_keys(&mut world, &story_key_input(crate::assets::Key::Up));
    assert_eq!(h.story_line, 0);
    // Up at the first line stays put.
    h.story_keys(&mut world, &story_key_input(crate::assets::Key::Up));
    assert_eq!(h.story_line, 0);
}

// Apply validates with the real story parser before writing: a broken story
// shows on the status line and the file is untouched; a valid one writes and
// refreshes the live preview without touching the world.jsonl dirty flag.
#[test]
fn story_apply_validates_then_writes() {
    let dir = std::env::temp_dir().join(format!("cn-story-apply-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tale.md");
    std::fs::write(&path, super::super::story::STARTER_STORY).unwrap();
    let src = path.to_string_lossy().to_string();

    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let mut h = hook(vec![story_import(&src)]);
    h.story_open = true;
    h.load_story(&mut world);
    assert_eq!(h.story_status, None);
    assert_eq!(h.story_path, src);
    assert!(h.story_lines.len() > 5, "the starter story loaded");

    // Break the story (no frontmatter): Apply rejects and writes nothing.
    h.story_lines = vec!["just prose, no frontmatter".to_string()];
    h.story_line = 0;
    h.seed_story_line(&mut world);
    h.apply_story(&mut world);
    assert!(h.story_status.is_some(), "parse failure shown");
    assert!(!h.rebuild_preview);
    let on_disk = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        on_disk,
        super::super::story::STARTER_STORY,
        "file untouched"
    );

    // A valid edit writes and requests the preview rebuild; the world.jsonl
    // dirty flag stays clear (no entry changed).
    h.story_lines = super::super::story::lines_of(super::super::story::STARTER_STORY);
    let last = h.story_lines.len() - 1;
    h.story_line = last;
    h.seed_story_line(&mut world);
    widget::seed_field(&mut world, story_panel::LINE_INPUT, "And they lived on.");
    h.apply_story(&mut world);
    assert_eq!(h.story_status, None);
    assert!(h.rebuild_preview, "preview refresh requested");
    assert!(!h.dirty, "no world.jsonl change");
    let on_disk = std::fs::read_to_string(&path).unwrap();
    assert!(on_disk.contains("And they lived on."));
    assert!(on_disk.ends_with('\n'));
    let _ = std::fs::remove_dir_all(&dir);
}

// A missing source file loads as an empty editable story with the error shown.
#[test]
fn story_load_missing_file_shows_status() {
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let mut h = hook(vec![story_import("/no/such/dir/story.md")]);
    h.story_open = true;
    h.load_story(&mut world);
    assert!(h.story_status.is_some());
    assert_eq!(h.story_lines, [""]);
}

// Create writes the starter file, adds the StoryImport entry (a normal world
// edit), and loads it for editing. Serialized via the cwd lock: the new file
// lands relative to the project root.
#[test]
fn story_create_writes_starter_and_adds_the_import() {
    let _guard = crate::test_support::lock();
    let dir = std::env::temp_dir().join(format!("cn-story-create-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let old = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();

    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let mut h = hook(Vec::new());
    h.story_open = true;
    h.load_story(&mut world);
    assert!(
        h.make_story_view([0.0, 0.0]).create,
        "no import: create mode"
    );
    h.create_story(&mut world);

    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "StoryImport");
    assert_eq!(h.entries[0]["args"]["source"], "story.md");
    assert!(h.dirty, "the new entry is a world edit");
    assert!(h.story_lines.len() > 5, "the starter story is loaded");
    assert!(!h.make_story_view([0.0, 0.0]).create);
    let written = std::fs::read_to_string(dir.join("story.md")).unwrap();
    assert_eq!(written, super::super::story::STARTER_STORY);
    // A second create is a no-op while an import exists.
    h.create_story(&mut world);
    assert_eq!(h.entries.len(), 1);

    std::env::set_current_dir(old).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

// Import panel

fn import_session() -> (EditorHook, World, std::path::PathBuf) {
    let mut world = World::new_empty();
    super::super::inject::editor_hud(&mut world);
    let mut h = hook(Vec::new());
    h.import_open = true;
    h.import_focus = true;
    h.focus_panel(PanelKey::Import);
    let dir = std::env::temp_dir().join(format!(
        "cn-import-{}-{}",
        std::process::id(),
        std::thread::current()
            .name()
            .unwrap_or("t")
            .replace(':', "_")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    (h, world, dir)
}

fn type_path(world: &mut World, path: &str) {
    widget::seed_field(world, import_panel::PATH_INPUT, path);
}

// A scene file resolves through the same dispatch `cn add` uses: one
// SceneImport entry, the world marked changed, and the field cleared.
#[test]
fn import_add_resolves_a_scene_file() {
    let (mut h, mut world, dir) = import_session();
    let glb = dir.join("crate_stack.glb");
    std::fs::write(&glb, b"glb bytes").unwrap();
    type_path(&mut world, &glb.to_string_lossy());
    h.add_import(&mut world);
    assert_eq!(h.import_status, None);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "SceneImport");
    assert_eq!(h.entries[0]["name"], "crate_stack");
    assert_eq!(
        h.entries[0]["args"]["source"],
        serde_json::Value::String(glb.to_string_lossy().to_string())
    );
    assert!(h.dirty && h.rebuild_preview);
    assert_eq!(
        widget::field_text(&world, import_panel::PATH_INPUT),
        "",
        "the path field clears on success"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// A story markdown resolves to a StoryImport; a colliding name is uniquified
// instead of erroring.
#[test]
fn import_add_uniquifies_a_colliding_name() {
    let (mut h, mut world, dir) = import_session();
    let md = dir.join("tale.md");
    std::fs::write(&md, super::super::story::STARTER_STORY).unwrap();
    h.entries.push(entry("tale", "PointLight"));
    type_path(&mut world, &md.to_string_lossy());
    h.add_import(&mut world);
    assert_eq!(h.import_status, None);
    assert_eq!(h.entries.len(), 2);
    assert_eq!(h.entries[1]["type"], "StoryImport");
    assert_eq!(h.entries[1]["name"], "tale_1", "renamed past the collision");
    let _ = std::fs::remove_dir_all(&dir);
}

// Failures land on the status line and commit nothing: a missing file, and an
// unknown extension.
#[test]
fn import_add_rejects_missing_files_and_unknown_extensions() {
    let (mut h, mut world, dir) = import_session();
    type_path(&mut world, "/no/such/thing.glb");
    h.add_import(&mut world);
    assert!(
        h.import_status
            .as_ref()
            .unwrap()
            .text()
            .contains("no such file")
    );
    assert!(h.entries.is_empty() && !h.dirty);

    let odd = dir.join("mystery.xyz");
    std::fs::write(&odd, b"?").unwrap();
    type_path(&mut world, &odd.to_string_lossy());
    h.add_import(&mut world);
    assert!(h.import_status.is_some(), "unknown extension rejected");
    assert!(h.entries.is_empty() && !h.dirty);
    let _ = std::fs::remove_dir_all(&dir);
}

// A `.hdr` into a world with no lighting environment adds one.
#[test]
fn import_add_resolves_an_hdr_to_an_environment_map() {
    let (mut h, mut world, dir) = import_session();
    let hdr = dir.join("studio.hdr");
    std::fs::write(&hdr, b"radiance").unwrap();
    type_path(&mut world, &hdr.to_string_lossy());
    h.add_import(&mut world);
    assert_eq!(h.import_status, None);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "EnvironmentMap");
    assert_eq!(h.entries[0]["name"], "studio");
    assert_eq!(
        h.entries[0]["args"]["source"],
        serde_json::Value::String(hdr.to_string_lossy().to_string())
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// A second `.hdr` retargets the world's existing map instead of appending one
// the runtime would ignore, and says so on the status line.
#[test]
fn import_add_retargets_an_existing_environment_map() {
    let (mut h, mut world, dir) = import_session();
    h.entries.push(serde_json::json!({
        "name": "env", "type": "EnvironmentMap", "args": {"source": "", "generator": "sky"}
    }));
    let hdr = dir.join("dusk.hdr");
    std::fs::write(&hdr, b"radiance").unwrap();
    type_path(&mut world, &hdr.to_string_lossy());
    h.add_import(&mut world);

    assert_eq!(h.entries.len(), 1, "no second map appended");
    assert_eq!(h.entries[0]["name"], "env", "the existing map is reused");
    assert_eq!(
        h.entries[0]["args"]["source"],
        serde_json::Value::String(hdr.to_string_lossy().to_string())
    );
    assert_eq!(h.entries[0]["args"]["generator"], "");
    assert!(h.dirty && h.rebuild_preview);
    // Reported as a notice, not an error: the Add succeeded.
    assert!(matches!(
        h.import_status,
        Some(import_panel::ImportStatus::Notice(_))
    ));
    assert_eq!(
        widget::field_text(&world, import_panel::PATH_INPUT),
        "",
        "the path field clears on success"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// Enter in the focused path field adds, like clicking the Add button.
#[test]
fn import_enter_key_adds() {
    let (mut h, mut world, dir) = import_session();
    let glb = dir.join("prop.glb");
    std::fs::write(&glb, b"glb").unwrap();
    type_path(&mut world, &glb.to_string_lossy());
    h.import_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(h.entries.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

// The list shows the world's file-backed entries and opens one in the
// standard edit form (alongside the Assets browse panel).
#[test]
fn import_rows_list_and_open_in_the_edit_form() {
    let (mut h, mut world, dir) = import_session();
    h.entries = vec![
        entry("lamp", "PointLight"),
        serde_json::json!({"name": "town", "type": "SceneImport", "args": {"source": "town.glb"}}),
        serde_json::json!({"name": "face", "type": "Font", "args": {"path": "face.ttf"}}),
        serde_json::json!({"name": "env", "type": "EnvironmentMap", "args": {"source": "sky.hdr"}}),
    ];
    let rows = h.import_rows();
    assert_eq!(rows.len(), 3, "only file-backed types list");
    assert_eq!(rows[0].entry, 1);
    assert_eq!(rows[0].caption, "town  (SceneImport)  town.glb");
    assert_eq!(rows[1].caption, "face  (Font)  face.ttf");
    assert_eq!(rows[2].caption, "env  (EnvironmentMap)  sky.hdr");

    h.apply_import_action(ImportAction::Open(0), &mut world);
    assert!(h.panel_open, "the Assets UI comes up with the form");
    assert!(h.form_open());
    assert_eq!(
        h.form_target,
        FormTarget::Entry(1),
        "the clicked entry is being edited"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// A browsed file lands in the path field as a project-relative path, ready for
// the user to confirm with Add (Browse never commits on its own). The dialog
// itself is not exercised: `browse_import` is a three-line wrapper over it, and
// everything the pick feeds runs through `accept_browsed_path`.
#[test]
fn import_browse_result_fills_the_path_field_relatively() {
    let _guard = crate::test_support::lock();
    let (mut h, mut world, dir) = import_session();
    let old = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();

    let assets = dir.join("assets");
    std::fs::create_dir_all(&assets).unwrap();
    let picked = assets.join("hero.glb");
    std::fs::write(&picked, b"glb").unwrap();
    h.import_status = Some(import_panel::ImportStatus::Error("stale error".to_string()));
    h.accept_browsed_path(&mut world, &picked);

    assert_eq!(
        widget::field_text(&world, import_panel::PATH_INPUT),
        "assets/hero.glb",
        "a file inside the project stores relatively"
    );
    assert!(h.import_focus, "the field takes focus, ready to Add");
    assert_eq!(h.import_status, None, "a stale error is cleared");
    assert!(h.entries.is_empty(), "Browse does not commit on its own");

    // Confirming with Add resolves the browsed path like any typed one.
    h.add_import(&mut world);
    assert_eq!(h.import_status, None);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "SceneImport");
    assert_eq!(h.entries[0]["args"]["source"], "assets/hero.glb");

    std::env::set_current_dir(old).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

// The Assets panel's origin-grouped tree

// A world whose GraphicsConfig pulls in companions, so the expansion has
// something to show without needing a scene file on disk.
fn expandable_hook() -> EditorHook {
    isolate_state_dir();
    let mut h = hook(vec![serde_json::json!({
        "name": "gfx", "type": "GraphicsConfig", "args": {}
    })]);
    h.panel_open = true;
    h
}

// The first asset the build generates that an authored line could override,
// as (group, index, name).
fn a_promotable_asset(h: &EditorHook) -> (usize, usize, String) {
    h.tree_groups
        .iter()
        .enumerate()
        .find_map(|(gi, g)| {
            g.assets
                .iter()
                .enumerate()
                .find_map(|(ai, a)| a.promote.as_ref().map(|_| (gi, ai, a.name.clone())))
        })
        .expect("an injected default can be promoted")
}

// The model is only cooked when the panel is actually showing: a closed panel
// must not pay for an expansion it never draws.
#[test]
fn the_tree_is_cooked_only_while_the_panel_is_up() {
    let mut h = expandable_hook();
    h.panel_open = false;
    assert!(h.tree_stale);
    h.refresh_tree_if_needed();
    assert!(
        h.tree_stale && h.tree_groups.is_empty(),
        "a hidden panel does not cook"
    );

    h.panel_open = true;
    h.refresh_tree_if_needed();
    assert!(!h.tree_stale);
    assert!(
        !h.tree_groups.is_empty(),
        "the injected companions are grouped"
    );
    assert_eq!(h.tree_status, None);
}

// An edit invalidates the model, and the refresh happens once for a burst of
// edits rather than once per edit.
#[test]
fn an_edit_restales_the_tree() {
    let mut h = expandable_hook();
    h.refresh_tree_if_needed();
    assert!(!h.tree_stale);
    h.mark_changed();
    assert!(h.tree_stale, "the expansion follows the entries");
    h.mark_changed();
    h.refresh_tree_if_needed();
    assert!(!h.tree_stale, "one refresh covers the burst");
}

#[test]
fn groups_fold_and_unfold() {
    let mut h = expandable_hook();
    let world = world_with_fields();
    h.refresh_tree_if_needed();
    let folded = h.tree_rows(&world).len();
    assert_eq!(folded, h.tree_groups.len(), "headers only");

    h.apply_panel(PanelAction::ToggleGroup(0), &mut world_with_fields());
    assert!(h.tree_rows(&world).len() > folded, "the group unfolded");
    h.apply_panel(PanelAction::ToggleGroup(0), &mut world_with_fields());
    assert_eq!(h.tree_rows(&world).len(), folded, "and folded again");
}

// The heart of the merged panel: a generated asset has no world.jsonl line, so
// clicking it opens a form seeded from what the expansion produced, and ONLY
// confirming appends that line -- keeping the generated name, which is what
// makes the new line override the expansion.
#[test]
fn editing_a_generated_asset_promotes_it_on_confirm() {
    let mut h = expandable_hook();
    let mut world = world_with_fields();
    h.refresh_tree_if_needed();
    let (gi, ai, name) = a_promotable_asset(&h);
    let before = h.entries.len();

    h.apply_panel(PanelAction::SelectRow(gi, ai), &mut world);
    assert!(h.form_open(), "a generated asset still opens the form");
    assert!(
        matches!(h.form_target, FormTarget::Promote(_)),
        "seeded from the expansion, not from an entry"
    );
    assert_eq!(
        widget::field_text(&world, form_panel::NAME_INPUT),
        name,
        "the name heading carries the generated name"
    );
    assert_eq!(
        h.entries.len(),
        before,
        "opening the form alone adds no line"
    );
    assert!(h.selection.contains(&name), "and the row selects");

    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), before + 1, "confirming appends the line");
    let added = h.entries.last().unwrap();
    assert_eq!(
        entry_name(added),
        Some(name.as_str()),
        "the promoted line keeps the generated name, so it shadows it"
    );
    assert!(h.dirty && h.tree_stale);
}

// After promoting, the asset relists once -- as an authored line under the
// world group -- rather than staying under its origin or appearing twice.
#[test]
fn a_promoted_asset_relists_as_authored_under_the_world_group() {
    let mut h = expandable_hook();
    let mut world = world_with_fields();
    h.refresh_tree_if_needed();
    let (gi, ai, name) = a_promotable_asset(&h);
    h.apply_panel(PanelAction::SelectRow(gi, ai), &mut world);
    h.apply_form(FormAction::Confirm, &mut world);
    h.refresh_tree_if_needed();

    let listings: Vec<&str> = h
        .tree_groups
        .iter()
        .filter(|g| g.assets.iter().any(|a| a.name == name))
        .map(|g| g.label.as_str())
        .collect();
    assert_eq!(
        listings,
        [asset_tree::WORLD_GROUP],
        "listed once, under World"
    );
    let promoted = h
        .tree_groups
        .iter()
        .flat_map(|g| &g.assets)
        .find(|a| a.name == name)
        .unwrap();
    assert!(promoted.promote.is_none(), "nothing left to promote");

    // Clicking it again now edits the line in place rather than re-promoting.
    let (g2, i2) = row_of(&h, &name);
    let before = h.entries.len();
    h.apply_panel(PanelAction::SelectRow(g2, i2), &mut world);
    assert!(matches!(h.form_target, FormTarget::Entry(_)));
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), before, "edited in place, not appended");
}

// The passes that emit unconditionally cannot be overridden by a copy, so their
// rows select but open no form -- and a form already open on something else
// closes rather than staying pointed at the previous asset.
#[test]
fn an_unconditional_expansion_selects_but_does_not_edit() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = world_with_fields();
    h.panel_open = true;
    seed_tree(
        &mut h,
        vec![TreeGroup {
            label: asset_tree::UNATTRIBUTED.to_string(),
            assets: vec![asset_tree::TreeAsset {
                name: "menu_tab_0".to_string(),
                asset_type: "TextLabel".to_string(),
                badge: asset_tree::Badge::Imported,
                promote: None,
            }],
        }],
    );
    click_row(&mut h, "lamp", &mut world);
    assert!(h.form_open(), "the authored line opens its form");

    click_row(&mut h, "menu_tab_0", &mut world);
    assert!(!h.form_open(), "a fixed expansion has nothing to edit");
    assert!(
        h.selection.contains("menu_tab_0"),
        "but it still selects in the viewport"
    );
    assert!(h.entries.len() == 1, "and nothing was appended");
}

// A world that does not cook reports why instead of showing an empty tree.
#[test]
fn a_broken_world_reports_its_error_in_the_status_line() {
    isolate_state_dir();
    let mut h = hook(vec![serde_json::json!({
        "name": "oops", "type": "NotARealAssetType", "args": {}
    })]);
    h.panel_open = true;
    h.refresh_tree_if_needed();
    assert!(h.tree_groups.is_empty());
    let status = h.tree_status.as_deref().expect("the failure surfaces");
    assert!(status.contains("NotARealAssetType"), "{status}");
}

// A committed edit becomes one undo step: undo restores the pre-edit list (and
// clears dirty when that list matches the on-disk state), redo replays it.
#[test]
fn undo_reverts_a_committed_edit_and_redo_replays_it() {
    let mut world = World::new_empty();
    let mut h = hook(vec![entry("a", "Sprite")]);
    assert!(!h.hud_state().undo && !h.hud_state().redo);

    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();
    assert!(h.dirty && h.hud_state().undo);

    h.undo(&mut world);
    assert_eq!(h.entries, vec![entry("a", "Sprite")]);
    assert!(!h.dirty, "back at the on-disk list: Save chip clears");
    assert!(
        h.rebuild_preview,
        "the restored list drives the live preview"
    );
    assert!(h.hud_state().redo);

    h.redo(&mut world);
    assert_eq!(h.entries, vec![entry("a", "Sprite"), entry("b", "Sprite")]);
    assert!(h.dirty, "the replayed edit is unsaved again");
}

// Editing from an undone state forks the timeline: the redo branch is gone.
#[test]
fn an_edit_after_undo_drops_the_redo_branch() {
    let mut world = World::new_empty();
    let mut h = hook(Vec::new());
    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();
    h.undo(&mut world);
    assert!(h.hud_state().redo);

    h.entries.push(entry("c", "Sprite"));
    h.mark_changed();
    assert!(!h.hud_state().redo, "the new edit invalidates redo");
    h.undo(&mut world);
    assert!(h.entries.is_empty());
}

// A mark_changed that changed nothing (e.g. an Apply that staged no edits)
// records no phantom undo step.
#[test]
fn a_no_change_mark_records_no_undo_step() {
    let mut h = hook(vec![entry("a", "Sprite")]);
    h.mark_changed();
    assert!(!h.hud_state().undo, "nothing changed, nothing to undo");
}

// The open form and row menu index into `entries`; a history jump drops them so
// they can never point at a removed or shifted row.
#[test]
fn undo_drops_entry_indexed_ui_state() {
    let mut world = World::new_empty();
    let mut h = hook(vec![entry("a", "Sprite")]);
    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();
    h.selected_type = Some("Sprite".to_string());
    h.form_target = FormTarget::Entry(1);
    h.row_menu = Some("b".to_string());

    h.undo(&mut world);
    assert_eq!(
        h.form_target,
        FormTarget::New,
        "the form no longer targets a live row"
    );
    assert_eq!(h.selected_type, None);
    assert_eq!(h.row_menu, None);
}

// Ctrl+Z / Ctrl+Y drive the history from the tick, but stand down while a text
// field owns the keyboard or the world holds the cursor (play mode).
#[test]
fn ctrl_z_y_step_history_unless_typing_or_playing() {
    use crate::assets::Key;
    let step = |h: &mut EditorHook, key: Key| {
        let mut world = world_with_input(FrameInput {
            viewport: [1280.0, 720.0],
            ctrl: true,
            captured_key: Some(key),
            ..Default::default()
        });
        h.tick(&mut world);
    };
    let mut h = hook(Vec::new());
    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();

    // Typing in the Story panel: the shortcut must not fire.
    h.story_focus = true;
    step(&mut h, Key::Z);
    assert_eq!(
        h.entries.len(),
        1,
        "suppressed while a text field is focused"
    );
    h.story_focus = false;

    // Play mode: the world owns the keyboard.
    h.world_capture = true;
    step(&mut h, Key::Z);
    assert_eq!(h.entries.len(), 1, "suppressed in play mode");
    h.world_capture = false;

    step(&mut h, Key::Z);
    assert!(h.entries.is_empty(), "Ctrl+Z undoes the edit");
    step(&mut h, Key::Y);
    assert_eq!(h.entries.len(), 1, "Ctrl+Y redoes it");
}

// A successful SAVE re-baselines dirty tracking: undoing past it re-dirties,
// redoing back to the saved list cleans the chip again.
#[test]
fn dirty_tracks_the_saved_list_across_history_jumps() {
    let mut world = World::new_empty();
    let mut h = hook(Vec::new());
    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();
    // Stand in for a successful SAVE (persist() would hit disk + the cook).
    h.dirty = false;
    h.saved = h.entries.clone();

    h.undo(&mut world);
    assert!(h.dirty, "behind the saved list is an unsaved state");
    h.redo(&mut world);
    assert!(!h.dirty, "redo back to the saved list clears the chip");
}

// Viewport picking test rig: a camera at `cam_pos` facing -Z (yaw 0, pitch 0),
// the injected typed fields (the pick flows open the edit form), and a
// PickIndex resource carrying the given (id, bb_min, bb_max) entries.
fn pick_world(
    cam_pos: [f32; 3],
    picks: Vec<(crate::ecs::asset_id::AssetId, [f32; 3], [f32; 3])>,
) -> World {
    let mut world = world_with_fields();
    world.add_component(crate::assets::Camera3D {
        position: cam_pos,
        view_matrix: concinnity_core::gfx::camera::view_matrix(cam_pos, 0.0, 0.0),
        fov_y_degrees: 90.0,
        near: 0.05,
        far: 200.0,
        yaw: 0.0,
        pitch: 0.0,
        desired_move: [0.0; 3],
        jump_requested: false,
        interact_requested: false,
        controller: None,
    });
    for s in highlight::outline_sprites() {
        world.add_component(s);
    }
    world.add_component(super::super::marquee::rect_sprite());
    world.insert_resource(crate::ecs::PickIndex {
        entries: picks
            .into_iter()
            .map(|(asset_id, bb_min, bb_max)| crate::ecs::PickEntry {
                asset_id,
                bb_min,
                bb_max,
            })
            .collect(),
    });
    world
}

fn click_at(world: &mut World, h: &mut EditorHook, pos: [f32; 2]) {
    click_at_mod(world, h, pos, false);
}

fn click_at_mod(world: &mut World, h: &mut EditorHook, pos: [f32; 2], shift: bool) {
    set_input(
        world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: pos[0],
            mouse_y: pos[1],
            left_click: true,
            left_button_down: true,
            shift,
            ..Default::default()
        },
    );
    h.tick(world);
}

// A button-up tick at `pos`: ends an armed marquee (or gizmo drag).
fn release_at(world: &mut World, h: &mut EditorHook, pos: [f32; 2]) {
    set_input(
        world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: pos[0],
            mouse_y: pos[1],
            ..Default::default()
        },
    );
    h.tick(world);
}

// A held-button move tick at `pos`: advances an armed marquee or gizmo drag.
fn drag_to(world: &mut World, h: &mut EditorHook, pos: [f32; 2]) {
    set_input(
        world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: pos[0],
            mouse_y: pos[1],
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(world);
}

// A click through the scene picks the nearest hit, brings up the assets UI,
// and opens the edit form on the picked authored entry.
#[test]
fn viewport_click_picks_the_nearest_prop_and_opens_its_form() {
    crate::ecs::asset_id::reset_interner();
    let near = crate::ecs::asset_id::intern("box_near");
    let far = crate::ecs::asset_id::intern("box_far");
    let mut world = pick_world(
        [0.0; 3],
        vec![
            // Both boxes straddle the -Z axis; the near one at z ~ -5.
            (far, [-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]),
            (near, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0]),
        ],
    );
    let mut h = hook(vec![
        entry("box_near", "Sprite"),
        entry("box_far", "Sprite"),
    ]);

    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("box_near"), "nearest hit wins");
    assert!(h.panel_open, "the assets UI comes up around the form");
    assert_eq!(
        h.form_target,
        FormTarget::Entry(0),
        "the form targets the picked entry"
    );
    assert_eq!(h.selected_type.as_deref(), Some("Sprite"));
}

// A second click on the same spot cycles to the occluded hit; a click away
// from any box clears the selection and the cycle. The boxes sit down-left of
// the camera axis so the clicks land in the screen region no default panel
// covers (the first pick opens the edit form, which claims center presses).
#[test]
fn repeat_viewport_clicks_cycle_and_empty_space_clears() {
    crate::ecs::asset_id::reset_interner();
    let near = crate::ecs::asset_id::intern("box_near");
    let far = crate::ecs::asset_id::intern("box_far");
    // The ray through pixel [200, 600] (fov 90, 1280x720) passes ~[-6.1, -3.3]
    // at depth 5 and ~[-12.2, -6.7] at depth 10; both boxes straddle it.
    let mut world = pick_world(
        [0.0; 3],
        vec![
            (near, [-7.1, -4.3, -6.0], [-5.1, -2.3, -4.0]),
            (far, [-13.2, -7.7, -11.0], [-11.2, -5.7, -9.0]),
        ],
    );
    let mut h = hook(vec![
        entry("box_near", "Sprite"),
        entry("box_far", "Sprite"),
    ]);

    click_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(h.selection.active(), Some("box_near"));
    click_at(&mut world, &mut h, [201.0, 601.0]);
    assert_eq!(
        h.selection.active(),
        Some("box_far"),
        "a repeat click reaches the occluded box"
    );
    assert_eq!(
        h.selection.iter().count(),
        1,
        "a plain click replaces, never accumulates"
    );
    click_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(
        h.selection.active(),
        Some("box_near"),
        "the cycle wraps back to the front"
    );

    // Aim into the gutter between the Preview panel and the edit form, well
    // away from both boxes: the press arms a marquee, and the still release
    // clears.
    click_at(&mut world, &mut h, [250.0, 450.0]);
    assert_eq!(
        h.selection.active(),
        Some("box_near"),
        "the selection survives until the release decides click vs marquee"
    );
    release_at(&mut world, &mut h, [250.0, 450.0]);
    assert_eq!(
        h.selection.active(),
        None,
        "empty space clears the selection"
    );
}

// Picking an asset the world does not declare opens no form until the tree
// knows how (if at all) it can be promoted: an unknown name is selectable only.
#[test]
fn viewport_click_on_an_unknown_asset_selects_without_a_form() {
    isolate_state_dir();
    crate::ecs::asset_id::reset_interner();
    let generated = crate::ecs::asset_id::intern("some_generated_asset");
    let mut world = pick_world(
        [0.0; 3],
        vec![(generated, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0])],
    );
    let mut h = hook(vec![entry("box_near", "Sprite")]);

    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("some_generated_asset"));
    assert!(h.panel_open, "the assets UI still comes up");
    assert_eq!(
        h.form_target,
        FormTarget::New,
        "nothing in the tree to seed a form from"
    );
    assert!(!h.form_open());
}

// Undo/redo invalidates the pick state along with the other entry-indexed UI.
#[test]
fn history_jumps_clear_the_pick_selection() {
    crate::ecs::asset_id::reset_interner();
    let id = crate::ecs::asset_id::intern("box_near");
    let mut world = pick_world([0.0; 3], vec![(id, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0])]);
    let mut h = hook(vec![entry("box_near", "Sprite")]);
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("box_near"));

    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();
    h.undo(&mut world);
    assert_eq!(
        h.selection.active(),
        None,
        "a history jump drops the selection"
    );
}

// The selection ring follows the picked asset's projected bounds and hides
// outside edit mode.
#[test]
fn selection_ring_tracks_the_picked_asset() {
    crate::ecs::asset_id::reset_interner();
    let id = crate::ecs::asset_id::intern("box_near");
    let mut world = pick_world([0.0; 3], vec![(id, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0])]);
    let mut h = hook(vec![entry("box_near", "Sprite")]);
    click_at(&mut world, &mut h, [640.0, 360.0]);

    let ring = |world: &World| {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == highlight::all_sprite_ids()[0])
            .cloned()
            .expect("outline sprite injected")
    };
    let s = ring(&world);
    assert!(s.visible, "the ring shows on pick");
    let (cx, cy) = (s.x + s.width * 0.5, s.y + s.height * 0.5);
    assert!(
        (cx - 640.0).abs() < 2.0 && (cy - 360.0).abs() < 2.0,
        "ring centered on the box, got ({cx}, {cy})"
    );
    assert!(s.border_width > 0.0 && s.tint[3] == 0.0, "border-only ring");

    // Play mode hides the ring; returning to edit mode restores it.
    h.world_capture = true;
    set_input(
        &mut world,
        FrameInput {
            viewport: [1280.0, 720.0],
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(!ring(&world).visible, "hidden in play mode");
    h.world_capture = false;
    h.tick(&mut world);
    assert!(ring(&world).visible, "back in edit mode it returns");
}

// The full translate-gizmo loop: pick a prop, grab its X tip handle, drag
// right, release. The live Transform follows during the drag; release commits
// the moved position to the authored entry as ONE undo step, and Ctrl-class
// undo restores the original position.
#[test]
fn gizmo_drag_moves_the_prop_and_commits_one_undo_step() {
    crate::ecs::asset_id::reset_interner();
    let id = crate::ecs::asset_id::intern("box_near");
    // Down-left of the camera axis so the pick, the handles, and the drag all
    // land in screen regions no default panel covers.
    let start = [-6.11f32, -3.3, -5.0];
    let mut world = pick_world(
        [0.0; 3],
        vec![(
            id,
            [start[0] - 1.0, start[1] - 1.0, start[2] - 1.0],
            [start[0] + 1.0, start[1] + 1.0, start[2] + 1.0],
        )],
    );
    let entity = world.push(crate::assets::Transform {
        position: start,
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    });
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(id, entity);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    for s in super::super::gizmo::sprites() {
        world.add_component(s);
    }

    let mut h = hook(vec![serde_json::json!({
        "name": "box_near", "type": "Prop", "args": { "position": start }
    })]);

    // Pick the prop (projects to ~[200, 600] for this camera).
    click_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(h.selection.active(), Some("box_near"));
    let layout = h
        .gizmo_layout(&world, [1280.0, 720.0])
        .expect("movable selection shows the gizmo");

    // Press the X tip handle: a drag starts, nothing re-picks.
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some(), "the tip press starts a drag");
    assert!(!h.dirty, "no entry change until release");

    // Drag 50 px right: the live Transform follows along world X.
    set_input(
        &mut world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: layout.tips[0][0] + 50.0,
            mouse_y: layout.tips[0][1],
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    let live = world
        .get::<crate::assets::Transform>(entity)
        .expect("entity alive")
        .position;
    assert!(live[0] > start[0] + 0.3, "moved right: {}", live[0]);
    assert!((live[1] - start[1]).abs() < 1e-3, "Y untouched");
    assert!((live[2] - start[2]).abs() < 1e-3, "Z untouched");

    // Release: the entry commits, one undo step, form refreshed.
    set_input(
        &mut world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: layout.tips[0][0] + 50.0,
            mouse_y: layout.tips[0][1],
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(h.gizmo_drag.is_none(), "release ends the drag");
    assert!(h.dirty, "the move is an unsaved edit");
    let committed = h.entries[0]["args"]["position"][0].as_f64().unwrap();
    assert!(
        committed > f64::from(start[0]) + 0.3,
        "entry follows: {committed}"
    );

    // One undo step restores the pre-drag position.
    h.undo(&mut world);
    let restored = h.entries[0]["args"]["position"][0].as_f64().unwrap();
    assert!((restored - f64::from(start[0])).abs() < 1e-3, "{restored}");
    assert!(!h.can_undo(), "the whole drag was one step");
}

// Shared rig for the rotate / scale drag tests: a prop down-left of the
// camera axis (panel-free screen region), its live Transform entity, and the
// EntityByName map the gizmo resolves through.
fn gizmo_rig(start: [f32; 3]) -> (World, crate::ecs::Entity, EditorHook) {
    crate::ecs::asset_id::reset_interner();
    let id = crate::ecs::asset_id::intern("box_near");
    let mut world = pick_world(
        [0.0; 3],
        vec![(
            id,
            [start[0] - 1.0, start[1] - 1.0, start[2] - 1.0],
            [start[0] + 1.0, start[1] + 1.0, start[2] + 1.0],
        )],
    );
    let entity = world.push(crate::assets::Transform {
        position: start,
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    });
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(id, entity);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    for s in super::super::gizmo::sprites() {
        world.add_component(s);
    }
    let h = hook(vec![serde_json::json!({
        "name": "box_near", "type": "Prop", "args": { "position": start }
    })]);
    (world, entity, h)
}

fn drag_input(pos: [f32; 2], held: bool) -> FrameInput {
    FrameInput {
        viewport: [1280.0, 720.0],
        mouse_x: pos[0],
        mouse_y: pos[1],
        left_button_down: held,
        ..Default::default()
    }
}

// Rotate mode: turning the mouse a quarter circle around the origin rotates
// the prop 90 degrees about the grabbed axis, committed as one undo step.
#[test]
fn gizmo_rotate_drag_turns_the_prop() {
    let start = [-6.11f32, -3.3, -5.0];
    let (mut world, entity, mut h) = gizmo_rig(start);
    h.gizmo_mode = gizmo::GizmoMode::Rotate;

    click_at(&mut world, &mut h, [200.0, 600.0]);
    let layout = h.gizmo_layout(&world, [1280.0, 720.0]).expect("gizmo up");
    // Grab the X tip (70 px right of the origin: screen angle 0)...
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some(), "rotate grab starts on the tip");
    // ...and swing the cursor to straight below the origin: +90 degrees of
    // screen angle. World X is perpendicular to the view (sign +1).
    set_input(
        &mut world,
        drag_input([layout.origin[0], layout.origin[1] + 70.0], true),
    );
    h.tick(&mut world);
    let live = world.get::<crate::assets::Transform>(entity).unwrap();
    assert!(
        (live.rotation_deg[0] - 90.0).abs() < 1.0,
        "quarter turn about X: {}",
        live.rotation_deg[0]
    );
    assert_eq!(live.position, start, "rotate leaves position alone");

    // Release commits rotation_deg; undo removes it again.
    set_input(
        &mut world,
        drag_input([layout.origin[0], layout.origin[1] + 70.0], false),
    );
    h.tick(&mut world);
    let committed = h.entries[0]["args"]["rotation_deg"][0].as_f64().unwrap();
    assert!((committed - 90.0).abs() < 1.0, "{committed}");
    h.undo(&mut world);
    assert!(
        h.entries[0]["args"].get("rotation_deg").is_none(),
        "one undo step restores the pre-drag entry"
    );
}

// Scale mode: dragging the X tip half its run further out scales X by ~1.5,
// leaving the other axes untouched.
#[test]
fn gizmo_scale_drag_stretches_one_axis() {
    let start = [-6.11f32, -3.3, -5.0];
    let (mut world, entity, mut h) = gizmo_rig(start);
    h.gizmo_mode = gizmo::GizmoMode::Scale;

    click_at(&mut world, &mut h, [200.0, 600.0]);
    let layout = h.gizmo_layout(&world, [1280.0, 720.0]).expect("gizmo up");
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some());
    set_input(
        &mut world,
        drag_input([layout.tips[0][0] + 35.0, layout.tips[0][1]], true),
    );
    h.tick(&mut world);
    let live = world.get::<crate::assets::Transform>(entity).unwrap();
    assert!(
        live.scale[0] > 1.3 && live.scale[0] < 1.7,
        "X stretched ~1.5x: {}",
        live.scale[0]
    );
    assert_eq!(live.scale[1], 1.0, "Y untouched");
    assert_eq!(live.position, start, "scale leaves position alone");

    set_input(
        &mut world,
        drag_input([layout.tips[0][0] + 35.0, layout.tips[0][1]], false),
    );
    h.tick(&mut world);
    let committed = h.entries[0]["args"]["scale"][0].as_f64().unwrap();
    assert!(committed > 1.3 && committed < 1.7, "{committed}");
    h.undo(&mut world);
    assert!(h.entries[0]["args"].get("scale").is_none());
}

// Two props with live Transforms at `s1` / `s2` (AABB half-extent `half`),
// wired like `gizmo_rig`, for the multi-select flows.
fn two_prop_rig(
    s1: [f32; 3],
    s2: [f32; 3],
    half: f32,
) -> (World, crate::ecs::Entity, crate::ecs::Entity, EditorHook) {
    crate::ecs::asset_id::reset_interner();
    let a = crate::ecs::asset_id::intern("box_a");
    let b = crate::ecs::asset_id::intern("box_b");
    let bb = |s: [f32; 3]| {
        (
            [s[0] - half, s[1] - half, s[2] - half],
            [s[0] + half, s[1] + half, s[2] + half],
        )
    };
    let (min1, max1) = bb(s1);
    let (min2, max2) = bb(s2);
    let mut world = pick_world([0.0; 3], vec![(a, min1, max1), (b, min2, max2)]);
    let transform = |p: [f32; 3]| crate::assets::Transform {
        position: p,
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    };
    let e1 = world.push(transform(s1));
    let e2 = world.push(transform(s2));
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(a, e1);
    by_name.insert(b, e2);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    for s in super::super::gizmo::sprites() {
        world.add_component(s);
    }
    let entry = |name: &str, p: [f32; 3]| serde_json::json!({ "name": name, "type": "Prop", "args": { "position": p } });
    let h = hook(vec![entry("box_a", s1), entry("box_b", s2)]);
    (world, e1, e2, h)
}

// Side-by-side props with a clear gap (so neither center ray clips the other
// box): box_a projects around [200, 600], box_b around [424, 598], both in
// the panel-free lower-left screen region.
const SIDE_A: [f32; 3] = [-6.11, -3.3, -5.0];
const SIDE_B: [f32; 3] = [-3.0, -3.3, -5.0];

// Shift-click toggles membership without disturbing the rest of the set.
#[test]
fn shift_click_toggles_selection_membership() {
    let (mut world, _, _, mut h) = two_prop_rig(SIDE_A, SIDE_B, 1.0);

    click_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(h.selection.active(), Some("box_a"));
    click_at_mod(&mut world, &mut h, [424.0, 598.0], true);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box_a", "box_b"],
        "shift-click adds the second box"
    );
    assert_eq!(h.selection.active(), Some("box_b"), "the newest is active");
    assert_eq!(
        h.form_target,
        FormTarget::Entry(1),
        "the form follows the active member"
    );

    click_at_mod(&mut world, &mut h, [424.0, 598.0], true);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box_a"],
        "a second shift-click removes it again"
    );
}

// A drag from empty space boxes both props; shift-drag adds instead of
// replacing; a sub-slop release is the plain clearing click.
#[test]
fn marquee_drag_selects_the_boxed_assets() {
    let (mut world, _, _, mut h) = two_prop_rig(SIDE_A, SIDE_B, 1.0);

    // Press empty space (the ray misses both boxes), drag across both
    // projections, release.
    click_at(&mut world, &mut h, [80.0, 450.0]);
    assert!(h.marquee.is_some(), "an empty-space press arms the marquee");
    drag_to(&mut world, &mut h, [560.0, 700.0]);
    let rect = world
        .query::<Sprite>()
        .find(|s| s.asset_id == super::super::marquee::RECT)
        .cloned()
        .expect("marquee sprite injected");
    assert!(rect.visible, "the rect shows once the drag clears the slop");
    assert_eq!(
        (rect.x, rect.y, rect.width, rect.height),
        (80.0, 450.0, 480.0, 250.0)
    );
    release_at(&mut world, &mut h, [560.0, 700.0]);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box_a", "box_b"],
        "both boxed props are selected"
    );
    assert!(h.marquee.is_none(), "release ends the marquee");
    assert!(
        !world
            .query::<Sprite>()
            .find(|s| s.asset_id == super::super::marquee::RECT)
            .unwrap()
            .visible,
        "the rect hides after release"
    );

    // A fresh plain click on box_a, then a shift-drag over box_b only: added.
    click_at(&mut world, &mut h, [200.0, 600.0]);
    release_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(h.selection.iter().collect::<Vec<_>>(), ["box_a"]);
    // Starts in the Preview / edit-form gutter, right of box_a's projection so
    // the box encloses box_b alone.
    click_at_mod(&mut world, &mut h, [250.0, 450.0], true);
    assert!(h.marquee.is_some());
    drag_to(&mut world, &mut h, [560.0, 700.0]);
    release_at(&mut world, &mut h, [560.0, 700.0]);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box_a", "box_b"],
        "shift-drag adds without replacing"
    );

    // A still empty-space click clears; a still shift-click does not.
    click_at_mod(&mut world, &mut h, [80.0, 450.0], true);
    release_at(&mut world, &mut h, [81.0, 450.0]);
    assert_eq!(h.selection.iter().count(), 2, "shift keeps the selection");
    click_at(&mut world, &mut h, [80.0, 450.0]);
    release_at(&mut world, &mut h, [81.0, 450.0]);
    assert_eq!(h.selection.active(), None, "plain still release clears");
}

// Every selection member gets a ring; the active member's is brighter.
#[test]
fn selection_rings_cover_every_member() {
    let (mut world, _, _, mut h) = two_prop_rig(SIDE_A, SIDE_B, 1.0);
    click_at(&mut world, &mut h, [200.0, 600.0]);
    click_at_mod(&mut world, &mut h, [424.0, 598.0], true);

    let ids = highlight::all_sprite_ids();
    let ring = |world: &World, i: usize| {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == ids[i])
            .cloned()
            .expect("ring pool injected")
    };
    let (r0, r1) = (ring(&world, 0), ring(&world, 1));
    assert!(r0.visible && r1.visible, "one ring per member");
    assert!(
        !ring(&world, 2).visible,
        "the rest of the pool stays hidden"
    );
    assert_ne!(
        r0.border_color, r1.border_color,
        "the active member's ring is distinguished"
    );
    let center = |r: &Sprite| (r.x + r.width * 0.5, r.y + r.height * 0.5);
    let (c0x, _) = center(&r0);
    let (c1x, _) = center(&r1);
    assert!(
        c0x < c1x,
        "rings follow selection order: box_a left of box_b"
    );
}

// The gizmo anchors at the selection centroid and a translate drag moves
// every member by the same world delta, committed as ONE undo step.
#[test]
fn multi_translate_moves_all_members_as_one_undo_step() {
    let (mut world, e1, e2, mut h) = two_prop_rig(SIDE_A, SIDE_B, 1.0);
    click_at(&mut world, &mut h, [200.0, 600.0]);
    click_at_mod(&mut world, &mut h, [424.0, 598.0], true);

    let layout = h
        .gizmo_layout(&world, [1280.0, 720.0])
        .expect("multi selection shows the gizmo");
    // The anchor is the centroid: between the two boxes, ~[312, 598].
    assert!(
        (layout.origin[0] - 312.0).abs() < 3.0 && (layout.origin[1] - 598.0).abs() < 3.0,
        "centroid anchor, got {:?}",
        layout.origin
    );

    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some(), "the tip press starts a drag");
    drag_to(
        &mut world,
        &mut h,
        [layout.tips[0][0] + 50.0, layout.tips[0][1]],
    );
    let p1 = world.get::<crate::assets::Transform>(e1).unwrap().position;
    let p2 = world.get::<crate::assets::Transform>(e2).unwrap().position;
    let (d1, d2) = (p1[0] - SIDE_A[0], p2[0] - SIDE_B[0]);
    assert!(d1 > 0.3, "box_a moved right: {d1}");
    assert!((d1 - d2).abs() < 1e-3, "one shared delta: {d1} vs {d2}");
    assert_eq!(p1[1], SIDE_A[1], "Y untouched");

    release_at(
        &mut world,
        &mut h,
        [layout.tips[0][0] + 50.0, layout.tips[0][1]],
    );
    assert!(h.dirty, "the move is an unsaved edit");
    for (i, s) in [(0, SIDE_A), (1, SIDE_B)] {
        let committed = h.entries[i]["args"]["position"][0].as_f64().unwrap();
        assert!(
            committed > f64::from(s[0]) + 0.3,
            "entry {i} follows: {committed}"
        );
    }

    h.undo(&mut world);
    for (i, s) in [(0, SIDE_A), (1, SIDE_B)] {
        let restored = h.entries[i]["args"]["position"][0].as_f64().unwrap();
        assert!(
            (restored - f64::from(s[0])).abs() < 1e-3,
            "entry {i}: {restored}"
        );
    }
    assert!(!h.can_undo(), "the whole multi-drag was one step");
}

// Rotate orbits member positions about the centroid (and spins each member),
// the group behavior of every mainstream editor; both writes land in the
// entries and one undo restores them.
#[test]
fn multi_rotate_orbits_members_about_the_centroid() {
    // Stacked vertically: box_a above the centroid, box_b below, so an X-axis
    // turn swings them through Z.
    let s1 = [-6.11, -2.3, -5.0];
    let s2 = [-6.11, -4.3, -5.0];
    let (mut world, e1, e2, mut h) = two_prop_rig(s1, s2, 0.8);
    h.gizmo_mode = gizmo::GizmoMode::Rotate;
    click_at(&mut world, &mut h, [200.0, 526.0]);
    assert_eq!(h.selection.active(), Some("box_a"));
    click_at_mod(&mut world, &mut h, [200.0, 670.0], true);
    assert_eq!(h.selection.iter().count(), 2);

    let layout = h.gizmo_layout(&world, [1280.0, 720.0]).expect("gizmo up");
    // Grab the X tip and swing a quarter circle to straight below the origin:
    // +90 degrees about world X.
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some(), "rotate grab starts on the tip");
    drag_to(
        &mut world,
        &mut h,
        [layout.origin[0], layout.origin[1] + 70.0],
    );

    let t1 = *world.get::<crate::assets::Transform>(e1).unwrap();
    let t2 = *world.get::<crate::assets::Transform>(e2).unwrap();
    assert!(
        (t1.rotation_deg[0] - 90.0).abs() < 1.0 && (t2.rotation_deg[0] - 90.0).abs() < 1.0,
        "both spin: {} / {}",
        t1.rotation_deg[0],
        t2.rotation_deg[0]
    );
    // +90 about X through the centroid [-6.11, -3.3, -5.0] carries the +Y
    // offset into +Z: box_a to z = -4, box_b to z = -6, both onto y = -3.3.
    assert!(
        (t1.position[1] + 3.3).abs() < 0.1 && (t1.position[2] + 4.0).abs() < 0.1,
        "box_a orbits: {:?}",
        t1.position
    );
    assert!(
        (t2.position[1] + 3.3).abs() < 0.1 && (t2.position[2] + 6.0).abs() < 0.1,
        "box_b orbits: {:?}",
        t2.position
    );

    release_at(
        &mut world,
        &mut h,
        [layout.origin[0], layout.origin[1] + 70.0],
    );
    let committed = h.entries[0]["args"]["rotation_deg"][0].as_f64().unwrap();
    assert!((committed - 90.0).abs() < 1.0, "{committed}");
    let z = h.entries[0]["args"]["position"][2].as_f64().unwrap();
    assert!((z + 4.0).abs() < 0.1, "the orbited position commits: {z}");

    h.undo(&mut world);
    for i in [0, 1] {
        assert!(
            h.entries[i]["args"].get("rotation_deg").is_none(),
            "one undo step restores entry {i}"
        );
    }
    let z = h.entries[0]["args"]["position"][2].as_f64().unwrap();
    assert!((z + 5.0).abs() < 1e-3, "position restored too: {z}");
    assert!(!h.can_undo(), "the whole multi-drag was one step");
}

// T / R / S switch the gizmo mode in edit mode, but never while typing.
#[test]
fn gizmo_mode_keys_switch_unless_typing() {
    let mut h = hook(Vec::new());
    let key = |h: &mut EditorHook, k: crate::assets::Key| {
        let mut world = world_with_input(FrameInput {
            viewport: [1280.0, 720.0],
            captured_key: Some(k),
            ..Default::default()
        });
        h.tick(&mut world);
    };
    key(&mut h, crate::assets::Key::R);
    assert_eq!(h.gizmo_mode, gizmo::GizmoMode::Rotate);
    key(&mut h, crate::assets::Key::S);
    assert_eq!(h.gizmo_mode, gizmo::GizmoMode::Scale);
    key(&mut h, crate::assets::Key::T);
    assert_eq!(h.gizmo_mode, gizmo::GizmoMode::Translate);

    // F toggles the fly camera through the same guard.
    key(&mut h, crate::assets::Key::F);
    assert!(h.fly, "F starts the fly camera");
    key(&mut h, crate::assets::Key::F);
    assert!(!h.fly, "F again stops it");

    // A focused text field keeps the keys for typing.
    h.story_focus = true;
    key(&mut h, crate::assets::Key::R);
    assert_eq!(h.gizmo_mode, gizmo::GizmoMode::Translate);
    key(&mut h, crate::assets::Key::F);
    assert!(!h.fly, "typing keeps F");
}

// A tree row click mirrors a viewport pick: plain replaces the selection and
// opens the clicked entry's edit form; with shift held it toggles membership
// instead.
#[test]
fn tree_row_click_selects_and_opens_the_form() {
    let mut world = world_with_fields();
    let mut h = hook(vec![entry("box", "Sprite"), entry("cam", "Camera3D")]);
    h.panel_open = true;
    seed_tree(&mut h, Vec::new());
    click_row(&mut h, "box", &mut world);
    assert_eq!(h.selection.active(), Some("box"));
    assert!(h.panel_open, "the assets UI comes up around the form");
    assert_eq!(
        h.form_target,
        FormTarget::Entry(0),
        "the form targets the clicked entry"
    );

    h.shift_held = true;
    click_row(&mut h, "cam", &mut world);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box", "cam"],
        "a shift click adds"
    );
    click_row(&mut h, "cam", &mut world);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box"],
        "a second shift click removes"
    );
}

// The row eye and the menu's Lock are editor-session state: they flip the
// hook's sets (the hidden set publishing as ids each tick) and never touch the
// entries.
#[test]
fn hide_and_lock_are_session_state_not_edits() {
    crate::ecs::asset_id::reset_interner();
    let id = crate::ecs::asset_id::intern("box");
    let mut world = world_with_input(FrameInput::default());
    let mut h = hook(vec![entry("box", "Sprite")]);
    h.panel_open = true;
    seed_tree(&mut h, Vec::new());
    let (g, i) = row_of(&h, "box");

    h.apply_panel(PanelAction::ToggleHide(g, i), &mut world);
    h.apply_panel(PanelAction::OpenRowMenu(g, i), &mut world);
    h.apply_panel(PanelAction::RowToggleLock, &mut world);
    assert!(h.hidden_assets.contains("box"));
    assert!(h.locked_assets.contains("box"));
    assert!(h.row_menu.is_none(), "picking Lock closes the menu");
    assert!(!h.dirty, "session toggles are not authored edits");

    h.tick(&mut world);
    let hidden = world
        .resource::<crate::ecs::EditorHidden>()
        .expect("the hook publishes the hidden set every tick");
    assert!(hidden.0.contains(&id), "names resolve to this world's ids");

    h.apply_panel(PanelAction::ToggleHide(g, i), &mut world);
    h.apply_panel(PanelAction::OpenRowMenu(g, i), &mut world);
    h.apply_panel(PanelAction::RowToggleLock, &mut world);
    assert!(h.hidden_assets.is_empty() && h.locked_assets.is_empty());
}

// A locked asset is skipped by viewport picking: the click passes through to
// empty space (arming the marquee) instead of selecting it.
#[test]
fn locked_assets_are_skipped_by_viewport_picking() {
    crate::ecs::asset_id::reset_interner();
    let near = crate::ecs::asset_id::intern("box_near");
    let mut world = pick_world([0.0; 3], vec![(near, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0])]);
    let mut h = hook(vec![entry("box_near", "Sprite")]);
    h.locked_assets.insert("box_near".to_string());

    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), None, "the locked box is not picked");
    assert!(h.marquee.is_some(), "the click fell through to empty space");
}

// A viewport pick unfolds the picked asset's group and scrolls its row into
// the tree's window.
#[test]
fn viewport_pick_reveals_the_tree_row() {
    let world = World::new_empty();
    let mut h = hook(Vec::new());
    h.panel_open = true;
    h.tree_stale = false;
    h.tree_groups = vec![TreeGroup {
        label: asset_tree::WORLD_GROUP.to_string(),
        assets: (0..30)
            .map(|i| asset_tree::TreeAsset {
                name: format!("a{i:02}"),
                asset_type: "Sprite".to_string(),
                badge: asset_tree::Badge::Authored,
                promote: None,
            })
            .collect(),
    }];

    h.reveal_in_tree("a25", &world);
    assert_eq!(h.tree_unfolded, vec![0], "the group unfolds");
    // Rows: header at 0, a25 at 26; the scroll clamps to the last window.
    assert_eq!(h.tree_scroll, 31 - panel::ROW_POOL);

    // A revealed row already inside the window leaves the scroll alone.
    h.reveal_in_tree("a20", &world);
    assert_eq!(h.tree_scroll, 31 - panel::ROW_POOL);
}

// Billboard test rig: the pick rig plus a PointLight entity indexed by name
// (as the loaders' name -> entity index would) and the injected billboard
// pools.
fn billboard_world(
    light_pos: [f32; 3],
    picks: Vec<(crate::ecs::asset_id::AssetId, [f32; 3], [f32; 3])>,
) -> World {
    let mut world = pick_world([0.0; 3], picks);
    for s in billboards::sprites() {
        world.add_component(s);
    }
    let entity = world.push(crate::assets::PointLight {
        position: light_pos,
        ..Default::default()
    });
    let id = crate::ecs::asset_id::intern("lamp");
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(id, entity);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    world
}

fn lamp_entry(pos: [f32; 3]) -> serde_json::Value {
    serde_json::json!({"name": "lamp", "type": "PointLight", "args": {"position": pos}})
}

// Clicking a light's billboard selects it by name through the normal pick
// flow, and the tick seeds the Transform the gizmo needs onto its entity.
#[test]
fn billboard_click_selects_the_light_and_seeds_its_transform() {
    crate::ecs::asset_id::reset_interner();
    let mut world = billboard_world([0.0, 0.0, -5.0], Vec::new());
    let mut h = hook(vec![lamp_entry([0.0, 0.0, -5.0])]);

    // The light projects to the viewport center (camera at origin facing -Z).
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("lamp"), "the icon press selects");
    assert!(h.panel_open, "the assets UI comes up around the form");
    assert_eq!(h.selected_type.as_deref(), Some("PointLight"));

    // The seeded Transform mirrors the authored position, so the gizmo's
    // member resolve works on the light.
    let entity = world
        .resource::<concinnity_core::ecs::EntityByName>()
        .unwrap()
        .0
        .values()
        .next()
        .copied()
        .unwrap();
    let t = world.get::<crate::assets::Transform>(entity).unwrap();
    assert_eq!(t.position, [0.0, 0.0, -5.0]);
    assert!(
        h.gizmo_layout(&world, [1280.0, 720.0]).is_some(),
        "the translate gizmo anchors on the selected light"
    );
}

// A mesh AABB in front of the billboard's anchor keeps the press; one behind
// it loses to the icon.
#[test]
fn billboard_and_mesh_overlap_prefers_the_nearer_hit() {
    crate::ecs::asset_id::reset_interner();
    let wall = crate::ecs::asset_id::intern("wall");
    // Wall at depth 2..3, light at depth 5: the wall is nearer.
    let mut world = billboard_world(
        [0.0, 0.0, -5.0],
        vec![(wall, [-1.0, -1.0, -3.0], [1.0, 1.0, -2.0])],
    );
    let mut h = hook(vec![entry("wall", "Sprite"), lamp_entry([0.0, 0.0, -5.0])]);
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("wall"), "the nearer mesh wins");

    // Wall at depth 9..10, light at depth 5: the icon is nearer.
    crate::ecs::asset_id::reset_interner();
    let wall = crate::ecs::asset_id::intern("wall");
    let mut world = billboard_world(
        [0.0, 0.0, -5.0],
        vec![(wall, [-1.0, -1.0, -10.0], [1.0, 1.0, -9.0])],
    );
    let mut h = hook(vec![entry("wall", "Sprite"), lamp_entry([0.0, 0.0, -5.0])]);
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("lamp"), "the nearer icon wins");
}

// Editor-hidden billboards neither draw nor pick; locked ones stay visible
// but pass the press through, both matching the mesh pick's rules.
#[test]
fn hidden_and_locked_billboards_follow_the_pick_rules() {
    crate::ecs::asset_id::reset_interner();
    let mut world = billboard_world([0.0, 0.0, -5.0], Vec::new());
    let mut h = hook(vec![lamp_entry([0.0, 0.0, -5.0])]);
    h.locked_assets.insert("lamp".to_string());
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), None, "a locked icon is pick-through");
    assert!(h.marquee.is_some(), "the click fell through to empty space");

    crate::ecs::asset_id::reset_interner();
    let mut world = billboard_world([0.0, 0.0, -5.0], Vec::new());
    let mut h = hook(vec![lamp_entry([0.0, 0.0, -5.0])]);
    h.hidden_assets.insert("lamp".to_string());
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), None, "a hidden asset draws no icon");
    assert!(h.marquee.is_some(), "the click fell through to empty space");
}

// Selecting a trigger volume draws its collider outline: dotted box segments
// for a cuboid, and nothing once the selection moves elsewhere.
#[test]
fn selected_trigger_volume_draws_its_box_outline() {
    crate::ecs::asset_id::reset_interner();
    let mut world = pick_world([0.0; 3], Vec::new());
    for s in billboards::sprites() {
        world.add_component(s);
    }
    let entity = world.push(crate::assets::TriggerVolume {
        position: [0.0, 0.0, -6.0],
        ..Default::default()
    });
    let id = crate::ecs::asset_id::intern("zone");
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(id, entity);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    let mut h = hook(vec![serde_json::json!({
        "name": "zone", "type": "TriggerVolume",
        "args": {"position": [0.0, 0.0, -6.0]}
    })]);

    // Click the volume's projected icon (viewport center): it selects and its
    // outline comes up.
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("zone"));
    let ids: std::collections::HashSet<_> = billboards::all_sprite_ids().into_iter().collect();
    let outline_shown = world
        .query::<Sprite>()
        .filter(|s| s.visible && ids.contains(&s.asset_id) && s.width < 4.0)
        .count();
    assert_eq!(
        outline_shown,
        billboards::BOX_EDGES * billboards::EDGE_SEGMENTS,
        "every dotted box segment is placed"
    );

    // Clearing the selection hides the outline again.
    h.selection.clear();
    h.tick(&mut world);
    let outline_shown = world
        .query::<Sprite>()
        .filter(|s| s.visible && ids.contains(&s.asset_id) && s.width < 4.0)
        .count();
    assert_eq!(outline_shown, 0, "no outline without a selected volume");
}

// Console commands mutate the working entries like their panel counterparts,
// and everything they do lands in the log sink.
#[test]
fn console_commands_edit_the_working_entries() {
    let mut h = hook(Vec::new());

    h.run_console_line("/add PhysicsConfig phys");
    assert_eq!(h.entries.len(), 1);
    assert_eq!(entry_name(&h.entries[0]), Some("phys"));
    assert_eq!(entry_type(&h.entries[0]), Some("PhysicsConfig"));
    assert!(h.dirty, "an added entry marks the world dirty");

    h.run_console_line("/del phys");
    assert!(h.entries.is_empty());

    h.run_console_line("/del ghost");
    h.run_console_line("just a note");
    let lines = h.console_sink.window(0, 64);
    let texts: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
    assert!(
        texts.contains(&"added 'phys' (PhysicsConfig)"),
        "got: {texts:?}"
    );
    assert!(
        texts.contains(&"removed 'phys' (PhysicsConfig)"),
        "got: {texts:?}"
    );
    assert!(
        texts.contains(&"no authored asset named 'ghost'"),
        "got: {texts:?}"
    );
    // A bare line echoes (as does every submitted command line).
    assert!(texts.contains(&"> just a note"), "got: {texts:?}");
    assert!(
        lines.iter().any(|l| l.severity == console::Severity::Error),
        "the missing name reports as an error"
    );
}

// Backtick opens the console focused-but-blurred for that frame (so the text
// system cannot type the backtick into the command line), closes it again on
// the next press, and stands down entirely while another field is typing.
#[test]
fn backtick_toggles_the_console_with_a_one_frame_blur() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    let input = FrameInput {
        captured_key: Some(crate::assets::Key::Backtick),
        ..Default::default()
    };

    h.drive_console_toggle(&input, &mut world);
    assert!(h.console_open && h.console_focus && h.console_blur);
    assert_eq!(h.panel_order.last(), Some(&PanelKey::Console));
    let (lines, total, first) = h.console_window();
    assert!(
        !h.make_console_view(&lines, total, first, "", [0.0, 0.0])
            .focus,
        "the opening frame never asserts field focus"
    );
    // The next frame clears the blur (the tick does this) and focus asserts.
    h.console_blur = false;
    let (lines, total, first) = h.console_window();
    assert!(
        h.make_console_view(&lines, total, first, "", [0.0, 0.0])
            .focus
    );

    h.drive_console_toggle(&input, &mut world);
    assert!(!h.console_open && !h.console_focus, "second press closes");

    // While another text field is focused, backtick is just a character.
    h.story_focus = true;
    h.drive_console_toggle(&input, &mut world);
    assert!(!h.console_open);
}

// The /del ghost completes against authored names, and Tab accepts it into
// the command line.
#[test]
fn console_ghost_completes_del_names_and_tab_accepts() {
    let mut h = hook(vec![entry("cube_red", "Prop")]);
    let mut world = World::new_empty();
    world.add_component(TextInput {
        asset_id: console_panel::INPUT,
        content: "/del cu".to_string(),
        caret: 7,
        ..Default::default()
    });

    assert_eq!(h.console_ghost(&world), "be_red");
    h.console_focus = true;
    let tab = FrameInput {
        captured_key: Some(crate::assets::Key::Tab),
        ..Default::default()
    };
    h.console_keys(&mut world, &tab);
    assert_eq!(
        widget::field_text(&world, console_panel::INPUT),
        "/del cube_red"
    );
    // With the name complete there is nothing left to ghost.
    assert_eq!(h.console_ghost(&world), "");
}

// The full tick path of a backtick open: the first frame shows the command
// line unfocused (so the text system cannot type the '`' into it), the next
// frame asserts focus.
#[test]
fn tick_opens_the_console_blurred_then_focuses() {
    isolate_state_dir();
    let mut world = world_with_input(FrameInput {
        captured_key: Some(crate::assets::Key::Backtick),
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    world.add_component(TextInput {
        asset_id: console_panel::INPUT,
        ..Default::default()
    });
    for id in console_panel::all_label_ids() {
        world.add_component(TextLabel {
            asset_id: id,
            ..Default::default()
        });
    }
    let mut h = hook(Vec::new());

    h.tick(&mut world);
    assert!(h.console_open);
    let input = world
        .query::<TextInput>()
        .find(|t| t.asset_id == console_panel::INPUT)
        .unwrap();
    assert!(
        input.visible && !input.focused,
        "the opening frame leaves the field blurred"
    );

    // Next frame, no key held: focus asserts.
    for i in world.query_mut::<FrameInput>() {
        i.captured_key = None;
    }
    h.tick(&mut world);
    let input = world
        .query::<TextInput>()
        .find(|t| t.asset_id == console_panel::INPUT)
        .unwrap();
    assert!(input.visible && input.focused);
}
