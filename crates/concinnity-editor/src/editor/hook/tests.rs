// src/editor/hook/tests.rs
//
// Unit + tick-level tests for the editor hook.

use super::*;
use crate::assets::{Sprite, TextInput, TextLabel};
use crate::editor::behavior::graph::CardKind;
use crate::editor::behavior::path;

fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), entries)
}

// The shared title-bar / close-button rects the routing derives for a panel
// (the shared geometry lives in `widget`).
fn title_rect_of(h: &EditorHook, key: PanelKey, vp: [f32; 2]) -> [f32; 4] {
    let o = h.origin(key, vp);
    widget::title_rect(o, registry::panel(key).size(h)[0])
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

fn entry_with_args(name: &str, ty: &str, args: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"name": name, "type": ty, "args": args})
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
    assert_eq!(
        h.sim.state,
        sim::SimState::Stopped,
        "editor holds the cursor at launch"
    );
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
    h.sim.state = sim::SimState::Playing;
    let mut world = world_with_input(FrameInput {
        escape: true,
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.tick(&mut world);
    assert_eq!(
        h.sim.state,
        sim::SimState::Paused,
        "Escape pauses play mode"
    );
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
    assert!(h.sim.playing(), "the checkbox click enters play mode");
    click(&mut h, &mut world, row_mid(0));
    assert!(!h.sim.playing(), "a second click leaves it");

    click(&mut h, &mut world, row_mid(1));
    assert!(h.fly, "the fly row starts the fly camera");
    assert!(!h.sim.playing());
    click(&mut h, &mut world, row_mid(0));
    assert!(
        h.sim.playing() && !h.fly,
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
    assert!(!h.sim.playing(), "the drag swallowed the click");
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
    let row = panel::row_rect(po, panel::PANEL_W, 1);
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
    let t = widget::title_rect(po, panel::PANEL_W);
    // The title bar's interior (clear of the corner / edge resize band) drags.
    let claimed = h.try_panel_press(
        PanelKey::Assets,
        t[0] + t[2] * 0.5,
        t[1] + t[3] * 0.5,
        vp,
        &mut world,
    );
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
    let x = form_panel::close_rect(h.origin(PanelKey::Edit, vp), form_panel::EDIT_W);
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

// An open Behavior panel claims only presses that land on it. Its chart views
// used to answer for the whole screen, which left every other panel unable to
// be moved, closed, or brought forward while one of them was showing.
#[test]
fn the_behavior_panel_leaves_the_other_panels_pressable_in_every_view() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"hide": {"target": "self"}}]}),
    )]);
    let vp = [1280.0, 720.0];
    h.view_open = true;
    // Behavior sits in front of the panels the press has to reach.
    h.focus_panel(PanelKey::Behavior);

    for mode in [ViewMode::Outline, ViewMode::Chart, ViewMode::Overview] {
        h.behavior_mode = mode;
        for key in [PanelKey::Preview, PanelKey::View] {
            let title = title_rect_of(&h, key, vp);
            let (mx, my) = (title[0] + 5.0, title[1] + 5.0);
            assert!(
                !h.try_panel_press(PanelKey::Behavior, mx, my, vp, &mut world),
                "{mode:?}: Behavior swallowed a press meant for {key:?}"
            );
            h.drag = None;
            assert!(
                h.try_panel_press(key, mx, my, vp, &mut world),
                "{mode:?}: {key:?} never saw the press"
            );
            h.drag = None;
            h.focus_panel(PanelKey::Behavior);
        }
    }
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
    // The title bar's interior (clear of the corner / edge resize band) drags.
    assert!(h.try_panel_press(
        PanelKey::Templates,
        t[0] + t[2] * 0.5,
        t[1] + t[3] * 0.5,
        vp,
        &mut world
    ));
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
    let apply = template_panel::apply_rect(
        h.origin(PanelKey::TemplateDetail, vp),
        h.effective_size(PanelKey::TemplateDetail)[0],
    );
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
    // Click the row's interior (clear of the edge resize band).
    assert!(h.try_panel_press(
        PanelKey::View,
        row[0] + row[2] * 0.5,
        row[1] + 5.0,
        vp,
        &mut world
    ));
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
    h.sim.state = sim::SimState::Playing;
    step(&mut h, Key::Z);
    assert_eq!(h.entries.len(), 1, "suppressed in play mode");
    h.sim.state = sim::SimState::Stopped;

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
    h.sim.state = sim::SimState::Playing;
    set_input(
        &mut world,
        FrameInput {
            viewport: [1280.0, 720.0],
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(!ring(&world).visible, "hidden in play mode");
    h.sim.state = sim::SimState::Stopped;
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

// How far `v` sits from the nearest multiple of `step`, in step units
// (transform math accumulates float error, so grid checks need a tolerance).
fn off_grid(v: f32, step: f32) -> f32 {
    (v / step - (v / step).round()).abs()
}

// With move snapping on, a translate drag lands on grid multiples of the step;
// holding Ctrl suspends the snap for the frame; release commits the snapped
// position.
#[test]
fn gizmo_translate_drag_snaps_to_the_grid_and_ctrl_suspends_it() {
    let start = [-6.11f32, -3.3, -5.0];
    let (mut world, entity, mut h) = gizmo_rig(start);
    h.snap.translate = super::super::snap::Snap {
        enabled: true,
        step: 0.25,
    };

    click_at(&mut world, &mut h, [200.0, 600.0]);
    let layout = h.gizmo_layout(&world, [1280.0, 720.0]).expect("gizmo up");
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some());

    let target = [layout.tips[0][0] + 50.0, layout.tips[0][1]];
    set_input(&mut world, drag_input(target, true));
    h.tick(&mut world);
    let delta = world
        .get::<crate::assets::Transform>(entity)
        .unwrap()
        .position[0]
        - start[0];
    assert!(delta > 0.2, "moved right: {delta}");
    assert!(off_grid(delta, 0.25) < 1e-4, "on the grid: {delta}");

    let mut ctrl = drag_input(target, true);
    ctrl.ctrl = true;
    set_input(&mut world, ctrl);
    h.tick(&mut world);
    let free = world
        .get::<crate::assets::Transform>(entity)
        .unwrap()
        .position[0]
        - start[0];
    assert!(free > 0.2, "still follows the cursor: {free}");
    assert!(
        off_grid(free, 0.25) > 1e-3,
        "unsnapped while Ctrl is held: {free}"
    );

    // Releasing Ctrl re-snaps the preview; the commit is what is shown.
    set_input(&mut world, drag_input(target, true));
    h.tick(&mut world);
    set_input(&mut world, drag_input(target, false));
    h.tick(&mut world);
    let committed = h.entries[0]["args"]["position"][0].as_f64().unwrap() as f32 - start[0];
    assert!(
        off_grid(committed, 0.25) < 1e-3,
        "the commit is on the grid: {committed}"
    );
}

// With rotate snapping on, the applied angle rounds to the step: a ~60 degree
// swing lands on 45.
#[test]
fn gizmo_rotate_drag_snaps_the_applied_angle() {
    let start = [-6.11f32, -3.3, -5.0];
    let (mut world, entity, mut h) = gizmo_rig(start);
    h.gizmo_mode = gizmo::GizmoMode::Rotate;
    h.snap.rotate = super::super::snap::Snap {
        enabled: true,
        step: 45.0,
    };

    click_at(&mut world, &mut h, [200.0, 600.0]);
    let layout = h.gizmo_layout(&world, [1280.0, 720.0]).expect("gizmo up");
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some());

    // Swing from the X tip (screen angle 0) to ~60 degrees of screen angle.
    let m = [layout.origin[0] + 35.0, layout.origin[1] + 60.6];
    set_input(&mut world, drag_input(m, true));
    h.tick(&mut world);
    let live = world.get::<crate::assets::Transform>(entity).unwrap();
    assert!(
        (live.rotation_deg[0] - 45.0).abs() < 1e-3,
        "snapped to 45: {}",
        live.rotation_deg[0]
    );

    set_input(&mut world, drag_input(m, false));
    h.tick(&mut world);
    let committed = h.entries[0]["args"]["rotation_deg"][0].as_f64().unwrap();
    assert!((committed - 45.0).abs() < 0.2, "{committed}");
}

// /snap drives the same settings the Preview panel rows toggle.
#[test]
fn console_snap_adjusts_and_reports_the_settings() {
    let mut world = World::new_empty();
    let mut h = hook(vec![]);
    h.run_console_line(&mut world, "/snap 0.25");
    assert!(h.snap.translate.enabled, "a step also enables the family");
    assert_eq!(h.snap.translate.step, 0.25);
    h.run_console_line(&mut world, "/snap rot 45");
    assert!(h.snap.rotate.enabled);
    assert_eq!(h.snap.rotate.step, 45.0);
    h.run_console_line(&mut world, "/snap off");
    assert!(!h.snap.translate.enabled && !h.snap.rotate.enabled);
    assert_eq!(h.snap.translate.step, 0.25, "off keeps the steps");
    let lines = h.console_sink.window(0, 100);
    assert!(
        lines
            .last()
            .is_some_and(|l| l.text == "snap: move 0.25 m (off), rotate 45 deg (off)"),
        "each /snap reports the resulting state"
    );
}

// Duplicating the selection clones each authored entry (args included) under
// a unique name, skips singletons, selects the copies, and is one undo step.
#[test]
fn duplicate_selection_clones_entries_and_selects_the_copies() {
    let mut world = World::new_empty();
    let mut h = hook(vec![
        serde_json::json!({
            "name": "box", "type": "Prop", "args": { "position": [1.0, 2.0, 3.0] }
        }),
        entry("phys", "PhysicsConfig"),
    ]);
    h.selection.set(vec!["box".to_string(), "phys".to_string()]);

    h.run_console_line(&mut world, "/dup");
    assert_eq!(h.entries.len(), 3, "the singleton is skipped");
    assert_eq!(entry_name(&h.entries[2]), Some("box_1"));
    assert_eq!(
        h.entries[2]["args"]["position"],
        serde_json::json!([1.0, 2.0, 3.0]),
        "the copy keeps the original's args"
    );
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        vec!["box_1"],
        "the copies become the selection"
    );
    assert!(h.dirty);
    let lines = h.console_sink.window(0, 16);
    assert!(lines.iter().any(|l| l.text == "duplicated 1"));

    h.undo(&mut world);
    assert_eq!(h.entries.len(), 2, "one undo step removes the whole batch");
    assert!(!h.can_undo());
}

// Ctrl+D duplicates through the tick, except while the Behavior panel is
// frontmost (its frame_keys own that shortcut for row duplication).
#[test]
fn ctrl_d_duplicates_unless_the_behavior_panel_owns_it() {
    let mut world = world_with_input(FrameInput {
        ctrl: true,
        captured_key: Some(crate::assets::Key::D),
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    for id in behavior_panel::all_field_ids() {
        world.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    let mut h = hook(vec![entry("box", "Sprite")]);
    h.selection.replace("box".to_string());
    h.tick(&mut world);
    assert_eq!(h.entries.len(), 2, "Ctrl+D duplicates the selection");

    registry::panel(PanelKey::Behavior).toggle(&mut h, &mut world);
    h.selection.replace("box".to_string());
    let before = h.entries.len();
    h.tick(&mut world);
    assert_eq!(
        h.entries.len(),
        before,
        "a frontmost Behavior panel keeps its own Ctrl+D"
    );
}

// Drop-to-floor lands an indexed member's bounds on the surface below it,
// rests a bounds-less member's origin on the ground-plane fallback, and
// commits the batch as one undo step.
#[test]
fn drop_to_floor_lands_the_selection_on_the_surface_below() {
    crate::ecs::asset_id::reset_interner();
    let a = crate::ecs::asset_id::intern("box_a");
    let _g = crate::ecs::asset_id::intern("ground");
    let lamp_id = crate::ecs::asset_id::intern("lamp");
    let mut world = pick_world(
        [0.0; 3],
        vec![
            (a, [-1.0, 4.0, -6.0], [1.0, 6.0, -4.0]),
            (_g, [-10.0, -1.0, -10.0], [10.0, 0.0, 10.0]),
        ],
    );
    let box_e = world.push(crate::assets::Transform {
        position: [0.0, 5.0, -5.0],
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    });
    // The lamp has no pick-index bounds and sits clear of the ground box, so
    // it exercises both the position-as-foot path and the y=0 fallback.
    let lamp_e = world.push(crate::assets::Transform {
        position: [100.0, 3.0, 0.0],
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    });
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(a, box_e);
    by_name.insert(lamp_id, lamp_e);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));

    let mut h = hook(vec![
        serde_json::json!({
            "name": "box_a", "type": "Prop", "args": { "position": [0.0, 5.0, -5.0] }
        }),
        serde_json::json!({
            "name": "ground", "type": "Prop", "args": { "position": [0.0, 0.0, 0.0] }
        }),
        serde_json::json!({
            "name": "lamp", "type": "PointLight", "args": { "position": [100.0, 3.0, 0.0] }
        }),
    ]);
    h.selection
        .set(vec!["box_a".to_string(), "lamp".to_string()]);

    h.run_console_line(&mut world, "/floor");
    assert_eq!(
        h.entries[0]["args"]["position"],
        serde_json::json!([0.0, 1.0, -5.0]),
        "the box bottom (1 below its position) rests on the ground top"
    );
    assert_eq!(
        h.entries[2]["args"]["position"],
        serde_json::json!([100.0, 0.0, 0.0]),
        "nothing below the lamp: its origin lands on the y=0 fallback"
    );
    let live = world.get::<crate::assets::Transform>(box_e).unwrap();
    assert_eq!(
        live.position,
        [0.0, 1.0, -5.0],
        "the live transform follows"
    );
    let lines = h.console_sink.window(0, 16);
    assert!(lines.iter().any(|l| l.text == "dropped 2"));

    h.undo(&mut world);
    assert_eq!(
        h.entries[0]["args"]["position"],
        serde_json::json!([0.0, 5.0, -5.0])
    );
    assert!(!h.can_undo(), "the whole drop was one step");

    // A selection with no eligible member (no live entity) drops nothing and
    // records no undo step.
    h.selection.set(vec!["ground".to_string()]);
    h.run_console_line(&mut world, "/floor");
    assert!(!h.can_undo(), "a no-op drop records nothing");
}

// The Content grid over a world with visual assets: cells list them with
// icon fallbacks (no thumbnails baked in tests), the type chip narrows, the
// search query ranks, and a cell click selects the asset.
#[test]
fn content_grid_lists_filters_and_selects_visual_assets() {
    let mut world = World::new_empty();
    let mut h = hook(vec![
        serde_json::json!({
            "name": "brick_tex", "type": "Texture",
            "args": { "generator": "brick", "resolution": 32 }
        }),
        serde_json::json!({
            "name": "brick_mat", "type": "Material", "args": { "roughness": 0.5 }
        }),
        entry("note", "TextLabel"),
    ]);
    h.content_open = true;
    h.tree_stale = true;
    h.refresh_tree_if_needed();

    let (cells, total) = h.content_cells(&world);
    assert_eq!(total, 2, "only the visual types are listed");
    let names: Vec<&str> = cells.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        names,
        ["brick_mat", "brick_tex"],
        "sorted, TextLabel absent"
    );
    assert!(
        cells.iter().all(|c| c.thumb.is_none()),
        "no baked thumbnails in tests: every cell falls back to its icon"
    );

    // The type chip narrows to one kind.
    while h.content_type_caption() != "Material" {
        h.cycle_content_type();
    }
    let (cells, total) = h.content_cells(&world);
    assert_eq!((cells.len(), total), (1, 1));
    assert_eq!(cells[0].asset_type, "Material");
    h.content_type = 0;

    // The search field ranks name matches.
    world.add_component(TextInput {
        asset_id: content_panel::SEARCH_INPUT,
        content: "tex".to_string(),
        ..Default::default()
    });
    let (cells, _) = h.content_cells(&world);
    assert_eq!(cells[0].name, "brick_tex", "the query's best match leads");

    // A cell click selects the asset by name.
    h.apply_content_action(
        crate::editor::content_panel::ContentAction::SelectCell(0),
        &mut world,
        [0.0, 0.0],
    );
    assert_eq!(h.selection.active(), Some("brick_tex"));
    let (cells, _) = h.content_cells(&world);
    assert!(cells[0].selected, "the grid highlights the selection");
}

// Dragging a mesh out of the Content grid places a Prop where the ghost
// lands: press a cell, pull into the viewport (ghost follows the surface
// below the cursor), release commits ONE undoable entry and selects it.
#[test]
fn drag_out_places_a_prop_where_the_ghost_lands() {
    crate::ecs::asset_id::reset_interner();
    let ground = crate::ecs::asset_id::intern("ground");
    let mut world = pick_world(
        [0.0; 3],
        vec![(ground, [-20.0, -1.0, -20.0], [20.0, 0.0, 20.0])],
    );
    let mut h = hook(vec![
        serde_json::json!({
            "name": "demo_ball", "type": "ProceduralMesh",
            "args": { "generator": "sphere" }
        }),
        serde_json::json!({
            "name": "ground", "type": "Prop",
            "args": { "mesh": "demo_ball", "position": [0.0, -0.5, 0.0] }
        }),
    ]);
    h.content_open = true;
    h.tree_stale = true;
    h.refresh_tree_if_needed();

    // Press the first grid cell: selects and arms the drag.
    let o = h.origin(PanelKey::Content, [1280.0, 720.0]);
    let cell = crate::editor::content_panel::cell_rect(o, 0);
    click_at(&mut world, &mut h, [cell[0] + 10.0, cell[1] + 10.0]);
    assert!(h.content_drag.is_some(), "the cell press arms a drag");
    assert_eq!(h.selection.active(), Some("demo_ball"));
    let before = h.entries.len();

    // Pull into the viewport: the ghost lands on the ground box below.
    set_input(&mut world, drag_input([640.0, 500.0], true));
    h.tick(&mut world);
    let pose = h
        .content_ghost_pose()
        .expect("the ghost has a landing point");
    assert!(pose[1].abs() < 1e-3, "landed on the ground top: {pose:?}");

    // Release commits one entry through the shared path.
    set_input(&mut world, drag_input([640.0, 500.0], false));
    h.tick(&mut world);
    assert!(h.content_drag.is_none());
    assert_eq!(h.entries.len(), before + 1);
    let placed = &h.entries[before];
    assert_eq!(entry_type(placed), Some("Prop"));
    assert_eq!(placed["args"]["mesh"], "demo_ball");
    assert!(
        placed["args"]["position"][1].as_f64().unwrap().abs() < 1e-3,
        "{placed}"
    );
    assert_eq!(h.selection.active(), entry_name(placed));
    assert!(h.dirty);
    h.undo(&mut world);
    assert_eq!(h.entries.len(), before, "one undo removes the placement");
}

// A press that never travels past the slop is just the selecting click.
#[test]
fn a_still_cell_press_places_nothing() {
    crate::ecs::asset_id::reset_interner();
    let mut world = pick_world([0.0; 3], vec![]);
    let mut h = hook(vec![serde_json::json!({
        "name": "demo_ball", "type": "ProceduralMesh", "args": { "generator": "box" }
    })]);
    h.content_open = true;
    h.tree_stale = true;
    h.refresh_tree_if_needed();
    let o = h.origin(PanelKey::Content, [1280.0, 720.0]);
    let cell = crate::editor::content_panel::cell_rect(o, 0);
    let at = [cell[0] + 10.0, cell[1] + 10.0];
    click_at(&mut world, &mut h, at);
    set_input(&mut world, drag_input(at, false));
    h.tick(&mut world);
    assert!(h.content_drag.is_none());
    assert_eq!(h.entries.len(), 1, "no placement from a plain click");
    assert!(!h.dirty);
    assert_eq!(h.selection.active(), Some("demo_ball"), "still selected");
}

// Dragging a Material onto a Prop assigns it instead of placing anything.
#[test]
fn material_drag_assigns_to_the_prop_under_the_cursor() {
    crate::ecs::asset_id::reset_interner();
    let crate_id = crate::ecs::asset_id::intern("crate_prop");
    let mut world = pick_world(
        [0.0; 3],
        vec![(crate_id, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0])],
    );
    let mut h = hook(vec![
        serde_json::json!({
            "name": "wood", "type": "Material", "args": { "roughness": 0.7 }
        }),
        serde_json::json!({
            "name": "crate_prop", "type": "Prop",
            "args": { "mesh": "demo", "position": [0.0, 0.0, -5.0] }
        }),
    ]);
    h.content_open = true;
    h.arm_content_drag("wood".to_string(), "Material".to_string(), [1000.0, 300.0]);

    set_input(&mut world, drag_input([640.0, 360.0], true));
    h.tick(&mut world);
    set_input(&mut world, drag_input([640.0, 360.0], false));
    h.tick(&mut world);
    assert_eq!(
        h.entries[1]["args"]["material"], "wood",
        "the hovered Prop gains the material"
    );
    assert_eq!(h.entries.len(), 2, "nothing was placed");
    assert!(h.dirty);
    h.undo(&mut world);
    assert!(h.entries[1]["args"].get("material").is_none());
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

// The row eye and lock are editor-session state: they flip the hook's sets (the
// hidden set publishing as ids each tick) and never touch the entries.
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
    h.apply_panel(PanelAction::ToggleLock(g, i), &mut world);
    assert!(h.hidden_assets.contains("box"));
    assert!(h.locked_assets.contains("box"));
    assert!(!h.dirty, "session toggles are not authored edits");

    h.tick(&mut world);
    let hidden = world
        .resource::<crate::ecs::EditorHidden>()
        .expect("the hook publishes the hidden set every tick");
    assert!(hidden.0.contains(&id), "names resolve to this world's ids");

    h.apply_panel(PanelAction::ToggleHide(g, i), &mut world);
    h.apply_panel(PanelAction::ToggleLock(g, i), &mut world);
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
    let mut world = World::new_empty();
    let mut h = hook(Vec::new());

    h.run_console_line(&mut world, "/add PhysicsConfig phys");
    assert_eq!(h.entries.len(), 1);
    assert_eq!(entry_name(&h.entries[0]), Some("phys"));
    assert_eq!(entry_type(&h.entries[0]), Some("PhysicsConfig"));
    assert!(h.dirty, "an added entry marks the world dirty");

    h.run_console_line(&mut world, "/del phys");
    assert!(h.entries.is_empty());

    h.run_console_line(&mut world, "/del ghost");
    h.run_console_line(&mut world, "just a note");
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

// ---------------------------------------------------------------------------
// Behavior panel

// An open Behavior panel over `entries`, with its value field injected.
fn behavior_session(entries: Vec<serde_json::Value>) -> (EditorHook, World) {
    let mut world = World::new_empty();
    for id in behavior_panel::all_field_ids() {
        world.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    let mut h = hook(entries);
    registry::panel(PanelKey::Behavior).toggle(&mut h, &mut world);
    (h, world)
}

fn behavior(name: &str, args: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"name": name, "type": "Behavior", "args": args})
}

// The outline row index of the first row with `label`.
fn behavior_row(h: &EditorHook, label: &str) -> usize {
    h.behavior_rows()
        .iter()
        .position(|r| r.label == label)
        .unwrap_or_else(|| panic!("no `{label}` row"))
}

fn select_behavior(h: &mut EditorHook, world: &mut World, label: &str) {
    let i = behavior_row(h, label);
    h.apply_behavior_action(BehaviorAction::Select(i), world, [0.0, 0.0]);
}

// The args of the open behavior, for asserting on what an action wrote.
fn open_args(h: &EditorHook) -> serde_json::Value {
    h.behavior_args()
}

// One press of the header's removal chip: the first arms it, the second carries
// it out.
fn press_remove(h: &mut EditorHook, world: &mut World) {
    h.apply_behavior_action(BehaviorAction::Remove, world, [0.0, 0.0]);
}

// Type into the name field, as the engine's text-input system would.
fn type_name(world: &mut World, text: &str) {
    widget::seed_field(world, behavior_panel::NAME_INPUT, text);
}

#[test]
fn behavior_panel_opens_on_the_first_behavior_and_steps_between_them() {
    let (mut h, mut world) = behavior_session(vec![
        entry("gfx", "GraphicsConfig"),
        behavior("greet", serde_json::json!({"on": "start"})),
        behavior("chase", serde_json::json!({"on": "tick"})),
    ]);
    let data = h.behavior_data();
    assert_eq!(
        (data.name.as_str(), data.index, data.total),
        ("greet", 0, 2)
    );

    h.apply_behavior_action(BehaviorAction::Step(1), &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_data().name, "chase");
    // Stepping past either end wraps rather than sticking.
    h.apply_behavior_action(BehaviorAction::Step(1), &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_data().name, "greet");
    h.apply_behavior_action(BehaviorAction::Step(-1), &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_data().name, "chase");
}

// A world with no behaviors is not a dead end: New appends one and opens it.
#[test]
fn behavior_new_appends_a_blank_behavior_and_opens_it() {
    let (mut h, mut world) = behavior_session(vec![entry("gfx", "GraphicsConfig")]);
    assert_eq!(h.behavior_data().total, 0);

    h.apply_behavior_action(BehaviorAction::New, &mut world, [0.0, 0.0]);
    let data = h.behavior_data();
    assert_eq!(data.total, 1);
    assert_eq!(data.index, 0);
    assert!(h.dirty && h.rebuild_preview, "adding one is a world edit");
    assert_eq!(open_args(&h), serde_json::json!({"on": "start", "do": []}));
    // A blank behavior still checks out, so the panel opens on a clean slate.
    assert!(matches!(h.behavior_status, Some(Status::Ok)));
}

// The status line is the world checker's own message, not a second opinion.
#[test]
fn behavior_status_reports_the_checkers_message() {
    let (h, _) = behavior_session(vec![behavior(
        "broken",
        serde_json::json!({"do": [{"despawn": {"target": {"bind": "nope"}}}]}),
    )]);
    let Some(Status::Error { message: e, .. }) = &h.behavior_status else {
        panic!("expected an error status, got {:?}", h.behavior_status);
    };
    assert!(e.contains("unbound name 'nope'"), "{e}");
    assert!(e.starts_with("Behavior 'broken'"), "{e}");
}

// The world-level checker is the one that runs, so a declared variable table is
// authoritative and a misspelled name is caught in the panel.
#[test]
fn behavior_status_enforces_the_declared_variable_table() {
    let vars = serde_json::json!({"name": "world_vars", "type": "Variables",
        "args": {"vars": [{"name": "health", "value": {"float": 100.0}}]}});
    let (h, _) = behavior_session(vec![
        vars,
        behavior(
            "heal",
            serde_json::json!({"do": [{"set": {"var": "helth", "value": {"float": 1.0}}}]}),
        ),
    ]);
    let Some(Status::Error { message: e, .. }) = &h.behavior_status else {
        panic!("expected an error status, got {:?}", h.behavior_status);
    };
    assert!(e.contains("undeclared variable 'helth'"), "{e}");
}

#[test]
fn behavior_picking_a_node_appends_it_and_refreshes_the_preview() {
    let (mut h, mut world) = behavior_session(vec![behavior("b", serde_json::json!({}))]);
    select_behavior(&mut h, &mut world, "do");
    h.apply_behavior_action(BehaviorAction::Palette, &mut world, [0.0, 0.0]);
    assert!(h.behavior_picking);

    let at = h
        .behavior_data()
        .picks
        .iter()
        .position(|p| p.verb == "hide")
        .expect("hide is offered");
    h.apply_behavior_action(BehaviorAction::Choose(at), &mut world, [0.0, 0.0]);
    assert!(!h.behavior_picking, "picking closes the palette");
    assert_eq!(
        open_args(&h)["do"],
        serde_json::json!([{"hide": {"target": "self"}}])
    );
    assert!(
        h.rebuild_preview,
        "the edited body runs in the live world straight away"
    );
    // A world-scoped `self` is exactly what the checker objects to, and it says so.
    let Some(Status::Error { message: e, .. }) = &h.behavior_status else {
        panic!("expected the scope error");
    };
    assert!(e.contains("`self` needs a `scope`"), "{e}");
}

// Selecting a row never changes it, however small the field: the source and a
// flag both stand still until something is picked for them.
#[test]
fn selecting_a_behavior_row_leaves_its_value_alone() {
    let before = serde_json::json!({"on": "start", "once": false});
    let (mut h, mut world) = behavior_session(vec![behavior("b", before.clone())]);
    for label in ["on", "once"] {
        select_behavior(&mut h, &mut world, label);
        assert_eq!(open_args(&h), before, "selecting `{label}` edited it");
    }
    assert!(!h.rebuild_preview, "and nothing was committed");
}

// The same fields reach their options through the palette instead.
#[test]
fn behavior_fixed_fields_are_set_from_the_palette() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "b",
        serde_json::json!({"on": "start", "once": false}),
    )]);
    for (label, verb, want) in [
        ("on", "tick", serde_json::json!("tick")),
        ("once", "true", serde_json::json!(true)),
    ] {
        select_behavior(&mut h, &mut world, label);
        h.apply_behavior_action(BehaviorAction::Palette, &mut world, [0.0, 0.0]);
        let at = h
            .behavior_data()
            .picks
            .iter()
            .position(|p| p.verb == verb)
            .unwrap_or_else(|| panic!("`{label}` offers `{verb}`"));
        h.apply_behavior_action(BehaviorAction::Choose(at), &mut world, [0.0, 0.0]);
        assert_eq!(open_args(&h)[label], want);
    }
}

#[test]
fn behavior_value_field_commits_on_enter_and_reports_a_bad_value() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "b",
        serde_json::json!({"delay": 0.0, "do": []}),
    )]);
    select_behavior(&mut h, &mut world, "delay");
    assert!(h.behavior_focus, "a typed row is ready to type into");

    widget::seed_field(&mut world, behavior_panel::VALUE_INPUT, "2.5");
    h.behavior_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(open_args(&h)["delay"], serde_json::json!(2.5));

    widget::seed_field(&mut world, behavior_panel::VALUE_INPUT, "soon");
    h.behavior_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(
        open_args(&h)["delay"],
        serde_json::json!(2.5),
        "a rejected value leaves the old one standing"
    );
    let Some(Status::Error { message: e, .. }) = &h.behavior_status else {
        panic!("expected a parse error");
    };
    assert!(e.contains("'soon' is not a number"), "{e}");
}

#[test]
fn behavior_delete_and_move_act_on_the_selected_member() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "b",
        serde_json::json!({"scope": ["Prop"], "do": [
            {"save": null}, {"hide": {"target": "self"}}]}),
    )]);
    select_behavior(&mut h, &mut world, "hide");
    h.apply_behavior_action(BehaviorAction::Move(-1), &mut world, [0.0, 0.0]);
    assert!(open_args(&h)["do"][0].get("hide").is_some());
    assert_eq!(
        h.behavior_row,
        Some(behavior_row(&h, "hide")),
        "the selection follows the node it moved"
    );

    h.apply_behavior_action(BehaviorAction::Delete, &mut world, [0.0, 0.0]);
    assert_eq!(open_args(&h)["do"].as_array().unwrap().len(), 1);
    assert!(
        h.behavior_row.is_none(),
        "the removed row's selection is dropped, not retargeted"
    );
}

// The toolbar's Del takes out a node; the header's Remove takes out the whole
// behavior. Removing one leaves the body of the others alone.
#[test]
fn behavior_remove_takes_the_open_behavior_not_a_node() {
    let body = serde_json::json!({"scope": ["Prop"], "do": [{"hide": {"target": "self"}}]});
    let (mut h, mut world) = behavior_session(vec![
        behavior("greet", body.clone()),
        behavior("chase", body.clone()),
    ]);
    select_behavior(&mut h, &mut world, "hide");

    press_remove(&mut h, &mut world);
    press_remove(&mut h, &mut world);
    assert_eq!(h.behavior_data().total, 1);
    assert_eq!(h.behavior_data().name, "chase");
    assert_eq!(open_args(&h), body, "the survivor's body is untouched");
    assert!(h.dirty && h.rebuild_preview, "removing one is a world edit");
    assert!(
        h.entries.iter().all(|e| entry_name(e) != Some("greet")),
        "the authored line is gone"
    );
}

// Destroying an authored asset takes two presses, and anything else the user
// does on the panel in between calls it off.
#[test]
fn behavior_remove_arms_first_and_any_other_press_cancels() {
    let (mut h, mut world) = behavior_session(vec![behavior("greet", serde_json::json!({}))]);
    press_remove(&mut h, &mut world);
    assert!(h.behavior_remove_armed, "the first press only arms");
    assert_eq!(h.behavior_data().total, 1, "and destroys nothing");

    h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    assert!(!h.behavior_remove_armed, "another press disarms it");
    press_remove(&mut h, &mut world);
    assert_eq!(
        h.behavior_data().total,
        1,
        "so the next press arms again rather than committing"
    );
    press_remove(&mut h, &mut world);
    assert_eq!(h.behavior_data().total, 0);
}

// The ordinal is clamped to what is left, so the panel always opens on a real
// behavior -- and on the empty-world prompt once the last one goes.
#[test]
fn behavior_remove_reopens_whatever_holds_that_ordinal() {
    let (mut h, mut world) = behavior_session(vec![
        behavior("a", serde_json::json!({})),
        behavior("b", serde_json::json!({})),
        behavior("c", serde_json::json!({})),
    ]);
    h.apply_behavior_action(BehaviorAction::Step(2), &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_data().name, "c");

    // The last of the three: the ordinal has to come back a place.
    press_remove(&mut h, &mut world);
    press_remove(&mut h, &mut world);
    let data = h.behavior_data();
    assert_eq!((data.name.as_str(), data.index, data.total), ("b", 1, 2));

    // Emptying the world leaves the prompt, not a dangling open behavior.
    for _ in 0..2 {
        press_remove(&mut h, &mut world);
        press_remove(&mut h, &mut world);
    }
    let data = h.behavior_data();
    assert_eq!((data.name.as_str(), data.total), ("", 0));
    assert!(h.behavior_status.is_none(), "and nothing to check");
}

// Removal is an ordinary entry edit, so the history covers it: the two-press
// arm guards the click, and Undo is still there behind it.
#[test]
fn behavior_remove_is_undoable() {
    let args = serde_json::json!({"on": "tick", "do": []});
    let (mut h, mut world) = behavior_session(vec![behavior("greet", args.clone())]);
    press_remove(&mut h, &mut world);
    press_remove(&mut h, &mut world);
    assert_eq!(h.behavior_data().total, 0);

    h.undo(&mut world);
    assert_eq!(h.behavior_data().total, 1);
    assert_eq!(h.behavior_data().name, "greet");
    assert_eq!(open_args(&h), args, "body and all");
}

#[test]
fn behavior_rename_commits_on_enter() {
    let (mut h, mut world) = behavior_session(vec![
        behavior("greet", serde_json::json!({})),
        behavior("chase", serde_json::json!({})),
    ]);
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    assert!(h.behavior_name_focus);
    assert_eq!(
        widget::field_text(&world, behavior_panel::NAME_INPUT),
        "greet",
        "the field opens on the name it is about to replace"
    );

    type_name(&mut world, "  welcome  ");
    h.behavior_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(h.behavior_data().name, "welcome", "trimmed on the way in");
    assert!(!h.behavior_name_focus, "committing gives up the keyboard");
    assert!(h.dirty && h.rebuild_preview, "renaming is a world edit");
    // The ordinal is untouched: renaming does not reorder the world.
    assert_eq!(h.behavior_data().index, 0);
}

// Two assets cannot share a name, so a taken one is suffixed until it is free
// and the field is put back in step with what actually landed.
#[test]
fn behavior_rename_keeps_the_name_unique() {
    let (mut h, mut world) = behavior_session(vec![
        behavior("greet", serde_json::json!({})),
        behavior("chase", serde_json::json!({})),
    ]);
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    type_name(&mut world, "chase");
    h.behavior_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(h.behavior_data().name, "chase_1");
    assert_eq!(
        widget::field_text(&world, behavior_panel::NAME_INPUT),
        "chase_1",
        "the field shows what the world holds, not what was typed"
    );

    // Committing a name unchanged is not a collision with itself.
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    h.behavior_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(h.behavior_data().name, "chase_1");
}

#[test]
fn behavior_rename_refuses_a_blank_name() {
    let (mut h, mut world) = behavior_session(vec![behavior("greet", serde_json::json!({}))]);
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    type_name(&mut world, "   ");
    h.behavior_keys(&mut world, &story_key_input(crate::assets::Key::Enter));

    assert_eq!(h.behavior_data().name, "greet", "nothing was written");
    assert!(!h.dirty, "and no edit was recorded");
    let Some(Status::Error { message: e, .. }) = &h.behavior_status else {
        panic!("expected the panel to say why, got {:?}", h.behavior_status);
    };
    assert!(e.contains("needs a name"), "{e}");
    assert_eq!(
        widget::field_text(&world, behavior_panel::NAME_INPUT),
        "greet",
        "the refused text is dropped rather than left to be committed later"
    );
}

// The checker quotes the behavior by name, so its verdict is re-read under the
// new one rather than left complaining about an asset the world no longer has.
#[test]
fn behavior_rename_reruns_the_checker_under_the_new_name() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "broken",
        serde_json::json!({"do": [{"despawn": {"target": {"bind": "nope"}}}]}),
    )]);
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    type_name(&mut world, "still_broken");
    h.behavior_keys(&mut world, &story_key_input(crate::assets::Key::Enter));

    let Some(Status::Error { message: e, .. }) = &h.behavior_status else {
        panic!("expected the error to survive the rename");
    };
    assert!(e.starts_with("Behavior 'still_broken'"), "{e}");
}

// Clicking away from a half-typed name throws it away rather than leaving it in
// the field for the next Enter to commit by surprise.
#[test]
fn behavior_name_reverts_when_it_loses_focus() {
    let (mut h, mut world) = behavior_session(vec![behavior("greet", serde_json::json!({}))]);
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    type_name(&mut world, "half typed");

    h.apply_behavior_action(BehaviorAction::Consume, &mut world, [0.0, 0.0]);
    assert!(!h.behavior_name_focus);
    assert_eq!(
        widget::field_text(&world, behavior_panel::NAME_INPUT),
        "greet"
    );
    h.behavior_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(h.behavior_data().name, "greet");
    assert!(!h.dirty);
}

// The two fields never hold the keyboard at once: whichever was pressed last
// owns it, so Enter always commits the field the user is looking at.
#[test]
fn behavior_name_and_value_fields_do_not_share_the_keyboard() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "b",
        serde_json::json!({"delay": 0.0, "do": []}),
    )]);
    select_behavior(&mut h, &mut world, "delay");
    assert!(h.behavior_focus && !h.behavior_name_focus);

    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    assert!(h.behavior_name_focus && !h.behavior_focus);

    h.apply_behavior_action(BehaviorAction::FocusValue, &mut world, [0.0, 0.0]);
    assert!(h.behavior_focus && !h.behavior_name_focus);
}

// The value field's contents survive the live-preview rebuild an edit triggers,
// so a half-typed value is not blanked out from under the user.
#[test]
fn behavior_value_field_is_carried_across_a_preview_rebuild() {
    let (_, mut world) = behavior_session(vec![behavior("b", serde_json::json!({}))]);
    widget::seed_field(&mut world, behavior_panel::VALUE_INPUT, "half typed");
    let snapshot = EditorHook::field_snapshot(&world);
    let mut fresh = World::new_empty();
    for id in behavior_panel::all_field_ids() {
        fresh.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    EditorHook::restore_fields(&mut fresh, &snapshot);
    assert_eq!(
        widget::field_text(&fresh, behavior_panel::VALUE_INPUT),
        "half typed"
    );
}

// The chart is a second view over the same rows, so switching to it keeps the
// selection and the toolbar keeps acting on the same node.
#[test]
fn behavior_view_cycles_through_the_three_views() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"hide": {"target": "self"}}]}),
    )]);
    assert_eq!(h.behavior_mode, ViewMode::Outline);
    select_behavior(&mut h, &mut world, "hide");
    let selected = h.behavior_row;

    for want in [ViewMode::Chart, ViewMode::Overview, ViewMode::Outline] {
        h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
        assert_eq!(h.behavior_mode, want);
        assert_eq!(
            h.behavior_row, selected,
            "the selection survives every switch"
        );
    }
}

// Clicking a card is clicking its row: the palette a card opens is the one its
// outline row offers, so there is no second editing path to keep in step.
#[test]
fn behavior_card_selects_the_row_it_stands_for() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [
            {"let": {"name": "t", "value": {"first": "q"}}},
            {"hide": {"target": "self"}},
        ]}),
    )]);
    h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    let card = h
        .behavior_data()
        .chart
        .cards
        .iter()
        .position(|c| c.title == "hide")
        .unwrap();

    h.apply_behavior_action(BehaviorAction::SelectCard(card), &mut world, [0.0, 0.0]);
    let rows = h.behavior_rows();
    assert_eq!(rows[h.behavior_row.unwrap()].label, "hide");
    // And the palette that selection offers is the node palette, so picking
    // from a card replaces the node the card draws.
    assert!(
        h.behavior_data()
            .picks
            .iter()
            .any(|p| p.verb == "set_transform"),
    );
}

// The overview maps the whole world, and clicking a behavior on it opens that
// behavior -- which is what makes the map an index rather than a picture.
#[test]
fn behavior_overview_opens_the_behavior_a_card_stands_for() {
    let (mut h, mut world) = behavior_session(vec![
        behavior(
            "award",
            serde_json::json!({"on": "start",
                "do": [{"set": {"var": "score", "value": {"int": 1}}}]}),
        ),
        behavior(
            "react",
            serde_json::json!({"on": {"variable": "score"}, "do": []}),
        ),
    ]);
    for _ in 0..2 {
        h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    }
    assert_eq!(h.behavior_mode, ViewMode::Overview);

    let overview = h.behavior_data().overview;
    let card = overview
        .cards
        .iter()
        .position(|c| c.title == "react")
        .expect("react is on the map");
    assert!(
        overview.cards.iter().any(|c| c.title == "score"),
        "the variable joining them is a card of its own"
    );

    h.apply_behavior_action(BehaviorAction::OpenCard(card), &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_index, 1);
    assert_eq!(h.behavior_data().name, "react");
    assert_eq!(
        h.behavior_mode,
        ViewMode::Chart,
        "and lands on the body it named"
    );
}

// The map is only built while it is showing: it walks every behavior in the
// world, which the other two views have no use for.
#[test]
fn behavior_overview_is_built_only_while_it_shows() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "a",
        serde_json::json!({"on": "start", "do": []}),
    )]);
    assert!(h.behavior_data().overview.cards.is_empty());
    for _ in 0..2 {
        h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    }
    assert!(!h.behavior_data().overview.cards.is_empty());
}

// The chart can grow a body it did not start empty. Appending goes through the
// card at the end of the chain, so a second node can be added without leaving
// for the outline -- which is what the chart could not do at all before.
#[test]
fn behavior_chart_appends_to_a_body_that_already_has_nodes() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [{"save": {}}]}),
    )]);
    h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    let tail = h
        .behavior_data()
        .chart
        .cards
        .iter()
        .position(|c| c.kind == CardKind::Add && c.path == [path::field("do")])
        .expect("the body's chain ends in a card that appends to it");

    h.apply_behavior_action(BehaviorAction::SelectCard(tail), &mut world, [0.0, 0.0]);
    let pick = h
        .behavior_data()
        .picks
        .iter()
        .position(|p| p.verb == "hide")
        .expect("it offers the node palette");
    h.apply_behavior_action(BehaviorAction::Choose(pick), &mut world, [0.0, 0.0]);

    assert_eq!(
        open_args(&h)["do"],
        serde_json::json!([{"save": {}}, {"hide": {"target": "self"}}]),
        "the node was appended after the one already there"
    );
    assert!(h.rebuild_preview, "and the live world has it");
}

// The settings a behavior declares once hang off no node, so nothing in the
// chart reached them: the trigger card settles them instead.
#[test]
fn behavior_chart_reaches_the_settings_the_behavior_declares() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "scope": ["Prop"], "do": [{"save": {}}]}),
    )]);
    h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    h.apply_behavior_action(BehaviorAction::SelectCard(0), &mut world, [0.0, 0.0]);

    let data = h.behavior_data();
    let listed: Vec<&str> = data
        .fields
        .iter()
        .map(|&i| data.rows[i].label.as_str())
        .collect();
    for want in ["on", "once", "delay", "cooldown", "scope"] {
        assert!(listed.contains(&want), "{want} in {listed:?}");
    }

    // And they are editable there, not just visible.
    let once = data.fields[listed.iter().position(|l| *l == "once").unwrap()];
    h.apply_behavior_action(BehaviorAction::Select(once), &mut world, [0.0, 0.0]);
    h.apply_behavior_action(BehaviorAction::Palette, &mut world, [0.0, 0.0]);
    let pick = h
        .behavior_data()
        .picks
        .iter()
        .position(|p| p.verb == "true")
        .expect("a flag offers its two options");
    h.apply_behavior_action(BehaviorAction::Choose(pick), &mut world, [0.0, 0.0]);
    assert_eq!(open_args(&h)["once"], serde_json::json!(true));
}

// Selecting a card lists that node's own settings, and picking one of them
// keeps the same node in the inspector rather than emptying it.
#[test]
fn behavior_inspector_holds_the_node_while_its_fields_are_selected() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "drip",
        serde_json::json!({"on": "start", "do": [
            {"spawn": {"template": "drop", "lifetime": 4.0}},
        ]}),
    )]);
    h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    let card = h
        .behavior_data()
        .chart
        .cards
        .iter()
        .position(|c| c.title == "spawn")
        .unwrap();
    h.apply_behavior_action(BehaviorAction::SelectCard(card), &mut world, [0.0, 0.0]);

    let data = h.behavior_data();
    assert_eq!(data.card, Some(card));
    let listed: Vec<&str> = data
        .fields
        .iter()
        .map(|&i| data.rows[i].label.as_str())
        .collect();
    assert!(listed.contains(&"lifetime"), "{listed:?}");

    // Selecting one of those settings holds the node it belongs to.
    let lifetime = data.fields[listed.iter().position(|l| *l == "lifetime").unwrap()];
    h.apply_behavior_action(BehaviorAction::Select(lifetime), &mut world, [0.0, 0.0]);
    let after = h.behavior_data();
    assert_eq!(after.card, Some(card), "the inspector holds the node");
    assert_eq!(after.fields, data.fields, "and lists the same settings");
    assert!(h.behavior_focus, "a typed setting is ready to type into");
}

// A node's own field is its own; the nodes nested inside it are cards of their
// own, so the inspector never doubles as a second way into the body.
#[test]
fn behavior_inspector_stops_at_the_nodes_a_branch_holds() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "gate",
        serde_json::json!({"on": "start", "do": [
            {"if": {"cond": {"bool": true}, "then": [{"save": null}], "else": []}},
        ]}),
    )]);
    h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    let card = h
        .behavior_data()
        .chart
        .cards
        .iter()
        .position(|c| c.title == "if")
        .unwrap();
    h.apply_behavior_action(BehaviorAction::SelectCard(card), &mut world, [0.0, 0.0]);
    let data = h.behavior_data();
    let listed: Vec<&str> = data
        .fields
        .iter()
        .map(|&i| data.rows[i].label.as_str())
        .collect();
    assert!(listed.contains(&"cond"), "{listed:?}");
    assert!(!listed.contains(&"save"), "{listed:?}");
}

// An empty branch is a card too, and picking from it appends the branch's first
// node -- the reason an empty `else` is drawn at all.
#[test]
fn behavior_empty_branch_card_appends_into_that_branch() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "gate",
        serde_json::json!({"on": "start", "do": [
            {"if": {"cond": {"bool": true}, "then": [{"save": {}}]}},
        ]}),
    )]);
    h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    // The card that appends into the (empty) `else`.
    let card = h
        .behavior_data()
        .chart
        .cards
        .iter()
        .position(|c| c.kind == CardKind::Add && c.path.last() == Some(&path::field("else")))
        .unwrap();
    h.apply_behavior_action(BehaviorAction::SelectCard(card), &mut world, [0.0, 0.0]);
    let pick = h
        .behavior_data()
        .picks
        .iter()
        .position(|p| p.verb == "hide")
        .unwrap();
    h.apply_behavior_action(BehaviorAction::Choose(pick), &mut world, [0.0, 0.0]);

    let body = open_args(&h);
    assert_eq!(
        body["do"][0]["if"]["else"][0]["hide"],
        serde_json::json!({"target": "self"}),
        "{body}",
    );
}

// In chart view the wheel pans the canvas instead of scrolling the outline, and
// stops at the chart's edge rather than running into empty space.
#[test]
fn behavior_wheel_pans_the_chart_within_its_extent() {
    let body: Vec<serde_json::Value> = (0..8).map(|_| serde_json::json!({"save": {}})).collect();
    let (mut h, mut world) = behavior_session(vec![behavior(
        "long",
        serde_json::json!({"on": "start", "do": body}),
    )]);
    h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    h.scroll_behavior(1.0);
    // This body is one row tall and wider than the canvas, so the wheel moves
    // along the axis that has room.
    assert!(h.behavior_pan[0] > 0.0, "{:?}", h.behavior_pan);
    assert_eq!(h.behavior_scroll, 0, "the outline's scroll is untouched");

    for _ in 0..200 {
        h.scroll_behavior(1.0);
    }
    let chart = h.behavior_data().chart;
    let canvas =
        behavior_panel::chart_canvas(h.effective_size(PanelKey::Behavior), ViewMode::Chart);
    assert_eq!(
        h.behavior_pan,
        crate::editor::behavior_chart::clamp_pan(h.behavior_pan, &chart, canvas)
    );
}

// Selecting a node off the right of the canvas brings its card into view, so
// stepping through a long body never leaves the selection off screen.
#[test]
fn behavior_selection_pans_an_off_canvas_card_into_view() {
    let body: Vec<serde_json::Value> = (0..12).map(|_| serde_json::json!({"save": {}})).collect();
    let (mut h, mut world) = behavior_session(vec![behavior(
        "long",
        serde_json::json!({"on": "start", "do": body}),
    )]);
    h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    // The last node, not the tail card past it: only a list member moves.
    let last = h
        .behavior_data()
        .chart
        .cards
        .iter()
        .rposition(|c| c.kind == CardKind::Node)
        .unwrap();
    h.apply_behavior_action(BehaviorAction::SelectCard(last), &mut world, [0.0, 0.0]);
    // Moving it earlier follows the node, which is what re-pans the canvas.
    h.apply_behavior_action(BehaviorAction::Move(-1), &mut world, [0.0, 0.0]);

    let data = h.behavior_data();
    let path = &data.rows[h.behavior_row.unwrap()].path;
    let card = data.chart.cards.iter().find(|c| &c.path == path).unwrap();
    let band = behavior_panel::chart_band(
        [0.0, 0.0],
        h.effective_size(PanelKey::Behavior),
        ViewMode::Chart,
    );
    let rect = crate::editor::behavior_chart::card_rect(card, band, h.behavior_pan);
    assert!(rect[0] >= band[0], "{rect:?} left of {band:?}");
    assert!(rect[0] + rect[2] <= band[0] + band[2] + 0.01, "{rect:?}");
}

// An open overlay draws above its own panel but still below the panel in front
// of it. This is what lets an opaque backing occlude what it covers, instead of
// each panel blanking the elements it happens to sit over.
#[test]
fn an_open_overlay_layers_above_its_panel_and_below_the_one_in_front() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"save": {}}]}),
    )]);
    // Nothing open: the palette sits at its panel's own layer.
    let flat = h.compute_layers();
    let panel_layer = flat[&behavior_panel::PANEL_BG];
    assert_eq!(flat[&behavior_panel::DROP_BG], panel_layer);

    select_behavior(&mut h, &mut world, "do");
    h.apply_behavior_action(BehaviorAction::Palette, &mut world, [0.0, 0.0]);
    let open = h.compute_layers();
    for id in behavior_panel::palette_ids() {
        assert!(
            open[&id] > open[&behavior_panel::PANEL_BG],
            "{id:?} does not draw above its own panel",
        );
        assert!(
            open[&id] > open[&behavior_panel::row_label(0)],
            "{id:?} does not draw above the rows it covers",
        );
    }
    // A panel focused in front still clears the whole band, overlay included.
    h.focus_panel(PanelKey::Preview);
    let stacked = h.compute_layers();
    let front = stacked[&preview::PANEL_BG];
    for id in behavior_panel::palette_ids() {
        assert!(stacked[&id] < front, "{id:?} escaped above the front panel");
    }
}

// Escape arrives as its own one-frame pulse rather than as a `Key`.
fn behavior_escape_input() -> FrameInput {
    FrameInput {
        escape: true,
        viewport: [1280.0, 720.0],
        ..Default::default()
    }
}

fn press_behavior_key(h: &mut EditorHook, world: &mut World, key: crate::assets::Key) {
    h.behavior_keys(world, &story_key_input(key));
}

// The title of the card the chart's selection belongs to.
fn selected_card_title(h: &EditorHook) -> Option<String> {
    let data = h.behavior_data();
    data.card
        .and_then(|i| data.chart.cards.get(i))
        .map(|c| c.title.clone())
}

fn selected_overview_title(h: &EditorHook) -> Option<String> {
    let data = h.behavior_data();
    h.behavior_overview_card
        .and_then(|i| data.overview.cards.get(i))
        .map(|c| c.title.clone())
}

// The outline is a list, so a step is one row. With nothing selected the first
// press starts from the end it comes from, and neither end wraps.
#[test]
fn behavior_arrows_step_the_outline_one_row_at_a_time() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"save": {}}, {"hide": {"target": "self"}}]}),
    )]);
    assert_eq!(h.behavior_row, None);

    press_behavior_key(&mut h, &mut world, crate::assets::Key::Down);
    assert_eq!(h.behavior_row, Some(0));
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Down);
    assert_eq!(h.behavior_row, Some(1));
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Up);
    assert_eq!(h.behavior_row, Some(0));
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Up);
    assert_eq!(h.behavior_row, Some(0), "the top of the list does not wrap");

    // Left and Right have nothing to follow in a list.
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Right);
    assert_eq!(h.behavior_row, Some(0));
}

// Stepping past the window scrolls it, so the selection is never off screen.
#[test]
fn behavior_arrows_scroll_the_outline_to_keep_the_selection_showing() {
    let body: Vec<serde_json::Value> = (0..30).map(|_| serde_json::json!({"save": {}})).collect();
    let (mut h, mut world) = behavior_session(vec![behavior(
        "long",
        serde_json::json!({"on": "start", "do": body}),
    )]);
    for _ in 0..25 {
        press_behavior_key(&mut h, &mut world, crate::assets::Key::Down);
    }
    let row = h.behavior_row.expect("a row is selected");
    assert_eq!(row, 24);
    assert!(h.behavior_scroll > 0, "the window followed the selection");
    assert!(row >= h.behavior_scroll, "row {row} is above the window");
}

// The chart is spatial: a sideways step follows the chain into a branch, and a
// vertical one crosses between the branches stacked under a branching node.
#[test]
fn behavior_arrows_follow_the_chart_chain_and_cross_its_branches() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [
            {"if": {
                "cond": {"bool": true},
                "then": [{"show": {"target": "self"}}],
                "else": [{"hide": {"target": "self"}}],
            }},
            {"save": {}},
        ]}),
    )]);
    h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_mode, ViewMode::Chart);

    // With nothing selected the chart starts at its first card, the trigger.
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Right);
    assert_eq!(selected_card_title(&h).as_deref(), Some("on tick"));
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Right);
    assert_eq!(selected_card_title(&h).as_deref(), Some("if"));
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Right);
    assert_eq!(selected_card_title(&h).as_deref(), Some("show"));

    press_behavior_key(&mut h, &mut world, crate::assets::Key::Down);
    assert_eq!(selected_card_title(&h).as_deref(), Some("hide"));
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Up);
    assert_eq!(selected_card_title(&h).as_deref(), Some("show"));
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Left);
    assert_eq!(selected_card_title(&h).as_deref(), Some("if"));
}

// The map opens on the behavior that was showing, steps between its cards, and
// Enter opens the behavior the card it lands on stands for.
#[test]
fn behavior_arrows_step_the_overview_and_enter_opens_a_behavior() {
    let (mut h, mut world) = behavior_session(vec![
        behavior(
            "award",
            serde_json::json!({"on": "start",
                "do": [{"set": {"var": "score", "value": {"int": 1}}}]}),
        ),
        behavior(
            "react",
            serde_json::json!({"on": {"variable": "score"}, "do": []}),
        ),
    ]);
    for _ in 0..2 {
        h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    }
    assert_eq!(h.behavior_mode, ViewMode::Overview);
    assert_eq!(
        selected_overview_title(&h).as_deref(),
        Some("award"),
        "the map opens on the behavior that was showing"
    );

    press_behavior_key(&mut h, &mut world, crate::assets::Key::Right);
    assert_eq!(selected_overview_title(&h).as_deref(), Some("score"));
    // A variable card stands for no behavior, so Enter leaves the map alone.
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    assert_eq!(h.behavior_mode, ViewMode::Overview);

    press_behavior_key(&mut h, &mut world, crate::assets::Key::Right);
    assert_eq!(selected_overview_title(&h).as_deref(), Some("react"));
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    assert_eq!(h.behavior_index, 1);
    assert_eq!(h.behavior_data().name, "react");
    assert_eq!(h.behavior_mode, ViewMode::Chart);
}

// With no field focused, Enter opens the selected row's palette; the palette
// then takes the arrows, and Enter inserts what it is highlighting.
#[test]
fn behavior_enter_opens_the_palette_and_its_arrows_pick_from_it() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    select_behavior(&mut h, &mut world, "do");
    assert!(!h.behavior_focus, "a list row takes no typed value");

    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    assert!(h.behavior_picking, "Enter opened the palette");
    assert_eq!(h.behavior_pick, 0);

    let second = h.behavior_data().picks[1].verb;
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Down);
    assert_eq!(h.behavior_pick, 1);
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);

    assert!(!h.behavior_picking, "picking closed the palette");
    let body = open_args(&h)["do"].clone();
    assert!(
        body[0].get(second).is_some(),
        "the highlighted option is the one that landed: {body:?}"
    );
}

// A row offering nothing has no palette, so Enter is left alone rather than
// arming one that never shows.
#[test]
fn behavior_enter_on_a_row_with_no_options_opens_nothing() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"let": {"name": "t", "value": {"int": 1}}}]}),
    )]);
    select_behavior(&mut h, &mut world, "name");
    assert!(h.behavior_data().picks.is_empty());
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    assert!(!h.behavior_picking);
}

// The highlight brings itself into the window, so a vocabulary longer than the
// palette shows is still reachable a press at a time.
#[test]
fn behavior_palette_highlight_scrolls_itself_into_the_window() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    select_behavior(&mut h, &mut world, "do");
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    let total = h.behavior_data().picks.len();
    assert!(
        total > behavior_panel::PICK_POOL,
        "the node vocabulary overflows the palette"
    );

    for _ in 0..behavior_panel::PICK_POOL {
        press_behavior_key(&mut h, &mut world, crate::assets::Key::Down);
    }
    assert_eq!(h.behavior_pick, behavior_panel::PICK_POOL);
    assert!(h.behavior_pick_scroll > 0, "the window followed it down");
    assert!(h.behavior_pick >= h.behavior_pick_scroll);
    assert!(h.behavior_pick < h.behavior_pick_scroll + behavior_panel::PICK_POOL);

    // And back up again, dragging the window with it.
    for _ in 0..behavior_panel::PICK_POOL {
        press_behavior_key(&mut h, &mut world, crate::assets::Key::Up);
    }
    assert_eq!(h.behavior_pick, 0);
    assert_eq!(h.behavior_pick_scroll, 0);
}

// Escape answers whichever state is waiting on a press, most consequential
// first: the open palette, then an armed removal, then the focused field.
#[test]
fn behavior_escape_clears_one_waiting_state_at_a_time() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"let": {"name": "t", "value": {"int": 1}}}]}),
    )]);
    select_behavior(&mut h, &mut world, "do");
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    assert!(h.behavior_picking);
    h.behavior_keys(&mut world, &behavior_escape_input());
    assert!(!h.behavior_picking, "the palette closed without picking");
    assert_eq!(open_args(&h)["do"].as_array().map(Vec::len), Some(1));

    press_remove(&mut h, &mut world);
    assert!(h.behavior_remove_armed);
    h.behavior_keys(&mut world, &behavior_escape_input());
    assert!(!h.behavior_remove_armed, "the armed removal was cancelled");
    assert_eq!(h.behavior_entries().len(), 1);

    select_behavior(&mut h, &mut world, "name");
    assert!(h.behavior_focus, "a text row takes the value field");
    h.behavior_keys(&mut world, &behavior_escape_input());
    assert!(!h.behavior_focus, "the value field gave the keyboard up");
}

// Escape gives the name field up without committing, reverting what was typed
// rather than leaving it to be committed by a later Enter.
#[test]
fn behavior_escape_reverts_an_abandoned_rename() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    type_name(&mut world, "half typed");
    h.behavior_keys(&mut world, &behavior_escape_input());

    assert!(!h.behavior_name_focus);
    assert_eq!(h.behavior_data().name, "chase");
    assert_eq!(
        widget::field_text(&world, behavior_panel::NAME_INPUT),
        "chase",
    );
}

// Left and Right are the caret's while the value field holds the keyboard, so
// they never also move the selection; Up and Down are free to.
#[test]
fn behavior_horizontal_keys_stay_with_the_caret_while_a_value_is_focused() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"let": {"name": "t", "value": {"int": 1}}}]}),
    )]);
    select_behavior(&mut h, &mut world, "name");
    assert!(h.behavior_focus);
    let row = h.behavior_row;

    for key in [crate::assets::Key::Left, crate::assets::Key::Right] {
        press_behavior_key(&mut h, &mut world, key);
        assert_eq!(h.behavior_row, row, "{key:?} moved the selection");
    }
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Down);
    assert_ne!(h.behavior_row, row, "Down still steps the outline");
}

// The name field is the asset's rather than the selection's, so it holds the
// arrows until Enter or Escape gives it up.
#[test]
fn behavior_name_field_holds_the_arrows_until_it_is_given_up() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"save": {}}]}),
    )]);
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Down);
    assert_eq!(h.behavior_row, None, "the arrows did not reach the outline");

    h.behavior_keys(&mut world, &behavior_escape_input());
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Down);
    assert_eq!(h.behavior_row, Some(0));
}

// Tab walks the same three-view cycle the header's button does, and the
// selection survives it, because all three views are over the one asset.
#[test]
fn behavior_tab_cycles_through_the_three_views() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"hide": {"target": "self"}}]}),
    )]);
    select_behavior(&mut h, &mut world, "hide");
    let selected = h.behavior_row;
    assert_eq!(h.behavior_mode, ViewMode::Outline);

    for want in [ViewMode::Chart, ViewMode::Overview, ViewMode::Outline] {
        press_behavior_key(&mut h, &mut world, crate::assets::Key::Tab);
        assert_eq!(h.behavior_mode, want);
        assert_eq!(
            h.behavior_row, selected,
            "the selection survives the switch"
        );
    }
}

// A half-typed rename is not a view switch: the name field holds Tab the same
// way it holds the arrows, until Enter or Escape gives the keyboard up.
#[test]
fn behavior_tab_leaves_the_view_alone_while_the_name_field_is_focused() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    type_name(&mut world, "half typed");
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Tab);
    assert_eq!(h.behavior_mode, ViewMode::Outline);
    assert!(h.behavior_name_focus, "the field kept the keyboard");

    h.behavior_keys(&mut world, &behavior_escape_input());
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Tab);
    assert_eq!(h.behavior_mode, ViewMode::Chart);
}

// The open palette is modal, so Tab neither switches the view out from under it
// nor picks anything.
#[test]
fn behavior_tab_does_nothing_while_the_palette_is_open() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    select_behavior(&mut h, &mut world, "do");
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    assert!(h.behavior_picking);

    press_behavior_key(&mut h, &mut world, crate::assets::Key::Tab);
    assert_eq!(h.behavior_mode, ViewMode::Outline);
    assert!(h.behavior_picking, "the palette is still up");
    assert_eq!(open_args(&h)["do"].as_array().map(Vec::len), Some(0));
}

// The checker says where, so the panel can point at it. A complaint about a
// field lands on that field's row rather than leaving the author to find it.
#[test]
fn behavior_status_points_at_the_row_the_checker_named() {
    let (h, _world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [
            {"save": {}},
            {"hide": {"target": {"int": 1}}},
        ]}),
    )]);
    let data = h.behavior_data();
    let view = h.make_behavior_view(&data, [0.0, 0.0]);
    assert!(
        view.status
            .and_then(behavior_panel::Status::error)
            .is_some(),
        "an entity field holding an int does not check out"
    );
    let row = view.fault_row.expect("the checker located it");
    assert_eq!(data.rows[row].label, "target");
    assert_eq!(
        data.rows[row].path,
        vec![
            crate::editor::behavior::path::field("do"),
            crate::editor::behavior::path::Step::Index(1),
            crate::editor::behavior::path::field("hide"),
            crate::editor::behavior::path::field("target"),
        ],
    );
}

// A rule about the asset as a whole has no one row to blame, so the banner says
// so and stays unclickable rather than sending the author somewhere arbitrary.
#[test]
fn behavior_status_with_nothing_to_blame_points_nowhere() {
    let (h, _world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [{"save": {}}]}),
    )]);
    let data = h.behavior_data();
    assert!(h.make_behavior_view(&data, [0.0, 0.0]).status == Some(&behavior_panel::Status::Ok));

    let (h, _world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "spawned", "do": []}),
    )]);
    let data = h.behavior_data();
    let view = h.make_behavior_view(&data, [0.0, 0.0]);
    assert!(
        view.status
            .and_then(behavior_panel::Status::error)
            .is_some()
    );
    // `on` is what the complaint blames, and the outline has a row for it.
    let row = view.fault_row.expect("the source row");
    assert_eq!(data.rows[row].label, "on");
}

// Going to the fault selects its row, which is what brings an off-screen one
// into view through the existing scroll and pan.
#[test]
fn behavior_go_to_fault_selects_the_faulting_row() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [
            {"save": {}},
            {"hide": {"target": {"int": 1}}},
        ]}),
    )]);
    assert_eq!(h.behavior_row, None);
    h.apply_behavior_action(BehaviorAction::GoToFault, &mut world, [0.0, 0.0]);
    let row = h.behavior_row.expect("the fault was selected");
    assert_eq!(h.behavior_rows()[row].label, "target");
}

// The overview maps the world rather than one body, so going to a fault steps to
// the view that can show it.
#[test]
fn behavior_go_to_fault_leaves_the_overview_for_the_body() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [{"hide": {"target": {"int": 1}}}]}),
    )]);
    for _ in 0..2 {
        h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    }
    assert_eq!(h.behavior_mode, ViewMode::Overview);

    h.apply_behavior_action(BehaviorAction::GoToFault, &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_mode, ViewMode::Chart);
    let row = h.behavior_row.expect("the fault was selected");
    assert_eq!(h.behavior_rows()[row].label, "target");
}

// The location is kept as a path rather than a row index, so a verdict left
// standing while the args change under it (the one path that does not re-check --
// a history jump) degrades to an ancestor of the fault instead of confidently
// marking whatever row has taken that index.
#[test]
fn a_stale_behavior_fault_never_points_off_its_own_path() {
    let (mut h, _world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [{"hide": {"target": {"int": 1}}}]}),
    )]);
    let located = {
        let data = h.behavior_data();
        let row = h.make_behavior_view(&data, [0.0, 0.0]).fault_row.unwrap();
        data.rows[row].path.clone()
    };

    // Grow the body ahead of the bad node without re-running the checker, so the
    // stored location now addresses a place the args no longer hold.
    let mut args = open_args(&h);
    args["do"]
        .as_array_mut()
        .unwrap()
        .insert(0, serde_json::json!({"save": {}}));
    let idx = h.behavior_entry().unwrap();
    h.entries[idx]
        .as_object_mut()
        .unwrap()
        .insert("args".to_string(), args);

    let data = h.behavior_data();
    let row = h.make_behavior_view(&data, [0.0, 0.0]).fault_row;
    let path = row.map(|i| data.rows[i].path.clone()).unwrap_or_default();
    assert!(
        crate::editor::behavior::path::starts_with(&located, &path),
        "pointed at {path:?}, which is not on the way to {located:?}",
    );
    assert_ne!(
        path, located,
        "the exact spot is gone, so it settled for less"
    );
}

// Type into the palette's filter, as the engine's text-input system would, then
// let the hook sample it the way its tick does.
fn type_filter(h: &mut EditorHook, world: &mut World, text: &str) {
    widget::seed_field(world, behavior_panel::FILTER_INPUT, text);
    h.sample_behavior_filter(world);
}

fn open_palette(h: &mut EditorHook, world: &mut World, row: &str) {
    select_behavior(h, world, row);
    h.apply_behavior_action(BehaviorAction::Palette, world, [0.0, 0.0]);
    assert!(h.behavior_picking, "the palette is open");
}

// The point of typing is that the first answer is the one wanted, so Enter after
// a query lands on the best match rather than on the vocabulary's first entry.
#[test]
fn behavior_palette_filter_narrows_and_enter_takes_the_best_match() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    open_palette(&mut h, &mut world, "do");
    let unfiltered = h.behavior_data().matches.len();

    type_filter(&mut h, &mut world, "foreach");
    let data = h.behavior_data();
    assert!(data.matches.len() < unfiltered, "the query narrowed it");
    assert_eq!(data.picks[data.matches[0]].verb, "for_each");

    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    assert!(
        open_args(&h)["do"][0].get("for_each").is_some(),
        "the best match is what landed: {:?}",
        open_args(&h)["do"],
    );
}

// A query is about the pick being made, so it never outlives it: the next palette
// opens on the whole vocabulary.
#[test]
fn behavior_palette_filter_clears_when_the_palette_closes() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    open_palette(&mut h, &mut world, "do");
    let unfiltered = h.behavior_data().matches.len();
    type_filter(&mut h, &mut world, "spawn");
    assert!(h.behavior_data().matches.len() < unfiltered);

    // Picking closes it, and the filter goes with it.
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    assert!(h.behavior_filter.is_empty());
    assert_eq!(
        widget::field_text(&world, behavior_panel::FILTER_INPUT),
        "",
        "the field was cleared too, not just the mirror"
    );

    // And so does dismissing.
    open_palette(&mut h, &mut world, "do");
    type_filter(&mut h, &mut world, "spawn");
    h.behavior_keys(&mut world, &behavior_escape_input());
    assert!(!h.behavior_picking);
    assert!(h.behavior_filter.is_empty());
    open_palette(&mut h, &mut world, "do");
    assert_eq!(h.behavior_data().matches.len(), unfiltered);
}

// Narrowing puts the highlight back at the top: a place in the old list may not
// even be in the new one, and Enter must never go dead.
#[test]
fn behavior_palette_filter_resets_the_highlight_it_may_have_excluded() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    open_palette(&mut h, &mut world, "do");
    for _ in 0..4 {
        press_behavior_key(&mut h, &mut world, crate::assets::Key::Down);
    }
    assert_eq!(h.behavior_pick, 4);

    // A query keeping fewer options than that would have stranded the highlight.
    type_filter(&mut h, &mut world, "spawn");
    assert_eq!(h.behavior_pick, 0);
    assert_eq!(h.behavior_pick_scroll, 0);
    assert!(h.behavior_data().matches.len() <= 4);

    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    assert!(
        open_args(&h)["do"][0].get("spawn").is_some(),
        "Enter still picked: {:?}",
        open_args(&h)["do"],
    );
}

// A query nothing answers keeps the palette up, because the field being typed
// into is inside it: collapsing would take away the only way to fix the typo.
#[test]
fn behavior_palette_survives_a_query_nothing_answers() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    open_palette(&mut h, &mut world, "do");
    type_filter(&mut h, &mut world, "zzzz");
    assert!(h.behavior_data().matches.is_empty());
    assert!(h.behavior_picking, "the palette is still up");

    // Enter has nothing to insert, and nothing is written.
    press_behavior_key(&mut h, &mut world, crate::assets::Key::Enter);
    assert!(h.behavior_picking);
    assert_eq!(open_args(&h)["do"].as_array().map(Vec::len), Some(0));

    // Correcting the query brings the options back.
    type_filter(&mut h, &mut world, "save");
    assert!(!h.behavior_data().matches.is_empty());
}

// While the palette is up its field holds the keyboard, so the editor's own
// letter shortcuts stand down rather than moving a gizmo behind it.
#[test]
fn behavior_palette_filter_holds_the_keyboard_off_the_shortcuts() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    assert!(!h.text_focus_active());
    open_palette(&mut h, &mut world, "do");
    assert!(
        h.text_focus_active(),
        "typing `s` into the filter must not reach the scale gizmo"
    );
}

// A panel's fields draw at its base layer, but the palette's backing is bumped
// above that -- so the field inside the palette has to be bumped with it or its
// text renders behind the box it sits in.
#[test]
fn behavior_palette_filter_field_draws_above_the_backing_it_sits_in() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": []}),
    )]);
    open_palette(&mut h, &mut world, "do");
    let layers = h.compute_layers();
    let field = layers[&behavior_panel::FILTER_INPUT];
    assert!(
        field > layers[&behavior_panel::PANEL_BG],
        "the field sank into its own panel"
    );
    assert_eq!(
        field,
        layers[&behavior_panel::DROP_BG],
        "the field and the backing it sits in share a layer"
    );
}

fn ctrl_key_input(key: crate::assets::Key) -> FrameInput {
    FrameInput {
        captured_key: Some(key),
        ctrl: true,
        viewport: [1280.0, 720.0],
        ..Default::default()
    }
}

fn body_verbs(h: &EditorHook) -> Vec<String> {
    open_args(h)["do"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|n| n.as_object()?.keys().next().cloned())
                .collect()
        })
        .unwrap_or_default()
}

// Duplicating puts the copy beside the original, carrying its whole subtree, and
// leaves the selection on what just landed so a follow-up acts on the copy.
#[test]
fn behavior_duplicate_copies_a_node_subtree_next_to_it() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [
            {"if": {"cond": {"bool": true}, "then": [{"hide": {"target": "self"}}]}},
            {"save": {}},
        ]}),
    )]);
    select_behavior(&mut h, &mut world, "if");
    h.apply_behavior_action(BehaviorAction::Duplicate, &mut world, [0.0, 0.0]);

    assert_eq!(body_verbs(&h), ["if", "if", "save"]);
    assert_eq!(
        open_args(&h)["do"][1]["if"]["then"][0]["hide"]["target"],
        serde_json::json!("self"),
        "the branch came with it"
    );
    let row = h.behavior_row.expect("the copy is selected");
    assert_eq!(
        h.behavior_rows()[row].element,
        Some(vec![
            crate::editor::behavior::path::field("do"),
            crate::editor::behavior::path::Step::Index(1),
        ]),
    );
}

// Ctrl+C then Ctrl+V is the same move spelled out, and what is held survives to
// be pasted again.
#[test]
fn behavior_ctrl_c_holds_a_node_and_ctrl_v_places_it() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [{"save": {}}, {"hide": {"target": "self"}}]}),
    )]);
    select_behavior(&mut h, &mut world, "hide");
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::C));
    assert!(h.behavior_clip.is_some(), "the node is held");
    assert_eq!(body_verbs(&h), ["save", "hide"], "copying wrote nothing");

    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::V));
    assert_eq!(body_verbs(&h), ["save", "hide", "hide"]);
    // Still held, so a second paste lands beside the first copy.
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::V));
    assert_eq!(body_verbs(&h), ["save", "hide", "hide", "hide"]);
}

// A duplicate is a paste of the selection, so it must not disturb what a Ctrl+C
// earlier put aside.
#[test]
fn behavior_duplicate_leaves_what_is_held_alone() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [{"save": {}}, {"hide": {"target": "self"}}]}),
    )]);
    select_behavior(&mut h, &mut world, "save");
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::C));

    select_behavior(&mut h, &mut world, "hide");
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::D));
    assert_eq!(body_verbs(&h), ["save", "hide", "hide"]);

    // What was held is still the `save`, and pastes as one.
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::V));
    assert_eq!(body_verbs(&h), ["save", "hide", "hide", "save"]);
}

// Carrying a node between behaviors is most of why the clipboard outlives the one
// it came from.
#[test]
fn behavior_clipboard_carries_a_node_to_another_behavior() {
    let (mut h, mut world) = behavior_session(vec![
        behavior(
            "chase",
            serde_json::json!({"on": "start", "do": [{"hide": {"target": "self"}}]}),
        ),
        behavior("greet", serde_json::json!({"on": "tick", "do": []})),
    ]);
    select_behavior(&mut h, &mut world, "hide");
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::C));

    h.apply_behavior_action(BehaviorAction::Step(1), &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_data().name, "greet");
    assert!(h.behavior_clip.is_some(), "opening another kept it");

    // The empty body's own row is the list, so a paste there appends.
    select_behavior(&mut h, &mut world, "do");
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::V));
    assert_eq!(body_verbs(&h), ["hide"]);
}

// A node does not belong in a list of component names, so nothing is written and
// the world is left as it was.
#[test]
fn behavior_paste_is_refused_by_a_list_of_another_kind() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "scope": ["Prop"], "do": [{"save": {}}]}),
    )]);
    select_behavior(&mut h, &mut world, "save");
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::C));

    select_behavior(&mut h, &mut world, "scope");
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::V));
    assert_eq!(open_args(&h)["scope"], serde_json::json!(["Prop"]));
    assert_eq!(body_verbs(&h), ["save"], "and the body is untouched too");
}

// A row that is not a member of any list has nothing to carry.
#[test]
fn behavior_copy_of_a_non_member_holds_nothing() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [{"save": {}}]}),
    )]);
    select_behavior(&mut h, &mut world, "on");
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::C));
    assert!(h.behavior_clip.is_none());
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::D));
    assert_eq!(
        body_verbs(&h),
        ["save"],
        "and there is nothing to duplicate"
    );
}

// The clipboard keys are about the selected node, so a field holding the keyboard
// keeps them: Ctrl+C while typing a value must not duplicate a node behind it.
#[test]
fn behavior_clipboard_keys_stand_down_while_a_field_is_focused() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [{"let": {"name": "t", "value": {"int": 1}}}]}),
    )]);
    select_behavior(&mut h, &mut world, "name");
    assert!(h.behavior_focus, "a text row takes the value field");
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::D));
    assert_eq!(body_verbs(&h), ["let"], "nothing was duplicated");

    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    h.behavior_keys(&mut world, &ctrl_key_input(crate::assets::Key::D));
    assert_eq!(body_verbs(&h), ["let"]);
}

// The edit goes through the same commit every other one does, so it is undoable.
#[test]
fn behavior_duplicate_is_undoable() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "start", "do": [{"save": {}}]}),
    )]);
    select_behavior(&mut h, &mut world, "save");
    h.apply_behavior_action(BehaviorAction::Duplicate, &mut world, [0.0, 0.0]);
    assert_eq!(body_verbs(&h), ["save", "save"]);
    h.undo(&mut world);
    assert_eq!(body_verbs(&h), ["save"]);
}

fn variables_session(entries: Vec<serde_json::Value>) -> (EditorHook, World) {
    let mut world = World::new_empty();
    for id in variables_panel::all_field_ids() {
        world.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    let mut h = hook(entries);
    registry::panel(PanelKey::Variables).toggle(&mut h, &mut world);
    (h, world)
}

fn var_rows(h: &EditorHook) -> Vec<(String, String, bool)> {
    h.variables_data()
        .rows
        .into_iter()
        .map(|r| (r.name, r.ty, r.at.is_some()))
        .collect()
}

fn select_var(h: &mut EditorHook, world: &mut World, name: &str) {
    let at = h
        .variables_data()
        .rows
        .iter()
        .position(|r| r.name == name)
        .unwrap_or_else(|| panic!("no `{name}` row"));
    h.apply_variables_action(VariablesAction::Select(at), world);
}

fn table_args(h: &EditorHook) -> serde_json::Value {
    h.variables_entry()
        .and_then(|i| h.entries[i].get("args").cloned())
        .unwrap_or(serde_json::Value::Null)
}

// A world with no table lists what its behaviors use, all undeclared, and says
// nothing is wrong: an undeclared name is only a problem once a table exists.
#[test]
fn variables_panel_lists_what_the_behaviors_use_before_any_table_exists() {
    let (h, _world) = variables_session(vec![
        behavior(
            "award",
            serde_json::json!({"on": "start",
                "do": [{"set": {"var": "score", "value": {"int": 1}}}]}),
        ),
        behavior(
            "react",
            serde_json::json!({"on": {"variable": "health"}, "do": []}),
        ),
    ]);
    assert_eq!(
        var_rows(&h),
        [
            ("health".to_string(), String::new(), false),
            ("score".to_string(), String::new(), false),
        ],
    );
    assert!(!h.variables_data().authoritative);
    assert!(
        h.variables_data().status.is_none(),
        "nothing is declared, so nothing is missing"
    );
}

// The first declaration creates the table, holding that variable: creating it
// empty would make it authoritative over a table that accounts for nothing.
#[test]
fn declaring_the_first_variable_creates_the_table_holding_it() {
    let (mut h, mut world) = variables_session(vec![behavior(
        "award",
        serde_json::json!({"on": "start", "do": [{"set": {"var": "score", "value": {"int": 1}}}]}),
    )]);
    assert!(h.variables_entry().is_none(), "no table yet");

    select_var(&mut h, &mut world, "score");
    h.apply_variables_action(VariablesAction::Declare, &mut world);

    let idx = h.variables_entry().expect("the table was created");
    assert_eq!(entry_type(&h.entries[idx]), Some("Variables"));
    assert_eq!(
        table_args(&h)["vars"],
        serde_json::json!([{"name": "score", "value": {"int": 0}}]),
        "created holding the name the behaviors already use",
    );
    assert_eq!(
        var_rows(&h),
        [("score".to_string(), "int".to_string(), true)]
    );
    assert!(
        h.variables_data().status.is_none(),
        "and nothing is missing"
    );
}

// A declared table is held to every name its behaviors use, so one it leaves out
// is a build error the panel has to say out loud.
#[test]
fn a_declared_table_warns_about_a_name_it_leaves_out() {
    let (mut h, mut world) = variables_session(vec![
        entry_with_args(
            "world_vars",
            "Variables",
            serde_json::json!({"vars": [{"name": "score", "value": {"int": 0}}]}),
        ),
        behavior(
            "hurt",
            serde_json::json!({"on": "start",
                "do": [{"set": {"var": "health", "value": {"int": 1}}}]}),
        ),
    ]);
    assert!(h.variables_data().authoritative);
    let status = h.variables_data().status.expect("it warns");
    assert!(status.contains("health"), "{status}");
    assert!(status.contains("authoritative"), "{status}");

    // Declaring it clears the warning.
    select_var(&mut h, &mut world, "health");
    h.apply_variables_action(VariablesAction::Declare, &mut world);
    assert!(h.variables_data().status.is_none());
    assert_eq!(
        var_rows(&h),
        [
            ("score".to_string(), "int".to_string(), true),
            ("health".to_string(), "int".to_string(), true),
        ],
    );
}

// Retyping steps through the literal kinds and rewrites the starting value with
// that type's own, so a declaration is never left holding one of another type.
#[test]
fn retyping_a_variable_steps_its_type_and_starting_value_together() {
    let (mut h, mut world) = variables_session(vec![entry_with_args(
        "world_vars",
        "Variables",
        serde_json::json!({"vars": [{"name": "score", "value": {"bool": true}}]}),
    )]);
    select_var(&mut h, &mut world, "score");
    let mut seen = vec![h.variables_data().rows[0].ty.clone()];
    for _ in 0..3 {
        h.apply_variables_action(VariablesAction::Retype, &mut world);
        seen.push(h.variables_data().rows[0].ty.clone());
    }
    assert_eq!(seen, ["bool", "int", "float", "vec3"]);
    assert!(
        table_args(&h)["vars"][0]["value"].get("vec3").is_some(),
        "{:?}",
        table_args(&h)["vars"][0],
    );
    // And it cycles rather than running out.
    h.apply_variables_action(VariablesAction::Retype, &mut world);
    assert_eq!(h.variables_data().rows[0].ty, "bool");
}

// The value field writes the declaration's starting value, parsed the way the
// Behavior panel parses a literal.
#[test]
fn typing_a_starting_value_writes_it_and_a_bad_one_is_refused() {
    let (mut h, mut world) = variables_session(vec![entry_with_args(
        "world_vars",
        "Variables",
        serde_json::json!({"vars": [{"name": "spawn", "value": {"vec3": [0, 0, 0]}}]}),
    )]);
    select_var(&mut h, &mut world, "spawn");
    h.apply_variables_action(VariablesAction::FocusValue, &mut world);
    widget::seed_field(&mut world, variables_panel::VALUE_INPUT, "1, 2, 3");
    h.variables_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(
        table_args(&h)["vars"][0]["value"]["vec3"],
        serde_json::json!([1.0, 2.0, 3.0]),
    );

    // Something that is not a vector leaves the declaration as it was, and the
    // field goes back to what the table holds.
    h.apply_variables_action(VariablesAction::FocusValue, &mut world);
    widget::seed_field(&mut world, variables_panel::VALUE_INPUT, "nonsense");
    h.variables_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(
        table_args(&h)["vars"][0]["value"]["vec3"],
        serde_json::json!([1.0, 2.0, 3.0]),
    );
    assert_eq!(
        widget::field_text(&world, variables_panel::VALUE_INPUT),
        "1, 2, 3",
    );
}

// Renaming commits on Enter; a blank name is refused and the field goes back to
// what the table holds.
#[test]
fn renaming_a_variable_commits_on_enter_and_refuses_a_blank() {
    let (mut h, mut world) = variables_session(vec![entry_with_args(
        "world_vars",
        "Variables",
        serde_json::json!({"vars": [{"name": "score", "value": {"int": 0}}]}),
    )]);
    select_var(&mut h, &mut world, "score");
    h.apply_variables_action(VariablesAction::FocusName, &mut world);
    widget::seed_field(&mut world, variables_panel::NAME_INPUT, "points");
    h.variables_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(
        table_args(&h)["vars"][0]["name"],
        serde_json::json!("points")
    );
    assert_eq!(
        h.variables_row,
        Some(0),
        "the selection followed the rename"
    );

    h.apply_variables_action(VariablesAction::FocusName, &mut world);
    widget::seed_field(&mut world, variables_panel::NAME_INPUT, "   ");
    h.variables_keys(&mut world, &story_key_input(crate::assets::Key::Enter));
    assert_eq!(
        table_args(&h)["vars"][0]["name"],
        serde_json::json!("points")
    );
    assert_eq!(
        widget::field_text(&world, variables_panel::NAME_INPUT),
        "points",
    );
}

// Removing a declaration drops the selection rather than retargeting it, and the
// name reappears as undeclared when a behavior still uses it.
#[test]
fn removing_a_declaration_leaves_the_name_the_behaviors_still_use() {
    let (mut h, mut world) = variables_session(vec![
        entry_with_args(
            "world_vars",
            "Variables",
            serde_json::json!({"vars": [{"name": "score", "value": {"int": 0}}]}),
        ),
        behavior(
            "award",
            serde_json::json!({"on": "start",
                "do": [{"set": {"var": "score", "value": {"int": 1}}}]}),
        ),
    ]);
    select_var(&mut h, &mut world, "score");
    h.apply_variables_action(VariablesAction::Remove, &mut world);

    assert_eq!(h.variables_row, None, "the selection was dropped");
    assert_eq!(table_args(&h)["vars"], serde_json::json!([]));
    assert_eq!(
        var_rows(&h),
        [("score".to_string(), String::new(), false)],
        "the behavior still uses it, so it is now missing",
    );
    assert!(h.variables_data().status.is_some(), "and the panel says so");
}

// The table's own malformed state is the checker's to report, so the panel quotes
// it rather than inventing its own wording.
#[test]
fn variables_panel_quotes_the_checkers_complaint_about_the_table() {
    let (h, _world) = variables_session(vec![entry_with_args(
        "world_vars",
        "Variables",
        serde_json::json!({"vars": [
            {"name": "score", "value": {"int": 0}},
            {"name": "score", "value": {"int": 1}},
        ]}),
    )]);
    let status = h.variables_data().status.expect("a duplicate is a fault");
    assert!(status.contains("duplicate variable 'score'"), "{status}");
}

// A variable's type is what the behaviors reading it type-check against, so
// changing it re-takes the Behavior panel's verdict too.
#[test]
fn retyping_a_variable_re_checks_the_behaviors_reading_it() {
    let (mut h, mut world) = variables_session(vec![
        entry_with_args(
            "world_vars",
            "Variables",
            serde_json::json!({"vars": [{"name": "score", "value": {"int": 0}}]}),
        ),
        behavior(
            "award",
            serde_json::json!({"on": "start",
                "do": [{"set": {"var": "score", "value": {"int": 1}}}]}),
        ),
    ]);
    registry::panel(PanelKey::Behavior).toggle(&mut h, &mut world);
    assert_eq!(h.behavior_status, Some(behavior_panel::Status::Ok));

    // The types cycle bool -> int -> float -> vec3, so two steps from int lands
    // on a type an int value no longer satisfies.
    select_var(&mut h, &mut world, "score");
    h.apply_variables_action(VariablesAction::Retype, &mut world);
    h.apply_variables_action(VariablesAction::Retype, &mut world);
    assert_eq!(h.variables_data().rows[0].ty, "vec3");
    let status = h
        .behavior_status
        .as_ref()
        .and_then(behavior_panel::Status::error)
        .expect("the behavior no longer checks out");
    assert!(status.contains("must be vec3"), "{status}");
}

// The overview maps behaviors through the variables they share, so a variable
// card is the way from a body into the table that declares it.
#[test]
fn an_overview_variable_card_opens_the_table_on_it() {
    let (mut h, mut world) = variables_session(vec![
        entry_with_args(
            "world_vars",
            "Variables",
            serde_json::json!({"vars": [{"name": "score", "value": {"int": 0}}]}),
        ),
        behavior(
            "award",
            serde_json::json!({"on": "start",
                "do": [{"set": {"var": "score", "value": {"int": 1}}}]}),
        ),
    ]);
    // Close the panel again so opening it from the map is what opens it.
    registry::panel(PanelKey::Variables).close(&mut h, &mut world);
    registry::panel(PanelKey::Behavior).toggle(&mut h, &mut world);
    for _ in 0..2 {
        h.apply_behavior_action(BehaviorAction::ToggleView, &mut world, [0.0, 0.0]);
    }
    let card = h
        .behavior_data()
        .overview
        .cards
        .iter()
        .position(|c| c.title == "score")
        .expect("the variable is on the map");

    h.apply_behavior_action(BehaviorAction::OpenVariable(card), &mut world, [0.0, 0.0]);
    assert!(h.variables_open, "the table opened");
    assert_eq!(h.panel_order.last(), Some(&PanelKey::Variables));
    let row = h.variables_row.expect("selected on the card's variable");
    assert_eq!(h.variables_data().rows[row].name, "score");
}

// The simulation transport: keys, chips, edit policy, and the trace exchange.

fn playing_hook(entries: Vec<serde_json::Value>) -> EditorHook {
    let mut h = hook(entries);
    h.sim.toggle_play();
    h
}

#[test]
fn transport_keys_play_pause_stop_and_step() {
    let mut h = hook(Vec::new());
    let key = |k, shift| FrameInput {
        ctrl: true,
        shift,
        captured_key: Some(k),
        ..Default::default()
    };
    h.sim_keys(&key(crate::assets::Key::P, false));
    assert!(h.sim.playing(), "Ctrl+P plays");
    h.sim_keys(&key(crate::assets::Key::P, false));
    assert_eq!(h.sim.state, sim::SimState::Paused, "Ctrl+P again pauses");
    h.sim_keys(&key(crate::assets::Key::Period, false));
    assert!(h.sim.take_run_frame(), "Ctrl+Period queues one step");
    h.sim_keys(&key(crate::assets::Key::P, true));
    assert_eq!(h.sim.state, sim::SimState::Stopped, "Ctrl+Shift+P stops");
    assert!(
        h.rebuild_preview,
        "Stop restores through the preview rebuild"
    );

    // A focused text field owns the keyboard.
    h.story_focus = true;
    h.sim_keys(&key(crate::assets::Key::P, false));
    assert_eq!(h.sim.state, sim::SimState::Stopped);
}

#[test]
fn transport_chips_drive_the_transport() {
    let mut h = hook(Vec::new());
    let mut world = World::new_empty();
    h.apply_top(HudAction::PlayPause, &mut world);
    assert!(h.sim.playing());
    h.apply_top(HudAction::Step, &mut world);
    assert_eq!(
        h.sim.state,
        sim::SimState::Paused,
        "Step while playing pauses"
    );
    h.apply_top(HudAction::Stop, &mut world);
    assert_eq!(h.sim.state, sim::SimState::Stopped);
    assert!(h.rebuild_preview);
}

#[test]
fn a_committed_edit_stops_the_simulation() {
    let mut h = playing_hook(vec![entry("box", "Prop")]);
    h.entries.push(entry("box2", "Prop"));
    h.mark_changed();
    assert_eq!(
        h.sim.state,
        sim::SimState::Stopped,
        "the rebuild discards the run, so the transport says so"
    );
}

#[test]
fn entering_play_ends_the_fly_camera_and_vice_versa() {
    let mut h = hook(Vec::new());
    h.toggle_fly();
    assert!(h.fly);
    h.sim_toggle_play();
    assert!(h.sim.playing() && !h.fly, "play takes the cursor from fly");
    h.toggle_fly();
    assert!(h.fly);
    assert_eq!(
        h.sim.state,
        sim::SimState::Paused,
        "fly pauses a running world"
    );
}

#[test]
fn the_trace_request_follows_the_live_debug_panels() {
    let mut h = hook(vec![behavior(
        "b",
        serde_json::json!({
            "on": "start", "do": [{"save": {}}],
        }),
    )]);
    let mut world = World::new_empty();
    h.drive_trace(&mut world);
    assert!(
        world.resource::<crate::ecs::TraceRequest>().is_none(),
        "no panel open, no request"
    );
    h.behavior_open = true;
    h.drive_trace(&mut world);
    assert!(world.resource::<crate::ecs::TraceRequest>().is_some());
    h.behavior_open = false;
    h.drive_trace(&mut world);
    assert!(
        world.resource::<crate::ecs::TraceRequest>().is_none(),
        "closing the panels withdraws it"
    );
}

// A world carrying one published trace tick for behavior `b`'s first node.
fn traced_world(id: crate::ecs::asset_id::AssetId, hit: bool) -> World {
    use crate::ecs::{ExecutionTrace, TraceEvent, TracePaths, TraceStep, TraceVal};
    let mut world = World::new_empty();
    let event = TraceEvent {
        behavior: id,
        node: 0,
    };
    world.insert_resource(TracePaths(vec![(
        id,
        vec![vec![TraceStep::Field("do"), TraceStep::Index(0)]],
    )]));
    world.insert_resource(ExecutionTrace {
        frame: 1,
        events: vec![event],
        vars: vec![("n".to_string(), TraceVal::Int(3))],
        locals: Vec::new(),
        hit: hit.then_some(event),
    });
    world
}

#[test]
fn trace_events_become_pulses_and_live_values() {
    crate::ecs::asset_id::reset_interner();
    let id = crate::ecs::asset_id::intern("b");
    let mut h = playing_hook(vec![behavior(
        "b",
        serde_json::json!({
            "on": "start", "do": [{"save": {}}],
        }),
    )]);
    h.behavior_open = true;
    let mut world = traced_world(id, false);
    h.drive_trace(&mut world);

    assert_eq!(h.behavior_pulses.len(), 1);
    assert_eq!(
        h.behavior_pulses[0].path,
        vec![path::field("do"), path::Step::Index(0)],
        "the pulse addresses the node the way a checker fault would"
    );
    assert_eq!(
        h.live_vars,
        vec![("n".to_string(), "int".to_string(), "3".to_string())]
    );
    let data = h.behavior_data();
    assert_eq!(
        data.pulse_cards.len(),
        1,
        "the node's card carries the pulse"
    );
    assert_eq!(data.pulse_rows.len(), 1, "so does its outline row");
    // The same frame again reports nothing new; the pulse just decays.
    h.drive_trace(&mut world);
    assert_eq!(h.behavior_pulses.len(), 1);

    // Live values reach the Variables panel and retitle its value column.
    let vdata = h.variables_data();
    assert!(vdata.live);
    assert!(
        vdata.rows.iter().any(|r| r.name == "n" && r.value == "3"),
        "{:?}",
        vdata.rows
    );
}

#[test]
fn a_breakpoint_hit_pauses_and_lands_on_the_node() {
    crate::ecs::asset_id::reset_interner();
    let id = crate::ecs::asset_id::intern("b");
    let mut h = playing_hook(vec![behavior(
        "b",
        serde_json::json!({
            "on": "start", "do": [{"save": {}}],
        }),
    )]);
    h.behavior_open = true;
    let mut world = traced_world(id, true);
    h.drive_trace(&mut world);
    assert_eq!(h.sim.state, sim::SimState::Paused, "the hit froze the run");
    let row = h.behavior_row.expect("the panel landed on the node");
    assert_eq!(
        h.behavior_rows()[row].path,
        vec![path::field("do"), path::Step::Index(0)]
    );
}

#[test]
fn stopping_clears_the_live_state() {
    crate::ecs::asset_id::reset_interner();
    let id = crate::ecs::asset_id::intern("b");
    let mut h = playing_hook(vec![behavior(
        "b",
        serde_json::json!({
            "on": "start", "do": [{"save": {}}],
        }),
    )]);
    h.behavior_open = true;
    let mut world = traced_world(id, false);
    h.drive_trace(&mut world);
    assert!(!h.behavior_pulses.is_empty() && !h.live_vars.is_empty());

    assert!(h.sim.stop());
    h.drive_trace(&mut world);
    assert!(h.behavior_pulses.is_empty(), "Stop shows authored data");
    assert!(h.live_vars.is_empty());
    assert!(!h.variables_data().live);
}

#[test]
fn ctrl_click_toggles_a_card_breakpoint() {
    let mut h = hook(vec![behavior(
        "b",
        serde_json::json!({
            "on": "start", "do": [{"save": {}}],
        }),
    )]);
    let mut world = World::new_empty();
    h.behavior_open = true;
    let data = h.behavior_data();
    let card = data
        .chart
        .cards
        .iter()
        .position(|c| c.kind == CardKind::Node)
        .expect("the body has a node card");

    h.ctrl_held = true;
    h.apply_behavior_action(BehaviorAction::SelectCard(card), &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_breakpoints.len(), 1);
    assert_eq!(h.behavior_breakpoints[0].0, "b", "held by behavior name");
    assert_eq!(
        h.behavior_data().break_cards,
        vec![card],
        "the card shows its marker"
    );
    h.apply_behavior_action(BehaviorAction::SelectCard(card), &mut world, [0.0, 0.0]);
    assert!(
        h.behavior_breakpoints.is_empty(),
        "a second toggle removes it"
    );

    // A plain click still selects.
    h.ctrl_held = false;
    h.apply_behavior_action(BehaviorAction::SelectCard(card), &mut world, [0.0, 0.0]);
    assert!(h.behavior_row.is_some());
}
