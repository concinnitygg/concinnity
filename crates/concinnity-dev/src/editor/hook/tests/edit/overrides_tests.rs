// src/editor/hook/tests/edit/overrides_tests.rs
//
// The override loop (`hook/edit/overrides.rs`): the minimal patch an edit to a
// generated asset writes, the origin a patched asset relists under, the
// single-field revert and apply pair and the bulk apply-all across a mixed
// patch, minimizing a patch back down to what actually differs, materializing
// a preset-backed Prefab so its entries become editable, the jump that walks
// the form to its next marked field, and the unapplied markers that follow an
// edit and an apply.

// The injected typed fields the form reads its controls back from.
// One Prefab with a single prop entry, one instance of it, and the patch line
// `patch` pinning fields on that instance's generated asset.

use crate::editor::hook::FormTarget;
use crate::editor::panels::asset_tree;
use crate::editor::panels::asset_tree::TreeGroup;
use crate::editor::panels::form;
use crate::editor::panels::panel::PanelAction;
use crate::editor::panels::story_panel;
use crate::editor::widget;
use concinnity_core::components::TextInput;
use concinnity_core::ecs::World;

use crate::editor::hook::tests::fixtures::{
    a_promotable_asset, click_row, entry, expandable_hook, hook, row_of, seed_tree, set_field,
    world_with_fields,
};
use crate::editor::hook::{EditorHook, entry_name};
use crate::editor::overrides;
use crate::editor::panels::form_panel::{self, FormAction};
use crate::test_support::isolate_state_dir;

fn prefab_hook(patch: serde_json::Value) -> EditorHook {
    isolate_state_dir();
    let mut h = hook(vec![
        serde_json::json!({"name":"box","type":"ProceduralMesh","args":{"generator":"box"}}),
        serde_json::json!({"name":"pair","type":"Prefab","args":{"props":[
            {"name":"a","kind":"prop","mesh":"box","position":[1.0,0.0,0.0]}]}}),
        serde_json::json!({"name":"i1","type":"Prop","args":{"prefab":"pair","position":[10.0,0.0,0.0]}}),
        serde_json::json!({"name":"i1_a","type":"Prop","args": patch}),
    ]);
    h.panel_open = true;
    h
}

fn entity_option(h: &EditorHook, label_prefix: &str) -> usize {
    let labels: Vec<String> = h
        .entity_menu_options()
        .into_iter()
        .map(|(_, l)| l)
        .collect();
    labels
        .iter()
        .position(|l| l.starts_with(label_prefix))
        .unwrap_or_else(|| panic!("no entity option starting {label_prefix:?}, got {labels:?}"))
}

fn entry_of<'a>(h: &'a EditorHook, name: &str) -> Option<&'a serde_json::Value> {
    h.entries.iter().find(|e| entry_name(e) == Some(name))
}

// Apply-all writes every path the prefab entry carries back into the
// definition and leaves the rest authored, reporting which ones it kept -- a
// silent drop would lose the instance's value.
#[test]
fn apply_all_writes_the_mappable_paths_and_keeps_the_rest() {
    let _guard = crate::test_support::lock();
    let mut h = prefab_hook(serde_json::json!({
        "position": [5.0, 0.0, 0.0],
        "cull_distance": 42.0
    }));
    let mut world = world_with_fields();
    h.open_asset_form("i1_a", &mut world);

    let k = entity_option(&h, "Apply all to Prefab");
    h.apply_form(FormAction::OpenEntityMenu, &mut world);
    h.apply_form(FormAction::PickEntityOption(k), &mut world);

    // World (5,0,0) under instance position (10,0,0) is local (-5,0,0).
    let def = entry_of(&h, "pair").expect("the definition stands");
    assert_eq!(
        def["args"]["props"][0]["position"],
        serde_json::json!([-5.0, 0.0, 0.0]),
        "the mappable path landed in the definition"
    );

    let patch = entry_of(&h, "i1_a").expect("the patch line survives its unmappable field");
    assert!(
        patch["args"].get("position").is_none(),
        "the applied path left the patch"
    );
    assert_eq!(patch["args"]["cull_distance"], 42.0);
    assert!(
        h.form_error
            .as_ref()
            .is_some_and(|e| e.contains("cull_distance")),
        "the kept path is reported, got {:?}",
        h.form_error
    );
}

// Apply-all over a fully mappable patch empties it, so the line goes and the
// whole thing is one undo step.
#[test]
fn apply_all_over_a_fully_mappable_patch_removes_the_line() {
    let _guard = crate::test_support::lock();
    let mut h = prefab_hook(serde_json::json!({"position": [5.0, 0.0, 0.0]}));
    let mut world = world_with_fields();
    let before = h.entries.len();
    h.open_asset_form("i1_a", &mut world);

    let k = entity_option(&h, "Apply all to Prefab");
    h.apply_form(FormAction::OpenEntityMenu, &mut world);
    h.apply_form(FormAction::PickEntityOption(k), &mut world);

    assert_eq!(
        h.entries.len(),
        before - 1,
        "the emptied patch line is gone"
    );
    assert!(h.form_error.is_none(), "nothing was kept back");

    h.undo(&mut world);
    assert_eq!(h.entries.len(), before);
    assert_eq!(
        entry_of(&h, "pair").unwrap()["args"]["props"][0]["position"],
        serde_json::json!([1.0, 0.0, 0.0]),
        "one undo restores the definition and the patch line together"
    );
}

// The baseline value the template hands the form for `key`, so a test can
// author a patch that agrees with the template rather than hard-coding the
// expansion's arithmetic.
fn baseline_of(h: &mut EditorHook, name: &str, key: &str) -> serde_json::Value {
    let mut world = world_with_fields();
    h.open_asset_form(name, &mut world);
    h.form_template
        .as_ref()
        .expect("a template-derived form")
        .baseline
        .get(key)
        .cloned()
        .unwrap_or(serde_json::Value::Null)
}

// Minimize strips patch fields that agree with the template. A patch that
// agrees on everything is not an override at all, so its line goes.
#[test]
fn minimize_removes_a_patch_that_matches_the_template() {
    let _guard = crate::test_support::lock();
    let mut probe = prefab_hook(serde_json::json!({"position": [5.0, 0.0, 0.0]}));
    let inherited = baseline_of(&mut probe, "i1_a", "position");

    let mut h = prefab_hook(serde_json::json!({ "position": inherited }));
    let mut world = world_with_fields();
    let before = h.entries.len();
    h.open_asset_form("i1_a", &mut world);

    let k = entity_option(&h, "Minimize override");
    h.apply_form(FormAction::OpenEntityMenu, &mut world);
    h.apply_form(FormAction::PickEntityOption(k), &mut world);

    assert_eq!(h.entries.len(), before - 1, "a no-op patch is not a patch");
    assert!(entry_of(&h, "i1_a").is_none());
}

// A patch that agrees on one field and differs on another keeps only the
// difference, so a legacy full copy shrinks to a real override.
#[test]
fn minimize_keeps_only_the_fields_that_differ() {
    let _guard = crate::test_support::lock();
    let mut probe = prefab_hook(serde_json::json!({"position": [5.0, 0.0, 0.0]}));
    let inherited = baseline_of(&mut probe, "i1_a", "position");

    let mut h = prefab_hook(serde_json::json!({
        "position": inherited,
        "cull_distance": 42.0
    }));
    let mut world = world_with_fields();
    let before = h.entries.len();
    h.open_asset_form("i1_a", &mut world);

    let k = entity_option(&h, "Minimize override");
    h.apply_form(FormAction::OpenEntityMenu, &mut world);
    h.apply_form(FormAction::PickEntityOption(k), &mut world);

    assert_eq!(
        h.entries.len(),
        before,
        "the line stays for the real override"
    );
    let args = &entry_of(&h, "i1_a").expect("the patch line").clone()["args"];
    assert_eq!(args["cull_distance"], 42.0);
    assert!(
        args.get("position").is_none(),
        "the inherited field was stripped, got {args}"
    );
}

// A preset-backed Prefab has no world line to apply into, so the entity menu
// offers to author one; materializing copies the preset's args verbatim.
#[test]
fn materializing_a_preset_prefab_authors_it_as_a_world_line() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let dir = crate::project::assets_dir()
        .expect("the harness opened a project")
        .join("prefabs");
    std::fs::create_dir_all(&dir).expect("preset dir");
    std::fs::write(
        dir.join("cn_test_prefab.json"),
        r#"{"args":{"props":[{"name":"a","kind":"prop","mesh":"box","position":[2.0,0.0,0.0]}]}}"#,
    )
    .expect("preset file");

    let mut h = hook(vec![
        serde_json::json!({"name":"box","type":"ProceduralMesh","args":{"generator":"box"}}),
        serde_json::json!({"name":"i1","type":"Prop","args":{"prefab":"cn_test_prefab","position":[10.0,0.0,0.0]}}),
        serde_json::json!({"name":"i1_a","type":"Prop","args":{"position":[5.0,0.0,0.0]}}),
    ]);
    h.panel_open = true;
    let mut world = world_with_fields();
    h.open_asset_form("i1_a", &mut world);

    let k = entity_option(&h, "Materialize Prefab");
    h.apply_form(FormAction::OpenEntityMenu, &mut world);
    h.apply_form(FormAction::PickEntityOption(k), &mut world);

    let def = entry_of(&h, "cn_test_prefab").expect("the definition was authored");
    assert_eq!(def["type"], "Prefab");
    assert_eq!(
        def["args"]["props"][0]["position"],
        serde_json::json!([2.0, 0.0, 0.0]),
        "the preset's entries came across verbatim"
    );

    let _ = std::fs::remove_file(dir.join("cn_test_prefab.json"));
}

// The jump walks the form's scroll window to the next overridden field and
// wraps back to the first, so a long form's marks are all reachable.
#[test]
fn jump_to_override_cycles_through_the_marked_fields() {
    let _guard = crate::test_support::lock();
    let mut h = prefab_hook(serde_json::json!({
        "position": [5.0, 0.0, 0.0],
        "scale": [2.0, 2.0, 2.0]
    }));
    let mut world = world_with_fields();
    h.open_asset_form("i1_a", &mut world);

    let marked: Vec<usize> = h
        .form_override_marks()
        .expect("marks for a template form")
        .iter()
        .enumerate()
        .filter(|(_, m)| **m != overrides::FieldOrigin::Inherited)
        .map(|(i, _)| i)
        .collect();
    assert!(
        marked.len() >= 2,
        "both pinned fields are marked, got {marked:?}"
    );

    let max = h.form_fields.len().saturating_sub(h.form_window());
    let mut seen = Vec::new();
    for _ in 0..marked.len() + 1 {
        h.jump_to_override(&mut world);
        seen.push(h.form_scroll);
    }
    assert!(
        seen.iter().all(|s| marked.contains(s) || *s == max),
        "every stop is a marked field (or the scroll clamp), got {seen:?} for {marked:?}"
    );
    assert!(
        seen.first() == seen.last() || seen.len() > marked.len(),
        "the walk wraps rather than sticking, got {seen:?}"
    );
}

// With nothing overridden there is no field to jump to, so the form stays put.
#[test]
fn jump_to_override_is_a_no_op_without_marks() {
    let _guard = crate::test_support::lock();
    let mut h = prefab_hook(serde_json::json!({}));
    let mut world = world_with_fields();
    h.open_asset_form("i1_a", &mut world);
    h.form_scroll = 0;

    h.jump_to_override(&mut world);
    assert_eq!(h.form_scroll, 0);
}

// Change one field of the open template-derived form through its control so a
// confirm has a divergence to commit: a bool toggles, any other visible
// scalar/text field gets a distinct value typed over it. Returns the changed
// field's dotted key.
fn diverge_a_form_field(h: &mut EditorHook, world: &mut World) -> String {
    if let Some(j) = h
        .form_fields
        .iter()
        .position(|f| matches!(f.kind, form::FieldKind::Bool))
    {
        let key = h.form_fields[j].key.clone();
        h.apply_form(FormAction::ToggleField(j), world);
        return key;
    }
    let window = h.form_window();
    let j = h
        .form_fields
        .iter()
        .position(|f| {
            matches!(
                f.kind,
                form::FieldKind::Str | form::FieldKind::Int | form::FieldKind::Float
            )
        })
        .filter(|&j| j < window)
        .expect("a visible scalar field to diverge");
    let key = h.form_fields[j].key.clone();
    widget::seed_field(world, form_panel::form_input(j), "42424");
    key
}

// The heart of the template loop: a generated asset has no world.jsonl line,
// so clicking it opens a form seeded from what the expansion produced. An
// unchanged confirm commits NOTHING (the asset stays pristine); diverging a
// field and confirming appends the minimal patch line -- keeping the generated
// name, which is what makes it patch the expansion.
#[test]
fn editing_a_generated_asset_writes_a_minimal_patch_on_confirm() {
    let _guard = crate::test_support::lock();
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
    assert!(
        h.form_template.is_some(),
        "the form knows its template baseline"
    );
    assert_eq!(
        widget::field_text(&world, form_panel::NAME_INPUT),
        name,
        "the name heading carries the generated name"
    );
    assert!(h.selection.contains(&name), "and the row selects");

    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(
        h.entries.len(),
        before,
        "an unchanged confirm authors nothing"
    );

    h.apply_panel(PanelAction::SelectRow(gi, ai), &mut world);
    let key = diverge_a_form_field(&mut h, &mut world);
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), before + 1, "a divergence appends the line");
    let added = h.entries.last().unwrap();
    assert_eq!(
        entry_name(added),
        Some(name.as_str()),
        "the patch line keeps the generated name, so it patches it"
    );
    let root = key.split('.').next().unwrap().to_string();
    let args = added.get("args").and_then(|a| a.as_object()).unwrap();
    assert_eq!(
        args.keys().collect::<Vec<_>>(),
        vec![&root],
        "only the diverged field is authored"
    );
    assert!(h.dirty && h.tree_stale);
}

// After a divergence is committed, the asset relists once -- still under its
// ORIGIN group, marked Overridden -- rather than moving to World or appearing
// twice. Clicking it again edits the patch line in place, template-aware.
#[test]
fn a_patched_asset_relists_under_its_origin_as_overridden() {
    let _guard = crate::test_support::lock();
    let mut h = expandable_hook();
    let mut world = world_with_fields();
    h.refresh_tree_if_needed();
    let (gi, ai, name) = a_promotable_asset(&h);
    let origin = h.tree_groups[gi].label.clone();
    h.apply_panel(PanelAction::SelectRow(gi, ai), &mut world);
    diverge_a_form_field(&mut h, &mut world);
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
        [origin.as_str()],
        "listed once, under its origin group"
    );
    let patched = h
        .tree_groups
        .iter()
        .flat_map(|g| &g.assets)
        .find(|a| a.name == name)
        .unwrap();
    assert_eq!(patched.badge, asset_tree::Badge::Overridden);

    // Clicking it again edits the patch line in place, still template-aware.
    let (g2, i2) = row_of(&h, &name);
    let before = h.entries.len();
    h.apply_panel(PanelAction::SelectRow(g2, i2), &mut world);
    assert!(matches!(h.form_target, FormTarget::Entry(_)));
    assert!(h.form_template.is_some());
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), before, "edited in place, not appended");
}

// The override loop over a prefab world: two instances of one Prefab, the
// first carrying a patch that pins its position.
fn two_instance_prefab_hook() -> EditorHook {
    isolate_state_dir();
    let mut h = hook(vec![
        serde_json::json!({"name":"box","type":"ProceduralMesh","args":{"generator":"box"}}),
        serde_json::json!({"name":"pair","type":"Prefab","args":{"props":[
            {"name":"a","kind":"prop","mesh":"box","position":[1.0,0.0,0.0]}]}}),
        serde_json::json!({"name":"i1","type":"Prop","args":{"prefab":"pair","position":[10.0,0.0,0.0]}}),
        serde_json::json!({"name":"i2","type":"Prop","args":{"prefab":"pair"}}),
        serde_json::json!({"name":"i1_a","type":"Prop","args":{"position":[5.0,0.0,0.0]}}),
    ]);
    h.panel_open = true;
    h
}

// The patched instance's form marks exactly the pinned field, and its
// override menu offers Revert plus Apply with the blast radius spelled out.
#[test]
fn an_overridden_field_marks_and_offers_revert_and_apply() {
    let _guard = crate::test_support::lock();
    let mut h = two_instance_prefab_hook();
    let mut world = world_with_fields();
    h.open_asset_form("i1_a", &mut world);
    assert!(h.form_template.is_some(), "i1_a derives from the prefab");

    let marks = h.form_override_marks().expect("marks for a template form");
    let j = h
        .form_fields
        .iter()
        .position(|f| f.key == "position")
        .expect("a position field");
    assert_eq!(marks[j], overrides::FieldOrigin::Overridden);
    assert!(
        marks
            .iter()
            .enumerate()
            .filter(|(i, _)| !h.form_fields[*i].key.starts_with("position"))
            .all(|(_, m)| *m == overrides::FieldOrigin::Inherited),
        "only the pinned field is marked"
    );

    let options = h.override_menu_options(j);
    let labels: Vec<&str> = options.iter().map(|(_, l)| l.as_str()).collect();
    assert_eq!(labels[0], "Revert 'position'");
    assert_eq!(labels[1], "Apply to Prefab 'pair' (updates 2 instances)");
}

// Reverting the field removes the patch line outright (it pinned nothing
// else), restoring the pristine instance -- and it is exactly one undo step.
#[test]
fn reverting_the_only_override_removes_the_patch_line_and_undoes_in_one_step() {
    let _guard = crate::test_support::lock();
    let mut h = two_instance_prefab_hook();
    let mut world = world_with_fields();
    let before = h.entries.len();
    h.open_asset_form("i1_a", &mut world);
    let j = h
        .form_fields
        .iter()
        .position(|f| f.key == "position")
        .unwrap();

    h.apply_form(FormAction::OpenOverrideMenu(j), &mut world);
    h.apply_form(FormAction::PickOverrideOption(0), &mut world);
    assert_eq!(h.entries.len(), before - 1, "the patch line is gone");
    assert!(
        !h.entries.iter().any(|e| entry_name(e) == Some("i1_a")),
        "the asset is pristine again"
    );
    assert!(h.form_open(), "the form re-derives instead of closing");
    assert!(
        h.form_override_marks()
            .unwrap()
            .iter()
            .all(|m| *m == overrides::FieldOrigin::Inherited)
    );

    h.undo(&mut world);
    assert_eq!(h.entries.len(), before, "one undo restores the patch line");
    assert!(h.entries.iter().any(|e| entry_name(e) == Some("i1_a")));
}

// Applying the override writes the value back into the Prefab definition
// through the inverse instance transform, drops the patch, and undoes as one
// step.
#[test]
fn applying_an_override_updates_the_prefab_entry_and_drops_the_patch() {
    let _guard = crate::test_support::lock();
    let mut h = two_instance_prefab_hook();
    let mut world = world_with_fields();
    let before = h.entries.len();
    h.open_asset_form("i1_a", &mut world);
    let j = h
        .form_fields
        .iter()
        .position(|f| f.key == "position")
        .unwrap();

    h.apply_form(FormAction::OpenOverrideMenu(j), &mut world);
    h.apply_form(FormAction::PickOverrideOption(1), &mut world);

    let def = h
        .entries
        .iter()
        .find(|e| entry_name(e) == Some("pair"))
        .unwrap();
    // World (5,0,0) under instance position (10,0,0) is local (-5,0,0).
    assert_eq!(
        def["args"]["props"][0]["position"],
        serde_json::json!([-5.0, 0.0, 0.0])
    );
    assert_eq!(
        h.entries.len(),
        before - 1,
        "the emptied patch line is gone; the template now carries the value"
    );

    h.undo(&mut world);
    let def = h
        .entries
        .iter()
        .find(|e| entry_name(e) == Some("pair"))
        .unwrap();
    assert_eq!(
        def["args"]["props"][0]["position"],
        serde_json::json!([1.0, 0.0, 0.0]),
        "one undo restores the definition and the patch line together"
    );
    assert_eq!(h.entries.len(), before);
}

// The entity menu on a patched instance offers Revert-all (and no Apply-all
// for a field set it can fully apply -- position maps, so it is offered too);
// Revert-all deletes the patch line.
#[test]
fn the_entity_menu_reverts_all_overrides() {
    let _guard = crate::test_support::lock();
    let mut h = two_instance_prefab_hook();
    let mut world = world_with_fields();
    let before = h.entries.len();
    h.open_asset_form("i1_a", &mut world);

    let labels: Vec<String> = h
        .entity_menu_options()
        .into_iter()
        .map(|(_, l)| l)
        .collect();
    assert!(
        labels.iter().any(|l| l == "Revert all overrides"),
        "{labels:?}"
    );
    assert!(
        labels
            .iter()
            .any(|l| l == "Apply all to Prefab 'pair' (updates 2 instances)"),
        "{labels:?}"
    );

    h.apply_form(FormAction::OpenEntityMenu, &mut world);
    let k = labels
        .iter()
        .position(|l| l == "Revert all overrides")
        .unwrap();
    h.apply_form(FormAction::PickEntityOption(k), &mut world);
    assert_eq!(h.entries.len(), before - 1);
    assert!(!h.entries.iter().any(|e| entry_name(e) == Some("i1_a")));

    h.undo(&mut world);
    assert_eq!(h.entries.len(), before, "revert-all is one undo step");
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

// Unapplied-edit markers: set by control edits, cleared by the panel's own
// open / apply, and surfaced as a "*" heading suffix.
#[test]
fn unapplied_markers_follow_edit_and_apply() {
    let mut h = hook(vec![entry("cube", "PointLight")]);
    let mut world = world_with_fields();
    // The Edit form: opening starts clean, a control edit marks, closing clears.
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::Entry(0));
    assert!(!h.form_touched);
    assert!(!h.panel_data(&world).form_title.ends_with('*'));
    h.apply_form(FormAction::CycleField(0), &mut world);
    assert!(h.form_touched, "a control edit marks the form");
    assert!(h.panel_data(&world).form_title.ends_with('*'));
    // Focus moves alone do not mark.
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::Entry(0));
    h.apply_form(FormAction::FocusName, &mut world);
    assert!(!h.form_touched, "focus is not an edit");
    h.form_touched = true;
    h.close_form();
    assert!(!h.form_touched, "closing discards the marker");
    // Lighting: re-seeding (open / apply / undo) clears the marker.
    h.lighting_touched = true;
    h.seed_lighting(&mut world);
    assert!(!h.lighting_touched);
    // Story: a changed line marks on commit; loading clears.
    h.story_lines = vec!["hello".to_string()];
    h.story_line = 0;
    world.add_component(TextInput {
        asset_id: story_panel::LINE_INPUT,
        ..Default::default()
    });
    set_field(&mut world, story_panel::LINE_INPUT, "hello edited");
    h.commit_story_line(&world);
    assert!(h.story_touched, "a changed line marks the story");
    // An unchanged commit does not re-mark after a clear.
    h.story_touched = false;
    h.commit_story_line(&world);
    assert!(!h.story_touched, "an identical line is not an edit");
    let dirty_view = h.make_story_view([0.0, 0.0]);
    assert!(!dirty_view.dirty);
}
