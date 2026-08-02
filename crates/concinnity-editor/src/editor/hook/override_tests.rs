// src/editor/hook/override_tests.rs
//
// The bulk and definition-level halves of the override loop
// (`hook/override_edit.rs`): apply-all across a mixed patch, minimizing a patch
// back down to what actually differs, materializing a preset-backed Prefab so
// its entries become editable, and the jump that walks the form to its next
// marked field. The single-field revert / apply pair is covered beside the form
// drive in `hook/tests.rs`.

use super::*;
use crate::test_support::isolate_state_dir;

fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), entries)
}

// The injected typed fields the form reads its controls back from.
fn world_with_fields() -> World {
    let mut world = World::new_empty();
    for id in panel::all_field_ids()
        .into_iter()
        .chain(form_panel::all_field_ids())
    {
        world.add_component(crate::assets::TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    world
}

// One Prefab with a single prop entry, one instance of it, and the patch line
// `patch` pinning fields on that instance's generated asset.
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

fn entry<'a>(h: &'a EditorHook, name: &str) -> Option<&'a serde_json::Value> {
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
    let def = entry(&h, "pair").expect("the definition stands");
    assert_eq!(
        def["args"]["props"][0]["position"],
        serde_json::json!([-5.0, 0.0, 0.0]),
        "the mappable path landed in the definition"
    );

    let patch = entry(&h, "i1_a").expect("the patch line survives its unmappable field");
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
        entry(&h, "pair").unwrap()["args"]["props"][0]["position"],
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
    assert!(entry(&h, "i1_a").is_none());
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
    let args = &entry(&h, "i1_a").expect("the patch line").clone()["args"];
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
    let dir = concinnity_core::paths::assets_dir().join("prefabs");
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

    let def = entry(&h, "cn_test_prefab").expect("the definition was authored");
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
