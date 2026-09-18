//! The /select console command's dispatch. Resolution itself is pure
//! (`editor/select_related.rs`); what is asserted here is what the dispatch adds:
//! which entry list each relationship is fed, that a hit replaces the selection
//! wholesale, and that a miss reports without disturbing what was selected.

use concinnity_core::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{select, selected};
use crate::test_support::isolate_state_dir;

fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), entries)
}

// A prop pair over a shared mesh, plus the material only the first references.
fn entries() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({"type":"Prop","args":{"$id":"p1","mesh":"box","material":"mat"}}),
        serde_json::json!({"type":"Prop","args":{"$id":"p2","mesh":"box"}}),
        serde_json::json!({"type":"Material","args":{"$id":"mat"}}),
    ]
}

fn log(h: &EditorHook) -> String {
    h.console_sink
        .window(0, 64)
        .iter()
        .map(|l| l.text.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

// `/select type` replaces the selection with every working entry of that type.
#[test]
fn select_type_replaces_the_selection() {
    let mut world = World::new();
    let mut h = hook(entries());
    select(&mut h, &["mat"]);

    h.run_console_line(&mut world, "/select type Prop");

    assert_eq!(
        selected(&h),
        vec!["p1", "p2"],
        "the props replace the material"
    );
    assert!(log(&h).contains("selected 2"));
}

// `/select using` replaces the selection with everything referencing the target.
#[test]
fn select_using_replaces_the_selection_with_the_referencing_assets() {
    let mut world = World::new();
    let mut h = hook(entries());

    h.run_console_line(&mut world, "/select using box");
    assert_eq!(selected(&h), vec!["p1", "p2"]);

    h.run_console_line(&mut world, "/select using mat");
    assert_eq!(
        selected(&h),
        vec!["p1"],
        "a narrower target replaces rather than extends"
    );
}

// A relationship that resolves to nothing reports it and leaves the selection
// alone, so a mistyped target cannot silently clear what the user had picked.
#[test]
fn a_relationship_with_no_matches_leaves_the_selection_alone() {
    let mut world = World::new();
    let mut h = hook(entries());
    select(&mut h, &["p1"]);

    h.run_console_line(&mut world, "/select using nothing");
    assert_eq!(selected(&h), vec!["p1"]);
    assert!(log(&h).contains("nothing references nothing"));

    h.run_console_line(&mut world, "/select type Nope");
    assert_eq!(selected(&h), vec!["p1"]);
    assert!(log(&h).contains("no assets of type Nope"));
}

// `/select origin` needs an active member to group from.
#[test]
fn select_origin_without_a_selection_errors() {
    let mut world = World::new();
    let mut h = hook(entries());

    h.run_console_line(&mut world, "/select origin");
    assert!(selected(&h).is_empty());
    assert!(log(&h).contains("nothing selected"));
}

// `/select origin` groups through a fresh cook, so what it gathers matches the
// outliner's grouping rather than the raw working list.
#[test]
fn select_origin_gathers_the_active_members_group() {
    isolate_state_dir();
    let _guard = crate::test_support::lock();
    let mut world = World::new();
    let mut h = hook(vec![
        serde_json::json!({"type":"PhysicsConfig","args":{"$id":"phys"}}),
        serde_json::json!({"type":"Camera3D","args":{"$id":"cam"}}),
    ]);
    select(&mut h, &["cam"]);

    h.run_console_line(&mut world, "/select origin");

    let members = selected(&h);
    assert!(
        members.iter().any(|n| n == "cam") && members.iter().any(|n| n == "phys"),
        "both authored assets share the World group, got {members:?}"
    );
    assert!(log(&h).contains("selected"));
}

// A name that cooks away into no group reports rather than clearing.
#[test]
fn select_origin_reports_a_name_no_group_lists() {
    isolate_state_dir();
    let _guard = crate::test_support::lock();
    let mut world = World::new();
    let mut h = hook(vec![
        serde_json::json!({"type":"PhysicsConfig","args":{"$id":"phys"}}),
    ]);
    select(&mut h, &["ghost"]);

    h.run_console_line(&mut world, "/select origin");

    assert_eq!(selected(&h), vec!["ghost"]);
    assert!(log(&h).contains("no origin group lists ghost"));
}
