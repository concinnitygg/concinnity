//! The editor's two hide mechanisms (`hook/hide.rs`): the per-asset hide and
//! lock that are session state rather than edits, and how an isolate composes
//! with them and with unhide.

use concinnity_core::components::FrameInput;
use concinnity_core::ecs::HiddenAssets;
use concinnity_host::thread::asset_id;

use super::fixtures::{entry, hook, row_of, seed_tree, world_with_input};

use crate::debug_hook::DebugHook;

use crate::editor::hook::tests::fixtures::select;
use crate::editor::panels::assets_panel::PanelAction;

// The row eye and lock are editor-session state: they flip the hook's sets (the
// hidden set publishing as ids each tick) and never touch the entries.
#[test]
fn hide_and_lock_are_session_state_not_edits() {
    asset_id::reset_interner();
    let id = asset_id::intern("box");
    let mut world = world_with_input(FrameInput::default());
    let mut h = hook(vec![entry("box", "Sprite")]);
    h.panel_open = true;
    seed_tree(&mut h, Vec::new());
    let (g, i) = row_of(&h, "box");

    h.apply_panel(PanelAction::ToggleHide(g, i), &mut world);
    h.apply_panel(PanelAction::ToggleLock(g, i), &mut world);
    assert!(h.hidden_assets.contains(&h.handle_for("box")));
    assert!(h.locked_assets.contains(&h.handle_for("box")));
    assert!(!h.dirty, "session toggles are not authored edits");

    h.tick(&mut world);
    let hidden = world
        .resource::<HiddenAssets>()
        .expect("the hook publishes the hidden set every tick");
    assert!(hidden.0.contains(&id), "names resolve to this world's ids");

    h.apply_panel(PanelAction::ToggleHide(g, i), &mut world);
    h.apply_panel(PanelAction::ToggleLock(g, i), &mut world);
    assert!(h.hidden_assets.is_empty() && h.locked_assets.is_empty());
}

// H adds to the manual hide set, Shift+H isolates without mutating it, and
// Ctrl+H clears both; the per-name test composes them (manual wins).
#[test]
fn hide_isolate_and_unhide_compose() {
    let mut h = hook(vec![entry("a", "Sprite"), entry("b", "Sprite")]);
    select(&mut h, &["a"]);
    h.hide_selected();
    assert!(h.hidden_assets.contains(&h.handle_for("a")));

    select(&mut h, &["b"]);
    h.toggle_isolate();
    assert!(h.isolate.is_some());
    assert!(
        h.handle_hidden(&h.handle_for("a")),
        "manual hide survives isolate"
    );
    assert!(
        !h.handle_hidden(&h.handle_for("b")),
        "the isolated selection stays visible"
    );

    h.toggle_isolate();
    assert!(h.isolate.is_none());
    assert!(
        h.handle_hidden(&h.handle_for("a")),
        "leaving isolate restores the manual set"
    );
    assert!(!h.handle_hidden(&h.handle_for("b")));

    h.unhide_all();
    assert!(h.hidden_assets.is_empty() && h.isolate.is_none());
}

// The hide set holds handles, so hiding an anonymous entry follows that entry
// when another removal relabels it, rather than staying on the label.
#[test]
fn a_hidden_anonymous_entry_stays_hidden_through_a_relabel() {
    let anon = || serde_json::json!({"type": "Sprite", "args": {}});
    let mut h = hook(vec![anon(), anon()]);
    select(&mut h, &["Sprite#1"]);
    h.hide_selected();
    h.entries.remove(0);
    assert!(
        h.handle_hidden(&h.handle_for("Sprite#0")),
        "the same entry, relabeled"
    );
    assert_eq!(h.hidden_assets.len(), 1);
}
