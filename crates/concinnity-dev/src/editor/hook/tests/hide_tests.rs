// src/editor/hook/tests/hide_tests.rs
//
// The editor's two hide mechanisms (`hook/hide.rs`): the per-asset hide and
// lock that are session state rather than edits, and how an isolate composes
// with them and with unhide.

use concinnity_core::components::FrameInput;
use concinnity_core::ecs::HiddenAssets;
use concinnity_host::thread::asset_id;

use super::fixtures::{entry, hook, row_of, seed_tree, world_with_input};

use crate::debug_hook::DebugHook;

use crate::editor::panels::panel::PanelAction;

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
    assert!(h.hidden_assets.contains("box"));
    assert!(h.locked_assets.contains("box"));
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
    h.selection.replace("a".to_string());
    h.hide_selected();
    assert!(h.hidden_assets.contains("a"));

    h.selection.replace("b".to_string());
    h.toggle_isolate();
    assert!(h.isolate.is_some());
    assert!(h.name_hidden("a"), "manual hide survives isolate");
    assert!(!h.name_hidden("b"), "the isolated selection stays visible");

    h.toggle_isolate();
    assert!(h.isolate.is_none());
    assert!(
        h.name_hidden("a"),
        "leaving isolate restores the manual set"
    );
    assert!(!h.name_hidden("b"));

    h.unhide_all();
    assert!(h.hidden_assets.is_empty() && h.isolate.is_none());
}
