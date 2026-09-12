// src/editor/hook/tests/behavior_keys_tests.rs
//
// The Behavior panel's keyboard (`hook/behavior_keys.rs`): the arrows stepping
// the outline, the chart chain and the overview, Enter opening the palette and
// picking from it, Escape unwinding one waiting state at a time, Tab cycling
// the views, and the clipboard verbs. Each also asserts what the keys stand
// down for: a focused field keeps them.

use concinnity_core::components::FrameInput;
use concinnity_core::components::InputKey;

use super::fixtures::{
    behavior, behavior_escape_input, behavior_session, open_args, press_behavior_key, press_remove,
    select_behavior, type_name,
};
use crate::editor::behavior;
use crate::editor::behavior::panel::{BehaviorAction, ViewMode};
use crate::editor::hook::EditorHook;

use crate::editor::widget;

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

    press_behavior_key(&mut h, &mut world, InputKey::Down);
    assert_eq!(h.behavior_row, Some(0));
    press_behavior_key(&mut h, &mut world, InputKey::Down);
    assert_eq!(h.behavior_row, Some(1));
    press_behavior_key(&mut h, &mut world, InputKey::Up);
    assert_eq!(h.behavior_row, Some(0));
    press_behavior_key(&mut h, &mut world, InputKey::Up);
    assert_eq!(h.behavior_row, Some(0), "the top of the list does not wrap");

    // Left and Right have nothing to follow in a list.
    press_behavior_key(&mut h, &mut world, InputKey::Right);
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
        press_behavior_key(&mut h, &mut world, InputKey::Down);
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
    press_behavior_key(&mut h, &mut world, InputKey::Right);
    assert_eq!(selected_card_title(&h).as_deref(), Some("on tick"));
    press_behavior_key(&mut h, &mut world, InputKey::Right);
    assert_eq!(selected_card_title(&h).as_deref(), Some("if"));
    press_behavior_key(&mut h, &mut world, InputKey::Right);
    assert_eq!(selected_card_title(&h).as_deref(), Some("show"));

    press_behavior_key(&mut h, &mut world, InputKey::Down);
    assert_eq!(selected_card_title(&h).as_deref(), Some("hide"));
    press_behavior_key(&mut h, &mut world, InputKey::Up);
    assert_eq!(selected_card_title(&h).as_deref(), Some("show"));
    press_behavior_key(&mut h, &mut world, InputKey::Left);
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

    press_behavior_key(&mut h, &mut world, InputKey::Right);
    assert_eq!(selected_overview_title(&h).as_deref(), Some("score"));
    // A variable card stands for no behavior, so Enter leaves the map alone.
    press_behavior_key(&mut h, &mut world, InputKey::Enter);
    assert_eq!(h.behavior_mode, ViewMode::Overview);

    press_behavior_key(&mut h, &mut world, InputKey::Right);
    assert_eq!(selected_overview_title(&h).as_deref(), Some("react"));
    press_behavior_key(&mut h, &mut world, InputKey::Enter);
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

    press_behavior_key(&mut h, &mut world, InputKey::Enter);
    assert!(h.behavior_picking, "Enter opened the palette");
    assert_eq!(h.behavior_pick, 0);

    let second = h.behavior_data().picks[1].verb;
    press_behavior_key(&mut h, &mut world, InputKey::Down);
    assert_eq!(h.behavior_pick, 1);
    press_behavior_key(&mut h, &mut world, InputKey::Enter);

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
    press_behavior_key(&mut h, &mut world, InputKey::Enter);
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
    press_behavior_key(&mut h, &mut world, InputKey::Enter);
    let total = h.behavior_data().picks.len();
    assert!(
        total > behavior::panel::PICK_POOL,
        "the node vocabulary overflows the palette"
    );

    for _ in 0..behavior::panel::PICK_POOL {
        press_behavior_key(&mut h, &mut world, InputKey::Down);
    }
    assert_eq!(h.behavior_pick, behavior::panel::PICK_POOL);
    assert!(h.behavior_pick_scroll > 0, "the window followed it down");
    assert!(h.behavior_pick >= h.behavior_pick_scroll);
    assert!(h.behavior_pick < h.behavior_pick_scroll + behavior::panel::PICK_POOL);

    // And back up again, dragging the window with it.
    for _ in 0..behavior::panel::PICK_POOL {
        press_behavior_key(&mut h, &mut world, InputKey::Up);
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
    press_behavior_key(&mut h, &mut world, InputKey::Enter);
    assert!(h.behavior_picking);
    h.behavior_keys(&mut world, &behavior_escape_input());
    assert!(!h.behavior_picking, "the palette closed without picking");
    assert_eq!(open_args(&h)["do"].as_array().map(Vec::len), Some(1));

    press_remove(&mut h, &mut world);
    assert!(h.behavior_remove_armed);
    h.behavior_keys(&mut world, &behavior_escape_input());
    assert!(!h.behavior_remove_armed, "the armed removal was canceled");
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
        widget::field_text(&world, behavior::panel::NAME_INPUT),
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

    for key in [InputKey::Left, InputKey::Right] {
        press_behavior_key(&mut h, &mut world, key);
        assert_eq!(h.behavior_row, row, "{key:?} moved the selection");
    }
    press_behavior_key(&mut h, &mut world, InputKey::Down);
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
    press_behavior_key(&mut h, &mut world, InputKey::Down);
    assert_eq!(h.behavior_row, None, "the arrows did not reach the outline");

    h.behavior_keys(&mut world, &behavior_escape_input());
    press_behavior_key(&mut h, &mut world, InputKey::Down);
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
        press_behavior_key(&mut h, &mut world, InputKey::Tab);
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
    press_behavior_key(&mut h, &mut world, InputKey::Tab);
    assert_eq!(h.behavior_mode, ViewMode::Outline);
    assert!(h.behavior_name_focus, "the field kept the keyboard");

    h.behavior_keys(&mut world, &behavior_escape_input());
    press_behavior_key(&mut h, &mut world, InputKey::Tab);
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
    press_behavior_key(&mut h, &mut world, InputKey::Enter);
    assert!(h.behavior_picking);

    press_behavior_key(&mut h, &mut world, InputKey::Tab);
    assert_eq!(h.behavior_mode, ViewMode::Outline);
    assert!(h.behavior_picking, "the palette is still up");
    assert_eq!(open_args(&h)["do"].as_array().map(Vec::len), Some(0));
}

fn ctrl_key_input(key: InputKey) -> FrameInput {
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
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::C));
    assert!(h.behavior_clip.is_some(), "the node is held");
    assert_eq!(body_verbs(&h), ["save", "hide"], "copying wrote nothing");

    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::V));
    assert_eq!(body_verbs(&h), ["save", "hide", "hide"]);
    // Still held, so a second paste lands beside the first copy.
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::V));
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
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::C));

    select_behavior(&mut h, &mut world, "hide");
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::D));
    assert_eq!(body_verbs(&h), ["save", "hide", "hide"]);

    // What was held is still the `save`, and pastes as one.
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::V));
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
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::C));

    h.apply_behavior_action(BehaviorAction::Step(1), &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_data().name, "greet");
    assert!(h.behavior_clip.is_some(), "opening another kept it");

    // The empty body's own row is the list, so a paste there appends.
    select_behavior(&mut h, &mut world, "do");
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::V));
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
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::C));

    select_behavior(&mut h, &mut world, "scope");
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::V));
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
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::C));
    assert!(h.behavior_clip.is_none());
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::D));
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
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::D));
    assert_eq!(body_verbs(&h), ["let"], "nothing was duplicated");

    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    h.behavior_keys(&mut world, &ctrl_key_input(InputKey::D));
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
