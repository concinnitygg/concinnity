// src/editor/hook/tests/edit/behavior_tests.rs
//
// The Behavior panel's actions (`hook/edit/behavior.rs`): opening a behavior
// and stepping between them, appending and removing one, the palette that
// fills a node's fields, the value and name fields' commit rules, the checker
// message the status line carries and the row it blames, the three views and
// the cards each draws, and the palette's filter. What the keyboard does with
// these actions is `tests/behavior_keys_tests.rs`.

use concinnity_core::components::InputKey;
use concinnity_core::components::TextInput;
use concinnity_core::ecs::World;

use crate::editor::behavior;
use crate::editor::behavior::graph::CardKind;
use crate::editor::behavior::panel::{BehaviorAction, Status, ViewMode};
use crate::editor::behavior::path;
use crate::editor::hook::tests::fixtures::{
    behavior, behavior_escape_input, behavior_row, behavior_session, entry, open_args,
    press_behavior_key, press_remove, select_behavior, story_key_input, type_name,
};
use crate::editor::hook::{EditorHook, entry_name};

use crate::editor::panels::registry::PanelKey;

use crate::editor::widget;

// Behavior panel

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

    widget::seed_field(&mut world, behavior::panel::VALUE_INPUT, "2.5");
    h.behavior_keys(&mut world, &story_key_input(InputKey::Enter));
    assert_eq!(open_args(&h)["delay"], serde_json::json!(2.5));

    widget::seed_field(&mut world, behavior::panel::VALUE_INPUT, "soon");
    h.behavior_keys(&mut world, &story_key_input(InputKey::Enter));
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
        widget::field_text(&world, behavior::panel::NAME_INPUT),
        "greet",
        "the field opens on the name it is about to replace"
    );

    type_name(&mut world, "  welcome  ");
    h.behavior_keys(&mut world, &story_key_input(InputKey::Enter));
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
    h.behavior_keys(&mut world, &story_key_input(InputKey::Enter));
    assert_eq!(h.behavior_data().name, "chase_1");
    assert_eq!(
        widget::field_text(&world, behavior::panel::NAME_INPUT),
        "chase_1",
        "the field shows what the world holds, not what was typed"
    );

    // Committing a name unchanged is not a collision with itself.
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    h.behavior_keys(&mut world, &story_key_input(InputKey::Enter));
    assert_eq!(h.behavior_data().name, "chase_1");
}

#[test]
fn behavior_rename_refuses_a_blank_name() {
    let (mut h, mut world) = behavior_session(vec![behavior("greet", serde_json::json!({}))]);
    h.apply_behavior_action(BehaviorAction::FocusName, &mut world, [0.0, 0.0]);
    type_name(&mut world, "   ");
    h.behavior_keys(&mut world, &story_key_input(InputKey::Enter));

    assert_eq!(h.behavior_data().name, "greet", "nothing was written");
    assert!(!h.dirty, "and no edit was recorded");
    let Some(Status::Error { message: e, .. }) = &h.behavior_status else {
        panic!("expected the panel to say why, got {:?}", h.behavior_status);
    };
    assert!(e.contains("needs a name"), "{e}");
    assert_eq!(
        widget::field_text(&world, behavior::panel::NAME_INPUT),
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
    h.behavior_keys(&mut world, &story_key_input(InputKey::Enter));

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
        widget::field_text(&world, behavior::panel::NAME_INPUT),
        "greet"
    );
    h.behavior_keys(&mut world, &story_key_input(InputKey::Enter));
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
    widget::seed_field(&mut world, behavior::panel::VALUE_INPUT, "half typed");
    let snapshot = EditorHook::field_snapshot(&world);
    let mut fresh = World::new();
    for id in behavior::panel::all_field_ids() {
        fresh.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    EditorHook::restore_fields(&mut fresh, &snapshot);
    assert_eq!(
        widget::field_text(&fresh, behavior::panel::VALUE_INPUT),
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
        behavior::panel::chart_canvas(h.effective_size(PanelKey::Behavior), ViewMode::Chart);
    assert_eq!(
        h.behavior_pan,
        crate::editor::behavior::chart::clamp_pan(h.behavior_pan, &chart, canvas)
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
    let band = behavior::panel::chart_band(
        [0.0, 0.0],
        h.effective_size(PanelKey::Behavior),
        ViewMode::Chart,
    );
    let rect = crate::editor::behavior::chart::card_rect(card, band, h.behavior_pan);
    assert!(rect[0] >= band[0], "{rect:?} left of {band:?}");
    assert!(rect[0] + rect[2] <= band[0] + band[2] + 0.01, "{rect:?}");
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
            .and_then(behavior::panel::Status::error)
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
    assert!(h.make_behavior_view(&data, [0.0, 0.0]).status == Some(&behavior::panel::Status::Ok));

    let (h, _world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "spawned", "do": []}),
    )]);
    let data = h.behavior_data();
    let view = h.make_behavior_view(&data, [0.0, 0.0]);
    assert!(
        view.status
            .and_then(behavior::panel::Status::error)
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
    widget::seed_field(world, behavior::panel::FILTER_INPUT, text);
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

    press_behavior_key(&mut h, &mut world, InputKey::Enter);
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
    press_behavior_key(&mut h, &mut world, InputKey::Enter);
    assert!(h.behavior_filter.is_empty());
    assert_eq!(
        widget::field_text(&world, behavior::panel::FILTER_INPUT),
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
        press_behavior_key(&mut h, &mut world, InputKey::Down);
    }
    assert_eq!(h.behavior_pick, 4);

    // A query keeping fewer options than that would have stranded the highlight.
    type_filter(&mut h, &mut world, "spawn");
    assert_eq!(h.behavior_pick, 0);
    assert_eq!(h.behavior_pick_scroll, 0);
    assert!(h.behavior_data().matches.len() <= 4);

    press_behavior_key(&mut h, &mut world, InputKey::Enter);
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
    press_behavior_key(&mut h, &mut world, InputKey::Enter);
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
    let field = layers[&behavior::panel::FILTER_INPUT];
    assert!(
        field > layers[&behavior::panel::PANEL_BG],
        "the field sank into its own panel"
    );
    assert_eq!(
        field,
        layers[&behavior::panel::DROP_BG],
        "the field and the backing it sits in share a layer"
    );
}
