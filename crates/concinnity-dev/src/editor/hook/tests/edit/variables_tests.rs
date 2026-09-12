// src/editor/hook/tests/edit/variables_tests.rs
//
// The Variables panel's actions (`hook/edit/variables.rs`): the rows the
// behaviors' usage supplies before any table exists, the table a first
// declaration creates, the warning a declared table earns for a name it leaves
// out, retyping and renaming a variable, the starting values it accepts, what
// removing a declaration leaves behind, and the re-check each change runs over
// the behaviors reading it.

use concinnity_core::components::InputKey;
use concinnity_core::components::TextInput;
use concinnity_core::ecs::World;

use crate::editor::behavior;
use crate::editor::behavior::panel::BehaviorAction;
use crate::editor::hook::tests::fixtures::{behavior, entry_with_args, hook, story_key_input};
use crate::editor::hook::{EditorHook, entry_type};

use crate::editor::panels::registry::{self, PanelKey};

use crate::editor::panels::variables_panel::{self, VariablesAction};
use crate::editor::widget;

fn variables_session(entries: Vec<serde_json::Value>) -> (EditorHook, World) {
    let mut world = World::new();
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
    h.variables_keys(&mut world, &story_key_input(InputKey::Enter));
    assert_eq!(
        table_args(&h)["vars"][0]["value"]["vec3"],
        serde_json::json!([1.0, 2.0, 3.0]),
    );

    // Something that is not a vector leaves the declaration as it was, and the
    // field goes back to what the table holds.
    h.apply_variables_action(VariablesAction::FocusValue, &mut world);
    widget::seed_field(&mut world, variables_panel::VALUE_INPUT, "nonsense");
    h.variables_keys(&mut world, &story_key_input(InputKey::Enter));
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
    h.variables_keys(&mut world, &story_key_input(InputKey::Enter));
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
    h.variables_keys(&mut world, &story_key_input(InputKey::Enter));
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
    assert_eq!(h.behavior_status, Some(behavior::panel::Status::Ok));

    // The types cycle bool -> int -> float -> vec3, so two steps from int lands
    // on a type an int value no longer satisfies.
    select_var(&mut h, &mut world, "score");
    h.apply_variables_action(VariablesAction::Retype, &mut world);
    h.apply_variables_action(VariablesAction::Retype, &mut world);
    assert_eq!(h.variables_data().rows[0].ty, "vec3");
    let status = h
        .behavior_status
        .as_ref()
        .and_then(behavior::panel::Status::error)
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
