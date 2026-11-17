use super::*;
use crate::test_support::asset;
use serde_json::json;

fn errors(assets: &[WorldJsonlAsset], vars: &DeclaredVars) -> Vec<String> {
    let mut errors = Vec::new();
    check_bindings(assets, vars, &mut errors);
    errors.iter().map(ToString::to_string).collect()
}

fn region(id: &str, action: &Value) -> WorldJsonlAsset {
    asset(id, RegisteredType::HitRegion, json!({ "action": action }))
}

fn state(values: &Value) -> WorldJsonlAsset {
    asset(
        "state",
        RegisteredType::UiStateValues,
        json!({ "values": values }),
    )
}

// Each failure names its asset; the ok cases name none.
fn failing(errs: &[String]) -> Vec<String> {
    let mut names: Vec<String> = errs
        .iter()
        .filter_map(|e| e.split('\'').nth(1).map(str::to_string))
        .collect();
    names.sort();
    names
}

#[test]
fn a_var_binding_names_a_declared_variable_and_fits_its_type() {
    let vars = DeclaredVars::from_args(&json!({"vars": [
        {"name": "score", "value": {"int": 0}},
        {"name": "lit", "value": {"bool": false}},
        {"name": "at", "value": {"vec3": [0, 0, 0]}},
    ]}));
    let world = [
        region("ok_add", &json!({"add": {"binding": "var:score", "by": 1}})),
        region("ok_flip", &json!({"flip": "var:lit"})),
        region(
            "ok_set",
            &json!({"set": {"binding": "var:score", "value": 4}}),
        ),
        region("typo", &json!({"flip": "var:typ"})),
        region("flip_int", &json!({"flip": "var:score"})),
        region(
            "set_float",
            &json!({"set": {"binding": "var:score", "value": 0.5}}),
        ),
        region("add_vec", &json!({"add": {"binding": "var:at", "by": 1}})),
    ];
    assert_eq!(
        failing(&errors(&world, &vars)),
        ["add_vec", "flip_int", "set_float", "typo"]
    );
}

#[test]
fn without_a_table_a_var_binding_names_a_variable_a_behavior_uses_whatever_its_type() {
    let world = [
        asset(
            "b",
            RegisteredType::Behavior,
            json!({"on": "tick", "do": [{"set": {"var": "beat", "value": {"int": 1}}}]}),
        ),
        region("ok", &json!({"add": {"binding": "var:beat", "by": 1}})),
        region("ok_flip", &json!({"flip": "var:beat"})),
        region("unused", &json!({"add": {"binding": "var:other", "by": 1}})),
    ];
    assert_eq!(
        failing(&errors(&world, &DeclaredVars::default())),
        ["unused"]
    );
}

#[test]
fn a_state_binding_names_a_declared_value_and_fits_its_type() {
    let world = [
        state(&json!([
            {"name": "grid", "value": true},
            {"name": "zoom", "value": 1.0},
            {"name": "title", "value": "Map"},
        ])),
        region("ok_flip", &json!({"flip": "state:grid"})),
        region(
            "ok_add",
            &json!({"add": {"binding": "state:zoom", "by": 0.5}}),
        ),
        region(
            "ok_set",
            &json!({"set": {"binding": "state:title", "value": "Atlas"}}),
        ),
        region("undeclared", &json!({"flip": "state:grd"})),
        region(
            "add_text",
            &json!({"add": {"binding": "state:title", "by": 1}}),
        ),
        region(
            "set_bool",
            &json!({"set": {"binding": "state:zoom", "value": true}}),
        ),
        asset(
            "label",
            RegisteredType::TextLabel,
            json!({"text_binding": "state:title"}),
        ),
        asset(
            "ok_drag",
            RegisteredType::HitRegion,
            json!({"slider": "state:zoom"}),
        ),
        asset(
            "drag_bool",
            RegisteredType::HitRegion,
            json!({"slider": "state:grid"}),
        ),
        asset(
            "panel",
            RegisteredType::ScrollPanel,
            json!({"groups": [{"collapsed_binding": "state:grid"}, {"collapsed_binding": "state:zoom"}]}),
        ),
    ];
    assert_eq!(
        failing(&errors(&world, &DeclaredVars::default())),
        ["add_text", "drag_bool", "panel", "set_bool", "undeclared"]
    );
}

#[test]
fn a_state_value_is_declared_once_with_a_value() {
    let world = [
        state(&json!([{"name": "a", "value": 1}, {"name": "a", "value": 2}])),
        state(&json!([{"name": "b", "value": null}])),
    ];
    let errs = errors(&world, &DeclaredVars::default());
    assert_eq!(errs.len(), 2, "{errs:?}");
    assert!(
        errs.iter()
            .any(|e| e.contains("`a` is declared more than once"))
    );
    assert!(errs.iter().any(|e| e.contains("`b` starts at no bool")));
}

#[test]
fn a_setting_is_stepped_and_set_as_its_row_allows() {
    let world = [
        region(
            "ok_step",
            &json!({"step": {"binding": "setting:window_mode", "by": -1}}),
        ),
        region(
            "ok_set",
            &json!({"set": {"binding": "setting:window_mode", "value": 0}}),
        ),
        region(
            "ok_slider",
            &json!({"set": {"binding": "setting:exposure", "value": 0.5}}),
        ),
        region("ok_flip", &json!({"flip": "setting:vsync"})),
        region(
            "far_step",
            &json!({"step": {"binding": "setting:window_mode", "by": 2_000_000_000}}),
        ),
        region(
            "zero_step",
            &json!({"step": {"binding": "setting:vsync", "by": 0}}),
        ),
        region(
            "negative",
            &json!({"set": {"binding": "setting:window_mode", "value": -1}}),
        ),
        region(
            "text",
            &json!({"set": {"binding": "setting:vsync", "value": "on"}}),
        ),
        region(
            "add",
            &json!({"add": {"binding": "setting:exposure", "by": 1}}),
        ),
        region(
            "slider_step",
            &json!({"step": {"binding": "setting:exposure", "by": 1}}),
        ),
        region(
            "past",
            &json!({"set": {"binding": "setting:exposure", "value": 2.0}}),
        ),
        asset(
            "label",
            RegisteredType::TextLabel,
            json!({"text_binding": "setting:vsync"}),
        ),
        asset(
            "ok_drag",
            RegisteredType::HitRegion,
            json!({"slider": "setting:exposure"}),
        ),
        asset(
            "drag_cycle",
            RegisteredType::HitRegion,
            json!({"slider": "setting:vsync"}),
        ),
        asset(
            "drag_rebind",
            RegisteredType::HitRegion,
            json!({"slider": "setting:key_forward"}),
        ),
    ];
    assert_eq!(
        failing(&errors(&world, &DeclaredVars::default())),
        [
            "add",
            "drag_cycle",
            "drag_rebind",
            "far_step",
            "negative",
            "past",
            "slider_step",
            "text",
            "zero_step"
        ]
    );
}

#[test]
fn an_unknown_setting_or_kind_is_reported() {
    let world = [region(
        "r",
        &json!({"step": {"binding": "setting:nope", "by": 1}}),
    )];
    let errs = errors(&world, &DeclaredVars::default());
    assert!(errs[0].contains("names no setting"), "{errs:?}");
}

#[test]
fn a_slider_region_fires_no_action() {
    let world = [
        asset(
            "both",
            RegisteredType::HitRegion,
            json!({"slider": "setting:exposure", "action": "quit"}),
        ),
        asset(
            "slider_only",
            RegisteredType::HitRegion,
            json!({"slider": "setting:exposure"}),
        ),
    ];
    let errs = errors(&world, &DeclaredVars::default());
    assert_eq!(failing(&errs), ["both"]);
    assert!(errs[0].contains("fires no `action`"), "{errs:?}");
}
