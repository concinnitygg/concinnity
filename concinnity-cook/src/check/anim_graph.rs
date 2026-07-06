// src/check/anim_graph.rs
//
// Structural validation of AnimGraph args: state/parameter/transition shape
// and value ranges. Cross-asset name lookups (target SkinnedMesh, clip
// Animations) are handled by the AnimGraph `CrossReferenced` impl, and the
// graph-ownership rules (one graph per mesh, no unreferenced clips) are
// world-global passes in crate::check::cross_reference.

use serde_json::Value;

const OPS: [&str; 6] = ["lt", "le", "gt", "ge", "eq", "ne"];

pub(crate) fn check(name: &str, args: &Value) -> Result<(), String> {
    let err = |detail: String| Err(format!("AnimGraph '{name}': {detail}"));

    let states = args
        .get("states")
        .and_then(|v| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[]);
    if states.is_empty() {
        return err("`states` must declare at least one state".into());
    }

    let mut state_names: Vec<&str> = Vec::with_capacity(states.len());
    for (i, state) in states.iter().enumerate() {
        let state_name = state.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if state_name.is_empty() {
            return err(format!("state #{i} has no `name`"));
        }
        if state_names.contains(&state_name) {
            return err(format!("duplicate state name '{state_name}'"));
        }
        state_names.push(state_name);
        if let Some(rate) = state.get("rate").and_then(|v| v.as_f64())
            && rate <= 0.0
        {
            return err(format!("state '{state_name}': `rate` must be positive"));
        }
    }

    let mut param_names: Vec<&str> = Vec::new();
    for (i, param) in args
        .get("parameters")
        .and_then(|v| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[])
        .iter()
        .enumerate()
    {
        let param_name = param.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if param_name.is_empty() {
            return err(format!("parameter #{i} has no `name`"));
        }
        if param_names.contains(&param_name) {
            return err(format!("duplicate parameter name '{param_name}'"));
        }
        param_names.push(param_name);
    }

    if let Some(initial) = args.get("initial").and_then(|v| v.as_str())
        && !initial.is_empty()
        && !state_names.contains(&initial)
    {
        return err(format!(
            "`initial` state '{initial}' is not a declared state"
        ));
    }

    for (i, tr) in args
        .get("transitions")
        .and_then(|v| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[])
        .iter()
        .enumerate()
    {
        for field in ["from", "to"] {
            let target = tr.get(field).and_then(|v| v.as_str()).unwrap_or("");
            if target.is_empty() {
                return err(format!("transition #{i} has no `{field}` state"));
            }
            if !state_names.contains(&target) {
                return err(format!(
                    "transition #{i}: `{field}` state '{target}' is not a declared state"
                ));
            }
        }
        if let Some(d) = tr.get("duration_secs").and_then(|v| v.as_f64())
            && d < 0.0
        {
            return err(format!(
                "transition #{i}: `duration_secs` must not be negative"
            ));
        }
        if let Some(e) = tr.get("exit_time").and_then(|v| v.as_f64())
            && !(0.0..=1.0).contains(&e)
        {
            return err(format!(
                "transition #{i}: `exit_time` must be within 0 to 1"
            ));
        }
        for (j, cond) in tr
            .get("conditions")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            let param = cond.get("parameter").and_then(|v| v.as_str()).unwrap_or("");
            if param.is_empty() {
                return err(format!("transition #{i} condition #{j} has no `parameter`"));
            }
            if !param_names.contains(&param) {
                return err(format!(
                    "transition #{i} condition #{j}: parameter '{param}' is not declared"
                ));
            }
            if let Some(op) = cond.get("op").and_then(|v| v.as_str())
                && !OPS.contains(&op)
            {
                return err(format!(
                    "transition #{i} condition #{j}: unknown op '{op}' (expected one of {})",
                    OPS.join(", ")
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Value {
        serde_json::json!({
            "target": "hero",
            "parameters": [{"name": "speed"}],
            "initial": "idle",
            "states": [
                {"name": "idle", "clip": "hero_idle"},
                {"name": "run", "clip": "hero_run"}
            ],
            "transitions": [
                {"from": "idle", "to": "run", "duration_secs": 0.2,
                 "conditions": [{"parameter": "speed", "op": "gt", "value": 0.5}]}
            ]
        })
    }

    #[test]
    fn valid_graph_passes() {
        assert!(check("g", &base()).is_ok());
    }

    #[test]
    fn empty_states_fails() {
        let e = check("g", &serde_json::json!({"target":"hero"})).unwrap_err();
        assert!(e.contains("at least one state"));
    }

    #[test]
    fn duplicate_state_name_fails() {
        let mut v = base();
        v["states"][1]["name"] = serde_json::json!("idle");
        assert!(check("g", &v).unwrap_err().contains("duplicate state"));
    }

    #[test]
    fn duplicate_parameter_name_fails() {
        let mut v = base();
        v["parameters"] = serde_json::json!([{"name":"speed"},{"name":"speed"}]);
        assert!(check("g", &v).unwrap_err().contains("duplicate parameter"));
    }

    #[test]
    fn non_positive_rate_fails() {
        let mut v = base();
        v["states"][0]["rate"] = serde_json::json!(0.0);
        assert!(check("g", &v).unwrap_err().contains("rate"));
    }

    #[test]
    fn unknown_initial_fails() {
        let mut v = base();
        v["initial"] = serde_json::json!("ghost");
        assert!(check("g", &v).unwrap_err().contains("ghost"));
    }

    #[test]
    fn transition_to_unknown_state_fails() {
        let mut v = base();
        v["transitions"][0]["to"] = serde_json::json!("ghost");
        assert!(check("g", &v).unwrap_err().contains("ghost"));
    }

    #[test]
    fn undeclared_condition_parameter_fails() {
        let mut v = base();
        v["transitions"][0]["conditions"][0]["parameter"] = serde_json::json!("nope");
        assert!(check("g", &v).unwrap_err().contains("nope"));
    }

    #[test]
    fn unknown_op_fails() {
        let mut v = base();
        v["transitions"][0]["conditions"][0]["op"] = serde_json::json!("between");
        assert!(check("g", &v).unwrap_err().contains("between"));
    }

    #[test]
    fn out_of_range_exit_time_fails() {
        let mut v = base();
        v["transitions"][0]["exit_time"] = serde_json::json!(1.5);
        assert!(check("g", &v).unwrap_err().contains("exit_time"));
    }

    #[test]
    fn negative_duration_fails() {
        let mut v = base();
        v["transitions"][0]["duration_secs"] = serde_json::json!(-0.1);
        assert!(check("g", &v).unwrap_err().contains("duration_secs"));
    }
}
