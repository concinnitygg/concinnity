// Structural validation of AnimationGraph args: state/parameter/transition shape
// and value ranges. Cross-asset name lookups (target SkinnedMesh, clip
// Animations) are handled by the AnimationGraph `CrossReferenced` impl, and the
// graph-ownership rules (one graph per mesh, no unreferenced clips) are
// world-global passes in crate::check::cross_reference.

use serde_json::Value;

const OPS: [&str; 6] = ["lt", "le", "gt", "ge", "eq", "ne"];

pub(crate) fn check(name: &str, args: &Value) -> Result<(), String> {
    let err = |detail: String| Err(format!("AnimationGraph '{name}': {detail}"));

    let states = args
        .get("states")
        .and_then(|v| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[]);
    if states.is_empty() {
        return err("`states` must declare at least one state".into());
    }

    let mut param_names_owned: Vec<&str> = Vec::new();
    for param in args
        .get("parameters")
        .and_then(|v| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[])
    {
        if let Some(param_name) = param.get("name").and_then(|v| v.as_str()) {
            param_names_owned.push(param_name);
        }
    }
    let declared_param = |name: &str| param_names_owned.contains(&name);

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
        let has_clip = state
            .get("clip")
            .and_then(|v| v.as_str())
            .is_some_and(|c| !c.is_empty());
        match (has_clip, state.get("blend")) {
            (true, Some(_)) => {
                return err(format!(
                    "state '{state_name}' sets both `clip` and `blend`; pick one"
                ));
            }
            (false, None) => {
                return err(format!("state '{state_name}' has no `clip` or `blend`"));
            }
            (false, Some(blend)) => {
                check_blend(name, state_name, blend, &declared_param)?;
            }
            (true, None) => {}
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

    // IK chains: three named joints, a usable pole, and a declared weight
    // parameter. Joint names resolve against the target skeleton at runtime
    // (the skeleton may come from a binary payload the check never sees).
    for (i, chain) in args
        .get("ik_chains")
        .and_then(|v| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[])
        .iter()
        .enumerate()
    {
        let joints = chain
            .get("joints")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        if joints.len() != 3 {
            return err(format!(
                "ik_chains[{i}]: `joints` must name exactly three joints (root, middle, end), \
                 got {}",
                joints.len()
            ));
        }
        for (j, joint) in joints.iter().enumerate() {
            if joint.as_str().is_none_or(|s| s.is_empty()) {
                return err(format!("ik_chains[{i}]: joints[{j}] must be a joint name"));
            }
        }
        if let Some(pole) = chain.get("pole").and_then(|v| v.as_array()) {
            let mag: f64 = pole.iter().filter_map(|v| v.as_f64()).map(|v| v * v).sum();
            if pole.len() != 3 || mag < 1.0e-8 {
                return err(format!(
                    "ik_chains[{i}]: `pole` must be a non-zero [x, y, z] bend direction"
                ));
            }
        }
        if let Some(param) = chain.get("weight_parameter").and_then(|v| v.as_str())
            && !param.is_empty()
            && !declared_param(param)
        {
            return err(format!(
                "ik_chains[{i}]: weight parameter '{param}' is not declared in `parameters`"
            ));
        }
    }

    Ok(())
}

// Structural validation of one blendspace node. Clip-name resolution stays
// with the cross-reference pass; this checks shape: known `kind`, declared
// parameters, ascending axis values, and complete 2D grids.
fn check_blend(
    graph: &str,
    state: &str,
    blend: &Value,
    declared_param: &impl Fn(&str) -> bool,
) -> Result<(), String> {
    let err = |detail: String| {
        Err(format!(
            "AnimationGraph '{graph}': state '{state}': {detail}"
        ))
    };
    let ascending = |v: &[f64]| v.windows(2).all(|w| w[0] < w[1]);
    let param_of = |field: &str| -> Result<(), String> {
        let p = blend.get(field).and_then(|v| v.as_str()).unwrap_or("");
        if p.is_empty() {
            return err(format!("blend has no `{field}`"));
        }
        if !declared_param(p) {
            return err(format!("blend {field} '{p}' is not a declared parameter"));
        }
        Ok(())
    };
    let axis_of = |field: &str| -> Result<Vec<f64>, String> {
        let values: Vec<f64> = blend
            .get(field)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
            .unwrap_or_default();
        if values.is_empty() {
            return Err(format!(
                "AnimationGraph '{graph}': state '{state}': blend `{field}` must be a non-empty \
                 number array"
            ));
        }
        if !ascending(&values) {
            return Err(format!(
                "AnimationGraph '{graph}': state '{state}': blend `{field}` must be strictly \
                 ascending"
            ));
        }
        Ok(values)
    };

    match blend.get("kind").and_then(|v| v.as_str()).unwrap_or("") {
        "blend1d" => {
            param_of("parameter")?;
            let points = blend
                .get("points")
                .and_then(|v| v.as_array())
                .map(|a| a.as_slice())
                .unwrap_or(&[]);
            if points.is_empty() {
                return err("blend has no `points`".into());
            }
            let mut values = Vec::with_capacity(points.len());
            for (j, point) in points.iter().enumerate() {
                let Some(value) = point.get("value").and_then(|v| v.as_f64()) else {
                    return err(format!("blend point #{j} has no numeric `value`"));
                };
                values.push(value);
                if point
                    .get("clip")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .is_empty()
                {
                    return err(format!("blend point #{j} has no `clip`"));
                }
            }
            if !ascending(&values) {
                return err("blend point `value`s must be strictly ascending".into());
            }
            Ok(())
        }
        "blend2d" => {
            param_of("parameter_x")?;
            param_of("parameter_y")?;
            let x_values = axis_of("x_values")?;
            let y_values = axis_of("y_values")?;
            let rows = blend
                .get("rows")
                .and_then(|v| v.as_array())
                .map(|a| a.as_slice())
                .unwrap_or(&[]);
            if rows.len() != y_values.len() {
                return err(format!(
                    "blend `rows` must have {} row(s), one per y value",
                    y_values.len()
                ));
            }
            for (j, row) in rows.iter().enumerate() {
                let cells = row.as_array().map(|a| a.as_slice()).unwrap_or(&[]);
                if cells.len() != x_values.len() {
                    return err(format!(
                        "blend `rows` row #{j} must have {} clip(s), one per x value",
                        x_values.len()
                    ));
                }
                for (k, cell) in cells.iter().enumerate() {
                    if cell.as_str().unwrap_or("").is_empty() {
                        return err(format!("blend `rows` row #{j} clip #{k} is empty"));
                    }
                }
            }
            Ok(())
        }
        "" => err("blend has no `kind` (`blend1d` or `blend2d`)".into()),
        other => err(format!(
            "unknown blend kind '{other}' (expected blend1d or blend2d)"
        )),
    }
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

    fn blend_base() -> Value {
        serde_json::json!({
            "target": "hero",
            "parameters": [{"name": "speed"}, {"name": "strafe"}],
            "states": [
                {"name": "locomotion", "blend": {"kind": "blend1d", "parameter": "speed",
                 "points": [
                     {"value": 0.0, "clip": "idle"},
                     {"value": 5.0, "clip": "run"}
                 ]}}
            ]
        })
    }

    #[test]
    fn valid_blend1d_passes() {
        assert!(check("g", &blend_base()).is_ok());
    }

    #[test]
    fn valid_blend2d_passes() {
        let mut v = blend_base();
        v["states"][0]["blend"] = serde_json::json!({
            "kind": "blend2d", "parameter_x": "speed", "parameter_y": "strafe",
            "x_values": [0.0, 5.0], "y_values": [-1.0, 1.0],
            "rows": [["a", "b"], ["c", "d"]]
        });
        assert!(check("g", &v).is_ok());
    }

    #[test]
    fn clip_and_blend_together_fails() {
        let mut v = blend_base();
        v["states"][0]["clip"] = serde_json::json!("idle");
        assert!(check("g", &v).unwrap_err().contains("pick one"));
    }

    #[test]
    fn state_without_clip_or_blend_fails() {
        let v = serde_json::json!({"target":"hero","states":[{"name":"empty"}]});
        assert!(check("g", &v).unwrap_err().contains("no `clip` or `blend`"));
    }

    #[test]
    fn unknown_blend_kind_fails() {
        let mut v = blend_base();
        v["states"][0]["blend"]["kind"] = serde_json::json!("radial");
        assert!(check("g", &v).unwrap_err().contains("radial"));
    }

    #[test]
    fn undeclared_blend_parameter_fails() {
        let mut v = blend_base();
        v["states"][0]["blend"]["parameter"] = serde_json::json!("nope");
        assert!(check("g", &v).unwrap_err().contains("nope"));
    }

    #[test]
    fn unsorted_blend_points_fail() {
        let mut v = blend_base();
        v["states"][0]["blend"]["points"][1]["value"] = serde_json::json!(-1.0);
        assert!(check("g", &v).unwrap_err().contains("ascending"));
    }

    #[test]
    fn blend_point_without_clip_fails() {
        let mut v = blend_base();
        v["states"][0]["blend"]["points"][1] = serde_json::json!({"value": 5.0});
        assert!(check("g", &v).unwrap_err().contains("no `clip`"));
    }

    #[test]
    fn mismatched_grid_rows_fail() {
        let mut v = blend_base();
        v["states"][0]["blend"] = serde_json::json!({
            "kind": "blend2d", "parameter_x": "speed", "parameter_y": "strafe",
            "x_values": [0.0, 5.0], "y_values": [-1.0, 1.0],
            "rows": [["a", "b"]]
        });
        assert!(check("g", &v).unwrap_err().contains("row(s)"));
    }

    #[test]
    fn descending_grid_axis_fails() {
        let mut v = blend_base();
        v["states"][0]["blend"] = serde_json::json!({
            "kind": "blend2d", "parameter_x": "speed", "parameter_y": "strafe",
            "x_values": [5.0, 0.0], "y_values": [-1.0, 1.0],
            "rows": [["a", "b"], ["c", "d"]]
        });
        assert!(check("g", &v).unwrap_err().contains("ascending"));
    }

    #[test]
    fn valid_ik_chain_passes() {
        let mut v = base();
        v["ik_chains"] = serde_json::json!([{
            "joints": ["hip", "knee", "foot"],
            "pole": [0.0, 0.0, 1.0],
            "weight_parameter": "speed"
        }]);
        assert!(check("g", &v).is_ok());
    }

    #[test]
    fn ik_chain_needs_exactly_three_named_joints() {
        let mut v = base();
        v["ik_chains"] = serde_json::json!([{"joints": ["hip", "foot"]}]);
        assert!(check("g", &v).unwrap_err().contains("exactly three"));
        v["ik_chains"] = serde_json::json!([{"joints": ["hip", "", "foot"]}]);
        assert!(check("g", &v).unwrap_err().contains("joints[1]"));
    }

    #[test]
    fn ik_chain_rejects_zero_pole_and_unknown_weight_parameter() {
        let mut v = base();
        v["ik_chains"] = serde_json::json!([{
            "joints": ["hip", "knee", "foot"],
            "pole": [0.0, 0.0, 0.0]
        }]);
        assert!(check("g", &v).unwrap_err().contains("non-zero"));
        v["ik_chains"] = serde_json::json!([{
            "joints": ["hip", "knee", "foot"],
            "weight_parameter": "ghost"
        }]);
        assert!(
            check("g", &v)
                .unwrap_err()
                .contains("'ghost' is not declared")
        );
    }
}
