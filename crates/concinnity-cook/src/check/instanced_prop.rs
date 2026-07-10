// Structural validation for InstancedProp args. Cross-asset mesh/material
// lookups are handled by build/pipeline.rs::validate_cross_references; this
// check enforces only the things we can see from the asset's own args.

// Soft cap on instances per cluster. The expansion path produces one
// DrawObject per instance, so a runaway cluster can blow out the draw list
// budget without warning.
pub(crate) const MAX_INSTANCES_PER_CLUSTER: usize = 16_384;

pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let instances = args
        .get("instances")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            format!(
                "Asset '{}': InstancedProp `instances` must be an array of transforms",
                name
            )
        })?;
    if instances.len() > MAX_INSTANCES_PER_CLUSTER {
        return Err(format!(
            "Asset '{}': InstancedProp has {} instances; current cap is {}. Split into multiple clusters.",
            name,
            instances.len(),
            MAX_INSTANCES_PER_CLUSTER
        ));
    }
    for (i, entry) in instances.iter().enumerate() {
        if !entry.is_object() {
            return Err(format!(
                "Asset '{}': InstancedProp instances[{}] must be an object",
                name, i
            ));
        }
        if let Some(p) = entry.get("position") {
            check_f32x3(p, &format!("Asset '{}': instances[{}].position", name, i))?;
        }
        if let Some(r) = entry.get("rotation_deg") {
            check_f32x3(
                r,
                &format!("Asset '{}': instances[{}].rotation_deg", name, i),
            )?;
        }
        if let Some(s) = entry.get("scale") {
            check_f32x3(s, &format!("Asset '{}': instances[{}].scale", name, i))?;
        }
    }
    Ok(())
}

fn check_f32x3(v: &serde_json::Value, label: &str) -> Result<(), String> {
    let arr = v
        .as_array()
        .ok_or_else(|| format!("{label} must be an array of 3 numbers"))?;
    if arr.len() < 3 {
        return Err(format!("{label} must have 3 elements, got {}", arr.len()));
    }
    for (i, e) in arr.iter().take(3).enumerate() {
        if e.as_f64().is_none() {
            return Err(format!("{label}[{i}] must be a number"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_instances_pass() {
        check("p", &json!({"instances": []})).expect("empty cluster");
    }

    #[test]
    fn full_transform_instances_pass() {
        let args = json!({"instances": [
            {"position": [1.0, 2.0, 3.0], "rotation_deg": [0, 90, 0], "scale": [1, 1, 1]},
            {}
        ]});
        check("p", &args).expect("valid instances");
    }

    #[test]
    fn missing_instances_errors() {
        let err = check("p", &json!({})).unwrap_err();
        assert!(err.contains("`instances` must be an array"), "got: {err}");
    }

    #[test]
    fn instances_over_the_cap_error() {
        let entries: Vec<serde_json::Value> = (0..MAX_INSTANCES_PER_CLUSTER + 1)
            .map(|_| json!({}))
            .collect();
        let err = check("p", &json!({"instances": entries})).unwrap_err();
        assert!(err.contains("16385 instances"), "got: {err}");
        assert!(err.contains("cap is 16384"), "got: {err}");
    }

    #[test]
    fn instances_at_the_cap_pass() {
        let entries: Vec<serde_json::Value> =
            (0..MAX_INSTANCES_PER_CLUSTER).map(|_| json!({})).collect();
        check("p", &json!({"instances": entries})).expect("at cap");
    }

    #[test]
    fn non_object_instance_errors() {
        let err = check("p", &json!({"instances": [{}, 3]})).unwrap_err();
        assert!(err.contains("instances[1] must be an object"), "got: {err}");
    }

    #[test]
    fn non_array_position_errors() {
        let args = json!({"instances": [{"position": "origin"}]});
        let err = check("p", &args).unwrap_err();
        assert!(
            err.contains("instances[0].position must be an array of 3 numbers"),
            "got: {err}"
        );
    }

    #[test]
    fn short_position_errors() {
        let args = json!({"instances": [{"position": [1.0, 2.0]}]});
        let err = check("p", &args).unwrap_err();
        assert!(err.contains("must have 3 elements, got 2"), "got: {err}");
    }

    #[test]
    fn non_numeric_position_component_errors() {
        let args = json!({"instances": [{"position": [1.0, null, 3.0]}]});
        let err = check("p", &args).unwrap_err();
        assert!(err.contains("position[1] must be a number"), "got: {err}");
    }

    #[test]
    fn extra_position_components_are_ignored() {
        let args = json!({"instances": [{"position": [1.0, 2.0, 3.0, "extra"]}]});
        check("p", &args).expect("only the first 3 components are validated");
    }

    #[test]
    fn bad_rotation_and_scale_are_validated_too() {
        let args = json!({"instances": [{"rotation_deg": [0, 0, "x"]}]});
        let err = check("p", &args).unwrap_err();
        assert!(
            err.contains("rotation_deg[2] must be a number"),
            "got: {err}"
        );

        let args = json!({"instances": [{"scale": 2.0}]});
        let err = check("p", &args).unwrap_err();
        assert!(
            err.contains("scale must be an array of 3 numbers"),
            "got: {err}"
        );
    }
}
