pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let has_model = !args
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .is_empty();
    let has_mesh = !args
        .get("mesh")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .is_empty();
    let has_prefab = !args
        .get("prefab")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .is_empty();
    if has_prefab && (has_model || has_mesh) {
        return Err(format!(
            "Asset '{}': Prop with 'prefab' cannot also set 'model' or 'mesh'",
            name
        ));
    }
    if !has_model && !has_mesh && !has_prefab {
        return Err(format!(
            "Asset '{}': Prop requires either a 'prefab', 'model', or 'mesh' field",
            name
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_only_passes() {
        assert!(check("p", &serde_json::json!({ "model": "m" })).is_ok());
    }

    #[test]
    fn prefab_only_passes() {
        assert!(check("p", &serde_json::json!({ "prefab": "pf" })).is_ok());
    }

    #[test]
    fn prefab_with_model_conflicts() {
        let err = check("p", &serde_json::json!({ "prefab": "pf", "model": "m" })).unwrap_err();
        assert!(err.contains("cannot also set"), "got: {err}");
    }

    #[test]
    fn prefab_with_mesh_conflicts() {
        let err = check("p", &serde_json::json!({ "prefab": "pf", "mesh": "m" })).unwrap_err();
        assert!(err.contains("cannot also set"), "got: {err}");
    }

    #[test]
    fn missing_all_sources_is_an_error() {
        let err = check("p", &serde_json::json!({})).unwrap_err();
        assert!(err.contains("requires either"), "got: {err}");
    }

    #[test]
    fn empty_string_source_is_treated_as_absent() {
        let err = check("p", &serde_json::json!({ "model": "" })).unwrap_err();
        assert!(err.contains("requires either"), "got: {err}");
    }
}
