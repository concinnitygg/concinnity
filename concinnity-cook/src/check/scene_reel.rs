pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    if let Some(entries) = args.get("scenes").and_then(|v| v.as_array()) {
        if entries.is_empty() {
            return Err(format!(
                "Asset '{}': SceneReel 'scenes' list is empty",
                name
            ));
        }
        for (i, entry) in entries.iter().enumerate() {
            if entry.as_str().map(|s| s.is_empty()).unwrap_or(true) {
                return Err(format!(
                    "Asset '{}': SceneReel 'scenes[{}]' must be a non-empty scene name string",
                    name, i
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_scene_list_passes() {
        let args = serde_json::json!({ "scenes": ["intro", "outro"] });
        assert!(check("reel", &args).is_ok());
    }

    #[test]
    fn no_scenes_key_passes() {
        // The scenes field is optional; absence is not a check-time error.
        assert!(check("reel", &serde_json::json!({})).is_ok());
    }

    #[test]
    fn empty_scene_list_is_an_error() {
        let err = check("reel", &serde_json::json!({ "scenes": [] })).unwrap_err();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn empty_scene_name_is_an_error() {
        let err = check("reel", &serde_json::json!({ "scenes": ["ok", ""] })).unwrap_err();
        assert!(err.contains("scenes[1]"), "got: {err}");
    }

    #[test]
    fn non_string_scene_entry_is_an_error() {
        let err = check("reel", &serde_json::json!({ "scenes": [123] })).unwrap_err();
        assert!(err.contains("scenes[0]"), "got: {err}");
    }
}
