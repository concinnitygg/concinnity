//! JSON-args source selection for the SdfVolume. The build pipeline and the
//! world checks read the field straight from the raw args JSON; the runtime
//! reads the typed struct instead, so the runtime tier carries no JSON parsing.

// Resolve an SdfVolume's distance-field source path from its raw args. One
// source serves every backend: the field is Slang, so there is nothing for a
// per-platform map to select between.
pub(crate) fn sdf_volume_source_path(args: &serde_json::Value) -> Option<String> {
    args.get("fragment_shader")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // One field serves every backend, so the path is read without a platform
    // and without an extension gate.
    #[test]
    fn sdf_source_path_is_the_one_declared_field() {
        let args = json!({ "fragment_shader": "shaders/chrome_blob.slang" });
        assert_eq!(
            sdf_volume_source_path(&args).as_deref(),
            Some("shaders/chrome_blob.slang")
        );
        assert_eq!(sdf_volume_source_path(&json!({})), None);
        assert_eq!(
            sdf_volume_source_path(&json!({"fragment_shader": ""})),
            None
        );
    }
}
