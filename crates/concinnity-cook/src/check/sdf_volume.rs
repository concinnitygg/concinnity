use concinnity_core::components::sdf_volume::SDF_PARAMS_LEN;

/// Check an `SdfVolume`'s authored args.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    check_args(args).map_err(|e| format!("Asset '{}': {}", name, e))
}

// Validate SdfVolume args without compiling.
fn check_args(args: &serde_json::Value) -> Result<(), String> {
    if crate::authoring::source_args::sdf_volume_source_path(args).is_none() {
        return Err(
            "SdfVolume requires a `fragment_shader` path to a `.slang` distance field \
             (declaring map + shade, or sampleVolume for a volumetric one)"
                .to_string(),
        );
    }
    if let Some(params) = args.get("params").and_then(|v| v.as_array())
        && params.len() > SDF_PARAMS_LEN
    {
        return Err(format!(
            "SdfVolume `params` is {} entries; max is {} \
                 (extra entries would be ignored)",
            params.len(),
            SDF_PARAMS_LEN
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_requires_a_distance_field() {
        assert!(check_args(&serde_json::json!({})).is_err());
        assert!(check_args(&serde_json::json!({"fragment_shader": ""})).is_err());
        assert!(check_args(&serde_json::json!({"fragment_shader": "shaders/blob.slang"})).is_ok());
    }

    #[test]
    fn check_rejects_oversized_params() {
        let mut params = vec![0.0; SDF_PARAMS_LEN + 1];
        params[0] = 1.0;
        let args = serde_json::json!({
            "fragment_shader": "shaders/blob.slang",
            "params": params,
        });
        assert!(check_args(&args).is_err());
    }

    #[test]
    fn check_accepts_short_params() {
        // Less than SDF_PARAMS_LEN is fine: the rest defaults to 0.
        let args = serde_json::json!({
            "fragment_shader": "shaders/blob.slang",
            "params": [1.0, 2.0, 3.0],
        });
        assert!(check_args(&args).is_ok());
    }

    // One field serves every backend, so a volume that validates for one build
    // validates for all of them. There is no per-backend source to be missing.
    #[test]
    fn check_does_not_depend_on_the_cooked_backend() {
        let args = serde_json::json!({ "fragment_shader": "shaders/blob.slang" });
        assert!(check_args(&args).is_ok());
    }

    // The named form prefixes the asset name so a world-wide report says which
    // asset failed.
    #[test]
    fn the_error_names_the_asset() {
        let err = check("blob", &serde_json::json!({})).unwrap_err();
        assert!(err.starts_with("Asset 'blob':"), "got: {err}");
    }
}
