use concinnity_core::components::sdf_volume::SDF_PARAMS_LEN;
use concinnity_core::platform::Platform;

/// Check an `SdfVolume`'s authored args against the platform the world is
/// cooked for.
pub(crate) fn check(
    name: &str,
    args: &serde_json::Value,
    platform: Platform,
) -> Result<(), String> {
    check_args(args, platform).map_err(|e| format!("Asset '{}': {}", name, e))
}

// Validate SdfVolume args without compiling.
fn check_args(args: &serde_json::Value, platform: Platform) -> Result<(), String> {
    if crate::authoring::source_args::sdf_volume_source_path(args, platform).is_none() {
        let platform_key = platform.key();
        return Err(format!(
            "SdfVolume requires a `fragment_shader` or a `fragment_shaders` \
             entry for backend \"{platform_key}\" (a path to a shader file \
             declaring map + shade)"
        ));
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
    fn check_requires_fragment_shader() {
        assert!(check_args(&serde_json::json!({}), Platform::Metal).is_err());
        assert!(check_args(&serde_json::json!({"fragment_shader": ""}), Platform::Metal).is_err());

        let args = serde_json::json!({"fragment_shader": "shaders/blob.metal"});
        assert!(check_args(&args, Platform::Metal).is_ok());
    }

    #[test]
    fn check_rejects_oversized_params() {
        let mut params = vec![0.0; SDF_PARAMS_LEN + 1];
        params[0] = 1.0;
        let args = serde_json::json!({
            "fragment_shader": "shaders/blob.metal",
            "params": params,
        });
        assert!(check_args(&args, Platform::Metal).is_err());
    }

    #[test]
    fn check_accepts_short_params() {
        // Less than SDF_PARAMS_LEN is fine: the rest defaults to 0.
        let args = serde_json::json!({
            "fragment_shader": "shaders/blob.metal",
            "params": [1.0, 2.0, 3.0],
        });
        assert!(check_args(&args, Platform::Metal).is_ok());
    }

    #[test]
    fn check_rejects_source_for_other_backend_only() {
        // A single path whose extension targets a different backend is "no
        // source for this platform": the build needs a shader for the backend
        // it cooks for, so validation fails rather than trying to read it.
        let args = serde_json::json!({ "fragment_shader": "shaders/blob.metal" });
        assert!(check_args(&args, Platform::Hlsl).is_err());
        assert!(check_args(&args, Platform::Glsl).is_err());
    }

    #[test]
    fn check_accepts_a_sources_map_holding_the_cooked_backend() {
        // A per-backend map validates for each backend it lists.
        let args = serde_json::json!({
            "fragment_shaders": {
                "metal": "shaders/blob.metal",
                "hlsl": "shaders/blob.hlsl",
                "glsl": "shaders/blob.glsl",
            }
        });
        for platform in [Platform::Metal, Platform::Hlsl, Platform::Glsl] {
            assert!(check_args(&args, platform).is_ok());
        }
    }

    #[test]
    fn check_rejects_sources_map_without_the_cooked_backend() {
        // A map lacking the cooked backend's entry has nothing to build here.
        let args = serde_json::json!({
            "fragment_shaders": { "metal": "shaders/blob.metal" }
        });
        assert!(check_args(&args, Platform::Metal).is_ok());
        assert!(check_args(&args, Platform::Hlsl).is_err());
    }

    // The named form prefixes the asset name so a world-wide report says which
    // asset failed.
    #[test]
    fn the_error_names_the_asset() {
        let err = check("blob", &serde_json::json!({}), Platform::Metal).unwrap_err();
        assert!(err.starts_with("Asset 'blob':"), "got: {err}");
    }
}
