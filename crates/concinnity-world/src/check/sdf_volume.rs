use concinnity_core::components::sdf_volume::SDF_PARAMS_LEN;

/// Check an `SdfVolume`'s authored args.
pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    check_args(args).map_err(|e| format!("Asset '{}': {}", name, e))
}

// Validate SdfVolume args without compiling.
fn check_args(args: &serde_json::Value) -> Result<(), String> {
    if crate::source_args::current_platform_source_arg(args).is_none() {
        let platform_key = concinnity_core::platform::Platform::current().key();
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

    // File extension matching the backend these tests compile against, so a
    // single `fragment_shader` path resolves as current-platform-compatible
    // on Metal, DirectX, and Vulkan alike.
    fn platform_ext() -> &'static str {
        concinnity_core::platform::Platform::current().key()
    }

    #[test]
    fn check_requires_fragment_shader() {
        let args = serde_json::json!({});
        assert!(check_args(&args).is_err());

        let args = serde_json::json!({"fragment_shader": ""});
        assert!(check_args(&args).is_err());

        let args =
            serde_json::json!({"fragment_shader": format!("shaders/blob.{}", platform_ext())});
        assert!(check_args(&args).is_ok());
    }

    #[test]
    fn check_rejects_oversized_params() {
        let mut params = vec![0.0; SDF_PARAMS_LEN + 1];
        params[0] = 1.0;
        let args = serde_json::json!({
            "fragment_shader": format!("shaders/blob.{}", platform_ext()),
            "params": params,
        });
        assert!(check_args(&args).is_err());
    }

    #[test]
    fn check_accepts_short_params() {
        // Less than SDF_PARAMS_LEN is fine: the rest defaults to 0.
        let args = serde_json::json!({
            "fragment_shader": format!("shaders/blob.{}", platform_ext()),
            "params": [1.0, 2.0, 3.0],
        });
        assert!(check_args(&args).is_ok());
    }

    #[test]
    fn check_rejects_source_for_other_backend_only() {
        // A single path whose extension targets a different backend is "no
        // source for this platform": the build needs a current-backend
        // shader, so validation fails rather than trying to read it.
        let other_ext = match platform_ext() {
            "metal" => "hlsl",
            _ => "metal",
        };
        let args = serde_json::json!({ "fragment_shader": format!("shaders/blob.{other_ext}") });
        assert!(check_args(&args).is_err());
    }

    #[test]
    fn check_accepts_sources_map_with_current_backend() {
        // A per-backend map that includes the current backend validates even
        // when it also lists other backends the build won't compile here.
        let args = serde_json::json!({
            "fragment_shaders": {
                "metal": "shaders/blob.metal",
                "hlsl": "shaders/blob.hlsl",
                "glsl": "shaders/blob.glsl",
            }
        });
        assert!(check_args(&args).is_ok());
    }

    #[test]
    fn check_rejects_sources_map_without_current_backend() {
        // A map lacking the current backend's entry has nothing to build here.
        let other_ext = match platform_ext() {
            "metal" => "hlsl",
            _ => "metal",
        };
        let args = serde_json::json!({
            "fragment_shaders": { other_ext: format!("shaders/blob.{other_ext}") }
        });
        assert!(check_args(&args).is_err());
    }
}
