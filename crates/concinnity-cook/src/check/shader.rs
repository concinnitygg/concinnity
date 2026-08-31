use concinnity_core::platform::Platform;

/// Check a `Shader`'s authored args against the platform the world is cooked
/// for.
pub(crate) fn check(
    name: &str,
    args: &serde_json::Value,
    platform: Platform,
) -> Result<(), String> {
    check_args(args, platform).map_err(|e| format!("Asset '{}': {}", name, e))
}

// Validate Shader args without compiling: the required vertex + fragment
// stages must each resolve a source for `platform` (the optional
// vertex_instanced stage is checked only when declared).
fn check_args(args: &serde_json::Value, platform: Platform) -> Result<(), String> {
    check_stage(args, "vertex", true, platform)?;
    check_stage(args, "fragment", true, platform)?;
    check_stage(args, "vertex_instanced", false, platform)
}

fn check_stage(
    args: &serde_json::Value,
    stage: &str,
    required: bool,
    platform: Platform,
) -> Result<(), String> {
    let Some(stage_args) = args.get(stage) else {
        if required {
            return Err(format!("Shader requires a `{stage}` stage"));
        }
        return Ok(());
    };
    if crate::authoring::source_args::stage_source_path(stage_args, platform).is_none() {
        // On Vulkan, missing sources are non-fatal: the runtime falls back to
        // built-in GLSL. See `compile_payload` for the matching carve-out.
        if platform == Platform::Glsl {
            return Ok(());
        }
        let key = platform.key();
        return Err(format!(
            "Shader stage `{stage}` requires a `source` or a `sources` entry for platform \"{key}\""
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Sources for every backend, so a stage declaring this resolves whichever
    // platform the check is asked about.
    fn every_platform() -> serde_json::Value {
        json!({"sources": {"metal": "a.metal", "hlsl": "a.hlsl", "glsl": "a.glsl"}})
    }

    #[test]
    fn check_requires_vertex_and_fragment_stages() {
        let ok = json!({"vertex": every_platform(), "fragment": every_platform()});
        assert!(check_args(&ok, Platform::Metal).is_ok());

        // The stage that IS declared resolves on every platform, so the error
        // can only be about the missing one.
        assert!(
            check_args(&json!({"fragment": every_platform()}), Platform::Metal)
                .unwrap_err()
                .contains("`vertex`"),
        );
        assert!(
            check_args(&json!({"vertex": every_platform()}), Platform::Metal)
                .unwrap_err()
                .contains("`fragment`"),
        );
    }

    #[test]
    fn a_stage_without_a_source_fails_everywhere_but_glsl() {
        // A declared stage with no source for the cooked platform: GLSL is a
        // non-fatal fallback while the other backends flag it.
        let missing = json!({"vertex": {}, "fragment": {}});
        assert!(check_args(&missing, Platform::Glsl).is_ok());
        for platform in [Platform::Metal, Platform::Hlsl] {
            let err = check_args(&missing, platform).unwrap_err();
            assert!(err.contains(platform.key()), "got: {err}");
        }
    }

    #[test]
    fn a_source_for_another_backend_does_not_satisfy_the_check() {
        // The map lists only Metal, so a DirectX cook has nothing to compile.
        let metal_only = json!({"sources": {"metal": "a.metal"}});
        let args = json!({"vertex": metal_only, "fragment": metal_only});
        assert!(check_args(&args, Platform::Metal).is_ok());
        assert!(check_args(&args, Platform::Hlsl).is_err());
    }

    #[test]
    fn undeclared_instanced_stage_is_fine_but_empty_declared_one_is_checked() {
        let base = json!({"vertex": every_platform(), "fragment": every_platform()});
        assert!(check_args(&base, Platform::Metal).is_ok());

        let mut with_empty_instanced = base.clone();
        with_empty_instanced["vertex_instanced"] = json!({});
        assert!(check_args(&with_empty_instanced, Platform::Glsl).is_ok());
        assert!(
            check_args(&with_empty_instanced, Platform::Metal)
                .unwrap_err()
                .contains("vertex_instanced")
        );
    }

    // The named form prefixes the asset name so a world-wide report says which
    // asset failed.
    #[test]
    fn the_error_names_the_asset() {
        let err = check("scene_shader", &json!({}), Platform::Metal).unwrap_err();
        assert!(err.starts_with("Asset 'scene_shader':"), "got: {err}");
    }
}
