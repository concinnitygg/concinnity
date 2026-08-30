use concinnity_core::platform::Platform;

/// Check a `Shader`'s authored args.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    check_args(args).map_err(|e| format!("Asset '{}': {}", name, e))
}

// Validate Shader args without compiling: the required vertex + fragment
// stages must each resolve a source for the current platform (the optional
// vertex_instanced stage is checked only when declared).
fn check_args(args: &serde_json::Value) -> Result<(), String> {
    check_stage(args, "vertex", true)?;
    check_stage(args, "fragment", true)?;
    check_stage(args, "vertex_instanced", false)
}

fn check_stage(args: &serde_json::Value, stage: &str, required: bool) -> Result<(), String> {
    let Some(stage_args) = args.get(stage) else {
        if required {
            return Err(format!("Shader requires a `{stage}` stage"));
        }
        return Ok(());
    };
    if crate::authoring::source_args::resolve_source_from_args(stage_args).is_none() {
        // On Linux/Vulkan, missing sources are non-fatal: the runtime falls
        // back to built-in GLSL. See `compile_payload` for the matching carve-out.
        let key = Platform::current().key();
        if key == "glsl" {
            return Ok(());
        }
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

    #[test]
    fn check_requires_vertex_and_fragment_stages() {
        let ok = json!({
            "vertex": {"sources": {"metal": "a.metal", "hlsl": "a.hlsl", "glsl": "a.glsl"}},
            "fragment": {"sources": {"metal": "a.metal", "hlsl": "a.hlsl", "glsl": "a.glsl"}}
        });
        assert!(check_args(&ok).is_ok());

        // The stage that IS declared resolves on every platform, so the error can
        // only be about the missing one. A bare `source` would instead resolve
        // only where its extension matches, making the assertion platform-specific.
        let resolvable =
            json!({"sources": {"metal": "a.metal", "hlsl": "a.hlsl", "glsl": "a.glsl"}});
        assert!(
            check_args(&json!({"fragment": resolvable}))
                .unwrap_err()
                .contains("`vertex`"),
        );
        assert!(
            check_args(&json!({"vertex": resolvable}))
                .unwrap_err()
                .contains("`fragment`"),
        );
    }

    #[test]
    fn check_requires_a_current_platform_source_per_stage() {
        // A declared stage with no current-platform source: GLSL/Vulkan is a
        // non-fatal fallback while the other backends flag it. Only one arm
        // runs per build, so branch on the active platform key.
        let missing = check_args(&json!({"vertex": {}, "fragment": {}}));
        if Platform::current().key() == "glsl" {
            assert!(missing.is_ok());
        } else {
            assert!(missing.is_err());
        }
    }

    #[test]
    fn undeclared_instanced_stage_is_fine_but_empty_declared_one_is_checked() {
        let base = json!({
            "vertex": {"sources": {"metal": "a.metal", "hlsl": "a.hlsl", "glsl": "a.glsl"}},
            "fragment": {"sources": {"metal": "a.metal", "hlsl": "a.hlsl", "glsl": "a.glsl"}}
        });
        assert!(check_args(&base).is_ok());

        let mut with_empty_instanced = base.clone();
        with_empty_instanced["vertex_instanced"] = json!({});
        let result = check_args(&with_empty_instanced);
        if Platform::current().key() == "glsl" {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("vertex_instanced"));
        }
    }
}
