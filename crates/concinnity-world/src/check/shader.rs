use concinnity_core::build::Platform;

pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    check_args(args).map_err(|e| format!("Asset '{}': {}", name, e))
}

// Validate ShaderStage args without compiling.
fn check_args(args: &serde_json::Value) -> Result<(), String> {
    if crate::source_args::resolve_source_from_args(args).is_none() {
        // On Linux/Vulkan, missing sources are non-fatal: the runtime falls
        // back to built-in GLSL. See `compile_payload` for the matching carve-out.
        let key = Platform::current().key();
        if key == "glsl" {
            return Ok(());
        }
        return Err(format!(
            "ShaderStage requires a `source` or a `sources` entry for platform \"{key}\""
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn check_requires_a_current_platform_source() {
        // With every platform declared, check passes on any backend.
        let ok = json!({"kind": "vertex", "sources": {"metal": "a.metal", "hlsl": "a.hlsl", "glsl": "a.glsl"}});
        assert!(check_args(&ok).is_ok());
        assert!(crate::source_args::resolve_source_from_args(&ok).is_some());

        // With nothing declared, GLSL/Vulkan is a non-fatal fallback while the
        // other backends flag the missing source. Only one arm runs per build,
        // so branch on the active platform key to stay deterministic.
        let missing = check_args(&json!({"kind": "vertex"}));
        if Platform::current().key() == "glsl" {
            assert!(missing.is_ok());
        } else {
            assert!(missing.is_err());
        }
    }
}
