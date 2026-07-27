// JSON-args source selection for the shader-backed asset types.
//
// A Shader stage / SdfVolume declares its shader source as a single path plus
// an optional per-platform map; the build pipeline and the world checks pick
// the building backend's entry straight from the raw args JSON. The runtime
// selects from the typed struct instead (`StageSourceExt` and the SdfVolume
// clamp in concinnity-core), so the runtime tier carries no JSON parsing.

use concinnity_core::build::Platform;

/// Resolve a shader stage source filename for `platform` from its raw stage args:
/// the `sources` map entry for the platform wins, then the single `source`
/// path when its file extension matches the platform.
pub fn stage_source_path(args: &serde_json::Value, platform: Platform) -> Option<String> {
    if let Some(obj) = args.get("sources").and_then(|v| v.as_object())
        && let Some(src) = obj.get(platform.key()).and_then(|v| v.as_str())
    {
        return Some(src.to_string());
    }
    let src = args
        .get("source")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())?;
    let ext = std::path::Path::new(src)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if platform.accepts_ext(ext) {
        Some(src.to_string())
    } else {
        None
    }
}

/// Resolves the shader source filename for the current platform from raw
/// stage args (a `Shader` stage sub-object or an SdfVolume).
pub fn resolve_source_from_args(args: &serde_json::Value) -> Option<String> {
    stage_source_path(args, Platform::current())
}

// True when this stage declares at least one source and every declared source
// (in `sources` and `source`) is an engine built-in shader. The bundled
// default shader set declares only `metal` + `hlsl` built-ins and no `glsl`,
// so on the Vulkan/GLSL backend it resolves to no source and renders via the
// backend's inline GLSL by design -- not a user mistake. A custom stage that
// merely forgot its `glsl` variant has at least one non-built-in source and is
// not covered, so the missing-source path still flags it.
pub fn declares_only_builtin_sources(args: &serde_json::Value) -> bool {
    use concinnity_core::build::shader::builtin_shader_source;
    let mut saw_any = false;
    let mut check = |name: &str| {
        if name.is_empty() {
            return true;
        }
        saw_any = true;
        builtin_shader_source(name).is_some()
    };
    if let Some(obj) = args.get("sources").and_then(|v| v.as_object()) {
        for v in obj.values() {
            if let Some(s) = v.as_str()
                && !check(s)
            {
                return false;
            }
        }
    }
    if let Some(s) = args.get("source").and_then(|v| v.as_str())
        && !check(s)
    {
        return false;
    }
    saw_any
}

/// Resolve an SdfVolume's fragment shader path for `platform` from its raw
/// args: the `fragment_shaders` map entry wins, then the single
/// `fragment_shader` path when its extension matches the platform.
pub fn sdf_volume_source_path(args: &serde_json::Value, platform: Platform) -> Option<String> {
    if let Some(obj) = args.get("fragment_shaders").and_then(|v| v.as_object())
        && let Some(src) = obj
            .get(platform.key())
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
    {
        return Some(src.to_string());
    }
    let src = args
        .get("fragment_shader")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())?;
    let ext = std::path::Path::new(src)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if platform.accepts_ext(ext) {
        Some(src.to_string())
    } else {
        None
    }
}

/// Resolve the raw fragment shader source an SdfVolume declares for the
/// current build backend.
pub fn current_platform_source_arg(args: &serde_json::Value) -> Option<String> {
    sdf_volume_source_path(args, Platform::current())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn shader_source_path_selects_per_platform() {
        // A sources-map entry for the requested platform wins.
        let args = json!({"sources": {"metal": "a.metal", "hlsl": "a.hlsl", "glsl": "a.glsl"}});
        assert_eq!(
            stage_source_path(&args, Platform::Metal),
            Some("a.metal".to_string())
        );
        assert_eq!(
            stage_source_path(&args, Platform::Hlsl),
            Some("a.hlsl".to_string())
        );
        assert_eq!(
            stage_source_path(&args, Platform::Glsl),
            Some("a.glsl".to_string())
        );

        // A single `source` is accepted when its extension matches the
        // platform, rejected when it is another backend's shader extension.
        let metal_only = json!({"source": "s.metal"});
        assert_eq!(
            stage_source_path(&metal_only, Platform::Metal),
            Some("s.metal".to_string())
        );
        assert_eq!(stage_source_path(&metal_only, Platform::Hlsl), None);

        // No source at all -> None.
        assert_eq!(
            stage_source_path(&json!({"kind": "vertex"}), Platform::Metal),
            None
        );
    }

    #[test]
    fn bundled_default_set_is_all_builtin() {
        // The vertex + fragment defaults the GraphicsConfig companion injects
        // declare only built-in metal/hlsl sources, so the GLSL fallback is
        // expected and must not be flagged.
        let vert = json!({
            "kind": "vertex",
            "sources": {"metal": "default.metal", "hlsl": "default_vert.hlsl"}
        });
        let frag = json!({
            "kind": "fragment",
            "sources": {"metal": "default.metal", "hlsl": "default_frag.hlsl"}
        });
        assert!(declares_only_builtin_sources(&vert));
        assert!(declares_only_builtin_sources(&frag));
    }

    #[test]
    fn custom_source_is_not_builtin() {
        // A custom stage that forgot its glsl variant has a non-built-in
        // source and stays flagged.
        let mixed = json!({
            "kind": "fragment",
            "sources": {"metal": "default.metal", "hlsl": "my_custom.hlsl"}
        });
        let custom = json!({"kind": "vertex", "source": "my_custom.metal"});
        assert!(!declares_only_builtin_sources(&mixed));
        assert!(!declares_only_builtin_sources(&custom));
    }

    #[test]
    fn no_declared_source_is_not_builtin() {
        // A stage declaring nothing is malformed, not an engine default.
        assert!(!declares_only_builtin_sources(&json!({"kind": "vertex"})));
        assert!(!declares_only_builtin_sources(
            &json!({"kind": "vertex", "source": ""})
        ));
    }

    #[test]
    fn sdf_source_path_prefers_map_over_single() {
        let args = json!({
            "fragment_shader": "shaders/single.metal",
            "fragment_shaders": { "metal": "shaders/from_map.metal" },
        });
        assert_eq!(
            sdf_volume_source_path(&args, Platform::Metal).as_deref(),
            Some("shaders/from_map.metal")
        );
    }
}
