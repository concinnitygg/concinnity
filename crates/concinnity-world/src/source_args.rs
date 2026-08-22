//! JSON-args source selection for the shader-backed asset types.
//!
//! A Shader stage / SdfVolume declares its shader source as a single path plus
//! an optional per-platform map; the build pipeline and the world checks pick
//! the building backend's entry straight from the raw args JSON. The runtime
//! selects from the typed struct instead (`StageSourceExt` and the SdfVolume
//! clamp in concinnity-core), so the runtime tier carries no JSON parsing.

use concinnity_core::platform::Platform;

// Resolve a shader stage source filename for `platform` from its raw stage args:
// the `sources` map entry for the platform wins, then the single `source`
// path when its file extension matches the platform.
pub(crate) fn stage_source_path(args: &serde_json::Value, platform: Platform) -> Option<String> {
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

// Resolve an SdfVolume's fragment shader path for `platform` from its raw
// args: the `fragment_shaders` map entry wins, then the single
// `fragment_shader` path when its extension matches the platform.
pub(crate) fn sdf_volume_source_path(
    args: &serde_json::Value,
    platform: Platform,
) -> Option<String> {
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
