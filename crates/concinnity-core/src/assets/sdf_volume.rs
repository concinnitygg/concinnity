// src/assets/sdf_volume.rs
//
// Runtime + build behavior for the SdfVolume asset. The authored schema (the
// SdfVolume struct, its Default, `cone_ratio`, and `SDF_PARAMS_LEN`) lives in
// concinnity-asset; this file keeps the `Component` impl, the `SourceBacked`
// impl, the build-time validation cook calls, the blob-residency helper the
// engine init uses, and the runtime step-count clamp bounds. The schema type +
// `SDF_PARAMS_LEN` are re-exported so `crate::assets::sdf_volume::*` paths (the
// render backends' uniform structs) keep resolving.

pub use concinnity_asset::{SDF_PARAMS_LEN, SdfVolume};

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// Hard cap on the per-volume cone-march step count. Matches the
/// runtime kernel's loop bound; values above this are clamped.
pub const SDF_MAX_STEPS_CEILING: u32 = 256;

/// Lower bound on the per-volume cone-march step count. Below this the
/// march doesn't have enough budget to converge on anything interesting.
pub const SDF_MAX_STEPS_FLOOR: u32 = 8;

// Resolve the fragment shader source path for the current build backend from a
// volume's `fragment_shaders` map (preferred) or its `fragment_shader`
// fallback. Mirrors the build-time `SourceBacked::source_path` selection.
fn current_platform_source(v: &SdfVolume) -> Option<String> {
    let args = serde_json::to_value(v).ok()?;
    current_platform_source_arg(&args)
}

impl Component for SdfVolume {
    const NAME: &'static str = "SdfVolume";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        // Extents must be positive: a zero or negative extent would
        // produce an inside-out bounding box no fragment ever enters.
        for axis in args.extent.iter_mut() {
            if !axis.is_finite() || *axis <= 0.0 {
                *axis = 1.0;
            }
        }
        if !args.max_gradient.is_finite() || args.max_gradient <= 0.0 {
            args.max_gradient = 1.0;
        }
        args.max_steps = args
            .max_steps
            .clamp(SDF_MAX_STEPS_FLOOR, SDF_MAX_STEPS_CEILING);
        if !args.max_distance.is_finite() || args.max_distance < 0.1 {
            args.max_distance = 0.1;
        }
        // Volumetrics are translucent: they don't write depth, so the
        // shadow pass has no surface to project. Force the flag off
        // rather than silently building an unusable shadow PSO.
        if args.volumetric {
            args.cast_shadows = false;
        }
        // Collapse the per-backend `fragment_shaders` map down to the single
        // `fragment_shader` for the current backend so the runtime sees a
        // concrete current-backend source path regardless of how the volume
        // was authored. In particular the DirectX raymarch pass filters
        // volumes by this path's extension; keeping it populated lets that
        // filter work for map-authored volumes without a backend-specific
        // change. No-op when the map has no entry for this backend.
        if let Some(src) = current_platform_source(&args) {
            args.fragment_shader = src;
        }
        args
    }

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
}

impl crate::build::SourceBacked for SdfVolume {
    fn source_path(args: &serde_json::Value, platform: crate::build::Platform) -> Option<String> {
        // Prefer the per-backend map entry for this platform.
        if let Some(obj) = args.get("fragment_shaders").and_then(|v| v.as_object())
            && let Some(src) = obj
                .get(platform.key())
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
        {
            return Some(src.to_string());
        }
        // Fall back to the single path, but only when its extension matches
        // this backend: a `.hlsl` path is not a source the Metal build needs.
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
}

/// Resolve the raw fragment shader source declared for the current build
/// backend, applying the `fragment_shaders` map / `fragment_shader` fallback.
pub fn current_platform_source_arg(args: &serde_json::Value) -> Option<String> {
    use crate::build::SourceBacked;
    <SdfVolume as SourceBacked>::source_path(args, crate::build::Platform::current())
}

/// Blob indices that hold an `SdfVolume` fragment-shader payload.
///
/// The graphics-system init drains `SdfVolume`s and reads their payload
/// bytes via the locator. The release sweep earlier in the same init
/// frees every blob whose contents have already been consumed, but
/// because the SDF drain runs *after* that sweep, any blob holding only
/// an SDF payload would be freed before being read. (When the world
/// has other small assets, the SDF shader bytes typically share a blob
/// with a kept asset and survive by accident; a world whose SDF shader
/// ends up alone in its blob exposes the bug as "SdfVolume payload
/// FileIo, skipping" with no surface drawn.) This helper lets the
/// release sweep keep SDF blobs resident, matching the
/// `audio_clip_blob_indices` pattern.
pub fn sdf_volume_blob_indices(
    ctx: &crate::ecs::PipelineContext,
) -> std::collections::HashSet<u32> {
    ctx.query::<SdfVolume>()
        .filter_map(|v| v.locator.as_ref().map(|l| l.blob_index))
        .collect()
}

/// Resolve the fragment shader source path the runtime should watch /
/// re-read for hot-reload. Mirrors `ShaderStage::resolve_runtime_source_path`
/// so the asset-hot-reload subsystem can subscribe to changes under
/// `concinnity-engine/assets/shaders/` for every live SdfVolume. Unused
/// today.
#[allow(dead_code)]
pub fn resolve_runtime_source_path(raw: &str) -> String {
    let p = std::path::Path::new(raw);
    if p.parent().map(|d| d.as_os_str().is_empty()).unwrap_or(true) {
        if let Some(path) = crate::paths::find_in_assets(raw) {
            return path;
        }
        return crate::paths::assets_dir()
            .join(raw)
            .to_string_lossy()
            .into_owned();
    }
    raw.to_string()
}

/// Validate `SdfVolume` args without compiling. Called by the check pass.
pub fn check(args: &serde_json::Value) -> Result<(), String> {
    if current_platform_source_arg(args).is_none() {
        let platform_key = crate::build::Platform::current().key();
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

    /// File extension matching the backend these tests compile against, so a
    /// single `fragment_shader` path resolves as current-platform-compatible
    /// on Metal, DirectX, and Vulkan alike.
    fn platform_ext() -> &'static str {
        crate::build::Platform::current().key()
    }

    #[test]
    fn defaults_are_sensible() {
        let v = SdfVolume::default();
        assert_eq!(v.centre, [0.0, 0.0, 0.0]);
        assert_eq!(v.extent, [1.0, 1.0, 1.0]);
        assert_eq!(v.max_gradient, 1.0);
        assert_eq!(v.max_steps, 64);
        assert_eq!(v.max_distance, 30.0);
        assert!(v.receive_shadows);
        assert!(!v.cast_shadows);
        assert!(v.visible);
        assert_eq!(v.params.len(), SDF_PARAMS_LEN);
        assert_eq!(v.cone_ratio(), 1.0);
    }

    #[test]
    fn from_args_clamps_steps() {
        let mut a = SdfVolume {
            max_steps: 1,
            ..Default::default()
        };
        let clamped = SdfVolume::from_args(a.clone());
        assert_eq!(clamped.max_steps, SDF_MAX_STEPS_FLOOR);

        a.max_steps = 9999;
        let clamped = SdfVolume::from_args(a);
        assert_eq!(clamped.max_steps, SDF_MAX_STEPS_CEILING);
    }

    #[test]
    fn from_args_repairs_bad_extent() {
        let a = SdfVolume {
            extent: [0.0, -1.0, f32::NAN],
            ..Default::default()
        };
        let fixed = SdfVolume::from_args(a);
        assert_eq!(fixed.extent, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn from_args_repairs_bad_gradient_and_distance() {
        let a = SdfVolume {
            max_gradient: -0.5,
            max_distance: f32::NAN,
            ..Default::default()
        };
        let fixed = SdfVolume::from_args(a);
        assert_eq!(fixed.max_gradient, 1.0);
        assert_eq!(fixed.max_distance, 0.1);
    }

    #[test]
    fn cone_ratio_inverts_gradient() {
        let v = SdfVolume {
            max_gradient: 2.0,
            ..Default::default()
        };
        assert!((v.cone_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn check_requires_fragment_shader() {
        let args = serde_json::json!({});
        assert!(check(&args).is_err());

        let args = serde_json::json!({"fragment_shader": ""});
        assert!(check(&args).is_err());

        let args =
            serde_json::json!({"fragment_shader": format!("shaders/blob.{}", platform_ext())});
        assert!(check(&args).is_ok());
    }

    #[test]
    fn check_rejects_oversized_params() {
        let mut params = vec![0.0; SDF_PARAMS_LEN + 1];
        params[0] = 1.0;
        let args = serde_json::json!({
            "fragment_shader": format!("shaders/blob.{}", platform_ext()),
            "params": params,
        });
        assert!(check(&args).is_err());
    }

    #[test]
    fn check_accepts_short_params() {
        // Less than SDF_PARAMS_LEN is fine: the rest defaults to 0.
        let args = serde_json::json!({
            "fragment_shader": format!("shaders/blob.{}", platform_ext()),
            "params": [1.0, 2.0, 3.0],
        });
        assert!(check(&args).is_ok());
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
        assert!(check(&args).is_err());
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
        assert!(check(&args).is_ok());
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
        assert!(check(&args).is_err());
    }

    #[test]
    fn source_path_prefers_map_over_single() {
        use crate::build::{Platform, SourceBacked};
        let args = serde_json::json!({
            "fragment_shader": "shaders/single.metal",
            "fragment_shaders": { "metal": "shaders/from_map.metal" },
        });
        assert_eq!(
            <SdfVolume as SourceBacked>::source_path(&args, Platform::Metal).as_deref(),
            Some("shaders/from_map.metal")
        );
    }

    #[test]
    fn from_args_collapses_map_to_current_backend() {
        // The runtime struct should carry the current backend's path in
        // `fragment_shader` so the DirectX path-extension filter still works
        // for map-authored volumes.
        // Include every backend so the collapse resolves regardless of which
        // backend this test build targets (metal / hlsl / glsl).
        let mut map = std::collections::BTreeMap::new();
        map.insert("metal".to_string(), "shaders/blob.metal".to_string());
        map.insert("hlsl".to_string(), "shaders/blob.hlsl".to_string());
        map.insert("glsl".to_string(), "shaders/blob.glsl".to_string());
        let a = SdfVolume {
            fragment_shaders: Some(map),
            ..Default::default()
        };
        let resolved = SdfVolume::from_args(a);
        assert_eq!(
            resolved.fragment_shader,
            format!("shaders/blob.{}", platform_ext())
        );
    }

    #[test]
    fn volumetric_forces_cast_shadows_off() {
        let a = SdfVolume {
            volumetric: true,
            cast_shadows: true,
            ..Default::default()
        };
        let fixed = SdfVolume::from_args(a);
        assert!(fixed.volumetric);
        assert!(
            !fixed.cast_shadows,
            "volumetric SDFs are translucent and must not cast hard shadows"
        );
    }

    #[test]
    fn volumetric_default_is_off() {
        let v = SdfVolume::default();
        assert!(!v.volumetric);
    }

    #[test]
    fn roundtrip_through_args() {
        let mut v = SdfVolume {
            centre: [1.0, 2.0, 3.0],
            extent: [4.0, 5.0, 6.0],
            fragment_shader: "shaders/foo.metal".to_string(),
            ..Default::default()
        };
        v.params[7] = 0.42;
        let json = serde_json::to_value(v.to_args()).expect("serialises");
        let back: SdfVolume = serde_json::from_value(json).expect("deserialises");
        let back = SdfVolume::from_args(back);
        assert_eq!(back.centre, [1.0, 2.0, 3.0]);
        assert_eq!(back.extent, [4.0, 5.0, 6.0]);
        assert_eq!(back.fragment_shader, "shaders/foo.metal");
        assert_eq!(back.params[7], 0.42);
    }
}
