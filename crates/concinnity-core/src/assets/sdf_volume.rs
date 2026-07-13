// src/assets/sdf_volume.rs
//
// Runtime behavior for the SdfVolume asset. The authored schema (the
// SdfVolume struct, its Default, `cone_ratio`, and `SDF_PARAMS_LEN`) lives in
// concinnity-asset; this file keeps the `Component` impl, the bake-time clamp,
// the blob-residency helper the engine init uses, and the runtime step-count
// clamp bounds. The JSON-args source selection and validation live in
// concinnity-world (`source_args`, `check::sdf_volume`). The schema type +
// `SDF_PARAMS_LEN` are re-exported so `crate::assets::sdf_volume::*` paths (the
// render backends' uniform structs) keep resolving.

pub use concinnity_asset::{SDF_PARAMS_LEN, SdfVolume};

use crate::ecs::asset_id::AssetId;
use crate::ecs::{Component, PayloadLocator};

/// Hard cap on the per-volume cone-march step count. Matches the
/// runtime kernel's loop bound; values above this are clamped.
pub const SDF_MAX_STEPS_CEILING: u32 = 256;

/// Lower bound on the per-volume cone-march step count. Below this the
/// march doesn't have enough budget to converge on anything interesting.
pub const SDF_MAX_STEPS_FLOOR: u32 = 8;

// Resolve the fragment shader source path for the current build backend from a
// volume's `fragment_shaders` map (preferred) or its `fragment_shader`
// fallback. Mirrors the build-time selection (concinnity-world `source_args`).
fn current_platform_source(v: &SdfVolume) -> Option<String> {
    let platform = crate::build::Platform::current();
    if let Some(map) = &v.fragment_shaders
        && let Some(src) = map.get(platform.key()).filter(|s| !s.is_empty())
    {
        return Some(src.clone());
    }
    if v.fragment_shader.is_empty() {
        return None;
    }
    let ext = std::path::Path::new(&v.fragment_shader)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if platform.accepts_ext(ext) {
        Some(v.fragment_shader.clone())
    } else {
        None
    }
}

// Normalize an authored volume for the runtime: clamp the raymarch knobs to
// sane bounds, force shadows off for translucent volumetrics (they write no
// depth), and collapse the per-backend `fragment_shaders` map to the current
// backend's `fragment_shader` (the DirectX raymarch pass filters volumes by
// that path's extension). Run by the build-side validator at bake time.
pub fn clamped(mut v: SdfVolume) -> SdfVolume {
    // Extents must be positive: a zero or negative extent would produce an
    // inside-out bounding box no fragment ever enters.
    for axis in v.extent.iter_mut() {
        if !axis.is_finite() || *axis <= 0.0 {
            *axis = 1.0;
        }
    }
    if !v.max_gradient.is_finite() || v.max_gradient <= 0.0 {
        v.max_gradient = 1.0;
    }
    v.max_steps = v
        .max_steps
        .clamp(SDF_MAX_STEPS_FLOOR, SDF_MAX_STEPS_CEILING);
    if !v.max_distance.is_finite() || v.max_distance < 0.1 {
        v.max_distance = 0.1;
    }
    if v.volumetric {
        v.cast_shadows = false;
    }
    if let Some(src) = current_platform_source(&v) {
        v.fragment_shader = src;
    }
    v
}

impl Component for SdfVolume {
    const NAME: &'static str = "SdfVolume";

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::result::CnResult> {
        Ok(postcard::from_bytes(bytes)?)
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
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
        let fixed = clamped(a.clone());
        assert_eq!(fixed.max_steps, SDF_MAX_STEPS_FLOOR);

        a.max_steps = 9999;
        let fixed = clamped(a);
        assert_eq!(fixed.max_steps, SDF_MAX_STEPS_CEILING);
    }

    #[test]
    fn from_args_repairs_bad_extent() {
        let a = SdfVolume {
            extent: [0.0, -1.0, f32::NAN],
            ..Default::default()
        };
        let fixed = clamped(a);
        assert_eq!(fixed.extent, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn from_args_repairs_bad_gradient_and_distance() {
        let a = SdfVolume {
            max_gradient: -0.5,
            max_distance: f32::NAN,
            ..Default::default()
        };
        let fixed = clamped(a);
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
        let resolved = clamped(a);
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
        let fixed = clamped(a);
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
        let json = serde_json::to_value(v.clone()).expect("serialises");
        let back: SdfVolume = serde_json::from_value(json).expect("deserialises");
        let back = clamped(back);
        assert_eq!(back.centre, [1.0, 2.0, 3.0]);
        assert_eq!(back.extent, [4.0, 5.0, 6.0]);
        assert_eq!(back.fragment_shader, "shaders/foo.metal");
        assert_eq!(back.params[7], 0.42);
    }
}
