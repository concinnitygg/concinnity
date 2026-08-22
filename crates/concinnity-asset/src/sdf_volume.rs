// Raymarched signed-distance-field volume schema.
//
// Authors a world-space bounding box plus a user-written fragment shader
// (containing the SDF `map` and per-point `shade` functions). At init the
// backend builds a per-volume render pipeline that sphere-traces the SDF inside
// the box; hits write opaque colour into `hdr_resolve` and update the main depth
// attachment so the raymarched surface composites with rasterised geometry
// naturally.
//
// The user writes one `.metal` file that defines two functions:
//
// ```metal
// float map(float3 p, constant SdfParams& params, float time);
// SdfSurface shade(float3 p, float3 normal,
//                  constant SdfParams& params, float time);
// ```
//
// The engine prepends a header (`raymarch_helpers.metal`: IQ primitive library,
// `sdfNormal`, `coneRaymarch`, PBR helpers) and appends a template
// (`raymarch_template.metal`: vertex + `fragment_main` that reconstructs the
// ray, samples main depth for early-out, calls the user's `map` + `shade`,
// applies PBR + shadow, writes colour + depth). The wrapped source compiles at
// runtime, matching how the water / fog / decal / particle passes load their own
// MSL. The build pipeline reads the user's source file and packs the raw bytes
// as this volume's payload, so production `cn run` worlds don't need the .metal
// file on disk at runtime: the bytes ride in the blob.

use crate::{AssetId, PayloadLocator};
use alloc::collections::BTreeMap;
use alloc::string::String;

/// Per-volume parameter slots packed into a single fixed-size uniform
/// block. The user shader casts the bound buffer to its own typed
/// struct; the engine just transports the bytes. Sized to comfortably
/// fit a flow-water shader (flow speed, wave coefficients, deep + shallow
/// colours, foam params, ...) without forcing schema design.
pub const SDF_PARAMS_LEN: usize = 32;

/// A raymarched signed-distance-field volume. It occupies a world-space
/// bounding box; a user-authored fragment shader sphere-traces an SDF inside
/// the box, composites correctly with the surrounding scene through the depth
/// buffer, and shades hits with the engine's lighting helpers.
///
/// The fragment shader is selected per backend: a `fragment_shaders` map keyed
/// by `"metal"` / `"hlsl"` / `"glsl"` lets one volume target multiple backends,
/// and the build only requires the entry for the backend it is building for. A
/// single `fragment_shader` path is the fallback when no map entry matches.
///
/// ```rust
/// # use concinnity_asset::SdfVolume;
/// SdfVolume {
///     centre: [0.0, 2.0, -4.0],
///     extent: [2.0, 2.0, 2.0],
///     max_gradient: 1.0,
///     max_steps: 64,
///     max_distance: 12.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SdfVolume {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// World-space centre of the bounding box.
    pub centre: [f32; 3],
    /// XYZ half-widths of the bounding box. The raymarch is clipped to the box,
    /// so the SDF only has to be well-defined inside this region.
    pub extent: [f32; 3],
    /// Single-platform fragment shader source path (e.g.
    /// `"shaders/chrome_blob.metal"`), resolved relative to the project's
    /// `assets/` at build time. Used when `fragment_shaders` has no entry for
    /// the building backend; the file extension must match the backend
    /// (`.metal` / `.hlsl`). The file defines the SDF's `map` and `shade`
    /// functions.
    #[serde(default)]
    pub fragment_shader: String,
    /// Per-backend fragment shader source paths keyed by `"metal"`, `"hlsl"`,
    /// or `"glsl"`. Takes priority over `fragment_shader`, letting one volume
    /// target multiple backends from a single declaration.
    #[serde(default)]
    pub fragment_shaders: Option<BTreeMap<String, String>>,
    /// Worst-case gradient of the SDF, used to size the cone-march step. `1.0`
    /// is correct for any well-formed SDF; higher values shorten the step but
    /// stay safe. Must be > 0.
    pub max_gradient: f32,
    /// Maximum cone-march steps per pixel. Clamped to `[8, 256]`.
    pub max_steps: u32,
    /// Maximum march distance in metres. Must be ≥ 0.1.
    pub max_distance: f32,
    /// Generic parameter block passed to the shader as a uniform buffer; the
    /// shader interprets it however it likes. Up to 32 values.
    pub params: [f32; SDF_PARAMS_LEN],
    /// When true, the volume casts shadows onto the surrounding scene. Disable
    /// for translucent / volumetric effects that shouldn't block light.
    pub cast_shadows: bool,
    /// When true (the default), the volume is shadowed by the scene. Set to
    /// false for unlit / always-bright effects (energy fields, etc.).
    pub receive_shadows: bool,
    /// When true, the volume renders as a participating medium (clouds, smoke,
    /// fog blobs, energy fields) instead of an opaque surface. The shader must
    /// define `sampleVolume(p, params, time)` returning per-point density,
    /// scattering colour, and emission instead of `map` / `shade`. Volumetrics
    /// never cast shadows (`cast_shadows` is forced off). The medium fills the
    /// whole bounding box, so don't overlap it with geometry it should render
    /// behind.
    pub volumetric: bool,
    /// When false the volume is skipped each frame.
    pub visible: bool,
    /// Injected at load time from the blob def. Carries the user
    /// shader source bytes packed at build time.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for SdfVolume {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            centre: [0.0, 0.0, 0.0],
            extent: [1.0, 1.0, 1.0],
            fragment_shader: String::new(),
            fragment_shaders: None,
            max_gradient: 1.0,
            max_steps: 64,
            max_distance: 30.0,
            params: [0.0; SDF_PARAMS_LEN],
            cast_shadows: false,
            receive_shadows: true,
            volumetric: false,
            visible: true,
            locator: None,
        }
    }
}

impl SdfVolume {
    /// Effective cone-march step ratio derived from the Lipschitz
    /// constant. A 1-Lipschitz SDF (gradient ≤ 1) cone-marches at
    /// ratio 1; larger gradients shorten the step proportionally.
    pub fn cone_ratio(&self) -> f32 {
        1.0 / self.max_gradient.max(f32::EPSILON)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn a_blank_volume_is_a_visible_unit_box_that_receives_shadows() {
        let v = SdfVolume::default();
        assert_eq!(v.centre, [0.0, 0.0, 0.0]);
        assert_eq!(v.extent, [1.0, 1.0, 1.0]);
        assert_eq!(v.max_steps, 64);
        assert_eq!(v.max_distance, 30.0);
        assert_eq!(v.params, [0.0; SDF_PARAMS_LEN]);
        assert!(v.visible);
        assert!(v.receive_shadows);
        // Raymarched surfaces do not write the shadow map by default.
        assert!(!v.cast_shadows);
        assert!(!v.volumetric);
        assert!(v.locator.is_none());
    }

    #[test]
    fn a_one_lipschitz_field_cone_marches_at_full_ratio() {
        assert_eq!(SdfVolume::default().cone_ratio(), 1.0);
    }

    #[test]
    fn a_steeper_gradient_shortens_the_step_proportionally() {
        let v = SdfVolume {
            max_gradient: 4.0,
            ..SdfVolume::default()
        };
        assert_eq!(v.cone_ratio(), 0.25);
    }

    #[test]
    fn a_zero_or_negative_gradient_cannot_divide_by_zero() {
        // An authored 0 would otherwise make the step ratio infinite and hang
        // the march, so the divisor is floored at epsilon.
        for max_gradient in [0.0, -1.0] {
            let v = SdfVolume {
                max_gradient,
                ..SdfVolume::default()
            };
            assert!(v.cone_ratio().is_finite(), "{max_gradient}");
            assert_eq!(v.cone_ratio(), 1.0 / f32::EPSILON);
        }
    }

    #[test]
    fn per_backend_shader_sources_parse_and_round_trip_through_postcard() {
        let v: SdfVolume = serde_json::from_str(
            r#"{"centre":[0,2,0],"extent":[3,3,3],"max_gradient":2.0,
                "fragment_shaders":{"metal":"blob.metal","hlsl":"blob.hlsl"},
                "cast_shadows":true,"visible":false}"#,
        )
        .unwrap();
        assert_eq!(v.cone_ratio(), 0.5);
        assert!(v.cast_shadows);
        assert!(!v.visible);
        let per_backend = v.fragment_shaders.as_ref().expect("per-backend sources");
        assert_eq!(per_backend["metal"], "blob.metal");
        assert_eq!(per_backend["hlsl"], "blob.hlsl");
        // The single-source field stays empty when the map is used.
        assert!(v.fragment_shader.is_empty());

        let bytes = postcard::to_allocvec(&v).unwrap();
        let back: SdfVolume = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.extent, [3.0, 3.0, 3.0]);
        assert_eq!(
            back.fragment_shaders.expect("per-backend sources")["metal"],
            "blob.metal"
        );
        // Identity and payload location are injected at load, never authored.
        assert_eq!(back.asset_id, AssetId::default());
        assert!(back.locator.is_none());
    }

    #[test]
    fn a_single_source_volume_leaves_the_per_backend_map_absent() {
        let v: SdfVolume = serde_json::from_str(r#"{"fragment_shader":"blob.metal"}"#).unwrap();
        assert_eq!(v.fragment_shader, "blob.metal".to_string());
        assert!(v.fragment_shaders.is_none());
        // `params` is a fixed-width uniform block, so a short array is a length
        // mismatch rather than a partial fill.
        assert!(serde_json::from_str::<SdfVolume>(r#"{"params":[1.5]}"#).is_err());
    }
}
