//! The SdfVolume asset: the authored schema (the struct, its `Default`,
//! `cone_ratio`, and `SDF_PARAMS_LEN`), the blob-residency helper the engine
//! init uses, and the runtime step-count clamp bounds. The
//! JSON-args source selection, validation, and the bake-time clamp live in
//! concinnity-cook (`authoring::source_args`, `check::sdf_volume`,
//! `authoring::validate::sdf_volume`).

use crate::ecs::PayloadLocator;
use alloc::string::String;

/// Per-volume parameter slots packed into a single fixed-size uniform
/// block. The user shader casts the bound buffer to its own typed
/// struct; the engine just transports the bytes. Sized to comfortably
/// fit a flow-water shader (flow speed, wave coefficients, deep + shallow
/// colors, foam params, ...) without forcing schema design.
pub const SDF_PARAMS_LEN: usize = 32;

/// A raymarched signed-distance-field volume. It occupies a world-space
/// bounding box; a user-authored fragment shader sphere-traces an SDF inside
/// the box, composites correctly with the surrounding scene through the depth
/// buffer, and shades hits with the engine's lighting helpers.
///
/// The distance field is one `.hlsl` file for every backend. The build
/// compiles it, so a field that does not compile fails `cn build` rather than
/// the renderer, and a shipped player needs no shader compiler of its own.
///
/// ```rust
/// # use concinnity_core::components::SdfVolume;
/// SdfVolume {
///     center: [0.0, 2.0, -4.0],
///     extent: [2.0, 2.0, 2.0],
///     max_gradient: 1.0,
///     max_steps: 64,
///     max_distance: 12.0,
///     ..Default::default()
/// };
/// ```
///
/// # The distance field
///
/// A field file defines functions, not an entry point. The engine owns the
/// march, the lighting and every binding, and calls the field from inside its
/// own pass. A surface volume's file defines the shape and its material:
///
/// ```hlsl
/// float map(float3 p, SdfParams params, float time);
/// SdfSurface shade(float3 p, float3 normal, SdfParams params, float time,
///                  float2 frag_uv);
/// ```
///
/// and a volumetric one defines the medium instead:
///
/// ```hlsl
/// VolumeSample sampleVolume(float3 p, SdfParams params, float time);
/// ```
///
/// `p` is a world-space point. Volumes may share one field file, so a field
/// that places its shape at `volume_center()` draws it in whichever volume
/// reads it:
///
/// ```hlsl
/// float map(float3 p, SdfParams params, float time)
/// {
///     return sdSphere(p - volume_center(), 0.5);
/// }
/// ```
///
/// Besides its own functions, a field can call:
///
/// - `sdf_param(params, i)`: parameter `i`, 0 to 31, of this volume's
///   `params`.
/// - `volume_center()`: the world-space center of the volume being drawn.
/// - `volume_extent()`: its half-widths.
/// - `sdSphere(p, r)`, `sdBox(p, b)`, `sdRoundBox(p, b, r)`, `sdTorus(p, t)`,
///   `sdCapsule(p, a, b, r)` and `sdPlane(p, n, h)`: distances to primitives
///   around the origin.
/// - `opSmoothUnion(a, b, k)`, `opSmoothSubtraction(d1, d2, k)` and
///   `opSmoothIntersection(a, b, k)`: two distances combined with a blend of
///   width `k`.
/// - `sampleSceneRefracted(frag_uv, normal, strength)`: the scene behind the
///   surface, bent by the normal, for a refractive `shade`.
///
/// `shade` returns an `SdfSurface`: `albedo`, `roughness`, `metallic`,
/// `emissive`, and `transmitted`, a color shown through the surface on top of
/// its lighting (zero for an opaque one). `sampleVolume` returns a
/// `VolumeSample`: `density` (0 is empty), `scattering`, which the sun's
/// light multiplies, and `emission`.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct SdfVolume {
    /// World-space center of the bounding box.
    pub center: [f32; 3],
    /// XYZ half-widths of the bounding box. The raymarch is clipped to the box,
    /// so the SDF only has to be well-defined inside this region.
    #[asset(default = [1.0, 1.0, 1.0])]
    pub extent: [f32; 3],
    /// Distance-field source path (e.g. `"shaders/chrome_blob.hlsl"`),
    /// resolved relative to the project's `assets/` at build time. The file
    /// defines `map` and `shade`, or `sampleVolume` for a volumetric volume.
    pub fragment_shader: String,
    /// Worst-case gradient of the SDF, used to size the cone-march step. `1.0`
    /// is correct for any well-formed SDF; higher values shorten the step but
    /// stay safe. Must be > 0.
    #[asset(default = 1.0)]
    pub max_gradient: f32,
    /// Maximum cone-march steps per pixel. Clamped to `[8, 256]`.
    #[asset(default = 64)]
    pub max_steps: u32,
    /// Maximum march distance in meters. Must be ≥ 0.1.
    #[asset(default = 30.0)]
    pub max_distance: f32,
    /// Generic parameter block passed to the shader as a uniform buffer; the
    /// shader interprets it however it likes. Up to 32 values.
    pub params: [f32; SDF_PARAMS_LEN],
    /// When true, the volume casts shadows onto the surrounding scene. Disable
    /// for translucent / volumetric effects that shouldn't block light.
    pub cast_shadows: bool,
    /// When true (the default), the volume is shadowed by the scene. Set to
    /// false for unlit / always-bright effects (energy fields, etc.).
    #[asset(default = true)]
    pub receive_shadows: bool,
    /// When true, the volume renders as a participating medium (clouds, smoke,
    /// fog blobs, energy fields) instead of an opaque surface. The shader must
    /// define `sampleVolume(p, params, time)` returning per-point density,
    /// scattering color, and emission instead of `map` / `shade`. Volumetrics
    /// never cast shadows (`cast_shadows` is forced off). The medium fills the
    /// whole bounding box, so don't overlap it with geometry it should render
    /// behind.
    pub volumetric: bool,
    /// When false the volume is skipped each frame.
    #[asset(default = true)]
    pub visible: bool,
    /// Injected at load time from the blob def. Carries the compiled distance
    /// field the build produced.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl SdfVolume {
    /// Effective cone-march step ratio derived from the Lipschitz
    /// constant. A 1-Lipschitz SDF (gradient ≤ 1) cone-marches at
    /// ratio 1; larger gradients shorten the step proportionally.
    pub fn cone_ratio(&self) -> f32 {
        1.0 / self.max_gradient.max(f32::EPSILON)
    }
}

/// Hard cap on the per-volume cone-march step count. Matches the
/// runtime kernel's loop bound; values above this are clamped.
pub const SDF_MAX_STEPS_CEILING: u32 = 256;

/// Lower bound on the per-volume cone-march step count. Below this the
/// march doesn't have enough budget to converge on anything interesting.
pub const SDF_MAX_STEPS_FLOOR: u32 = 8;

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
) -> alloc::collections::BTreeSet<u32> {
    ctx.query::<SdfVolume>()
        .filter_map(|v| v.locator.as_ref().map(|l| l.blob_index))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn a_short_params_array_is_rejected_rather_than_partially_filled() {
        assert!(serde_json::from_str::<SdfVolume>(r#"{"params":[1.5]}"#).is_err());
    }
}
