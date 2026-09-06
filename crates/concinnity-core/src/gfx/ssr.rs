//! Screen-space reflection (SSR) configuration. Backend-agnostic resolve of the
//! authored `PostProcessConfig` SSR fields into clamped settings, plus the
//! per-frame GPU uniform. The screen-space ray-march itself lives in each
//! backend's shader; this module owns only the parameter math so it can be
//! unit-tested without a GPU.

use crate::gfx::camera::{camera_to_world, view_ray_scale};

use crate::gfx::render_types::SsrParams;

// Upper bound on `intensity`. The resolve pass mixes the reflection over the
// base shading by a Fresnel-weighted amount, so a value above 1.0 would just
// over-brighten grazing edges; 1.0 is full physically-weighted reflection.
const MAX_INTENSITY: f32 = 1.0;

// Smallest usable march distance: a ray shorter than this finds nothing.
const MIN_DISTANCE: f32 = 1.0;
// Largest march distance. Caps the per-pixel work; beyond this the reflection
// would be too unreliable (and too expensive) to be worth marching.
const MAX_DISTANCE: f32 = 200.0;

// Number of ray-march samples the resolve shader takes. The step length is
// `max_distance / MARCH_STEPS`, so a longer ray spends a longer stride rather
// than more samples. Must match `SSR_MAX_STEPS` in the SSR resolve MSL.
const MARCH_STEPS: f32 = 48.0;

// View-space intersection tolerance as a multiple of the march stride. A ray
// point is a hit when it lands behind the scene surface by less than this:
// wide enough to catch a crossing between two samples, tight enough not to
// punch through thin geometry.
const THICKNESS_SCALE: f32 = 2.5;

/// Canonical roughness cut for sharp reflections: surfaces rougher than this get
/// no screen-space / ray-traced reflection. One value drives four shaders that
/// must agree for the reflection pipeline to be self-consistent:
///
/// - the SSR resolve gate (ssr.metal)
/// - the RT-reflection resolve gate (rt_reflections.slang)
/// - the roughness blur ramp (reflection_composite.metal)
/// - the forward double-count fade (`REFL_RESOLVE_CUT` in main_shading.slang)
///
/// All four shaders are compiled offline, so each declares the literal itself
/// and a unit test locks every declaration to this value (the engine shaders in
/// reflection_shaders_lock_shared_roughness_cut, main_shading.slang alongside this
/// module). As an MSL `constant` it folds at compile time: sharing it costs
/// nothing at runtime.
pub const REFLECTION_ROUGHNESS_CUT: f32 = 0.6;

/// Clamped SSR tunables resolved from the authored asset fields. Held by the
/// backend and turned into a per-frame [`SsrParams`] once the camera is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsrSettings {
    /// Reflection blend strength multiplier in `[0, 1]`.
    pub intensity: f32,
    /// World-space distance the reflection ray marches before giving up.
    pub max_distance: f32,
}

impl SsrSettings {
    /// Clamp the authored intensity / distance into a safe range.
    pub fn resolve(intensity: f32, max_distance: f32) -> Self {
        Self {
            intensity: intensity.clamp(0.0, MAX_INTENSITY),
            max_distance: max_distance.clamp(MIN_DISTANCE, MAX_DISTANCE),
        }
    }

    /// Build the per-frame GPU uniform from these settings and the active
    /// camera. `fov_y_radians` is the vertical field of view and `aspect` the
    /// viewport width / height ratio: together they give the view-ray scale
    /// the resolve pass needs to project a view-space ray point to a UV.
    /// `inv_view_rot` is the view-space to world-space rotation and `cam_pos` the
    /// world camera position (together the rigid camera-to-world transform), and
    /// `prefilter_mip_count` the IBL prefilter cubemap mip count (0 = no IBL); the
    /// resolve uses these to sample the cubemap (or a reflection probe) as a
    /// reflection fallback, through the sky's inverse rotation `sky_rot`.
    pub fn params(
        &self,
        fov_y_radians: f32,
        aspect: f32,
        inv_view_rot: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        prefilter_mip_count: f32,
        sky_rot: [[f32; 4]; 3],
    ) -> SsrParams {
        let stride = self.max_distance / MARCH_STEPS;
        // The resolve rebuilds the world-space surface position the reflection
        // probe box-projects against, so it needs the full camera-to-world.
        let inv_view = camera_to_world(inv_view_rot, cam_pos);
        let (tan_half_fov_y, aspect) = view_ray_scale(fov_y_radians, aspect);
        SsrParams {
            intensity: self.intensity,
            max_distance: self.max_distance,
            tan_half_fov_y,
            aspect,
            stride,
            thickness: stride * THICKNESS_SCALE,
            prefilter_mip_count,
            _pad: 0.0,
            inv_view,
            sky_rot,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::camera::MIN_ASPECT;
    use crate::gfx::transform::IDENTITY;
    use crate::sky::SkyOrientation;

    #[test]
    fn resolve_clamps_intensity_and_distance() {
        let s = SsrSettings::resolve(5.0, 1.0e6);
        assert_eq!(s.intensity, MAX_INTENSITY);
        assert_eq!(s.max_distance, MAX_DISTANCE);

        let s = SsrSettings::resolve(-2.0, -10.0);
        assert_eq!(s.intensity, 0.0);
        assert_eq!(s.max_distance, MIN_DISTANCE);
    }

    #[test]
    fn resolve_passes_through_in_range_values() {
        let s = SsrSettings::resolve(0.7, 40.0);
        assert_eq!(s.intensity, 0.7);
        assert_eq!(s.max_distance, 40.0);
    }

    #[test]
    fn params_derive_stride_and_thickness_from_distance() {
        let s = SsrSettings::resolve(0.7, 48.0);
        let p = s.params(
            core::f32::consts::FRAC_PI_2,
            1.6,
            IDENTITY,
            [0.0; 3],
            6.0,
            SkyOrientation::IDENTITY_ROWS,
        );
        // 48 units over 48 steps -> a 1-unit stride.
        assert!((p.stride - 1.0).abs() < 1.0e-5);
        assert!((p.thickness - THICKNESS_SCALE).abs() < 1.0e-5);
        // A 90-degree vertical FOV has tan(45 deg) == 1.
        assert!((p.tan_half_fov_y - 1.0).abs() < 1.0e-5);
        assert_eq!(p.aspect, 1.6);
    }

    #[test]
    fn params_floor_a_degenerate_aspect() {
        let s = SsrSettings::resolve(0.7, 40.0);
        let p = s.params(
            core::f32::consts::FRAC_PI_2,
            0.0,
            IDENTITY,
            [0.0; 3],
            0.0,
            SkyOrientation::IDENTITY_ROWS,
        );
        assert!(p.aspect >= MIN_ASPECT);
    }

    #[test]
    fn params_pass_through_ibl_fallback_inputs() {
        let s = SsrSettings::resolve(0.7, 40.0);
        let p = s.params(
            core::f32::consts::FRAC_PI_2,
            1.6,
            IDENTITY,
            [0.0; 3],
            7.0,
            SkyOrientation::IDENTITY_ROWS,
        );
        assert_eq!(p.prefilter_mip_count, 7.0);
        // Identity rotation + a zero camera position leaves the matrix identity.
        assert_eq!(p.inv_view, IDENTITY);
    }

    #[test]
    fn params_assemble_camera_to_world_translation_column() {
        // inv_view's translation column must be the world camera position so the
        // resolve can lift a reconstructed view-space surface point to the right
        // world point for reflection-probe box projection.
        let s = SsrSettings::resolve(0.7, 40.0);
        let p = s.params(
            core::f32::consts::FRAC_PI_2,
            1.6,
            IDENTITY,
            [3.0, 4.0, 5.0],
            6.0,
            SkyOrientation::IDENTITY_ROWS,
        );
        assert_eq!(p.inv_view[3], [3.0, 4.0, 5.0, 1.0]);
        // The rotation columns are untouched.
        assert_eq!(p.inv_view[0], [1.0, 0.0, 0.0, 0.0]);
    }
}
