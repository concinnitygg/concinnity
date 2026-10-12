//! Screen-space reflection (SSR) configuration. Backend-agnostic resolve of the
//! authored `PostProcessConfig` SSR fields into clamped settings, plus the
//! per-frame GPU uniform. The screen-space ray-march itself lives in each
//! backend's shader; this module owns only the parameter math so it can be
//! unit-tested without a GPU.

use crate::components::PostProcessConfig;
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

// Number of ray-march samples the resolve shader takes (`SSR_MAX_STEPS` in
// ssr.hlsl). The step length is `max_distance / MAX_STEPS`, so a longer ray
// spends a longer stride rather than more samples.
const MAX_STEPS: u32 = 48;

// View-space intersection tolerance as a multiple of the march stride. A ray
// point is a hit when it lands behind the scene surface by less than this:
// wide enough to catch a crossing between two samples, tight enough not to
// punch through thin geometry.
const THICKNESS_SCALE: f32 = 2.5;

/// Canonical roughness cut for sharp reflections: surfaces rougher than this get
/// no screen-space / ray-traced reflection. One value drives four shaders that
/// must agree for the reflection pipeline to be self-consistent:
///
/// - the SSR resolve gate (`ssr.hlsl`)
/// - the RT-reflection resolve gate (`rt_reflections.hlsl`)
/// - the roughness blur ramp (`reflection.hlsl`)
/// - the forward double-count fade (`main_shading.hlsl`)
///
/// Each declares the literal as a `static const`, and a unit test locks every
/// declaration to this value.
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

/// Per-frame camera inputs for building the SSR GPU uniform.
/// `fov_y_radians` / `aspect` give the view-ray scale the resolve projects a
/// view-space ray point to a UV with. `inv_view_rot` (the view-to-world
/// rotation) and `cam_pos` form the camera-to-world transform, and
/// `prefilter_mip_count` is the IBL prefilter cubemap mip count (0 = no IBL);
/// the resolve uses these to sample the cubemap (or a reflection probe) as a
/// reflection fallback, through the sky's inverse rotation `sky_rot`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsrParamsInputs {
    /// Vertical field of view in radians.
    pub fov_y_radians: f32,
    /// Viewport aspect ratio, width over height.
    pub aspect: f32,
    /// View-to-world rotation, column-major.
    pub inv_view_rot: [[f32; 4]; 4],
    /// World-space camera position.
    pub cam_pos: [f32; 3],
    /// IBL cubemap mip count; 0 when there is no IBL fallback.
    pub prefilter_mip_count: f32,
    /// Rows of the sky's inverse rotation.
    pub sky_rot: [[f32; 4]; 3],
    /// The scale the HDR chain is written at; the absolute fallback radiance
    /// is multiplied by it to match the scene taps.
    pub pre_exposure: f32,
}

impl SsrSettings {
    /// Resolve the SSR tunables into clamped settings, or `None` when the `ssr`
    /// toggle is off.
    ///
    /// Deliberately NOT gated on the intensity, unlike SSAO / SSGI: when a
    /// resolve is active the forward pass hands its glossy dielectric specular
    /// over to the resolve's composite (`ViewUniforms::reflections_enabled`), so
    /// a zero-intensity resolve still carries that term and dropping the pass
    /// changes the image. Measured at 1.53M of 3.28M pixels on a glossy floor.
    pub fn from_config(cfg: &PostProcessConfig) -> Option<Self> {
        cfg.ssr
            .then(|| Self::resolve(cfg.ssr_intensity, cfg.ssr_max_distance))
    }

    /// Clamp the authored intensity / distance into a safe range.
    pub fn resolve(intensity: f32, max_distance: f32) -> Self {
        Self {
            intensity: intensity.clamp(0.0, MAX_INTENSITY),
            max_distance: max_distance.clamp(MIN_DISTANCE, MAX_DISTANCE),
        }
    }

    /// Build the per-frame GPU uniform from these settings and the active
    /// camera.
    pub fn params(&self, inputs: SsrParamsInputs) -> SsrParams {
        let SsrParamsInputs {
            fov_y_radians,
            aspect,
            inv_view_rot,
            cam_pos,
            prefilter_mip_count,
            sky_rot,
            pre_exposure,
        } = inputs;
        let stride = self.max_distance / MAX_STEPS as f32;
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
            pre_exposure,
            inv_view,
            sky_rot,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::camera::MIN_ASPECT;
    use crate::render::shader_consts;
    use crate::sky::SkyOrientation;
    use crate::transform::IDENTITY;

    #[test]
    fn the_roughness_cut_matches_the_shader() {
        let src = crate::render::shaders::REFLECTION_CUT;
        assert_eq!(
            shader_consts::float(src, "REFLECTION_ROUGHNESS_CUT"),
            REFLECTION_ROUGHNESS_CUT
        );
    }

    #[test]
    fn the_march_step_count_matches_the_shader() {
        let src = crate::render::shaders::SSR;
        assert_eq!(
            shader_consts::uint(src, "SSR_MAX_STEPS"),
            MAX_STEPS as usize
        );
    }

    #[test]
    fn from_config_follows_the_toggle() {
        assert!(SsrSettings::from_config(&PostProcessConfig::default()).is_some());
        let off = PostProcessConfig {
            ssr: false,
            ..Default::default()
        };
        assert!(SsrSettings::from_config(&off).is_none());
    }

    #[test]
    fn from_config_keeps_a_zero_intensity() {
        // An active resolve owns the glossy dielectric specular the forward pass
        // hands over, so a zero-intensity resolve still changes the image.
        let cfg = PostProcessConfig {
            ssr: true,
            ssr_intensity: 0.0,
            ..Default::default()
        };
        assert!(SsrSettings::from_config(&cfg).is_some());
    }

    #[test]
    fn from_config_resolves_and_clamps_when_enabled() {
        let cfg = PostProcessConfig {
            ssr: true,
            ssr_intensity: 9.0,
            ssr_max_distance: 1.0e6,
            ..Default::default()
        };
        let s = SsrSettings::from_config(&cfg).expect("ssr on");
        assert_eq!(s.intensity, 1.0);
        assert!(s.max_distance > 0.0 && s.max_distance.is_finite());
    }

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
        let p = s.params(SsrParamsInputs {
            fov_y_radians: core::f32::consts::FRAC_PI_2,
            aspect: 1.6,
            inv_view_rot: IDENTITY,
            cam_pos: [0.0; 3],
            prefilter_mip_count: 6.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
        });
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
        let p = s.params(SsrParamsInputs {
            fov_y_radians: core::f32::consts::FRAC_PI_2,
            aspect: 0.0,
            inv_view_rot: IDENTITY,
            cam_pos: [0.0; 3],
            prefilter_mip_count: 0.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
        });
        assert!(p.aspect >= MIN_ASPECT);
    }

    #[test]
    fn params_pass_through_ibl_fallback_inputs() {
        let s = SsrSettings::resolve(0.7, 40.0);
        let p = s.params(SsrParamsInputs {
            fov_y_radians: core::f32::consts::FRAC_PI_2,
            aspect: 1.6,
            inv_view_rot: IDENTITY,
            cam_pos: [0.0; 3],
            prefilter_mip_count: 7.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
        });
        assert_eq!(p.prefilter_mip_count, 7.0);
        assert_eq!(p.pre_exposure, 1.0);
        // Identity rotation + a zero camera position leaves the matrix identity.
        assert_eq!(p.inv_view, IDENTITY);
    }

    #[test]
    fn params_assemble_camera_to_world_translation_column() {
        // inv_view's translation column must be the world camera position so the
        // resolve can lift a reconstructed view-space surface point to the right
        // world point for reflection-probe box projection.
        let s = SsrSettings::resolve(0.7, 40.0);
        let p = s.params(SsrParamsInputs {
            fov_y_radians: core::f32::consts::FRAC_PI_2,
            aspect: 1.6,
            inv_view_rot: IDENTITY,
            cam_pos: [3.0, 4.0, 5.0],
            prefilter_mip_count: 6.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
        });
        assert_eq!(p.inv_view[3], [3.0, 4.0, 5.0, 1.0]);
        // The rotation columns are untouched.
        assert_eq!(p.inv_view[0], [1.0, 0.0, 0.0, 0.0]);
    }
}
