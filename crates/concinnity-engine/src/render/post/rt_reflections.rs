//! Hardware ray-traced reflection configuration. Backend-agnostic resolve of the
//! authored `Reflections` fields into clamped settings, plus the per-frame
//! GPU uniform. The acceleration-structure build and the inline ray-trace itself
//! live in the backend (Metal); this module owns only the parameter math so it
//! can be unit-tested without a GPU.
//!
//! RT reflections replace SSR's screen-space resolve: they reuse the same
//! authored `intensity` / `max_distance` tunables (so a world toggling
//! from SSR to RT keeps the same look knobs). Each glossy pixel first walks its
//! ray through SSGI's depth pyramid; only what the screen cannot answer (off
//! screen, behind geometry, or a hit it cannot trust on a mirror-like surface)
//! is traced against the scene BVH, so reflected geometry that is off-screen
//! still appears.

use crate::components::Reflections;
use crate::gfx::camera::{camera_to_world, view_ray_scale};

use crate::gfx::render_types::{LevelSurfacesGpu, RtParams};

// Upper bound on `intensity`. The kernel mixes the reflection over the base
// shading by a Fresnel-weighted amount, so a value above 1.0 would just
// over-brighten grazing edges; 1.0 is full physically-weighted reflection.
const MAX_INTENSITY: f32 = 1.0;

// Smallest usable ray reach: a ray shorter than this finds nothing.
const MIN_DISTANCE: f32 = 1.0;
// Largest ray reach. Unlike SSR's screen-march this is a true world-space
// `t_max` on the BVH traversal, so it can reach farther than the SSR cap
// without the per-step cost; still bounded so a stray value can't explode it.
const MAX_DISTANCE: f32 = 1000.0;

/// Threads per side of a classify group: the classify dispatches
/// [`classify_groups`] of them over the trace target.
pub const CLASSIFY_GROUP: u32 = 8;

/// Threads per trace group. The classify writes the trace's group count into
/// the ray arguments, so the host dispatches the trace indirectly.
pub const TRACE_GROUP: u32 = 64;

/// Most trace groups the classify asks for: the per-dimension dispatch limit
/// every API guarantees. The trace strides over a longer list.
pub const TRACE_MAX_GROUPS: u32 = 65_535;

/// The ray arguments each frame starts from, ahead of the classify: no trace
/// groups in x, one in y and z, and no listed pixels.
pub const RAY_ARGS_RESET: [u32; 4] = [0, 1, 1, 0];

/// Threads per group of the sky's pass over the listed misses. Mirrors
/// `REFLECTION_MISS_GROUP`.
pub const MISS_GROUP: u32 = 64;

/// Groups an indirect dispatch of the sky's pass may launch along x. Mirrors
/// `REFLECTION_MISS_MAX_GROUPS`.
pub const MISS_MAX_GROUPS: u32 = 65_535;

/// The listed misses' indirect arguments every frame starts from: no group in
/// x, one in y and z, and no ray listed.
pub const MISS_ARGS_RESET: [u32; 4] = [0, 1, 1, 0];

/// The classify's group count over a `width` x `height` trace target.
pub fn classify_groups(width: u32, height: u32) -> (u32, u32) {
    (
        width.div_ceil(CLASSIFY_GROUP),
        height.div_ceil(CLASSIFY_GROUP),
    )
}

/// Whether `params` has the trace list its sharp sky misses for the sky's pass.
pub fn sky_listed(params: &RtParams) -> bool {
    params.sky[0] > 0.5
}

/// `params` with every miss settled from the prefiltered sky, for a frame
/// whose sky pass cannot shade listed rays.
pub fn without_listed_sky(params: &RtParams) -> RtParams {
    RtParams {
        sky: [0.0; 4],
        ..*params
    }
}

/// Entries the ray list holds for a `width` x `height` trace target: one per
/// texel, enough for a frame in which every pixel is owed a ray.
pub fn ray_list_len(width: u32, height: u32) -> u64 {
    u64::from(width.max(1)) * u64::from(height.max(1))
}

/// Clamped RT-reflection tunables resolved from the authored asset fields. Held
/// by the backend and turned into a per-frame [`RtParams`] once the camera and
/// sun are known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtReflectionSettings {
    /// Reflection blend strength multiplier in `[0, 1]`.
    pub intensity: f32,
    /// World-space distance the reflection ray travels before it misses.
    pub max_distance: f32,
    /// Per-axis render-resolution divisor the trace target is sized by.
    pub divisor: u32,
    /// Whether a reflected hit casts a sun-shadow ray.
    pub sun_shadows: bool,
    /// Whether water traces its own per-pixel reflection in the transparent
    /// pass, rather than taking the probe, sky or planar reflection.
    pub water_trace: bool,
}

/// Per-frame camera + sun inputs for building the RT-reflection GPU uniform.
/// `fov_y_radians` / `aspect` give the view-ray scale used to rebuild a
/// view-space position from the SSR pre-pass G-buffer. `view` (the
/// world-to-view matrix) and `cam_pos` (the world camera position) give the
/// camera-to-world transform that lifts the reconstructed hit point + normal
/// into the BVH's world space. `sun_dir` is the world-space unit direction
/// toward the sun and `sun_color` its radiance; `prefilter_mip_count` is the IBL
/// cubemap mip count (0 = no IBL) for the miss fallback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtParamsInputs {
    /// Vertical field of view in radians.
    pub fov_y_radians: f32,
    /// Viewport aspect ratio, width over height.
    pub aspect: f32,
    /// World-to-view matrix, column-major. Only its rotation is read.
    pub view: [[f32; 4]; 4],
    /// World-space camera position.
    pub cam_pos: [f32; 3],
    /// World-space unit direction toward the sun.
    pub sun_dir: [f32; 3],
    /// Sun radiance, linear RGB.
    pub sun_color: [f32; 3],
    /// IBL cubemap mip count; 0 when there is no IBL fallback.
    pub prefilter_mip_count: f32,
    /// Rows of the sky's inverse rotation, so a missed ray samples the same
    /// oriented sky the main pass shows. Identity when the sky does not turn.
    pub sky_rot: [[f32; 4]; 3],
    /// The scale the HDR chain is written at.
    pub pre_exposure: f32,
    /// Levels in the depth pyramid the classify walks screen rays through this
    /// frame; 0 when none was built, and every glossy pixel then traces.
    pub screen_levels: u32,
    /// The level surfaces rays meet analytically, and the waves on them
    /// ([`ReflectionLayers::frame`](super::reflection_layer::ReflectionLayers::frame)).
    pub level: LevelSurfacesGpu,
    /// Whether a ray that leaves the scene with a sharp lobe is listed for
    /// the sky's own pass, which runs this frame, rather than taking the
    /// prefiltered sky.
    pub sky_listed: bool,
}

impl RtReflectionSettings {
    /// Resolve the ray-traced-reflection tunables into clamped settings, or
    /// `None` when `ray_traced` is off. Shares the `intensity` /
    /// `max_distance` fields with screen-space reflections; the backend additionally gates on GPU ray-tracing
    /// support. Not gated on the intensity, for the same specular-handover
    /// reason as [`SsrSettings::from_config`](super::ssr::settings::SsrSettings::from_config).
    pub fn from_config(cfg: &Reflections) -> Option<Self> {
        cfg.ray_traced.then(|| Self {
            divisor: cfg.ray_traced_resolution.scale_divisor(),
            sun_shadows: cfg.ray_traced_shadows,
            ..Self::resolve(cfg.intensity, cfg.max_distance)
        })
    }

    /// Clamp the authored intensity / distance into a safe range, tracing at
    /// full resolution with sun shadows and water traced.
    pub fn resolve(intensity: f32, max_distance: f32) -> Self {
        Self {
            intensity: intensity.clamp(0.0, MAX_INTENSITY),
            max_distance: max_distance.clamp(MIN_DISTANCE, MAX_DISTANCE),
            divisor: 1,
            sun_shadows: true,
            water_trace: true,
        }
    }

    /// The trace target's extent for a `width` x `height` render, never below
    /// one texel per axis.
    pub fn trace_extent(&self, width: u32, height: u32) -> (u32, u32) {
        let d = self.divisor.max(1);
        ((width / d).max(1), (height / d).max(1))
    }

    /// Build the per-frame GPU uniform from these settings, the active camera,
    /// and the sun.
    pub fn params(&self, inputs: RtParamsInputs) -> RtParams {
        let RtParamsInputs {
            fov_y_radians,
            aspect,
            view,
            cam_pos,
            sun_dir,
            sun_color,
            prefilter_mip_count,
            sky_rot,
            pre_exposure,
            screen_levels,
            level,
            sky_listed,
        } = inputs;
        let inv_view = camera_to_world(view, cam_pos);
        let (tan_half_fov_y, aspect) = view_ray_scale(fov_y_radians, aspect);
        RtParams {
            intensity: self.intensity,
            max_distance: self.max_distance,
            tan_half_fov_y,
            aspect,
            prefilter_mip_count,
            sun_shadows: if self.sun_shadows { 1.0 } else { 0.0 },
            trace_divisor: self.divisor.max(1) as f32,
            pre_exposure,
            cam_pos: [cam_pos[0], cam_pos[1], cam_pos[2], screen_levels as f32],
            sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            sun_color: [sun_color[0], sun_color[1], sun_color[2], 0.0],
            inv_view,
            sky_rot,
            level,
            sky: [if sky_listed { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::sky::SkyOrientation;
    use crate::gfx::camera::MIN_ASPECT;
    use crate::gfx::render_types::{LevelLayerGpu, MAX_LEVEL_LAYERS};
    use crate::render::post::reflection_layer::NO_LEVEL_SURFACES;
    use crate::transform::IDENTITY;

    #[test]
    fn the_miss_list_constants_match_the_shader() {
        use crate::render::shader::consts;
        let src = crate::render::shader::embedded::REFLECTION_MISS;
        assert_eq!(
            consts::uint(src, "REFLECTION_MISS_GROUP"),
            MISS_GROUP as usize
        );
        assert_eq!(
            consts::uint(src, "REFLECTION_MISS_MAX_GROUPS"),
            MISS_MAX_GROUPS as usize
        );
    }

    #[test]
    fn the_group_sizes_match_the_shader() {
        use crate::render::shader::consts;
        let src = crate::render::shader::embedded::RT_REFLECTIONS;
        assert_eq!(
            consts::uint(src, "RT_CLASSIFY_GROUP"),
            CLASSIFY_GROUP as usize
        );
        assert_eq!(consts::uint(src, "RT_TRACE_GROUP"), TRACE_GROUP as usize);
        assert_eq!(
            consts::uint(src, "RT_TRACE_MAX_GROUPS"),
            TRACE_MAX_GROUPS as usize
        );
    }

    #[test]
    fn classify_groups_cover_every_texel() {
        assert_eq!(classify_groups(1920, 1080), (240, 135));
        assert_eq!(classify_groups(1921, 1081), (241, 136));
        assert_eq!(classify_groups(1, 1), (1, 1));
    }

    #[test]
    fn the_ray_list_holds_one_entry_per_texel() {
        assert_eq!(ray_list_len(960, 540), 960 * 540);
        assert_eq!(ray_list_len(0, 0), 1);
    }

    #[test]
    fn from_config_follows_the_toggle() {
        assert!(RtReflectionSettings::from_config(&Reflections::default()).is_some());
        let off = Reflections {
            ray_traced: false,
            ..Default::default()
        };
        assert!(RtReflectionSettings::from_config(&off).is_none());
    }

    #[test]
    fn from_config_keeps_a_zero_intensity() {
        let cfg = Reflections {
            ray_traced: true,
            intensity: 0.0,
            ..Default::default()
        };
        assert!(RtReflectionSettings::from_config(&cfg).is_some());
    }

    #[test]
    fn from_config_reuses_ssr_tunables_when_enabled() {
        let cfg = Reflections {
            ray_traced: true,
            intensity: 9.0,
            max_distance: 1.0e6,
            ..Default::default()
        };
        let s = RtReflectionSettings::from_config(&cfg).expect("rt on");
        // Reuses the SSR intensity / distance fields, clamped by the RT resolve.
        assert_eq!(s.intensity, 1.0);
        assert!(s.max_distance > 0.0 && s.max_distance.is_finite());
    }

    #[test]
    fn from_config_carries_trace_resolution_and_shadows() {
        use crate::components::PassResolution;
        let cfg = Reflections {
            ray_traced_resolution: PassResolution::Quarter,
            ray_traced_shadows: false,
            ..Default::default()
        };
        let s = RtReflectionSettings::from_config(&cfg).expect("rt on");
        assert_eq!(s.divisor, 4);
        assert!(!s.sun_shadows);
    }

    #[test]
    fn trace_extent_divides_and_floors_at_one_texel() {
        let s = RtReflectionSettings {
            divisor: 2,
            ..RtReflectionSettings::resolve(0.7, 40.0)
        };
        assert_eq!(s.trace_extent(1921, 1080), (960, 540));
        assert_eq!(s.trace_extent(1, 1), (1, 1));
        let full = RtReflectionSettings::resolve(0.7, 40.0);
        assert_eq!(full.trace_extent(1921, 1080), (1921, 1080));
    }

    #[test]
    fn params_flag_sun_shadows() {
        let inputs = RtParamsInputs {
            fov_y_radians: 1.0,
            aspect: 1.0,
            view: IDENTITY,
            cam_pos: [0.0; 3],
            sun_dir: [0.0, 1.0, 0.0],
            sun_color: [1.0; 3],
            prefilter_mip_count: 0.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
            screen_levels: 0,
            level: NO_LEVEL_SURFACES,
            sky_listed: false,
        };
        let on = RtReflectionSettings::resolve(0.7, 40.0);
        assert_eq!(on.params(inputs).sun_shadows, 1.0);
        let off = RtReflectionSettings {
            sun_shadows: false,
            ..on
        };
        assert_eq!(off.params(inputs).sun_shadows, 0.0);
    }

    #[test]
    fn params_carry_the_trace_divisor() {
        let inputs = RtParamsInputs {
            fov_y_radians: 1.0,
            aspect: 1.0,
            view: IDENTITY,
            cam_pos: [0.0; 3],
            sun_dir: [0.0, 1.0, 0.0],
            sun_color: [1.0; 3],
            prefilter_mip_count: 0.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
            screen_levels: 0,
            level: NO_LEVEL_SURFACES,
            sky_listed: false,
        };
        let full = RtReflectionSettings::resolve(0.7, 40.0);
        assert_eq!(full.params(inputs).trace_divisor, 1.0);
        let quarter = RtReflectionSettings { divisor: 4, ..full };
        assert_eq!(quarter.params(inputs).trace_divisor, 4.0);
    }

    #[test]
    fn params_carry_the_screen_trace_levels() {
        let inputs = RtParamsInputs {
            fov_y_radians: 1.0,
            aspect: 1.0,
            view: IDENTITY,
            cam_pos: [1.0, 2.0, 3.0],
            sun_dir: [0.0, 1.0, 0.0],
            sun_color: [1.0; 3],
            prefilter_mip_count: 0.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
            screen_levels: 7,
            level: NO_LEVEL_SURFACES,
            sky_listed: false,
        };
        let s = RtReflectionSettings::resolve(0.7, 40.0);
        assert_eq!(s.params(inputs).cam_pos, [1.0, 2.0, 3.0, 7.0]);
        let none = RtParamsInputs {
            screen_levels: 0,
            ..inputs
        };
        assert_eq!(s.params(none).cam_pos[3], 0.0);
    }

    #[test]
    fn params_carry_the_level_layer() {
        let inputs = RtParamsInputs {
            fov_y_radians: 1.0,
            aspect: 1.0,
            view: IDENTITY,
            cam_pos: [0.0; 3],
            sun_dir: [0.0, 1.0, 0.0],
            sun_color: [1.0; 3],
            prefilter_mip_count: 0.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
            screen_levels: 0,
            level: LevelSurfacesGpu {
                layers: [LevelLayerGpu {
                    shape: [1.0, 2.0, 3.0, 4.0],
                    extent: [0.0, 0.0, 5.0, 1.0],
                    body: [0.1, 0.2, 0.3, 5.0],
                    surface: [0.5, 0.25, 0.0, 0.0],
                }; MAX_LEVEL_LAYERS],
                ..NO_LEVEL_SURFACES
            },
            sky_listed: true,
        };
        let p = RtReflectionSettings::resolve(0.7, 40.0).params(inputs);
        assert_eq!(p.level, inputs.level);
        assert_eq!(p.sky, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn resolve_clamps_intensity_and_distance() {
        let s = RtReflectionSettings::resolve(5.0, 1.0e6);
        assert_eq!(s.intensity, MAX_INTENSITY);
        assert_eq!(s.max_distance, MAX_DISTANCE);

        let s = RtReflectionSettings::resolve(-2.0, -10.0);
        assert_eq!(s.intensity, 0.0);
        assert_eq!(s.max_distance, MIN_DISTANCE);
    }

    #[test]
    fn resolve_passes_through_in_range_values() {
        let s = RtReflectionSettings::resolve(0.7, 60.0);
        assert_eq!(s.intensity, 0.7);
        assert_eq!(s.max_distance, 60.0);
    }

    #[test]
    fn params_carry_camera_and_sun_inputs() {
        let s = RtReflectionSettings::resolve(0.8, 40.0);
        let p = s.params(RtParamsInputs {
            fov_y_radians: core::f32::consts::FRAC_PI_2,
            aspect: 1.6,
            view: IDENTITY,
            cam_pos: [3.0, 4.0, 5.0],
            sun_dir: [0.0, 1.0, 0.0],
            sun_color: [1.0, 0.9, 0.8],
            prefilter_mip_count: 6.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
            screen_levels: 0,
            level: NO_LEVEL_SURFACES,
            sky_listed: false,
        });
        assert_eq!(p.intensity, 0.8);
        assert_eq!(p.max_distance, 40.0);
        // A 90-degree vertical FOV has tan(45 deg) == 1.
        assert!((p.tan_half_fov_y - 1.0).abs() < 1.0e-5);
        assert_eq!(p.aspect, 1.6);
        assert_eq!(p.prefilter_mip_count, 6.0);
        assert_eq!(p.cam_pos, [3.0, 4.0, 5.0, 0.0]);
        assert_eq!(p.sun_dir, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(p.sun_color, [1.0, 0.9, 0.8, 0.0]);
    }

    #[test]
    fn params_assemble_camera_to_world_translation_column() {
        // inv_view's translation column must be the world camera position so the
        // reconstructed view-space hit point lifts to the right world point.
        let s = RtReflectionSettings::resolve(0.8, 40.0);
        let p = s.params(RtParamsInputs {
            fov_y_radians: core::f32::consts::FRAC_PI_2,
            aspect: 1.6,
            view: IDENTITY,
            cam_pos: [3.0, 4.0, 5.0],
            sun_dir: [0.0, 1.0, 0.0],
            sun_color: [1.0, 1.0, 1.0],
            prefilter_mip_count: 6.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
            screen_levels: 0,
            level: NO_LEVEL_SURFACES,
            sky_listed: false,
        });
        assert_eq!(p.inv_view[3], [3.0, 4.0, 5.0, 1.0]);
        // The rotation columns are untouched.
        assert_eq!(p.inv_view[0], [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn params_floor_a_degenerate_aspect() {
        let s = RtReflectionSettings::resolve(0.7, 40.0);
        let p = s.params(RtParamsInputs {
            fov_y_radians: core::f32::consts::FRAC_PI_2,
            aspect: 0.0,
            view: IDENTITY,
            cam_pos: [0.0, 0.0, 0.0],
            sun_dir: [0.0, 1.0, 0.0],
            sun_color: [1.0, 1.0, 1.0],
            prefilter_mip_count: 0.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
            screen_levels: 0,
            level: NO_LEVEL_SURFACES,
            sky_listed: false,
        });
        assert!(p.aspect >= MIN_ASPECT);
    }

    #[test]
    fn unlisting_the_sky_keeps_every_other_field() {
        let s = RtReflectionSettings::resolve(0.7, 40.0);
        let listed = s.params(RtParamsInputs {
            fov_y_radians: core::f32::consts::FRAC_PI_2,
            aspect: 1.5,
            view: IDENTITY,
            cam_pos: [1.0, 2.0, 3.0],
            sun_dir: [0.0, 1.0, 0.0],
            sun_color: [1.0, 1.0, 1.0],
            prefilter_mip_count: 6.0,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 2.0,
            screen_levels: 4,
            level: NO_LEVEL_SURFACES,
            sky_listed: true,
        });
        assert!(sky_listed(&listed));
        let unlisted = without_listed_sky(&listed);
        assert!(!sky_listed(&unlisted));
        let restored = RtParams {
            sky: listed.sky,
            ..unlisted
        };
        assert_eq!(bytemuck::bytes_of(&restored), bytemuck::bytes_of(&listed));
    }
}
