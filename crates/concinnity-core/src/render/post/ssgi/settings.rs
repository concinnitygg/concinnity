//! Screen-space global illumination (SSGI) configuration. Backend-agnostic
//! resolve of the authored `PostProcessConfig` SSGI fields into clamped
//! settings, plus the per-frame GPU uniform. SSGI reuses the depth + normal
//! pre-pass G-buffer SSR reads, but integrates bounced radiance over a
//! cosine-weighted hemisphere instead of along a single reflection vector,
//! accumulates it over frames, and adds the result on top of the IBL ambient
//! term. This module owns only the parameter math so it can be unit-tested
//! without a GPU.

use crate::components::{IndirectLighting, PostProcessConfig};
use crate::gfx::camera::view_ray_scale;

use crate::gfx::render_types::SsgiParams;

// Upper bound on `intensity`. The composite pass adds the gathered indirect
// radiance on top of the existing shading, so this is an additive multiplier
// rather than a `[0, 1]` blend: values above 1 exaggerate the bounce.
const MAX_INTENSITY: f32 = 4.0;

// Smallest usable ray reach: a ray shorter than this finds nothing.
const MIN_DISTANCE: f32 = 0.5;
// Largest ray reach. SSGI is a near-field effect (the far field is the IBL
// term's job), so the reach is capped well below SSR's.
const MAX_DISTANCE: f32 = 100.0;

// Hemisphere rays traced per pixel each frame. The accumulation averages rays
// across frames, so more rays per frame buy faster convergence after a
// disocclusion or a lighting change rather than a smoother steady state. The
// default is the authored `PostProcessConfig.ssgi_rays` default, owned by the
// schema.
#[cfg(test)]
pub(crate) const DEFAULT_RAYS: u32 = crate::components::DEFAULT_SSGI_RAYS;
const MIN_RAYS: u32 = 1;
/// Most hemisphere rays a pixel may trace per frame.
pub const MAX_RAYS: u32 = 4;

// View-space intersection tolerance as a fraction of the ray's reach. A ray is
// a hit where it passes behind the scene surface by less than this: wide enough
// that a ray skimming a surface lands on it, tight enough not to catch thin
// geometry the ray passes behind.
const THICKNESS_FRACTION: f32 = 1.0 / 6.0;

/// Clamped SSGI tunables resolved from the authored asset fields. Held by the
/// backend and turned into a per-frame [`SsgiParams`] once the camera is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsgiSettings {
    /// Indirect-bounce blend strength multiplier in `[0, MAX_INTENSITY]`.
    pub intensity: f32,
    /// View-space distance a hemisphere ray travels before it misses.
    pub max_distance: f32,
    /// Hemisphere rays traced per pixel each frame, clamped to
    /// `[MIN_RAYS, MAX_RAYS]`.
    pub rays: u32,
    /// Render-resolution divisor for the trace and its accumulation: 1 is full
    /// resolution, 2 is half (a quarter of the pixels), 4 a quarter. The
    /// composite is a depth-aware filter, so it upsamples the lower-resolution
    /// accumulation back to full resolution as it goes.
    pub gi_scale: u32,
}

/// Where a pass's accumulation stands this frame: the part of the uniform the
/// settings cannot know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsgiFrame {
    /// The frame counter the ray directions are drawn from.
    pub frame: u32,
    /// Whether last frame's accumulation can be reprojected.
    pub history_valid: bool,
    /// Levels in the closest-depth pyramid.
    pub levels: u32,
}

impl SsgiSettings {
    /// Resolve the SSGI tunables into clamped settings, or `None` when
    /// `indirect_lighting` is not `Ssgi`, or its intensity scales the gathered
    /// bounce to zero, so the backend can skip the SSGI passes.
    pub fn from_config(cfg: &PostProcessConfig) -> Option<Self> {
        (cfg.indirect_lighting == IndirectLighting::Ssgi)
            .then(|| {
                Self::resolve(
                    cfg.ssgi_intensity,
                    cfg.ssgi_max_distance,
                    cfg.ssgi_rays,
                    cfg.ssgi_resolution.scale_divisor(),
                )
            })
            .filter(|s| s.contributes())
    }

    /// Clamp the authored tunables into safe ranges.
    pub fn resolve(intensity: f32, max_distance: f32, rays: u32, gi_scale: u32) -> Self {
        Self {
            intensity: intensity.clamp(0.0, MAX_INTENSITY),
            max_distance: max_distance.clamp(MIN_DISTANCE, MAX_DISTANCE),
            rays: rays.clamp(MIN_RAYS, MAX_RAYS),
            gi_scale: gi_scale.max(1),
        }
    }

    /// Trace dimensions for a given render resolution: the render size divided
    /// by `gi_scale`, never below 1x1.
    pub fn gi_dimensions(&self, render_w: u32, render_h: u32) -> (u32, u32) {
        (
            (render_w / self.gi_scale).max(1),
            (render_h / self.gi_scale).max(1),
        )
    }

    /// Whether the pass can contribute anything to the frame. The composite
    /// scales the accumulated indirect term by `intensity` and blends it
    /// additively, so a zero intensity adds exactly zero and every SSGI draw is
    /// dead weight. Backends gate `FrameGraphInputs::ssgi_enabled` on this so
    /// the graph drops the node.
    pub fn contributes(&self) -> bool {
        self.intensity > 0.0
    }

    /// Build the per-frame GPU uniform from these settings, the active camera,
    /// and where the pass's accumulation stands. `fov_y_radians` is the
    /// vertical field of view and `aspect` the viewport width / height ratio:
    /// together they give the view-ray scale the trace needs to project a
    /// view-space ray point to a UV.
    pub fn params(&self, fov_y_radians: f32, aspect: f32, frame: SsgiFrame) -> SsgiParams {
        let (tan_half_fov_y, aspect) = view_ray_scale(fov_y_radians, aspect);
        SsgiParams {
            intensity: self.intensity,
            max_distance: self.max_distance,
            tan_half_fov_y,
            aspect,
            thickness: self.max_distance * THICKNESS_FRACTION,
            history_valid: if frame.history_valid { 1.0 } else { 0.0 },
            rays: self.rays,
            frame: frame.frame,
            levels: frame.levels,
            gi_scale: self.gi_scale,
            _pad: [0; 2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::PassResolution;
    use crate::gfx::camera::MIN_ASPECT;

    const FRAME: SsgiFrame = SsgiFrame {
        frame: 7,
        history_valid: true,
        levels: 5,
    };

    #[test]
    fn from_config_follows_indirect_lighting() {
        assert!(SsgiSettings::from_config(&PostProcessConfig::default()).is_some());
        let ibl = PostProcessConfig {
            indirect_lighting: IndirectLighting::Ibl,
            ..Default::default()
        };
        assert!(SsgiSettings::from_config(&ibl).is_none());
    }

    #[test]
    fn from_config_drops_an_inert_intensity() {
        let inert = PostProcessConfig {
            indirect_lighting: IndirectLighting::Ssgi,
            ssgi_intensity: 0.0,
            ..Default::default()
        };
        assert!(SsgiSettings::from_config(&inert).is_none());

        let faint = PostProcessConfig {
            indirect_lighting: IndirectLighting::Ssgi,
            ssgi_intensity: 0.05,
            ..Default::default()
        };
        assert!(SsgiSettings::from_config(&faint).is_some());
    }

    #[test]
    fn from_config_carries_resolution_and_rays() {
        let cfg = PostProcessConfig {
            indirect_lighting: IndirectLighting::Ssgi,
            ssgi_resolution: PassResolution::Quarter,
            ssgi_rays: 2,
            ..Default::default()
        };
        let s = SsgiSettings::from_config(&cfg).expect("ssgi on");
        assert_eq!(s.rays, 2);
        assert_eq!(s.gi_scale, 4);
    }

    #[test]
    fn from_config_resolves_and_clamps_when_enabled() {
        let cfg = PostProcessConfig {
            indirect_lighting: IndirectLighting::Ssgi,
            ssgi_intensity: 99.0,
            ssgi_max_distance: 1.0e6,
            ..Default::default()
        };
        let s = SsgiSettings::from_config(&cfg).expect("ssgi on");
        assert_eq!(s.intensity, 4.0);
        assert!(s.max_distance > 0.0 && s.max_distance.is_finite());
    }

    #[test]
    fn resolve_clamps_intensity_and_distance() {
        let s = SsgiSettings::resolve(9.0, 1.0e6, DEFAULT_RAYS, 1);
        assert_eq!(s.intensity, MAX_INTENSITY);
        assert_eq!(s.max_distance, MAX_DISTANCE);

        let s = SsgiSettings::resolve(-2.0, -10.0, DEFAULT_RAYS, 1);
        assert_eq!(s.intensity, 0.0);
        assert_eq!(s.max_distance, MIN_DISTANCE);
    }

    #[test]
    fn zero_intensity_does_not_contribute() {
        // A world may author `indirect_lighting: ssgi` and dial the intensity to
        // zero; the settings still resolve, so presence alone cannot gate the pass.
        assert!(!SsgiSettings::resolve(0.0, 8.0, DEFAULT_RAYS, 1).contributes());
        assert!(SsgiSettings::resolve(0.05, 8.0, DEFAULT_RAYS, 1).contributes());
        assert!(!SsgiSettings::resolve(-1.0, 8.0, DEFAULT_RAYS, 1).contributes());
    }

    #[test]
    fn resolve_passes_through_in_range_values() {
        let s = SsgiSettings::resolve(0.6, 8.0, DEFAULT_RAYS, 2);
        assert_eq!(s.intensity, 0.6);
        assert_eq!(s.max_distance, 8.0);
        assert_eq!(s.rays, DEFAULT_RAYS);
        assert_eq!(s.gi_scale, 2);
    }

    #[test]
    fn resolve_clamps_rays_and_scale() {
        // Over-range rays clamp to the maximum; a zero scale floors to full
        // resolution (1).
        let s = SsgiSettings::resolve(0.6, 8.0, 9999, 0);
        assert_eq!(s.rays, MAX_RAYS);
        assert_eq!(s.gi_scale, 1);
        let s = SsgiSettings::resolve(0.6, 8.0, 0, 4);
        assert_eq!(s.rays, MIN_RAYS);
        assert_eq!(s.gi_scale, 4);
    }

    #[test]
    fn the_default_ray_count_is_in_range() {
        assert!((MIN_RAYS..=MAX_RAYS).contains(&DEFAULT_RAYS));
    }

    #[test]
    fn gi_dimensions_divide_by_scale_and_floor_at_one() {
        let full = SsgiSettings::resolve(0.6, 8.0, DEFAULT_RAYS, 1);
        assert_eq!(full.gi_dimensions(1920, 1080), (1920, 1080));
        let half = SsgiSettings::resolve(0.6, 8.0, DEFAULT_RAYS, 2);
        assert_eq!(half.gi_dimensions(1920, 1080), (960, 540));
        // A tiny render target never collapses below 1x1.
        assert_eq!(half.gi_dimensions(1, 1), (1, 1));
    }

    #[test]
    fn params_carry_the_tunables_the_camera_and_the_frame() {
        let s = SsgiSettings::resolve(0.6, 12.0, 2, 2);
        let p = s.params(core::f32::consts::FRAC_PI_2, 1.6, FRAME);
        assert_eq!(p.intensity, 0.6);
        assert_eq!(p.max_distance, 12.0);
        // A 90-degree vertical FOV has tan(45 deg) == 1.
        assert!((p.tan_half_fov_y - 1.0).abs() < 1.0e-5);
        assert_eq!(p.aspect, 1.6);
        assert!((p.thickness - 2.0).abs() < 1.0e-5);
        assert_eq!(p.rays, 2);
        assert_eq!(p.gi_scale, 2);
        assert_eq!((p.frame, p.levels, p.history_valid), (7, 5, 1.0));

        let fresh = SsgiFrame {
            history_valid: false,
            ..FRAME
        };
        assert_eq!(s.params(1.0, 1.0, fresh).history_valid, 0.0);
    }

    #[test]
    fn the_thickness_scales_with_the_reach() {
        let near = SsgiSettings::resolve(0.6, 6.0, 1, 2).params(1.0, 1.0, FRAME);
        let far = SsgiSettings::resolve(0.6, 12.0, 1, 2).params(1.0, 1.0, FRAME);
        assert!((far.thickness - 2.0 * near.thickness).abs() < 1.0e-5);
    }

    #[test]
    fn params_floor_a_degenerate_aspect() {
        let s = SsgiSettings::resolve(0.6, 8.0, DEFAULT_RAYS, 1);
        let p = s.params(core::f32::consts::FRAC_PI_2, 0.0, FRAME);
        assert!(p.aspect >= MIN_ASPECT);
    }
}
