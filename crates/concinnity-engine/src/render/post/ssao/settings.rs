//! Screen-space ambient occlusion (GTAO) configuration. Backend-agnostic
//! resolve of the authored `AmbientOcclusion` SSAO fields into clamped
//! settings, plus the per-frame GPU uniform.

use crate::components::AmbientOcclusion;
use crate::gfx::camera::view_ray_scale;

use crate::gfx::render_types::SsaoParams;

// Upper bound on `intensity` so a stray asset value cannot drive the ambient
// term fully black across the whole frame.
const MAX_INTENSITY: f32 = 4.0;

// Smallest usable radius. A zero or negative radius would make every horizon
// search degenerate, so the authored value is floored here.
const MIN_RADIUS: f32 = 1.0e-3;

// The search radius in pixels at which the occlusion has faded out, and the one
// at which it is at full strength. A radius under a pixel resolves nothing, and
// fading over a range keeps the cutoff from showing as a line of constant depth.
const FADED_RADIUS_PX: f32 = 1.0;
const FULL_RADIUS_PX: f32 = 2.0;

/// Clamped SSAO tunables resolved from the authored asset fields. Held by the
/// backend and turned into a per-frame [`SsaoParams`] once the camera is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsaoSettings {
    /// World-space hemisphere radius the horizon search covers.
    pub radius: f32,
    /// Occlusion strength multiplier applied to the integrated visibility.
    pub intensity: f32,
}

impl SsaoSettings {
    /// Resolve the SSAO tunables into clamped settings, or `None` when the
    /// `enabled` toggle is off -- or on with an intensity that cannot darken
    /// anything -- so the backend can skip the SSAO passes entirely.
    pub fn from_config(cfg: &AmbientOcclusion) -> Option<Self> {
        cfg.enabled
            .then(|| Self::resolve(cfg.radius, cfg.intensity))
            .filter(Self::contributes)
    }

    /// Clamp the authored radius / intensity into a safe range.
    pub fn resolve(radius: f32, intensity: f32) -> Self {
        Self {
            radius: radius.max(MIN_RADIUS),
            intensity: intensity.clamp(0.0, MAX_INTENSITY),
        }
    }

    /// Whether the pass can darken anything. The kernel returns
    /// `pow(visibility, max(intensity, 0.0))`, so a zero intensity resolves to
    /// 1.0 everywhere -- the same value the renderer's 1x1 white fallback binds
    /// when SSAO is off. Resolving to `None` on this keeps a zero-intensity
    /// world on that already-supported path instead of paying the kernel and
    /// the blur to compute white.
    pub fn contributes(&self) -> bool {
        self.intensity > 0.0
    }

    /// Build the per-frame GPU uniform from these settings and the active
    /// camera. `fov_y_radians` is the vertical field of view and `aspect` the
    /// viewport width / height ratio: together they give the view-ray scale
    /// the kernel needs to rebuild view-space positions from linear depth.
    /// `height_px` is the render height, which sets how far out the radius
    /// still spans a pixel.
    pub fn params(&self, fov_y_radians: f32, aspect: f32, height_px: u32) -> SsaoParams {
        let (tan_half_fov_y, aspect) = view_ray_scale(fov_y_radians, aspect);
        let reach = self.depth_at_radius_px(tan_half_fov_y, height_px, FADED_RADIUS_PX);
        let full = self.depth_at_radius_px(tan_half_fov_y, height_px, FULL_RADIUS_PX);
        let fade_scale = -1.0 / (reach - full);
        SsaoParams {
            radius: self.radius,
            intensity: self.intensity,
            tan_half_fov_y,
            aspect,
            reach,
            fade_scale,
            fade_bias: -reach * fade_scale,
            _pad: 0.0,
        }
    }

    // The view depth at which the radius spans `px` pixels of a `height_px`
    // tall render: the viewport spans `2 * tan_half_fov_y * depth` there.
    fn depth_at_radius_px(self, tan_half_fov_y: f32, height_px: u32, px: f32) -> f32 {
        self.radius * height_px.max(1) as f32 / (2.0 * tan_half_fov_y.max(1.0e-6) * px)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_follows_the_toggle() {
        assert!(SsaoSettings::from_config(&AmbientOcclusion::default()).is_some());
        let off = AmbientOcclusion {
            enabled: false,
            ..Default::default()
        };
        assert!(SsaoSettings::from_config(&off).is_none());
    }

    #[test]
    fn from_config_drops_an_inert_intensity() {
        // The backends read presence, so a toggle left on with a value that
        // cannot change a pixel has to resolve away here or the passes run for
        // nothing. Verified pixel-for-pixel neutral.
        let inert = AmbientOcclusion {
            enabled: true,
            intensity: 0.0,
            ..Default::default()
        };
        assert!(SsaoSettings::from_config(&inert).is_none());

        let faint = AmbientOcclusion {
            enabled: true,
            intensity: 0.05,
            ..Default::default()
        };
        assert!(SsaoSettings::from_config(&faint).is_some());
    }

    #[test]
    fn from_config_resolves_and_clamps_when_enabled() {
        let cfg = AmbientOcclusion {
            enabled: true,
            radius: -1.0,
            intensity: 99.0,
        };
        let s = SsaoSettings::from_config(&cfg).expect("ssao on");
        assert!(s.radius > 0.0);
        assert_eq!(s.intensity, 4.0);
    }

    #[test]
    fn zero_intensity_does_not_contribute() {
        assert!(!SsaoSettings::resolve(0.5, 0.0).contributes());
        assert!(SsaoSettings::resolve(0.5, 0.05).contributes());
    }

    #[test]
    fn a_negative_intensity_clamps_to_no_contribution() {
        assert!(!SsaoSettings::resolve(0.5, -1.0).contributes());
    }

    #[test]
    fn resolve_floors_radius_and_clamps_intensity() {
        let s = SsaoSettings::resolve(-1.0, 100.0);
        assert!(s.radius >= MIN_RADIUS);
        assert_eq!(s.intensity, MAX_INTENSITY);

        let s = SsaoSettings::resolve(0.75, -2.0);
        assert_eq!(s.radius, 0.75);
        assert_eq!(s.intensity, 0.0);
    }

    #[test]
    fn resolve_passes_through_in_range_values() {
        let s = SsaoSettings::resolve(0.5, 1.0);
        assert_eq!(s.radius, 0.5);
        assert_eq!(s.intensity, 1.0);
    }

    // The occlusion strength the kernel applies at view depth `d`.
    fn fade(p: &SsaoParams, d: f32) -> f32 {
        (d * p.fade_scale + p.fade_bias).clamp(0.0, 1.0)
    }

    #[test]
    fn the_reach_is_where_the_radius_spans_one_pixel() {
        // tan(45 deg) == 1, so a 1000 px render spans 2 * depth there: a 0.5 m
        // radius covers one pixel at 250 m.
        let p = SsaoSettings::resolve(0.5, 1.0).params(core::f32::consts::FRAC_PI_2, 1.0, 1000);
        assert!((p.reach - 250.0).abs() < 1.0e-3, "{}", p.reach);
        let doubled =
            SsaoSettings::resolve(1.0, 1.0).params(core::f32::consts::FRAC_PI_2, 1.0, 2000);
        assert!((doubled.reach - 1000.0).abs() < 1.0e-2, "{}", doubled.reach);
    }

    #[test]
    fn the_occlusion_fades_out_between_two_pixels_and_one() {
        let p = SsaoSettings::resolve(0.5, 1.0).params(core::f32::consts::FRAC_PI_2, 1.0, 1000);
        // Two pixels at 125 m, one at 250 m.
        assert_eq!(fade(&p, 1.0), 1.0);
        assert!((fade(&p, 125.0) - 1.0).abs() < 1.0e-5);
        assert!((fade(&p, 187.5) - 0.5).abs() < 1.0e-5);
        assert!(fade(&p, 250.0).abs() < 1.0e-5);
        assert_eq!(fade(&p, 10_000.0), 0.0);
    }

    #[test]
    fn the_fade_falls_steadily_with_depth() {
        let p = SsaoSettings::resolve(0.5, 1.0).params(0.9, 1.7, 1080);
        let mut last = fade(&p, 0.0);
        for i in 1..=400 {
            let f = fade(&p, p.reach * i as f32 / 200.0);
            assert!(f <= last, "rose at step {i}");
            assert!(last - f <= 0.02, "jumped at step {i}");
            last = f;
        }
    }

    #[test]
    fn a_zero_height_render_keeps_the_params_finite() {
        let p = SsaoSettings::resolve(0.5, 1.0).params(0.9, 1.7, 0);
        assert!(p.reach.is_finite() && p.reach > 0.0);
        assert!(p.fade_scale.is_finite() && p.fade_bias.is_finite());
    }

    #[test]
    fn params_compute_tan_half_fov() {
        let s = SsaoSettings::resolve(0.5, 1.0);
        // A 90-degree vertical FOV has tan(45 deg) == 1.
        let p = s.params(core::f32::consts::FRAC_PI_2, 1.6, 1080);
        assert!((p.tan_half_fov_y - 1.0).abs() < 1.0e-5);
        assert_eq!(p.aspect, 1.6);
        assert_eq!(p.radius, 0.5);
        assert_eq!(p.intensity, 1.0);
    }

    #[test]
    fn params_floor_a_degenerate_aspect() {
        let s = SsaoSettings::resolve(0.5, 1.0);
        let p = s.params(core::f32::consts::FRAC_PI_2, 0.0, 1080);
        assert!(p.aspect >= MIN_RADIUS);
    }
}
