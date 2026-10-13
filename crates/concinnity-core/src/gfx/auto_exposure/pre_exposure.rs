//! Pre-exposure: the scale every pass writing scene radiance into the
//! half-float HDR targets applies, so the stored values stay in FP16's normal
//! range however bright or dim the scene is.
//!
//! The rules every backend follows:
//!
//! - A pass writing the frame's HDR chain shades in absolute radiance and
//!   multiplies what it writes by [`PreExposure::value`].
//! - A pass reading the frame's chain back into absolute shading (a
//!   refraction, a reflection, a raymarched surface showing the scene through)
//!   divides the tap by it. A pass that only filters or composites the chain
//!   stays in the pre-exposed scale.
//! - Temporal history written last frame is multiplied by
//!   [`PreExposure::history_scale`] when read, so a change in pre-exposure
//!   neither flashes nor lags.
//! - Auto-exposure measures luminance with the pre-exposure divided out.
//! - Bloom and the composite take [`PreExposure::post_params`]: the exposure
//!   with the pre-exposure divided out, and the pre-exposure itself for the
//!   bloom prefilter's firefly weight.
//! - The main pass clears to the authored background times the pre-exposure
//!   ([`PreExposure::clear_color`]).
//! - A planar mirror is drawn every frame, at the frame's pre-exposure, and
//!   its readers divide it out like a read of the chain. Probe faces and the
//!   sky and irradiance cubes are read across frames, so they are stored at
//!   fixed scales instead (`render::radiance_storage`).

use crate::gfx::render_types::PostProcessParams;

// Bounds a pre-exposure is kept within: wide enough for every exposure the EV
// limits allow, narrow enough that dividing by it stays finite.
const MIN_VALUE: f32 = 1.0 / 16_777_216.0;
const MAX_VALUE: f32 = 16_777_216.0;

/// This frame's pre-exposure and the one the previous frame's history was
/// written at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreExposure {
    current: f32,
    previous: f32,
}

impl Default for PreExposure {
    fn default() -> Self {
        Self::UNIT
    }
}

impl PreExposure {
    /// No scaling: the HDR chain holds absolute radiance.
    pub const UNIT: Self = Self {
        current: 1.0,
        previous: 1.0,
    };

    /// A first frame at `value`, with no history to rescale.
    pub fn new(value: f32) -> Self {
        let v = sanitize(value);
        Self {
            current: v,
            previous: v,
        }
    }

    /// Moves to the next frame, written at `value`.
    pub fn advance(&mut self, value: f32) {
        self.previous = self.current;
        self.current = sanitize(value);
    }

    /// The scale this frame's HDR chain is written at.
    pub fn value(&self) -> f32 {
        self.current
    }

    /// What last frame's history is multiplied by to read at this frame's
    /// pre-exposure.
    pub fn history_scale(&self) -> f32 {
        self.current / self.previous
    }

    /// The color the main pass clears its HDR target to for an authored
    /// background `clear`: its radiance scaled, its alpha kept.
    pub fn clear_color(&self, clear: [f32; 4]) -> [f32; 4] {
        let [r, g, b, a] = clear;
        let v = self.current;
        [r * v, g * v, b * v, a]
    }

    /// The post-process block bloom and the composite read: its `exposure`
    /// scales the pre-exposed chain, so the pre-exposure is divided out of
    /// it, and `pre_exposure` carries the value itself.
    pub fn post_params(&self, post: &PostProcessParams) -> PostProcessParams {
        PostProcessParams {
            exposure: post.exposure / self.current,
            pre_exposure: self.current,
            ..*post
        }
    }
}

fn sanitize(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value.clamp(MIN_VALUE, MAX_VALUE)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn post() -> PostProcessParams {
        PostProcessParams {
            bloom_intensity: 0.5,
            bloom_threshold: 1.0,
            bloom_knee: 0.5,
            exposure: 0.5,
            vignette: 0.2,
            lut_strength: 1.0,
            hdr_output: 0.0,
            pq_output: 0.0,
            fxaa: 1.0,
            pre_exposure: 1.0,
        }
    }

    #[test]
    fn unit_scales_nothing() {
        let p = PreExposure::default();
        assert_eq!(p.value(), 1.0);
        assert_eq!(p.history_scale(), 1.0);
    }

    #[test]
    fn first_frame_has_no_history_to_rescale() {
        let p = PreExposure::new(8.0);
        assert_eq!(p.value(), 8.0);
        assert_eq!(p.history_scale(), 1.0);
    }

    #[test]
    fn history_rescales_by_the_ratio() {
        let mut p = PreExposure::new(2.0);
        p.advance(8.0);
        assert_eq!(p.value(), 8.0);
        assert_eq!(p.history_scale(), 4.0);
        // Last frame's history, written at 2, reads as if written at 8.
        let radiance = 0.25_f32;
        assert_eq!(radiance * 2.0 * p.history_scale(), radiance * p.value());
        p.advance(8.0);
        assert_eq!(p.history_scale(), 1.0);
        p.advance(1.0);
        assert_eq!(p.history_scale(), 0.125);
    }

    #[test]
    fn invalid_values_fall_back_to_unit() {
        for bad in [0.0, -3.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(PreExposure::new(bad).value(), 1.0, "{bad}");
        }
        let mut p = PreExposure::new(4.0);
        p.advance(f32::NAN);
        assert_eq!(p.value(), 1.0);
        assert_eq!(p.history_scale(), 0.25);
    }

    #[test]
    fn extreme_values_are_clamped_finite() {
        assert_eq!(PreExposure::new(1e30).value(), MAX_VALUE);
        assert_eq!(PreExposure::new(1e-30).value(), MIN_VALUE);
        let mut p = PreExposure::new(1e-30);
        p.advance(1e30);
        assert!(p.history_scale().is_finite());
        assert!((1.0 / p.value()).is_finite());
    }

    #[test]
    fn post_params_fold_the_pre_exposure_out_of_the_exposure() {
        let p = PreExposure::new(16.0);
        let folded = p.post_params(&post());
        assert_eq!(folded.exposure, 0.5 / 16.0);
        assert_eq!(folded.pre_exposure, 16.0);
        assert_eq!(folded.bloom_threshold, 1.0);
        assert_eq!(folded.fxaa, 1.0);
        // What the composite tonemaps is unchanged: radiance x pre x
        // (exposure / pre) == radiance x exposure.
        let radiance = 3.0_f32;
        assert_eq!(
            radiance * p.value() * folded.exposure,
            radiance * post().exposure
        );
    }

    #[test]
    fn the_clear_color_scales_radiance_but_not_alpha() {
        let p = PreExposure::new(4.0);
        assert_eq!(p.clear_color([0.1, 0.25, 0.5, 1.0]), [0.4, 1.0, 2.0, 1.0]);
        let c = [0.3, 0.2, 0.1, 0.5];
        assert_eq!(PreExposure::UNIT.clear_color(c), c);
    }

    #[test]
    fn unit_post_params_are_the_authored_block() {
        let unit = PreExposure::UNIT.post_params(&post());
        assert_eq!(unit.exposure, post().exposure);
        assert_eq!(unit.pre_exposure, 1.0);
    }
}
