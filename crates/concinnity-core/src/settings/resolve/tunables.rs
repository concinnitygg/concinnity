//! The composite's authored tunables, assembled from the assets that own them.

use crate::components::{AaMode, Bloom, Exposure, Vignette};
use crate::gfx::render_types::PostProcessTunables;

/// Resolve the authored composite tunables into the GPU-facing
/// `PostProcessTunables`: clamps each one and converts the exposure (stops)
/// into the linear multiplier the shaders expect. The composite's
/// display-output flags are not authored, so they are absent here: the backend
/// adds them to the full `PostProcessParams` once it has negotiated EDR
/// support with the display.
pub fn post_process_tunables(
    bloom: &Bloom,
    exposure: &Exposure,
    vignette: &Vignette,
    lut_strength: f32,
    aa_mode: AaMode,
) -> PostProcessTunables {
    PostProcessTunables {
        bloom_intensity: bloom.intensity.max(0.0),
        bloom_threshold: bloom.threshold.max(0.0),
        bloom_knee: bloom.knee.max(0.0),
        exposure: exposure.multiplier(),
        vignette: vignette.strength.clamp(0.0, 1.0),
        lut_strength: lut_strength.clamp(0.0, 1.0),
        fxaa: aa_mode.fxaa_flag(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::AntiAliasing;

    fn tunables(bloom: Bloom, vignette: f32, lut: f32, aa: AaMode) -> PostProcessTunables {
        post_process_tunables(
            &bloom,
            &Exposure::default(),
            &Vignette { strength: vignette },
            lut,
            aa,
        )
    }

    #[test]
    fn defaults_resolve_to_the_renderer_fallback() {
        // The renderer's no-asset fallback has to resolve to the same thing.
        let defaults = post_process_tunables(
            &Bloom::default(),
            &Exposure::default(),
            &Vignette::default(),
            1.0,
            AntiAliasing::default().mode,
        );
        assert_eq!(defaults, PostProcessTunables::DEFAULT);
    }

    #[test]
    fn negative_and_overrange_inputs_are_clamped() {
        let bloom = Bloom {
            intensity: -3.0,
            threshold: -1.0,
            knee: -0.2,
        };
        let p = tunables(bloom, 5.0, -2.0, AaMode::Fxaa);
        assert_eq!(p.bloom_intensity, 0.0);
        assert_eq!(p.bloom_threshold, 0.0);
        assert_eq!(p.bloom_knee, 0.0);
        assert_eq!(p.vignette, 1.0);
        assert_eq!(p.lut_strength, 0.0);
    }

    #[test]
    fn the_fxaa_gate_follows_the_aa_mode() {
        assert_eq!(tunables(Bloom::default(), 0.0, 1.0, AaMode::Off).fxaa, 0.0);
        assert_eq!(tunables(Bloom::default(), 0.0, 1.0, AaMode::Taa).fxaa, 1.0);
    }
}
