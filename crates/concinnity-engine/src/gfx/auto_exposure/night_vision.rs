// Night vision: the rods' share of sight at the luminance the eye is adapted
// to. The scene's radiance is relative to its sun, so it is read in absolute
// units by taking the sun's intensity as full sunlight.

use concinnity_core::math::{exp2, log2};

/// Illuminance of full sunlight at the ground, in lux: what a world's sun
/// intensity stands for when the scene's radiance is read in cd/m2.
pub const SUNLIGHT_LUX: f32 = 100_000.0;

/// Reflected-light meter calibration constant `K` (ISO 2720), in cd s/m2: a
/// scene of mean luminance `L` cd/m2 meters at `EV100 = log2(L * 100 / K)`.
pub const METER_CALIBRATION: f32 = 12.5;

/// Luminance, in cd/m2, at and below which sight is scotopic (rods only).
pub const SCOTOPIC_LUMINANCE: f32 = 0.01;

/// Luminance, in cd/m2, at and above which sight is photopic (cones only).
pub const PHOTOPIC_LUMINANCE: f32 = 3.0;

/// The EV100 a scene of mean luminance `luminance` cd/m2 meters at.
pub fn ev100(luminance: f32) -> f32 {
    log2(luminance * 100.0 / METER_CALIBRATION)
}

/// The mean luminance, in cd/m2, of a scene that meters at `ev100`.
pub fn luminance_at_ev100(ev100: f32) -> f32 {
    exp2(ev100) * METER_CALIBRATION / 100.0
}

/// The EV100 of a scene whose mean luminance is `2^log_lum` in scene units,
/// lit by a sun of intensity `sunlight`, or `None` without a sun to scale by.
pub fn scene_ev100(log_lum: f32, sunlight: f32) -> Option<f32> {
    (sunlight.is_finite() && sunlight > 0.0 && log_lum.is_finite())
        .then(|| ev100(exp2(log_lum) * (SUNLIGHT_LUX / sunlight)))
}

/// The rods' share of sight adapted to `luminance` cd/m2: 0 from
/// [`PHOTOPIC_LUMINANCE`] up, 1 from [`SCOTOPIC_LUMINANCE`] down, and a
/// smoothstep in log luminance through the mesopic range between.
pub fn rod_share(luminance: f32) -> f32 {
    if luminance.is_nan() {
        return 0.0;
    }
    let lo = log2(SCOTOPIC_LUMINANCE);
    let hi = log2(PHOTOPIC_LUMINANCE);
    let t = ((hi - log2(luminance.max(f32::MIN_POSITIVE))) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How far the composite turns the image to night vision: `strength` times
/// the rods' share at a scene mean of `2^log_lum` under a sun of intensity
/// `sunlight`.
pub fn night_vision_weight(strength: f32, log_lum: f32, sunlight: f32) -> f32 {
    scene_ev100(log_lum, sunlight).map_or(0.0, |ev| {
        strength.clamp(0.0, 1.0) * rod_share(luminance_at_ev100(ev))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ev100_and_luminance_invert_each_other() {
        // K = 12.5: EV100 0 is 0.125 cd/m2, and sunny 16 (EV100 15) ~4096.
        assert!((luminance_at_ev100(0.0) - 0.125).abs() < 1e-6);
        assert!((luminance_at_ev100(15.0) - 4096.0).abs() < 1e-2);
        assert!((ev100(1.0) - 3.0).abs() < 1e-5);
        for ev in [-6.0, -2.5, 0.0, 7.0, 15.0] {
            assert!((ev100(luminance_at_ev100(ev)) - ev).abs() < 1e-4, "{ev}");
        }
    }

    // A white surface under the sun is 100000 / pi cd/m2 whatever the sun's
    // intensity, since the shading divides the light by pi too.
    #[test]
    fn scene_luminance_scales_by_the_sun() {
        let white = |sun: f32| {
            let log_lum = log2(sun / core::f32::consts::PI);
            luminance_at_ev100(scene_ev100(log_lum, sun).unwrap())
        };
        for sun in [0.5, 1.0, 3.2, 10.0] {
            let l = white(sun);
            assert!(
                (l / (SUNLIGHT_LUX / core::f32::consts::PI) - 1.0).abs() < 1e-3,
                "{sun}: {l}"
            );
        }
        assert!(scene_ev100(-4.0, 0.0).is_none());
        assert!(scene_ev100(-4.0, f32::NAN).is_none());
        assert!(scene_ev100(f32::NAN, 1.0).is_none());
    }

    #[test]
    fn the_rods_take_over_through_the_mesopic_range() {
        for l in [3.0, 10.0, 1500.0, 1e6, f32::INFINITY] {
            assert_eq!(rod_share(l), 0.0, "{l}");
        }
        for l in [0.01, 0.0025, 1e-6, 0.0, -1.0] {
            assert_eq!(rod_share(l), 1.0, "{l}");
        }
        assert_eq!(rod_share(f32::NAN), 0.0);
        let mid = rod_share(exp2(
            0.5 * (log2(SCOTOPIC_LUMINANCE) + log2(PHOTOPIC_LUMINANCE)),
        ));
        assert!((mid - 0.5).abs() < 1e-5, "{mid}");
        let mut last = 0.0;
        for i in 0..=60 {
            let s = rod_share(exp2(4.0 - i as f32 * 0.25));
            assert!(s >= last - 1e-6, "{i}: {s} after {last}");
            last = s;
        }
    }

    // The composite's rods' luminance (Durand and Dorsey, from linear sRGB
    // through CIE XYZ), before its white scale.
    fn rods(c: [f32; 3]) -> f32 {
        let dot = |w: [f32; 3]| c[0] * w[0] + c[1] * w[1] + c[2] * w[2];
        let x = dot([0.4124, 0.3576, 0.1805]);
        let y = dot([0.2126, 0.7152, 0.0722]);
        let z = dot([0.0193, 0.1192, 0.9505]);
        y * (1.33 * (1.0 + (y + z) / x) - 1.68)
    }

    // The shader's constants are what they say: white keeps its luminance,
    // and the tint is chromaticity (0.25, 0.25) at unit luminance.
    #[test]
    fn composite_constants_match_their_derivation() {
        let src = crate::render::shader::embedded::COMPOSITE;
        let scale = crate::render::shader::consts::float(src, "SCOTOPIC_WHITE_SCALE");
        assert!((scale * rods([1.0; 3]) - 1.0).abs() < 1e-5, "{scale}");
        // Under the rods red goes dark and blue stays light.
        assert!(scale * rods([1.0, 0.0, 0.0]) < 0.5 * 0.2126);
        assert!(scale * rods([0.0, 0.0, 1.0]) > 2.0 * 0.0722);

        let (x, y) = (0.25_f32, 0.25_f32);
        let xyz = [x / y, 1.0, (1.0 - x - y) / y];
        let rgb = [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
        .map(|r| r[0] * xyz[0] + r[1] * xyz[1] + r[2] * xyz[2]);
        let text = crate::render::shader::consts::literal(src, "SCOTOPIC_TINT");
        let tint: alloc::vec::Vec<f32> = text
            .trim_start_matches("float3(")
            .trim_end_matches(')')
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        assert_eq!(tint.len(), 3);
        for i in 0..3 {
            assert!((tint[i] - rgb[i]).abs() < 1e-3, "{i}: {tint:?} vs {rgb:?}");
        }
        let luma = 0.2126 * tint[0] + 0.7152 * tint[1] + 0.0722 * tint[2];
        assert!((luma - 1.0).abs() < 1e-3, "{luma}");
        assert!(tint[2] > tint[1] && tint[1] > tint[0]);
    }

    // The planet's sun is 3.2: its day mean (2^-4.4) is photopic, its civil
    // dusk (2^-13.3) at the mesopic edge, and its moonlit night (2^-23.6)
    // scotopic.
    #[test]
    fn day_dusk_and_night_under_a_planet_sun() {
        assert_eq!(night_vision_weight(1.0, -4.4, 3.2), 0.0);
        assert!(night_vision_weight(1.0, -13.3, 3.2) < 0.01);
        assert_eq!(night_vision_weight(1.0, -23.6, 3.2), 1.0);
        assert_eq!(night_vision_weight(0.6, -23.6, 3.2), 0.6);
        // Off by default, clamped, and nothing without a sun.
        assert_eq!(night_vision_weight(0.0, -23.6, 3.2), 0.0);
        assert_eq!(night_vision_weight(5.0, -23.6, 3.2), 1.0);
        assert_eq!(night_vision_weight(-1.0, -23.6, 3.2), 0.0);
        assert_eq!(night_vision_weight(1.0, -23.6, 0.0), 0.0);
    }
}
