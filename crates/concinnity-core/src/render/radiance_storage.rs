//! Radiance kept in half-float targets: the cap every scene writer applies,
//! and the fixed scales the baked reflection cubes are stored at.
//!
//! The scene chain is written at the frame's pre-exposure, so it always sits
//! near the middle of half float's range. Probe faces and the atmosphere's
//! sky cube are drawn once and read for many frames, so they cannot carry a
//! frame's exposure. Stored absolute, a moonlit scene (about 1e-7 of the
//! sunlight) falls into half float's subnormal range, where it bands, and the
//! Milky Way (about 3e-8) below its smallest step, where it vanishes. Each
//! cube kind is written at its own constant scale instead and divided back
//! on read. Mirrored in `shaders/hdr_store.hlsl`, `probe_types.hlsl` and
//! `atmosphere.hlsl`.

use crate::math::log2;

/// The largest value a channel of a half-float scene target is written at.
/// A brighter color is scaled down whole, keeping its hue.
pub const HDR_STORE_MAX: f32 = 60000.0;

/// The scale a reflection probe's faces are drawn at, and its cube stored
/// at. The largest power of two at which a color the store cap scales down
/// still exceeds the prefilter's firefly clamp (12) in luminance, so the cap
/// never changes what a probe's prefiltered cube holds by day; a moonlit face
/// stays out of the subnormal range.
pub const PROBE_STORAGE_SCALE: f32 = 256.0;

/// The scale the atmosphere's sky cube is stored at. The cube holds no sun
/// disc, so a daylight sky stays far under the cap (about 58 after scaling)
/// while the Milky Way's faint floor keeps distinct half-float steps.
pub const SKY_CUBE_STORAGE_SCALE: f32 = 1024.0;

/// A probe face's clear: the background radiance `clear` (alpha kept) at
/// [`PROBE_STORAGE_SCALE`].
pub fn probe_clear_color(clear: [f32; 4]) -> [f32; 4] {
    let [r, g, b, a] = clear;
    let s = PROBE_STORAGE_SCALE;
    [r * s, g * s, b * s, a]
}

/// `rgb` capped to [`HDR_STORE_MAX`] in its brightest channel, its hue kept.
/// An infinite channel is capped alone. Mirrors `hdr_store_cap`.
pub fn store_cap(rgb: [f32; 3]) -> [f32; 3] {
    let peak = rgb[0].max(rgb[1]).max(rgb[2]);
    if peak <= HDR_STORE_MAX {
        rgb
    } else if peak.is_infinite() {
        rgb.map(|c| c.min(HDR_STORE_MAX))
    } else {
        rgb.map(|c| c * (HDR_STORE_MAX / peak))
    }
}

/// The environment cube rows (`SkyOrientation::sample_rows`) with the cube's
/// storage scale in row 0's `w`, as its base-2 logarithm: the sky cube's
/// when the atmosphere supplies the cube, 0 (scale 1) for an environment
/// map's. Readers multiply what they sample by `exp2(-w)`.
pub fn with_sky_cube_scale(mut rows: [[f32; 4]; 3], atmosphere: bool) -> [[f32; 4]; 3] {
    rows[0][3] = if atmosphere {
        log2(SKY_CUBE_STORAGE_SCALE)
    } else {
        0.0
    };
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::shader_consts::float;

    // The half-float value nearest `x`, ties to even, subnormals included.
    fn half(x: f32) -> f32 {
        let a = f64::from(x.abs());
        if a == 0.0 {
            return 0.0;
        }
        if a >= 65520.0 {
            return f32::INFINITY.copysign(x);
        }
        let exponent = a.log2().floor().max(-14.0) as i32;
        let step = 2f64.powi(exponent - 10);
        ((a / step).round_ties_even() * step).copysign(f64::from(x)) as f32
    }

    // `x` written at `scale` into half float and read back.
    fn round_trip(x: f32, scale: f32) -> f32 {
        half(store_cap([x * scale; 3])[0]) / scale
    }

    fn worst_error(values: impl Iterator<Item = f32>, scale: f32) -> f32 {
        values
            .map(|x| (round_trip(x, scale) - x).abs() / x)
            .fold(0.0, f32::max)
    }

    #[test]
    fn the_half_mirror_rounds_like_the_gpu() {
        assert_eq!(half(1.0), 1.0);
        assert_eq!(half(65504.0), 65504.0);
        assert_eq!(half(2f32.powi(-24)), 2f32.powi(-24));
        assert_eq!(half(2f32.powi(-26)), 0.0);
        assert_eq!(half(1.0 + 2f32.powi(-11)), 1.0);
        assert!(half(70000.0).is_infinite());
    }

    // Moonlit ground (a full moon gives 2.5e-6 of the sunlight) bands into a
    // handful of steps in an absolute probe face and keeps its gradient in
    // one stored at the probe scale.
    #[test]
    fn a_moonlit_probe_face_keeps_its_gradient() {
        let ramp = || (0..64).map(|i| 5e-8 + i as f32 * 2e-9);
        let mut absolute: alloc::vec::Vec<f32> = ramp().map(half).collect();
        absolute.dedup();
        assert!(absolute.len() <= 4, "{} steps", absolute.len());
        assert!(worst_error(ramp(), PROBE_STORAGE_SCALE) < 0.02);
    }

    // The Milky Way's heart (8.6e-9 of a 3.2 sun) and its 3 % floor round to
    // nothing in an absolute cube and survive the sky cube's scale.
    #[test]
    fn the_milky_way_survives_the_sky_cube() {
        let heart = 8.6e-9 * 3.2;
        let floor = heart * 0.03;
        assert_eq!(half(heart), 0.0);
        assert_eq!(half(floor), 0.0);
        assert!(worst_error([heart].into_iter(), SKY_CUBE_STORAGE_SCALE) < 0.002);
        assert!(worst_error([floor].into_iter(), SKY_CUBE_STORAGE_SCALE) < 0.05);
    }

    // By day the scales change nothing the prefilter keeps: its firefly clamp
    // (12, scaled with the cube) still fits half float, and a color the store
    // cap scales down stays above it, so the clamp makes it the same.
    #[test]
    fn day_radiance_fits_both_scales() {
        let firefly = 12.0;
        let min_luminance_share = 0.0722;
        for scale in [PROBE_STORAGE_SCALE, SKY_CUBE_STORAGE_SCALE] {
            assert!(firefly * scale <= 65504.0);
        }
        assert!(HDR_STORE_MAX / PROBE_STORAGE_SCALE * min_luminance_share > firefly);
        const { assert!(HDR_STORE_MAX / SKY_CUBE_STORAGE_SCALE > 50.0) };
        assert!((round_trip(1.0, SKY_CUBE_STORAGE_SCALE) - 1.0).abs() < 1e-3);
        assert!((round_trip(1.0, PROBE_STORAGE_SCALE) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn the_store_cap_keeps_the_hue() {
        assert_eq!(store_cap([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
        let capped = store_cap([120000.0, 60000.0, 30000.0]);
        assert_eq!(capped, [60000.0, 30000.0, 15000.0]);
        assert_eq!(
            store_cap([f32::INFINITY, 1.0, 70000.0]),
            [HDR_STORE_MAX, 1.0, HDR_STORE_MAX]
        );
    }

    #[test]
    fn a_probe_face_clears_at_its_scale() {
        let s = PROBE_STORAGE_SCALE;
        assert_eq!(
            probe_clear_color([0.5, 0.25, 0.0, 1.0]),
            [0.5 * s, 0.25 * s, 0.0, 1.0]
        );
    }

    #[test]
    fn the_rows_carry_the_sky_cube_scale() {
        let rows = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let sky = with_sky_cube_scale(rows, true);
        assert_eq!(2f32.powf(-sky[0][3]), 1.0 / SKY_CUBE_STORAGE_SCALE);
        assert_eq!(sky[1..], rows[1..]);
        assert_eq!(sky[0][..3], rows[0][..3]);
        assert_eq!(with_sky_cube_scale(sky, false), rows);
    }

    #[test]
    fn the_constants_match_the_shaders() {
        use crate::render::shaders::{ATMOSPHERE, HDR_STORE, PROBE_TYPES};
        assert_eq!(float(HDR_STORE, "HDR_STORE_MAX"), HDR_STORE_MAX);
        assert_eq!(
            float(PROBE_TYPES, "PROBE_STORAGE_SCALE"),
            PROBE_STORAGE_SCALE
        );
        assert_eq!(
            float(ATMOSPHERE, "SKY_CUBE_STORAGE_SCALE"),
            SKY_CUBE_STORAGE_SCALE
        );
    }
}
