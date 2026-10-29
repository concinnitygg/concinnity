//! The Milky Way's shape: a band around the galactic plane, thicker and
//! brighter toward its core. Mirrors `milky_way_envelope` in `night_sky.hlsl`;
//! the shader breaks the band up with clumps and dark dust lanes on top.

use concinnity_core::math::exp;

/// The band's relative brightness at galactic latitude `latitude` and
/// longitude `longitude` (radians, 0 at the core, in `[-pi, pi]`): 1 at the
/// core's heart, falling away from the plane and toward the anticenter, over
/// a faint floor of unresolved stars.
pub(crate) fn milky_way_envelope(latitude: f32, longitude: f32) -> f32 {
    let core = exp(-(longitude * longitude) / (0.5 * 0.5));
    let width = 0.1 + 0.12 * core;
    let band = exp(-0.5 * (latitude * latitude) / (width * width));
    let along = 0.3 + 0.7 * exp(-(longitude * longitude) / (1.6 * 1.6));
    let bulge =
        exp(-(longitude * longitude) / (0.35 * 0.35) - (latitude * latitude) / (0.16 * 0.16));
    (band * along + 0.6 * bulge) / 1.6 + 0.03
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_band_is_brightest_at_its_core_and_fades_off_the_plane() {
        let heart = milky_way_envelope(0.0, 0.0);
        assert!((heart - 1.03).abs() < 1e-3, "{heart}");
        // Off the plane it falls to the floor.
        assert!(milky_way_envelope(0.6, 0.0) < 0.1 * heart);
        assert!(milky_way_envelope(1.4, 2.0) < 0.04);
        // Along the plane it dims toward the anticenter but stays a band.
        let anticenter = milky_way_envelope(0.0, core::f32::consts::PI);
        assert!(
            anticenter < 0.3 * heart && anticenter > 0.15,
            "{anticenter}"
        );
        // Symmetric about the plane and the core.
        assert_eq!(milky_way_envelope(0.2, 0.7), milky_way_envelope(-0.2, -0.7));
        // Thicker near the core than far from it.
        let half = |l: f32| milky_way_envelope(0.15, l) / milky_way_envelope(0.0, l);
        assert!(half(0.0) > half(2.5));
    }
}
