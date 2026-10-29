use super::*;
use crate::components::Atmosphere;
use crate::render::environment::atmosphere::optics::transmittance;
use concinnity_core::fanout::Inline;
use std::sync::OnceLock;

fn earth() -> &'static DaylightCurves {
    static CURVES: OnceLock<DaylightCurves> = OnceLock::new();
    CURVES.get_or_init(|| {
        DaylightCurves::bake(
            &AtmosphereSettings::resolve(&Atmosphere::default(), None),
            &Inline,
        )
    })
}

fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

#[test]
fn the_constants_match_the_shader() {
    use crate::render::shader::consts::uint;
    let src = crate::render::shader::embedded::DAYLIGHT_TYPES;
    assert_eq!(uint(src, "DAYLIGHT_KNOTS"), DAYLIGHT_KNOTS);
}

// The knots run from straight down to straight up, densest at the horizon,
// and every cosine maps back onto the knot it came from.
#[test]
fn the_knots_crowd_round_the_horizon_and_map_back() {
    assert_eq!(knot_mu(0), -1.0);
    assert_eq!(knot_mu(DAYLIGHT_KNOTS - 1), 1.0);
    for i in 1..DAYLIGHT_KNOTS {
        assert!(knot_mu(i) > knot_mu(i - 1));
    }
    let middle = DAYLIGHT_KNOTS / 2;
    let near = knot_mu(middle) - knot_mu(middle - 1);
    let far = knot_mu(DAYLIGHT_KNOTS - 1) - knot_mu(DAYLIGHT_KNOTS - 2);
    assert!(near * 10.0 < far, "{near} {far}");
    for i in 0..DAYLIGHT_KNOTS {
        assert!((knot_coord(knot_mu(i)) - i as f32).abs() < 1e-4, "{i}");
    }
    assert_eq!(knot_coord(-3.0), 0.0);
    assert_eq!(knot_coord(3.0), (DAYLIGHT_KNOTS - 1) as f32);
}

// A sample on a knot is that knot, and between two it moves straight from
// one to the other.
#[test]
fn sampling_is_linear_between_knots() {
    let curves = earth();
    for i in [3, DAYLIGHT_KNOTS / 2, DAYLIGHT_KNOTS - 1] {
        let at = curves.sample(knot_mu(i));
        let knot = curves.knots()[i];
        for c in 0..3 {
            assert!(
                (at.irradiance[c] - knot.irradiance[c]).abs() < 1e-6 * (1.0 + knot.irradiance[c])
            );
        }
    }
    let i = DAYLIGHT_KNOTS / 2 + 4;
    let between = curves.sample(knot_mu(i) + (knot_mu(i + 1) - knot_mu(i)) * 0.25);
    let (a, b) = (curves.knots()[i], curves.knots()[i + 1]);
    // The coordinate is not linear in the cosine, so only its bracket holds.
    let lo = a.zenith[2].min(b.zenith[2]);
    let hi = a.zenith[2].max(b.zenith[2]);
    assert!((lo..=hi).contains(&between.zenith[2]));
}

// The sun's light at the ground is the transmittance straight out of the air
// by day, and nothing once it has set.
#[test]
fn the_ground_gets_the_sunlight_the_air_lets_through() {
    let s = AtmosphereSettings::resolve(&Atmosphere::default(), None);
    let curves = earth();
    let noon = curves.sample(1.0).transmittance;
    let straight = transmittance(&s, s.bottom_radius + 1e-3, 1.0);
    for c in 0..3 {
        assert!((noon[c] - straight[c]).abs() < 1e-3, "{c}");
    }
    assert_eq!(curves.sample(-0.05).transmittance, [0.0; 3]);
    let mut last = 0.0;
    for i in DAYLIGHT_KNOTS / 2..DAYLIGHT_KNOTS {
        let t = luminance(curves.knots()[i].transmittance);
        assert!(t >= last, "{i}");
        last = t;
    }
    // A low sun is reddened.
    let low = curves.sample(0.05).transmittance;
    assert!(low[0] > low[2] * 2.0, "{low:?}");
}

// The sky lights a level surface by a modest share of the direct sunlight
// at noon, dims through twilight, and is dark once the sun is well down.
#[test]
fn the_sky_light_follows_the_sun_down() {
    let curves = earth();
    let noon = curves.sample(1.0);
    let share = luminance(noon.irradiance) / luminance(noon.transmittance);
    assert!((0.04..0.4).contains(&share), "{share}");
    let mut last = f32::MAX;
    for mu in [1.0, 0.5, 0.1, 0.0, -0.1, -0.2] {
        let e = luminance(curves.sample(mu).irradiance);
        assert!(e < last, "{mu}: {e}");
        last = e;
    }
    assert!(luminance(curves.sample(-0.5).irradiance) < luminance(noon.irradiance) * 1e-4);
    // Twilight: no sun, but the sky still lights the ground.
    assert!(luminance(curves.sample(-0.05).irradiance) > 0.0);
}

// Overhead the day sky is blue; the horizon is paler than the zenith.
#[test]
fn the_day_sky_is_blue_overhead_and_paler_at_the_horizon() {
    let noon = earth().sample(0.8);
    assert!(noon.zenith[2] > noon.zenith[0] * 1.5, "{:?}", noon.zenith);
    let tint = |c: [f32; 3]| c[2] / c[0];
    assert!(tint(noon.horizon) < tint(noon.zenith));
}

// The curves describe the ground under the sky drawn over it: with an
// apparent radius, the bigger ground's.
#[test]
fn an_apparent_sky_is_baked_over_its_own_ground() {
    let small = AtmosphereSettings::resolve(
        &Atmosphere {
            apparent_radius_km: Some(6360.0),
            ..Default::default()
        },
        Some(([0.0, -50_000.0, 0.0], 50_000.0)),
    );
    let curves = DaylightCurves::bake(&small, &Inline);
    let reference = earth();
    for mu in [1.0, 0.3, 0.02] {
        let (a, b) = (curves.sample(mu), reference.sample(mu));
        let e = (luminance(a.irradiance), luminance(b.irradiance));
        assert!((e.0 - e.1).abs() < e.1 * 0.05 + 1e-6, "{mu}: {e:?}");
    }
}

// Spreading the bake over threads, in any order, bakes the same curves.
#[test]
fn a_spread_bake_matches_one_on_a_single_thread() {
    use concinnity_core::fanout::test_pools::{Reversed, Threads};
    let s = AtmosphereSettings::resolve(&Atmosphere::default(), None);
    assert_eq!(&DaylightCurves::bake(&s, &Threads(4)), earth());
    assert_eq!(&DaylightCurves::bake(&s, &Reversed(4)), earth());
}
