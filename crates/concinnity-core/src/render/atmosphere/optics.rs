//! Light lost along a straight path through the air: the CPU mirror of the
//! transmittance table, which the sun's color is taken from.

use super::apparent::{table_view_angle, zenith_horizon};
use super::lut::TRANSMITTANCE_STEPS;
use super::settings::AtmosphereSettings;
use crate::math::{acos, asin, cos, exp, sqrt};

/// Distance from radius `r` along a ray whose cosine to the local up is `mu`
/// to where it leaves the sphere of `radius`, or 0 when it starts outside and
/// misses. Mirrors `ray_exit` in atmosphere_common.hlsl.
pub(super) fn ray_exit(r: f32, mu: f32, radius: f32) -> f32 {
    let disc = r * r * (mu * mu - 1.0) + radius * radius;
    (-r * mu + sqrt(disc.max(0.0))).max(0.0)
}

/// Whether a ray from radius `r` with cosine `mu` hits the ground sphere.
/// Mirrors `ray_hits_ground` in atmosphere_common.hlsl.
pub(super) fn ray_hits_ground(r: f32, mu: f32, bottom: f32) -> bool {
    mu < 0.0 && r * r * (mu * mu - 1.0) + bottom * bottom >= 0.0
}

/// Optical depth (extinction integrated over distance) from radius `r` with
/// cosine `mu` to the top of the air, midpoint-sampled in
/// [`TRANSMITTANCE_STEPS`] steps exactly as the transmittance kernel takes it.
pub fn optical_depth(s: &AtmosphereSettings, r: f32, mu: f32) -> [f32; 3] {
    let length = ray_exit(r, mu, s.top_radius);
    let dt = length / TRANSMITTANCE_STEPS as f32;
    let mut depth = [0.0_f32; 3];
    for i in 0..TRANSMITTANCE_STEPS {
        let t = (i as f32 + 0.5) * dt;
        let rt = sqrt(t * t + 2.0 * r * mu * t + r * r);
        let ext = s.extinction_at(rt - s.bottom_radius);
        for (d, e) in depth.iter_mut().zip(ext) {
            *d += e * dt;
        }
    }
    depth
}

/// The fraction of light surviving the path from radius `r` with cosine `mu`
/// out of the air, per color: 0 when the path runs into the ground.
pub fn transmittance(s: &AtmosphereSettings, r: f32, mu: f32) -> [f32; 3] {
    if ray_hits_ground(r, mu, s.bottom_radius) {
        return [0.0; 3];
    }
    optical_depth(s, r, mu).map(|d| exp(-d))
}

/// How much of the sun's disc is above the horizon seen from radius `r`, the
/// sun's center at cosine `mu` to the local up: 1 with the whole disc up, 0
/// once it has set, eased across the disc's own width between.
pub fn sun_visible_fraction(s: &AtmosphereSettings, r: f32, mu: f32) -> f32 {
    let sin_h = s.bottom_radius / r.max(s.bottom_radius);
    let horizon = -sqrt((1.0 - sin_h * sin_h).max(0.0));
    let radius = crate::math::acos(s.sun_cos_radius).max(1e-4);
    let above = asin(mu.clamp(-1.0, 1.0)) - asin(horizon);
    let x = ((above + radius) / (2.0 * radius)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// The sunlight reaching radius `r` from a sun at cosine `mu`, as a factor of
/// the light above the air: the transmittance of the path, scaled by the
/// part of the disc still above the horizon.
pub(super) fn sun_transmittance(s: &AtmosphereSettings, r: f32, mu: f32) -> [f32; 3] {
    let visible = sun_visible_fraction(s, r, mu);
    if visible <= 0.0 {
        return [0.0; 3];
    }
    // A sun straddling the horizon is lit through the grazing path.
    let sin_h = s.bottom_radius / r.max(s.bottom_radius);
    let horizon = -sqrt((1.0 - sin_h * sin_h).max(0.0));
    let mu = mu.max(horizon + 1e-4);
    transmittance(s, r, mu).map(|t| t * visible)
}

/// [`sun_transmittance`] for a camera at radius `r` over `view`'s ground
/// whose sky is built over `table`'s: the disc sets at the true horizon, and
/// its light crosses the table's air from as high above that ground, at the
/// angle the horizon warp moves the sun to.
pub(super) fn sun_transmittance_seen(
    view: &AtmosphereSettings,
    table: &AtmosphereSettings,
    r: f32,
    mu: f32,
) -> [f32; 3] {
    let visible = sun_visible_fraction(view, r, mu);
    if visible <= 0.0 {
        return [0.0; 3];
    }
    let r_table = table.bottom_radius + (r - view.bottom_radius);
    let angle = table_view_angle(
        acos(mu.clamp(-1.0, 1.0)),
        zenith_horizon(view.bottom_radius, r),
        zenith_horizon(table.bottom_radius, r_table),
    );
    let sin_h = table.bottom_radius / r_table.max(table.bottom_radius);
    let horizon = -sqrt((1.0 - sin_h * sin_h).max(0.0));
    transmittance(table, r_table, cos(angle).max(horizon + 1e-4)).map(|t| t * visible)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Atmosphere;

    fn earth() -> AtmosphereSettings {
        AtmosphereSettings::resolve(&Atmosphere::default(), None)
    }

    // Straight up from the ground the integral has a closed form: each layer
    // contributes its coefficient times its scale height (the top of the air
    // is far enough up to call infinite), and the ozone tent its area. The
    // midpoint rule's 2.5 km steps undersample the thin haze layer by a few
    // percent, as the table does.
    #[test]
    fn the_zenith_depth_matches_the_closed_form() {
        let s = earth();
        let d = optical_depth(&s, s.bottom_radius, 1.0);
        for (i, depth) in d.into_iter().enumerate() {
            let expected = s.rayleigh_scattering[i] * s.rayleigh_scale_height
                + s.mie_extinction[i] * s.mie_scale_height
                + s.ozone_absorption[i] * s.ozone_half_width;
            assert!(
                (depth - expected).abs() < expected * 0.03,
                "{i}: {depth} vs {expected}"
            );
        }
    }

    // Noon is nearly white, and a low sun is dimmer and redder: blue loses
    // the most on the long grazing path.
    #[test]
    fn a_low_sun_is_dimmer_and_redder() {
        let s = earth();
        let r = s.bottom_radius + 0.001;
        let noon = transmittance(&s, r, 1.0);
        let low = transmittance(&s, r, 0.05);
        assert!(noon[0] > 0.9 && noon[2] > 0.7, "{noon:?}");
        assert!(low[0] > low[1] && low[1] > low[2], "{low:?}");
        assert!((0..3).all(|i| low[i] < noon[i]));
        assert!(low[2] < 0.25 * low[0], "{low:?}");
    }

    // A ray into the ground carries nothing; one from above the air loses
    // nothing.
    #[test]
    fn the_ground_blocks_and_space_is_clear() {
        let s = earth();
        assert_eq!(transmittance(&s, s.bottom_radius + 1.0, -0.5), [0.0; 3]);
        let space = transmittance(&s, s.top_radius + 10.0, 0.3);
        assert_eq!(space, [1.0; 3]);
    }

    // The disc sets over its own width: whole above the horizon, half on it,
    // none below, and the light follows.
    #[test]
    fn the_sun_fades_out_across_the_horizon() {
        let s = earth();
        let r = s.bottom_radius;
        let radius = crate::math::acos(s.sun_cos_radius);
        assert_eq!(
            sun_visible_fraction(&s, r, crate::math::sin(radius * 2.0)),
            1.0
        );
        assert!((sun_visible_fraction(&s, r, 0.0) - 0.5).abs() < 1e-3);
        assert_eq!(
            sun_visible_fraction(&s, r, -crate::math::sin(radius * 2.0)),
            0.0
        );
        assert_eq!(sun_transmittance(&s, r, -0.2), [0.0; 3]);
        let setting = sun_transmittance(&s, r, 0.0);
        assert!(setting[0] > 0.0 && setting[0] < 0.5, "{setting:?}");
    }

    // From high up the horizon dips, so the sun stays up longer.
    #[test]
    fn altitude_lowers_the_horizon() {
        let s = earth();
        let mu = -0.05;
        assert_eq!(sun_visible_fraction(&s, s.bottom_radius, mu), 0.0);
        assert_eq!(sun_visible_fraction(&s, s.bottom_radius + 20.0, mu), 1.0);
    }

    // Over a 50 km planet the low sun stays pale; seen through an Earth-sized
    // sky it is red, and it still sets at the true horizon.
    #[test]
    fn an_apparent_ground_reddens_a_small_planets_sun() {
        let small = AtmosphereSettings::resolve(
            &Atmosphere::default(),
            Some(([0.0, -50_000.0, 0.0], 50_000.0)),
        );
        let table = small.with_ground_radius(6360.0);
        let r = small.bottom_radius + 0.002;
        let mu = 3.0_f32.to_radians().sin();
        let own = sun_transmittance(&small, r, mu);
        let seen = sun_transmittance_seen(&small, &table, r, mu);
        let ratio = |t: [f32; 3]| t[0] / t[2].max(1e-6);
        assert!(ratio(own) < 2.0, "{own:?}");
        assert!(ratio(seen) > 3.0 * ratio(own), "{seen:?} vs {own:?}");
        // Below the small planet's horizon the sun is gone, as before.
        assert_eq!(sun_transmittance_seen(&small, &table, r, -0.1), [0.0; 3]);
        // A sun overhead is nearly white either way.
        let noon = sun_transmittance_seen(&small, &table, r, 1.0);
        assert!(noon.iter().all(|&t| t > 0.7), "{noon:?}");
    }

    #[test]
    fn ray_exit_reaches_the_sphere() {
        let r = 6360.0;
        let top = 6460.0;
        assert!((ray_exit(r, 1.0, top) - 100.0).abs() < 1e-2);
        let t = ray_exit(r, 0.0, top);
        assert!((t - sqrt(top * top - r * r)).abs() < 1e-1);
        assert!(ray_hits_ground(r + 1.0, -0.9, r));
        assert!(!ray_hits_ground(r + 1.0, 0.1, r));
    }
}
