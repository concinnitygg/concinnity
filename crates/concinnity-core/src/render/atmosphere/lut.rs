//! The lookup tables' sizes and parameterizations: how a texel maps to the
//! height and angles it holds, and back. Each function mirrors one in
//! atmosphere_common.hlsl.

use crate::math::{acos, cos, sqrt};

/// Transmittance table width (view angle) in texels.
pub const TRANSMITTANCE_WIDTH: u32 = 256;
/// Transmittance table height (altitude) in texels.
pub const TRANSMITTANCE_HEIGHT: u32 = 64;
/// Multiple-scattering table edge (sun angle by altitude) in texels.
pub const MULTISCATTER_SIZE: u32 = 32;
/// Sky-view table width (azimuth from the sun) in texels.
pub const SKY_VIEW_WIDTH: u32 = 192;
/// Sky-view table height (view elevation) in texels.
pub const SKY_VIEW_HEIGHT: u32 = 108;
/// Aerial-perspective volume edge in froxels, across the screen and in depth.
pub const AERIAL_SIZE: u32 = 32;
/// How far the aerial-perspective volume reaches, in kilometers. Surfaces past
/// it take its last slice.
pub const AERIAL_MAX_DISTANCE_KM: f32 = 32.0;
/// Edge of each face of the irradiance cube the atmosphere writes, in texels.
pub const IRRADIANCE_FACE: u32 = 16;
/// Edge of each face of the sky cube the reflections are prefiltered from,
/// in texels.
pub const SKY_CUBE_FACE: u32 = 512;

/// Midpoint samples along each transmittance path. `TRANSMITTANCE_STEPS` in
/// atmosphere_common.hlsl.
pub(super) const TRANSMITTANCE_STEPS: u32 = 40;

// A [0, 1] table coordinate to the texel-center range, so 0 and 1 land on the
// first and last texel centers and the ends are never filtered against the
// clamp.
fn unit_to_texels(unit: f32, size: u32) -> f32 {
    let n = size as f32;
    unit * (n - 1.0) / n + 0.5 / n
}

// The inverse of `unit_to_texels`.
fn texels_to_unit(uv: f32, size: u32) -> f32 {
    let n = size as f32;
    (uv - 0.5 / n) * n / (n - 1.0)
}

/// The transmittance table coordinate holding radius `r` and view cosine
/// `mu` (Bruneton's mapping: the distance to the top of the air across its
/// range at that height, and the height across the horizon's reach).
pub fn transmittance_uv(bottom: f32, top: f32, r: f32, mu: f32) -> [f32; 2] {
    let h = sqrt((top * top - bottom * bottom).max(0.0));
    let rho = sqrt((r * r - bottom * bottom).max(0.0));
    let disc = r * r * (mu * mu - 1.0) + top * top;
    let d = (-r * mu + sqrt(disc.max(0.0))).max(0.0);
    let d_min = top - r;
    let d_max = rho + h;
    let x_mu = (d - d_min) / (d_max - d_min).max(1e-6);
    let x_r = rho / h.max(1e-6);
    [
        unit_to_texels(x_mu, TRANSMITTANCE_WIDTH),
        unit_to_texels(x_r, TRANSMITTANCE_HEIGHT),
    ]
}

/// The radius and view cosine a transmittance table coordinate holds.
pub fn transmittance_r_mu(bottom: f32, top: f32, uv: [f32; 2]) -> (f32, f32) {
    let x_mu = texels_to_unit(uv[0], TRANSMITTANCE_WIDTH);
    let x_r = texels_to_unit(uv[1], TRANSMITTANCE_HEIGHT);
    let h = sqrt((top * top - bottom * bottom).max(0.0));
    let rho = h * x_r;
    let r = sqrt(rho * rho + bottom * bottom);
    let d_min = top - r;
    let d_max = rho + h;
    let d = d_min + x_mu * (d_max - d_min);
    let mu = if d == 0.0 {
        1.0
    } else {
        ((h * h - rho * rho - d * d) / (2.0 * r * d)).clamp(-1.0, 1.0)
    };
    (r, mu)
}

// Angle between the zenith and the horizon seen from radius `r`, and the
// horizon's angular depth below the level, in radians.
pub(super) fn horizon_angles(bottom: f32, r: f32) -> (f32, f32) {
    let v_horizon = sqrt((r * r - bottom * bottom).max(0.0));
    let beta = acos((v_horizon / r).clamp(-1.0, 1.0));
    (core::f32::consts::PI - beta, beta)
}

/// The sky-view table coordinate of a view ray from radius `r` whose cosine
/// to the local up is `view_cos` and whose azimuth from the sun's has cosine
/// `azimuth_cos`. The upper half of the table holds the rays above the
/// horizon, the lower half those into the ground, each squeezed toward the
/// horizon where the sky changes fastest; the azimuth is squeezed toward the
/// sun, where the haze's glow is.
pub fn sky_view_uv(bottom: f32, r: f32, view_cos: f32, azimuth_cos: f32) -> [f32; 2] {
    let (zenith_horizon, beta) = horizon_angles(bottom, r);
    let view_angle = acos(view_cos.clamp(-1.0, 1.0));
    let v = if view_angle < zenith_horizon {
        let c = 1.0 - sqrt((1.0 - view_angle / zenith_horizon).max(0.0));
        0.5 * c
    } else {
        let c = sqrt(((view_angle - zenith_horizon) / beta.max(1e-6)).max(0.0));
        0.5 + 0.5 * c
    };
    let azimuth = acos(azimuth_cos.clamp(-1.0, 1.0)) / core::f32::consts::PI;
    [
        unit_to_texels(sqrt(azimuth.max(0.0)), SKY_VIEW_WIDTH),
        unit_to_texels(v, SKY_VIEW_HEIGHT),
    ]
}

/// The view cosine and azimuth cosine a sky-view table coordinate holds, the
/// inverse of [`sky_view_uv`].
pub fn sky_view_direction(bottom: f32, r: f32, uv: [f32; 2]) -> (f32, f32) {
    let (zenith_horizon, beta) = horizon_angles(bottom, r);
    let u = texels_to_unit(uv[0], SKY_VIEW_WIDTH);
    let v = texels_to_unit(uv[1], SKY_VIEW_HEIGHT);
    let view_angle = if v < 0.5 {
        let c = 1.0 - 2.0 * v;
        zenith_horizon * (1.0 - c * c)
    } else {
        let c = 2.0 * v - 1.0;
        zenith_horizon + beta * c * c
    };
    let azimuth = u * u * core::f32::consts::PI;
    (cos(view_angle), cos(azimuth))
}

/// Distance in kilometers of the center of aerial-perspective slice `slice`:
/// slices spread quadratically, so the near ones are thin.
pub fn aerial_slice_distance(slice: u32) -> f32 {
    let w = (slice as f32 + 0.5) / AERIAL_SIZE as f32;
    AERIAL_MAX_DISTANCE_KM * w * w
}

/// The aerial volume's depth coordinate for a surface `distance` kilometers
/// away, the inverse of [`aerial_slice_distance`].
pub fn aerial_slice_of(distance: f32) -> f32 {
    sqrt((distance / AERIAL_MAX_DISTANCE_KM).max(0.0)).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOTTOM: f32 = 6360.0;
    const TOP: f32 = 6460.0;

    #[test]
    fn the_constants_match_the_shader() {
        use crate::render::shader_consts::{float, uint};
        let src = crate::render::shaders::ATMOSPHERE_TYPES;
        let pairs = [
            ("TRANSMITTANCE_STEPS", TRANSMITTANCE_STEPS),
            ("TRANSMITTANCE_WIDTH", TRANSMITTANCE_WIDTH),
            ("TRANSMITTANCE_HEIGHT", TRANSMITTANCE_HEIGHT),
            ("MULTISCATTER_SIZE", MULTISCATTER_SIZE),
            ("SKY_VIEW_WIDTH", SKY_VIEW_WIDTH),
            ("SKY_VIEW_HEIGHT", SKY_VIEW_HEIGHT),
            ("AERIAL_SIZE", AERIAL_SIZE),
            ("IRRADIANCE_FACE", IRRADIANCE_FACE),
            ("SKY_CUBE_FACE", SKY_CUBE_FACE),
        ];
        for (name, value) in pairs {
            assert_eq!(uint(src, name), value as usize, "{name}");
        }
        assert_eq!(float(src, "AERIAL_MAX_DISTANCE_KM"), AERIAL_MAX_DISTANCE_KM);
    }

    // Every texel center maps to a height and angle that map back to it.
    #[test]
    fn the_transmittance_mapping_round_trips() {
        for y in 0..TRANSMITTANCE_HEIGHT {
            for x in (0..TRANSMITTANCE_WIDTH).step_by(5) {
                let uv = [
                    (x as f32 + 0.5) / TRANSMITTANCE_WIDTH as f32,
                    (y as f32 + 0.5) / TRANSMITTANCE_HEIGHT as f32,
                ];
                let (r, mu) = transmittance_r_mu(BOTTOM, TOP, uv);
                assert!((BOTTOM..=TOP + 1e-2).contains(&r), "{r}");
                let back = transmittance_uv(BOTTOM, TOP, r, mu);
                assert!(
                    (back[0] - uv[0]).abs() < 2e-3 && (back[1] - uv[1]).abs() < 2e-3,
                    "{uv:?} -> ({r}, {mu}) -> {back:?}"
                );
            }
        }
    }

    // The table's ends are the ground and the top of the air, and straight up
    // and the horizon at the ground.
    #[test]
    fn the_transmittance_table_spans_the_shell() {
        let first = [
            0.5 / TRANSMITTANCE_WIDTH as f32,
            0.5 / TRANSMITTANCE_HEIGHT as f32,
        ];
        let (r, mu) = transmittance_r_mu(BOTTOM, TOP, first);
        assert!(
            (r - BOTTOM).abs() < 1e-2 && (mu - 1.0).abs() < 1e-4,
            "{r} {mu}"
        );
        let last = [1.0 - first[0], first[1]];
        let (_, mu) = transmittance_r_mu(BOTTOM, TOP, last);
        assert!(mu.abs() < 1e-3, "the ground's horizon: {mu}");
        let top = [first[0], 1.0 - first[1]];
        let (r, _) = transmittance_r_mu(BOTTOM, TOP, top);
        assert!((r - TOP).abs() < 1e-1, "{r}");
    }

    #[test]
    fn the_sky_view_mapping_round_trips() {
        for r in [BOTTOM + 0.001, BOTTOM + 2.0, BOTTOM + 40.0] {
            for y in 0..SKY_VIEW_HEIGHT {
                for x in (0..SKY_VIEW_WIDTH).step_by(7) {
                    let uv = [
                        (x as f32 + 0.5) / SKY_VIEW_WIDTH as f32,
                        (y as f32 + 0.5) / SKY_VIEW_HEIGHT as f32,
                    ];
                    let (view_cos, azimuth_cos) = sky_view_direction(BOTTOM, r, uv);
                    let back = sky_view_uv(BOTTOM, r, view_cos, azimuth_cos);
                    assert!(
                        (back[0] - uv[0]).abs() < 3e-3 && (back[1] - uv[1]).abs() < 3e-3,
                        "r {r}: {uv:?} -> {back:?}"
                    );
                }
            }
        }
    }

    // The table's middle row is the horizon: above it the rays clear the
    // ground, below it they strike it.
    #[test]
    fn the_sky_view_horizon_splits_the_table() {
        let r = BOTTOM + 1.0;
        let (zenith_horizon, _) = horizon_angles(BOTTOM, r);
        let up = sky_view_uv(BOTTOM, r, cos(zenith_horizon - 0.01), 1.0);
        let down = sky_view_uv(BOTTOM, r, cos(zenith_horizon + 0.01), 1.0);
        assert!(up[1] < 0.5 && down[1] > 0.5, "{up:?} {down:?}");
        let zenith = sky_view_uv(BOTTOM, r, 1.0, 1.0);
        assert!((zenith[1] - unit_to_texels(0.0, SKY_VIEW_HEIGHT)).abs() < 1e-5);
        let toward_sun = sky_view_uv(BOTTOM, r, 0.3, 1.0);
        let away = sky_view_uv(BOTTOM, r, 0.3, -1.0);
        assert!(toward_sun[0] < 0.01 && away[0] > 0.99);
    }

    #[test]
    fn aerial_slices_round_trip_and_crowd_the_camera() {
        for s in 0..AERIAL_SIZE {
            let d = aerial_slice_distance(s);
            let w = aerial_slice_of(d);
            assert!((w - (s as f32 + 0.5) / AERIAL_SIZE as f32).abs() < 1e-5);
        }
        assert!(aerial_slice_distance(0) < 0.01);
        assert!(aerial_slice_distance(AERIAL_SIZE - 1) > 0.9 * AERIAL_MAX_DISTANCE_KM);
        assert_eq!(aerial_slice_of(1000.0), 1.0);
    }
}
