//! The lookup tables' sizes and parameterizations: how a texel maps to the
//! height and angles it holds, and back. Each function mirrors one in
//! atmosphere_common.hlsl.

use super::optics::ray_enter;
use crate::math::{acos, asin, cos, floor, sqrt};

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

/// The angles from the zenith the sky-view table spans from radius `r`:
/// where its rays start to meet the air (the zenith inside it, the air's
/// rim seen from above it), the horizon, and the horizon's depth below the
/// level. Mirrors `atmo_sky_angles` in atmosphere_common.hlsl.
pub fn sky_angles(bottom: f32, top: f32, r: f32) -> [f32; 3] {
    let (zenith_horizon, beta) = horizon_angles(bottom, r);
    let rim = if r > top {
        core::f32::consts::PI - asin((top / r).min(1.0))
    } else {
        0.0
    };
    [rim.min(zenith_horizon), zenith_horizon, beta]
}

/// The sky-view table coordinate of a view ray from radius `r` whose angle
/// from the local up is `view_angle` and whose azimuth from the sun's has
/// cosine `azimuth_cos`. The upper half of the table holds the rays from
/// the air's rim (the zenith inside the air) down to the horizon, the lower
/// half those into the ground, each squeezed toward the horizon where the
/// sky changes fastest; the azimuth is squeezed toward the sun, where the
/// haze's glow is. A ray above the rim reads the rim.
pub fn sky_view_uv(bottom: f32, top: f32, r: f32, view_angle: f32, azimuth_cos: f32) -> [f32; 2] {
    let [rim, zenith_horizon, beta] = sky_angles(bottom, top, r);
    let v = if view_angle < zenith_horizon {
        let x = ((view_angle - rim) / (zenith_horizon - rim).max(1e-6)).max(0.0);
        0.5 * (1.0 - sqrt((1.0 - x).max(0.0)))
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

/// The view angle and azimuth cosine a sky-view table coordinate holds, the
/// inverse of [`sky_view_uv`].
pub fn sky_view_direction(bottom: f32, top: f32, r: f32, uv: [f32; 2]) -> (f32, f32) {
    let [rim, zenith_horizon, beta] = sky_angles(bottom, top, r);
    let u = texels_to_unit(uv[0], SKY_VIEW_WIDTH);
    let v = texels_to_unit(uv[1], SKY_VIEW_HEIGHT);
    let view_angle = if v < 0.5 {
        let c = 1.0 - 2.0 * v;
        rim + (zenith_horizon - rim) * (1.0 - c * c)
    } else {
        let c = 2.0 * v - 1.0;
        zenith_horizon + beta * c * c
    };
    let azimuth = u * u * core::f32::consts::PI;
    (view_angle, cos(azimuth))
}

/// A cubic B-spline read of a table `size` texels long at coordinate `uv`,
/// as two linear taps: their coordinates and their weights. The spline bends
/// smoothly through the texel centers, where a linear read creases. Mirrors
/// `atmo_bspline_taps` in atmosphere_common.hlsl, one axis at a time.
pub fn bspline_taps(uv: f32, size: u32) -> ([f32; 2], [f32; 2]) {
    let n = size as f32;
    let x = uv * n - 0.5;
    let i = floor(x);
    let f = x - i;
    let (f2, f3) = (f * f, f * f * f);
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let h0 = (i - 0.5 + w1 / g0) / n;
    let h1 = (i + 1.5 + w3 / (1.0 - g0).max(1e-6)) / n;
    ([h0, h1], [g0, 1.0 - g0])
}

/// The sky-view table's row taps at row coordinate `v`, kept to the half of
/// the table, sky or ground, that `v` is in, so the horizon is never blurred
/// across. Mirrors `atmo_sky_view_sample` in atmosphere_common.hlsl.
pub fn sky_view_row_taps(v: f32) -> ([f32; 2], [f32; 2]) {
    let n = SKY_VIEW_HEIGHT as f32;
    let (lo, hi) = if v < 0.5 {
        (0.5 / n, 0.5 - 0.5 / n)
    } else {
        (0.5 + 0.5 / n, 1.0 - 0.5 / n)
    };
    let (h, g) = bspline_taps(v, SKY_VIEW_HEIGHT);
    (h.map(|y| y.clamp(lo, hi)), g)
}

/// Where the aerial volume follows a view ray from radius `r` whose cosine
/// to the local up is `mu`, as kilometers from the camera: where the ray
/// enters the air (0 from inside it) and how far past that its slices reach,
/// or `None` when it misses the air. The reach ends where the ray would meet
/// a sphere `depth` under the ground, or at its closest approach to the
/// center if it passes over that sphere, so a ray that strikes the ground
/// keeps its last slices about the ground whatever its length; and never
/// past [`AERIAL_MAX_DISTANCE_KM`]. Mirrors `atmo_aerial_span` in
/// atmosphere_common.hlsl.
pub fn aerial_span(bottom: f32, top: f32, depth: f32, r: f32, mu: f32) -> Option<[f32; 2]> {
    let enter = ray_enter(r, mu, top);
    if enter < 0.0 {
        return None;
    }
    let mut end = ray_enter(r, mu, (bottom - depth).max(0.0));
    if end < 0.0 {
        end = if -r * mu > enter {
            -r * mu
        } else {
            f32::INFINITY
        };
    }
    Some([enter, (end - enter).clamp(0.0, AERIAL_MAX_DISTANCE_KM)])
}

/// Distance in kilometers past the ray's entry into the air of the center
/// of aerial-perspective slice `slice`, for a ray whose slices reach
/// `reach`: slices spread quadratically, so the near ones are thin.
pub fn aerial_slice_distance(slice: u32, reach: f32) -> f32 {
    let w = (slice as f32 + 0.5) / AERIAL_SIZE as f32;
    reach * w * w
}

/// The aerial volume's depth coordinate for a surface `distance` kilometers
/// past the ray's entry, the inverse of [`aerial_slice_distance`].
pub fn aerial_slice_of(distance: f32, reach: f32) -> f32 {
    sqrt((distance / reach.max(1e-6)).max(0.0)).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::exp;

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
        let bottom = 50.0;
        let top = 50.0 * 6460.0 / 6360.0;
        let cases = [
            (BOTTOM, TOP, BOTTOM + 0.001),
            (BOTTOM, TOP, BOTTOM + 2.0),
            (BOTTOM, TOP, BOTTOM + 40.0),
            (bottom, top, bottom + 150.0),
            (bottom, top, top + 0.01),
        ];
        for (b, t, r) in cases {
            for y in 0..SKY_VIEW_HEIGHT {
                for x in (0..SKY_VIEW_WIDTH).step_by(7) {
                    let uv = [
                        (x as f32 + 0.5) / SKY_VIEW_WIDTH as f32,
                        (y as f32 + 0.5) / SKY_VIEW_HEIGHT as f32,
                    ];
                    let (view_angle, azimuth_cos) = sky_view_direction(b, t, r, uv);
                    let back = sky_view_uv(b, t, r, view_angle, azimuth_cos);
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
        let up = sky_view_uv(BOTTOM, TOP, r, zenith_horizon - 0.01, 1.0);
        let down = sky_view_uv(BOTTOM, TOP, r, zenith_horizon + 0.01, 1.0);
        assert!(up[1] < 0.5 && down[1] > 0.5, "{up:?} {down:?}");
        let zenith = sky_view_uv(BOTTOM, TOP, r, 0.0, 1.0);
        assert!((zenith[1] - unit_to_texels(0.0, SKY_VIEW_HEIGHT)).abs() < 1e-5);
        let toward_sun = sky_view_uv(BOTTOM, TOP, r, 1.2, 1.0);
        let away = sky_view_uv(BOTTOM, TOP, r, 1.2, -1.0);
        assert!(toward_sun[0] < 0.01 && away[0] > 0.99);
    }

    // From above the air the upper half spans only the rays that cross it:
    // the first row grazes the air's top and every ray above it reads that
    // row, so the limb gets the whole half.
    #[test]
    fn from_above_the_air_the_table_spans_the_limb() {
        let bottom = 50.0;
        let top = 50.786;
        let r = 200.0;
        let [rim, zenith_horizon, _] = sky_angles(bottom, top, r);
        let limb = (zenith_horizon - rim).to_degrees();
        assert!((limb - 0.23).abs() < 0.01, "{limb}");
        let first = unit_to_texels(0.0, SKY_VIEW_HEIGHT);
        assert_eq!(sky_view_uv(bottom, top, r, 0.0, 1.0)[1], first);
        assert_eq!(sky_view_uv(bottom, top, r, rim - 0.1, 1.0)[1], first);
        let (grazing, _) = sky_view_direction(bottom, top, r, [0.5, first]);
        assert!((grazing - rim).abs() < 1e-5);
        let mid = sky_view_uv(bottom, top, r, 0.5 * (rim + zenith_horizon), 1.0)[1];
        assert!(mid > first && mid < 0.5, "{mid}");
        // Inside the air the rim is the zenith; at the top they meet.
        assert_eq!(sky_angles(bottom, top, top - 0.01)[0], 0.0);
        let at_top = sky_angles(bottom, top, top + 1e-4)[0];
        assert!(
            (at_top - core::f32::consts::FRAC_PI_2).abs() < 0.01,
            "{at_top}"
        );
    }

    // A linear read of `table` at `uv`, as the sampler takes it, clamped.
    fn linear(table: &[f32], uv: f32) -> f32 {
        let x = (uv * table.len() as f32 - 0.5).clamp(0.0, (table.len() - 1) as f32);
        let i = (floor(x) as usize).min(table.len() - 2);
        let f = x - i as f32;
        table[i] * (1.0 - f) + table[i + 1] * f
    }

    fn spline(table: &[f32], uv: f32) -> f32 {
        let (h, g) = bspline_taps(uv, table.len() as u32);
        g[0] * linear(table, h[0]) + g[1] * linear(table, h[1])
    }

    // The two taps carry the whole weight, and a straight ramp reads back
    // exactly, as a linear read would.
    #[test]
    fn the_bspline_keeps_a_ramp() {
        let ramp: [f32; 32] = core::array::from_fn(|i| 3.0 + 0.5 * i as f32);
        for k in 0..200 {
            let uv = (4.0 + k as f32 * 0.12) / 32.0;
            let (h, g) = bspline_taps(uv, 32);
            assert!((g[0] + g[1] - 1.0).abs() < 1e-6);
            assert!(g[0] > 0.0 && g[1] >= 0.0 && h[0] <= h[1]);
            let expected = linear(&ramp, uv);
            assert!((spline(&ramp, uv) - expected).abs() < 1e-4, "{uv}");
        }
    }

    // Over a curved table the linear read's slope jumps at every texel
    // center; the spline's does not.
    #[test]
    fn the_bspline_has_no_crease_at_the_texel_centers() {
        let curve: [f32; 32] = core::array::from_fn(|i| exp(i as f32 * 0.3));
        let e = 1e-3 / 32.0;
        for c in 6..26 {
            let uv = (c as f32 + 0.5) / 32.0;
            let kink = |read: &dyn Fn(f32) -> f32| {
                let left = (read(uv) - read(uv - e)) / e;
                let right = (read(uv + e) - read(uv)) / e;
                (right - left).abs() / right.abs()
            };
            assert!(kink(&|u| linear(&curve, u)) > 0.1, "{c}");
            assert!(kink(&|u| spline(&curve, u)) < 0.01, "{c}");
        }
    }

    // The row taps never cross the horizon between the table's halves.
    #[test]
    fn the_sky_view_row_taps_keep_to_their_half() {
        let n = SKY_VIEW_HEIGHT as f32;
        for k in 0..=1000 {
            let v = k as f32 / 1000.0;
            let (h, _) = sky_view_row_taps(v);
            for y in h {
                assert!((y < 0.5) == (v < 0.5), "{v}: {y}");
                assert!((0.5 / n..=1.0 - 0.5 / n).contains(&y), "{v}: {y}");
            }
        }
    }

    #[test]
    fn aerial_slices_round_trip_and_crowd_the_camera() {
        for reach in [AERIAL_MAX_DISTANCE_KM, 0.8] {
            for s in 0..AERIAL_SIZE {
                let d = aerial_slice_distance(s, reach);
                let w = aerial_slice_of(d, reach);
                assert!((w - (s as f32 + 0.5) / AERIAL_SIZE as f32).abs() < 1e-5);
            }
            assert!(aerial_slice_distance(0, reach) < 0.01);
            assert!(aerial_slice_distance(AERIAL_SIZE - 1, reach) > 0.9 * reach);
            assert_eq!(aerial_slice_of(1000.0, reach), 1.0);
        }
    }

    // Near the ground of an Earth-sized planet the slices start at the
    // camera, and every ray but the steep ones keeps the full reach: a ray
    // ends no nearer than a scale height under the ground.
    #[test]
    fn near_the_ground_the_aerial_reach_is_the_full_reach() {
        for h in [0.002, 0.1, 1.0] {
            for i in 0..=40 {
                let mu = -1.0 + i as f32 / 20.0;
                let [enter, reach] = aerial_span(BOTTOM, TOP, 8.0, BOTTOM + h, mu).expect("inside");
                assert_eq!(enter, 0.0);
                assert!(reach >= 8.0, "{h} {mu}: {reach}");
                if mu >= -0.2 {
                    assert_eq!(reach, AERIAL_MAX_DISTANCE_KM, "{h} {mu}");
                }
            }
        }
    }

    // From above a thin air a ray starts where it enters the air and ends
    // just under the ground it strikes; past the limb it ends at its closest
    // approach, so the reach runs on without a jump across the limb, and a
    // ray over the air has none.
    #[test]
    fn from_above_the_air_the_reach_ends_at_the_ground() {
        let (bottom, top, depth, r) = (50.0, 50.786, 0.063, 200.0);
        let down = aerial_span(bottom, top, depth, r, -1.0).expect("down");
        assert!((down[0] - (r - top)).abs() < 1e-3, "{down:?}");
        assert!((down[1] - (top - bottom + depth)).abs() < 1e-3, "{down:?}");
        assert_eq!(aerial_span(bottom, top, depth, r, 0.0), None);
        let floor = bottom - depth;
        let tangent = -crate::math::sqrt(1.0 - (floor / r) * (floor / r));
        let reach = |mu: f32| aerial_span(bottom, top, depth, r, mu).expect("in the air")[1];
        // A chord's length near tangency goes as the root of the angle, so the
        // gap closes as the rays close in on the tangent from either side.
        let gap = |d: f32| (reach(tangent - d) - reach(tangent + d)).abs();
        assert!(gap(1e-4) < 5.0, "{}", gap(1e-4));
        assert!(gap(1e-6) < 0.6, "{}", gap(1e-6));
        assert!(gap(1e-6) < gap(1e-4));
    }
}
