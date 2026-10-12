//! Procedural stars: at most one per cell of a grid laid over the faces of a
//! cube around the celestial sphere, kept with a probability that makes them
//! uniform over the sky, with magnitudes drawn so faint stars far outnumber
//! bright ones, and colored by their temperature. Mirrors the star half of
//! night_sky.hlsl.
//!
//! A star is drawn as a small Gaussian integrated over each pixel's square,
//! so the light it puts into the image is the same wherever it falls within
//! a pixel: a sub-pixel jitter moves the light between pixels without
//! changing its total, and a temporal filter sees a steady point.

use super::hash::{night_hash, night_unit};
use crate::math::{exp, ln, powf, sqrt};

/// Cells along each edge of a cube face.
pub const STAR_GRID: u32 = 128;
/// Stars brighter than [`NAKED_EYE_LIMIT`] over the whole sky.
pub const NAKED_EYE_STARS: f32 = 9000.0;
/// The magnitude [`NAKED_EYE_STARS`] counts down to.
pub const NAKED_EYE_LIMIT: f32 = 6.5;
/// How fast the count grows with magnitude: each magnitude fainter holds
/// `10^COUNT_SLOPE` times as many stars as the one before.
pub const COUNT_SLOPE: f32 = 0.5;
/// The brightest a star is drawn.
pub const BRIGHTEST_MAGNITUDE: f32 = -1.5;
/// The faintest-star magnitudes a sky may ask for. The upper end keeps a
/// face-center cell's chance of holding a star at or under 1.
pub const LIMIT_RANGE: [f32; 2] = [2.0, 8.0];
/// Width of a star's Gaussian, in pixels.
pub const STAR_SIGMA_PX: f32 = 0.6;
/// How far from the edges of a cube face its border cells keep their stars,
/// as a fraction of a cell: a star's footprint never crosses into the next
/// face, whose cells the lookup does not visit.
pub const FACE_MARGIN: f32 = 0.25;
/// How much of a star's blackbody tint is kept.
pub const STAR_SATURATION: f32 = 0.7;

/// Mean stars per steradian down to magnitude `limit`.
pub fn stars_per_steradian(limit: f32) -> f32 {
    let limit = limit.clamp(LIMIT_RANGE[0], LIMIT_RANGE[1]);
    NAKED_EYE_STARS * powf(10.0, COUNT_SLOPE * (limit - NAKED_EYE_LIMIT))
        / (4.0 * core::f32::consts::PI)
}

/// The cube face `dir` points through and its coordinates on that face, each
/// in `[-1, 1]`. Face `2a + s` lies along axis `a`, positive when `s` is 0;
/// `u` runs along the next axis and `v` the one after.
pub fn cube_face(dir: [f32; 3]) -> (u32, f32, f32) {
    let abs = dir.map(f32::abs);
    let axis = if abs[0] >= abs[1] && abs[0] >= abs[2] {
        0
    } else if abs[1] >= abs[2] {
        1
    } else {
        2
    };
    let major = abs[axis].max(1e-20);
    let face = axis as u32 * 2 + u32::from(dir[axis] < 0.0);
    (
        face,
        dir[(axis + 1) % 3] / major,
        dir[(axis + 2) % 3] / major,
    )
}

/// The direction (not normalized) through `(u, v)` on cube face `face`.
pub fn face_direction(face: u32, u: f32, v: f32) -> [f32; 3] {
    let axis = (face / 2) as usize;
    let mut d = [0.0; 3];
    d[axis] = if face.is_multiple_of(2) { 1.0 } else { -1.0 };
    d[(axis + 1) % 3] = u;
    d[(axis + 2) % 3] = v;
    d
}

/// The solid angle of cell `(i, j)` of a face, taken at its center.
pub fn cell_solid_angle(i: u32, j: u32) -> f32 {
    let w = 2.0 / STAR_GRID as f32;
    let u = -1.0 + (i as f32 + 0.5) * w;
    let v = -1.0 + (j as f32 + 0.5) * w;
    let k = 1.0 + u * u + v * v;
    w * w / (k * sqrt(k))
}

/// One star.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Star {
    /// Unit direction in the celestial sphere's frame.
    pub direction: [f32; 3],
    /// Apparent magnitude: 5 magnitudes fainter is 100 times dimmer.
    pub magnitude: f32,
    /// Surface temperature in kelvin, which sets its color.
    pub temperature: f32,
}

/// The star in cell `(i, j)` of face `face`, if it holds one. `seed` is the
/// sky's 24-bit seed lane ([`super::seed_lane`]) and `density` its stars per
/// steradian ([`stars_per_steradian`]).
pub fn star_in_cell(
    seed: u32,
    density: f32,
    limit: f32,
    face: u32,
    i: u32,
    j: u32,
) -> Option<Star> {
    let n = STAR_GRID;
    let h0 = night_hash(((face * n + j) * n + i).wrapping_add(night_hash(seed)));
    if night_unit(h0) >= density * cell_solid_angle(i, j) {
        return None;
    }
    let h1 = night_hash(h0);
    let h2 = night_hash(h1);
    let h3 = night_hash(h2);
    let h4 = night_hash(h3);
    let border = i == 0 || j == 0 || i == n - 1 || j == n - 1;
    let margin = if border { FACE_MARGIN } else { 0.0 };
    let place = |h: u32| margin + (1.0 - 2.0 * margin) * night_unit(h);
    let w = 2.0 / n as f32;
    let u = -1.0 + (i as f32 + place(h1)) * w;
    let v = -1.0 + (j as f32 + place(h2)) * w;
    let d = face_direction(face, u, v);
    let len = sqrt(1.0 + u * u + v * v);
    Some(Star {
        direction: d.map(|c| c / len),
        magnitude: magnitude_of(night_unit(h3), limit),
        temperature: temperature_of(night_unit(h4)),
    })
}

/// The magnitude at quantile `u` of the stars down to `limit`: the count
/// brighter than `m` grows as `10^(COUNT_SLOPE m)`, so most stars sit near
/// the limit and a few are bright.
pub fn magnitude_of(u: f32, limit: f32) -> f32 {
    let limit = limit.clamp(LIMIT_RANGE[0], LIMIT_RANGE[1]);
    let log10 = ln(u.max(1e-7)) / core::f32::consts::LN_10;
    (limit + log10 / COUNT_SLOPE).max(BRIGHTEST_MAGNITUDE)
}

/// The temperature at quantile `u`: 3000 K to 30000 K, most stars cool and
/// sun-like and a few hot and blue.
pub fn temperature_of(u: f32) -> f32 {
    3000.0 * powf(10.0, powf(u, 1.6))
}

/// A blackbody's color at `temperature` kelvin, sampled at a red, green and
/// blue wavelength, with [`STAR_SATURATION`] of its tint kept and scaled to
/// unit luminance.
pub fn blackbody_color(temperature: f32) -> [f32; 3] {
    const C2: f32 = 14_388.0;
    const WAVELENGTHS_UM: [f32; 3] = [0.610, 0.550, 0.465];
    let planck = WAVELENGTHS_UM.map(|l| {
        let l5 = l * l * l * l * l;
        1.0 / (l5 * (exp(C2 / (l * temperature)) - 1.0))
    });
    let lum = luminance(planck).max(1e-30);
    planck.map(|c| 1.0 + STAR_SATURATION * (c / lum - 1.0))
}

/// Rec. 709 luminance.
pub fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// The share of a star's light a pixel takes along one axis, the star
/// `offset_px` pixels from the pixel's center: its Gaussian integrated over
/// the pixel's width.
pub fn footprint(offset_px: f32) -> f32 {
    let k = 1.0 / (STAR_SIGMA_PX * core::f32::consts::SQRT_2);
    0.5 * (erf((offset_px + 0.5) * k) - erf((offset_px - 0.5) * k))
}

// An odd, saturating fit to the error function (within 4e-4); the footprint
// telescopes, so its total stays exact whatever the fit's error.
fn erf(x: f32) -> f32 {
    let x = (x * (core::f32::consts::FRAC_2_SQRT_PI + 0.100_91 * x * x)).clamp(-15.0, 15.0);
    1.0 - 2.0 / (exp(2.0 * x) + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3::dot;

    fn all_stars(seed: u32, limit: f32) -> alloc::vec::Vec<Star> {
        let density = stars_per_steradian(limit);
        let mut out = alloc::vec::Vec::new();
        for face in 0..6 {
            for j in 0..STAR_GRID {
                for i in 0..STAR_GRID {
                    out.extend(star_in_cell(seed, density, limit, face, i, j));
                }
            }
        }
        out
    }

    // The naked-eye sky holds about 9000 stars, and three times as many a
    // magnitude fainter.
    #[test]
    fn the_sky_holds_the_counts_the_magnitudes_ask_for() {
        let naked = all_stars(11, 6.5).len() as f32;
        assert!((naked / 9000.0 - 1.0).abs() < 0.05, "{naked}");
        let deep = all_stars(11, 7.5).len() as f32;
        let ratio = deep / naked;
        assert!((ratio - powf(10.0, 0.5)).abs() < 0.2, "{ratio}");
    }

    // Faint stars far outnumber bright ones: each magnitude bin holds about
    // 10^0.5 times the one brighter than it, and only a handful are brighter
    // than magnitude 1.
    #[test]
    fn magnitudes_follow_the_count_slope() {
        let stars = all_stars(3, 6.5);
        let bin = |lo: f32| {
            stars
                .iter()
                .filter(|s| s.magnitude >= lo && s.magnitude < lo + 1.0)
                .count() as f32
        };
        for lo in [3.0, 4.0, 5.0] {
            let ratio = bin(lo) / bin(lo - 1.0);
            assert!((ratio - 3.16).abs() < 0.6, "bin {lo}: {ratio}");
        }
        let bright = stars.iter().filter(|s| s.magnitude < 1.0).count();
        assert!((5..60).contains(&bright), "{bright}");
        assert!(stars.iter().all(|s| s.magnitude >= BRIGHTEST_MAGNITUDE));
        assert!(stars.iter().all(|s| s.magnitude <= 6.5));
    }

    // Over the whole sky the stars are spread evenly, the cube's corners
    // neither crowded nor bare.
    #[test]
    fn stars_are_uniform_over_the_sphere() {
        let stars = all_stars(5, 7.0);
        let n = stars.len() as f32;
        for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.577, 0.577, 0.577]] {
            // A 30-degree cap around a face center and around a corner hold
            // the cap's share.
            let cap = stars
                .iter()
                .filter(|s| dot(s.direction, axis) > 0.866)
                .count() as f32;
            let share = (1.0 - 0.866) / 2.0;
            assert!((cap / (n * share) - 1.0).abs() < 0.12, "{axis:?}: {cap}");
        }
    }

    // A star lies in the cell it came from, the face lookup finds it there,
    // and the same seed gives the same sky.
    #[test]
    fn a_star_sits_in_its_own_cell() {
        let density = stars_per_steradian(8.0);
        let mut found = 0;
        for face in 0..6 {
            for (i, j) in [(0, 0), (5, 9), (64, 64), (127, 3)] {
                let Some(star) = star_in_cell(9, density, 8.0, face, i, j) else {
                    continue;
                };
                found += 1;
                let (f, u, v) = cube_face(star.direction);
                assert_eq!(f, face);
                let cell = |x: f32| ((x + 1.0) * 0.5 * STAR_GRID as f32) as u32;
                assert_eq!((cell(u), cell(v)), (i, j));
                assert_eq!(star_in_cell(9, density, 8.0, face, i, j), Some(star));
            }
        }
        assert!(found > 6, "{found}");
        assert_ne!(all_stars(1, 4.0), all_stars(2, 4.0));
    }

    #[test]
    fn hot_stars_are_blue_and_cool_ones_red() {
        let cool = blackbody_color(3000.0);
        let sun = blackbody_color(5800.0);
        let hot = blackbody_color(25_000.0);
        assert!(cool[0] > cool[2] * 1.5, "{cool:?}");
        assert!(hot[2] > hot[0] * 1.5, "{hot:?}");
        assert!(sun.iter().all(|c| (c - 1.0).abs() < 0.1), "{sun:?}");
        for c in [cool, sun, hot] {
            assert!((luminance(c) - 1.0).abs() < 1e-4);
        }
        assert!((temperature_of(0.0) - 3000.0).abs() < 1.0);
        assert!((temperature_of(1.0) - 30_000.0).abs() < 10.0);
    }

    // A star's light over the pixels around it adds up to all of it wherever
    // it sits within a pixel, so a sub-pixel jitter cannot make it shimmer.
    #[test]
    fn a_star_keeps_its_light_under_subpixel_jitter() {
        for k in 0..16 {
            let (ox, oy) = (k as f32 * 0.0625 - 0.5, 0.31 - k as f32 * 0.04);
            let mut total = 0.0;
            let mut peak: f32 = 0.0;
            for py in -3..=3 {
                for px in -3..=3 {
                    let w = footprint(px as f32 - ox) * footprint(py as f32 - oy);
                    total += w;
                    peak = peak.max(w);
                }
            }
            assert!((total - 1.0).abs() < 1e-4, "{k}: {total}");
            // Never collapsed into one pixel, never smeared thin.
            assert!(peak > 0.15 && peak < 0.5, "{k}: {peak}");
        }
        assert!(footprint(3.0) < 1e-3);
        assert!((erf(1.0) - 0.8427).abs() < 4e-4);
    }
}
