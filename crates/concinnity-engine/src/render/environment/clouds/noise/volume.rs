//! The CPU mirror of the noise volume, line for line, so its tiling and
//! ranges are tested here. Four channels, each tiling across the volume:
//!
//! - r: the base shape, Perlin noise eroded by Worley cells (puffy, billowed
//!   blobs), `SHAPE_CELLS` cells across at its coarsest octave;
//! - g: the detail, Worley octaves the edges are worn by, `DETAIL_CELLS`
//!   across at its coarsest;
//! - b: the weather's smooth field, Perlin octaves `WEATHER_CELLS` across;
//! - a: the weather's cells, one Worley octave as many across.

use super::super::settings::{DETAIL_CELLS, SHAPE_CELLS, WEATHER_CELLS};
use super::{NOISE_SIZE, hash3f};
use concinnity_core::math::{SaturatingFrom, floor, sqrt};

// Lattice cell `c` wrapped into a tile `period` cells across.
fn wrap(c: i32, period: i32) -> u32 {
    u32::saturating_from(c.rem_euclid(period))
}

fn wrap3(c: [i32; 3], period: i32) -> [u32; 3] {
    c.map(|v| wrap(v, period))
}

fn cell_of(p: f32) -> i32 {
    // The volume's coordinates stay within a few periods of the tile.
    #[expect(clippy::cast_possible_truncation, reason = "floored and small")]
    let c = floor(p) as i32;
    c
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// Gradient noise at `p` (in cells) over a tile `period` cells across, in about
// [-1, 1]. `cloud_perlin`.
fn perlin(p: [f32; 3], period: i32) -> f32 {
    let cell = p.map(cell_of);
    let f: [f32; 3] = core::array::from_fn(|i| p[i] - cell[i] as f32);
    let corner = |dx: i32, dy: i32, dz: i32| {
        let h = hash3f(wrap3([cell[0] + dx, cell[1] + dy, cell[2] + dz], period));
        let g = h.map(|v| v * 2.0 - 1.0);
        dot3(g, [f[0] - dx as f32, f[1] - dy as f32, f[2] - dz as f32])
    };
    let u = f.map(fade);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u[0]);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u[0]);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u[0]);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u[0]);
    lerp(lerp(x00, x10, u[1]), lerp(x01, x11, u[1]), u[2])
}

// One minus the distance from `p` (in cells) to the nearest of one point
// scattered per cell, over a tile `period` cells across, in [0, 1]: bright
// cell centers, dark edges. `cloud_worley`.
fn worley(p: [f32; 3], period: i32) -> f32 {
    let cell = p.map(cell_of);
    let mut nearest = 1.0f32;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let c = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                let h = hash3f(wrap3(c, period));
                let d: [f32; 3] = core::array::from_fn(|i| c[i] as f32 + h[i] - p[i]);
                nearest = nearest.min(dot3(d, d));
            }
        }
    }
    1.0 - sqrt(nearest).min(1.0)
}

// Octaves of `noise` at `cells` across, doubling, weighted by `weights`.
fn fbm(p: [f32; 3], cells: f32, weights: &[f32], noise: fn([f32; 3], i32) -> f32) -> f32 {
    let mut sum = 0.0;
    let mut total = 0.0;
    let mut scale = cells;
    for &w in weights {
        // Whole cells across, so every octave tiles.
        #[expect(clippy::cast_possible_truncation, reason = "a handful of cells")]
        let period = scale as i32;
        sum += w * noise(p.map(|v| v * scale), period);
        total += w;
        scale *= 2.0;
    }
    sum / total
}

fn remap(v: f32, lo: f32, hi: f32) -> f32 {
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
}

// The four channels at `p`, the volume's coordinates in tiles (one tile is
// `[0, 1)` along each axis), each in `[0, 1]`. `cloud_noise_at`.
fn noise_channels(p: [f32; 3]) -> [f32; 4] {
    let perlin01 = |q: [f32; 3], period: i32| perlin(q, period) * 0.5 + 0.5;
    let shape_perlin = fbm(p, SHAPE_CELLS, &[0.5, 0.25, 0.125], perlin01);
    let shape_worley = fbm(p, SHAPE_CELLS, &[0.625, 0.25, 0.125], worley);
    let shape = remap(shape_perlin, shape_worley - 1.0, 1.0);
    let detail = fbm(p, DETAIL_CELLS, &[0.625, 0.25, 0.125], worley);
    let weather = fbm(p, WEATHER_CELLS, &[0.5, 0.3, 0.2], perlin01);
    let cells = fbm(p, WEATHER_CELLS, &[1.0], worley);
    [shape, detail, weather.clamp(0.0, 1.0), cells]
}

// The texel at `(x, y, z)` as the volume stores it, eight bits a channel.
fn noise_texel(x: u32, y: u32, z: u32) -> [u8; 4] {
    let size = NOISE_SIZE as f32;
    let p = [x, y, z].map(|c| (c as f32 + 0.5) / size);
    noise_channels(p).map(|v| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "rounded within [0, 255]"
        )]
        let q = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        q
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use concinnity_core::approx::close;

    // Every channel tiles: a point and the same point one tile over read the
    // same, so the volume wraps without a seam.
    #[test]
    fn every_channel_tiles() {
        for p in [[0.13, 0.71, 0.42], [0.99, 0.01, 0.5], [0.0, 0.0, 0.0]] {
            let a = noise_channels(p);
            for axis in 0..3 {
                let mut q = p;
                q[axis] += 1.0;
                assert!(close(noise_channels(q), a, 1e-4), "{p:?} along {axis}");
            }
        }
    }

    #[test]
    fn the_channels_stay_in_range_and_vary() {
        let samples: Vec<[f32; 4]> = (0..64)
            .map(|i| {
                let t = i as f32 / 64.0;
                noise_channels([t, (t * 7.3) % 1.0, (t * 3.1) % 1.0])
            })
            .collect();
        for c in 0..4 {
            let (lo, hi) = samples.iter().fold((f32::MAX, f32::MIN), |(lo, hi), s| {
                (lo.min(s[c]), hi.max(s[c]))
            });
            assert!(lo >= 0.0 && hi <= 1.0, "channel {c}: [{lo}, {hi}]");
            assert!(hi - lo > 0.2, "channel {c} barely varies: [{lo}, {hi}]");
        }
    }

    // Worley is brightest at a cell's point and falls off away from it.
    #[test]
    fn worley_peaks_at_the_cell_points() {
        let cell = [2, 1, 3];
        let h = hash3f(wrap3(cell, 4));
        let at_point: [f32; 3] = core::array::from_fn(|i| cell[i] as f32 + h[i]);
        assert!(close(worley(at_point, 4), 1.0, 1e-5));
        let off = [at_point[0] + 0.3, at_point[1], at_point[2]];
        assert!(worley(off, 4) < 0.75);
    }

    #[test]
    fn perlin_vanishes_on_the_lattice() {
        assert!(close(perlin([3.0, 1.0, 2.0], 4), 0.0, 1e-6));
    }

    #[test]
    fn texels_are_quantized_channels() {
        let t = noise_texel(5, 77, 101);
        let c = noise_channels([5.5, 77.5, 101.5].map(|v| v / NOISE_SIZE as f32));
        for i in 0..4 {
            assert!((f32::from(t[i]) / 255.0 - c[i]).abs() <= 0.5 / 255.0 + 1e-6);
        }
    }
}
