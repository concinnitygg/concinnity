// Fields a ground texture is drawn from, sampled over one tile in tile units
// (u and v in [0, 1)) and wrapping at its edges, so the texture repeats
// without a seam.

// A well-mixed 32-bit hash of a lattice point and a seed.
pub(super) fn hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut h =
        x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

// A hash mapped to [0, 1).
pub(super) fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}

// Lattice coordinate `i` wrapped onto a lattice `cells` across.
fn wrap(i: i64, cells: u32) -> u32 {
    u32::try_from(i.rem_euclid(i64::from(cells.max(1)))).unwrap_or(0)
}

// The lattice cell holding `x` (in cells) and how far into it `x` lies.
fn split(x: f32) -> (i64, f32) {
    let f = x.floor();
    // In range: callers sample within a few tiles of the origin.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a floored lattice coordinate"
    )]
    let i = f as i64;
    (i, x - f)
}

// Quintic ease, flat at both ends so the noise has no lattice creases.
fn ease(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Value noise in [0, 1] with `cells` lattice cells across the tile.
pub(super) fn value_noise(u: f32, v: f32, cells: u32, seed: u32) -> f32 {
    let n = cells.max(1);
    let (ix, fx) = split(u * n as f32);
    let (iy, fy) = split(v * n as f32);
    let at = |dx: i64, dy: i64| unit(hash(wrap(ix + dx, n), wrap(iy + dy, n), seed));
    let (sx, sy) = (ease(fx), ease(fy));
    let top = at(0, 0) + (at(1, 0) - at(0, 0)) * sx;
    let bottom = at(0, 1) + (at(1, 1) - at(0, 1)) * sx;
    top + (bottom - top) * sy
}

/// Fractal value noise in [0, 1]: `octaves` layers from `cells` across, each
/// twice as fine and half as strong as the last.
pub(super) fn fbm(u: f32, v: f32, cells: u32, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut weight = 0.0;
    let mut amplitude = 1.0;
    let mut n = cells.max(1);
    for octave in 0..octaves.max(1) {
        sum +=
            amplitude * value_noise(u, v, n, seed.wrapping_add(octave.wrapping_mul(0x9e37_79b9)));
        weight += amplitude;
        amplitude *= 0.5;
        n = n.saturating_mul(2);
    }
    sum / weight
}

/// The scattered point nearest a sample, one point jittered in each of a
/// lattice's cells: distances in cell widths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Nearest {
    /// Distance to the nearest point.
    pub(super) f1: f32,
    /// Distance to the second nearest.
    pub(super) f2: f32,
    /// The nearest point's hash, stable for that point wherever it is sampled.
    pub(super) id: u32,
    /// From the nearest point to the sample.
    pub(super) offset: [f32; 2],
}

/// The points scattered one to a cell over a lattice `cells` across, nearest
/// to (`u`, `v`).
pub(super) fn nearest(u: f32, v: f32, cells: u32, seed: u32) -> Nearest {
    let n = cells.max(1);
    let (ix, fx) = split(u * n as f32);
    let (iy, fy) = split(v * n as f32);
    let mut best = Nearest {
        f1: f32::MAX,
        f2: f32::MAX,
        id: 0,
        offset: [0.0; 2],
    };
    for dy in -1..=1_i64 {
        for dx in -1..=1_i64 {
            let id = hash(wrap(ix + dx, n), wrap(iy + dy, n), seed);
            let px = dx as f32 + unit(id);
            let py = dy as f32 + unit(hash(id, 0x5bd1_e995, seed));
            let offset = [fx - px, fy - py];
            let d = (offset[0] * offset[0] + offset[1] * offset[1]).sqrt();
            if d < best.f1 {
                best.f2 = best.f1;
                best.f1 = d;
                best.id = id;
                best.offset = offset;
            } else if d < best.f2 {
                best.f2 = d;
            }
        }
    }
    best
}

// Hermite ramp from 0 at `a` to 1 at `b`.
pub(super) fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A pebble or flake lying at a scattered point: how far into its outline a
/// sample lies (1 at its middle, 0 at its rim and outside), the outline an
/// ellipse `radius` wide (in cell widths) and `aspect` as long across, turned
/// by the point's own angle.
pub(super) fn ellipse_inside(near: &Nearest, radius: f32, aspect: f32) -> f32 {
    let angle = unit(hash(near.id, 0x68e3_1da4, 0)) * core::f32::consts::TAU;
    let (s, c) = angle.sin_cos();
    let [x, y] = near.offset;
    let along = (x * c + y * s) / radius;
    let across = (y * c - x * s) / (radius * aspect);
    (1.0 - (along * along + across * across)).max(0.0)
}

/// `values` blurred over a square `2 * radius + 1` texels wide, wrapping, on a
/// `size` by `size` grid.
pub(super) fn box_blur(values: &[f32], size: usize, radius: usize) -> Vec<f32> {
    let width = (2 * radius + 1) as f32;
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0; src.len()];
        for line in 0..size {
            let at = |i: usize| {
                let i = i % size;
                if horizontal {
                    src[line * size + i]
                } else {
                    src[i * size + line]
                }
            };
            let mut sum: f32 = (0..=2 * radius).map(|k| at(k + size - radius)).sum();
            for i in 0..size {
                let index = if horizontal {
                    line * size + i
                } else {
                    i * size + line
                };
                out[index] = sum / width;
                sum += at(i + radius + 1) - at(i + size - radius);
            }
        }
        out
    };
    pass(&pass(values, true), false)
}

/// Lattice cells across a tile for features `cell_m` meters apart.
pub(super) fn cells(cell_m: f32) -> u32 {
    let n = (super::TILE_METERS / cell_m.max(1e-4))
        .round()
        .clamp(1.0, 4096.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to a small positive count"
    )]
    let n = n as u32;
    n
}

/// The texel `x` (in tile units) falls in, on a tile `size` texels across.
pub(super) fn texel(x: f32, size: u32) -> u32 {
    let t = (x * size as f32).floor().max(0.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a texel index"
    )]
    let t = t as u32;
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_noise_wraps_at_the_tile_edge() {
        for k in 0..16 {
            let v = k as f32 / 16.0;
            let a = value_noise(0.0, v, 7, 3);
            let b = value_noise(1.0, v, 7, 3);
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn fbm_stays_in_the_unit_range() {
        for k in 0..256 {
            let x = k as f32 / 256.0;
            let n = fbm(x, 1.0 - x, 3, 4, 9);
            assert!((0.0..=1.0).contains(&n), "{n}");
        }
    }

    #[test]
    fn the_nearest_point_wraps_and_orders_its_distances() {
        let a = nearest(0.001, 0.5, 5, 1);
        let b = nearest(1.001, 0.5, 5, 1);
        assert_eq!(a.id, b.id);
        assert!((a.f1 - b.f1).abs() < 1e-4);
        assert!(a.f1 <= a.f2);
    }

    #[test]
    fn an_ellipse_peaks_at_its_point_and_ends_at_its_rim() {
        let near = Nearest {
            f1: 0.0,
            f2: 1.0,
            id: 42,
            offset: [0.0, 0.0],
        };
        assert_eq!(ellipse_inside(&near, 0.2, 0.5), 1.0);
        let far = Nearest {
            offset: [0.3, 0.3],
            ..near
        };
        assert_eq!(ellipse_inside(&far, 0.2, 0.5), 0.0);
    }

    #[test]
    fn a_blur_keeps_the_mean_and_flattens_a_spike() {
        let size = 8;
        let mut values = vec![0.0; size * size];
        values[0] = 64.0;
        let blurred = box_blur(&values, size, 1);
        let sum: f32 = blurred.iter().sum();
        assert!((sum - 64.0).abs() < 1e-3);
        assert!((blurred[0] - 64.0 / 9.0).abs() < 1e-4);
        // The spike's blur wraps onto the far row and column.
        assert!(blurred[size * size - 1] > 0.0);
    }
}
