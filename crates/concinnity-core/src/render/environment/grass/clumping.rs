//! How a layer's blades crowd into clumps. Each blade belongs to the nearest
//! center of a jittered lattice of clump-sized cells, and is drawn toward
//! that center by a share that shrinks from the middle of its cell to the
//! cell's edge, so blades crowd the middle, thin toward the edges and never
//! leave their cell: the ground between clumps grows sparser but never bare,
//! and every blade is kept.
//!
//! A blade's place in its cell is its spot: its distance to the nearest
//! center over its distance to the second nearest, 0 at the center and 1 on
//! the cell's edge. `clump_at`, `gather_scale` and `gathered_density`
//! are the shaders' `grass_clump_at`, `grass_gather_scale` and
//! `grass_gathered_density` line for line.

#[cfg(test)]
use super::planet::density::grass_hash;
#[cfg(test)]
use crate::math::{SaturatingFrom, floor, sqrt};

/// How far the strongest clumping draws a blade at its clump's center toward
/// it: the share of its distance it keeps there is `1 - GATHER_STRENGTH`.
/// Matches `GRASS_GATHER` in the shaders.
const GATHER_STRENGTH: f32 = 0.65;

/// The clump a point belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clump {
    /// The clump's hash, which its shared height, lean and hue come from.
    pub hash: u32,
    /// Its center, in the point's coordinates.
    pub center: [f32; 2],
    /// The point's spot in the clump's cell: 0 at the center, 1 on its edge.
    pub spot: f32,
}

// The kernel's `grass_hash2`.
#[cfg(test)]
fn hash2(c: [i32; 2]) -> u32 {
    grass_hash(
        c[0].cast_unsigned()
            .wrapping_add(grass_hash(c[1].cast_unsigned())),
    )
}

// The kernel's `grass_unit`.
#[cfg(test)]
fn unit(h: u32, salt: u32) -> f32 {
    (grass_hash(h ^ salt) >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// The clump point `p` (meters on the ground's plane) belongs to, for clumps
/// `clump_size` across salted by `seed` and `salt`.
#[cfg(test)]
fn clump_at(p: [f32; 2], clump_size: f32, seed: u32, salt: u32) -> Clump {
    let q = [p[0] / clump_size, p[1] / clump_size];
    let base = [
        i32::saturating_from(floor(q[0])),
        i32::saturating_from(floor(q[1])),
    ];
    let (mut best, mut second) = (f32::MAX, f32::MAX);
    let mut clump = Clump {
        hash: 0,
        center: p,
        spot: 0.0,
    };
    for dz in -1..=1 {
        for dx in -1..=1 {
            let cell = [base[0] + dx, base[1] + dz];
            let h = grass_hash(hash2(cell) ^ seed ^ salt) ^ 0x9e37_79b9;
            let c = [cell[0] as f32 + unit(h, 1), cell[1] as f32 + unit(h, 2)];
            let d = (c[0] - q[0]) * (c[0] - q[0]) + (c[1] - q[1]) * (c[1] - q[1]);
            if d < best {
                second = best;
                best = d;
                clump.hash = h;
                clump.center = [c[0] * clump_size, c[1] * clump_size];
            } else if d < second {
                second = d;
            }
        }
    }
    clump.spot = sqrt(best / second.max(1e-12)).min(1.0);
    clump
}

/// The share of its distance to its clump's center a blade at `spot` keeps
/// under `clumping`: `1 - GATHER_STRENGTH * clumping` at the center, rising
/// to 1 at the cell's edge.
fn gather_scale(spot: f32, clumping: f32) -> f32 {
    let center = 1.0 - GATHER_STRENGTH * clumping;
    center + (1.0 - center) * spot
}

/// Where a blade at `p` grows once drawn into `clump` under `clumping`.
pub fn gathered(p: [f32; 2], clump: &Clump, clumping: f32) -> [f32; 2] {
    let s = gather_scale(clump.spot, clumping);
    [
        clump.center[0] + (p[0] - clump.center[0]) * s,
        clump.center[1] + (p[1] - clump.center[1]) * s,
    ]
}

/// How densely blades grow at `spot` under `clumping`, as a multiple of the
/// layer's mean: the blades drawn to a spot came from the larger one whose
/// gathered place it is, so the density is that ring's area over this one's.
#[cfg(test)]
fn gathered_density(spot: f32, clumping: f32) -> f32 {
    let center = 1.0 - GATHER_STRENGTH * clumping;
    let pull = 1.0 - center;
    // The spot `from` a blade left: `from * gather_scale(from) = spot`.
    let from = 2.0 * spot / (center + sqrt(center * center + 4.0 * pull * spot));
    1.0 / ((center + pull * from) * (center + 2.0 * pull * from))
}

/// The farthest any blade of a layer moves from where it would grow without
/// clumping, in meters. A blade at spot `t` moves `t (1 - t)` of its distance
/// to the second nearest center, at most a quarter of it, times the pull, and
/// the second nearest center lies within 2.25 clumps.
pub fn max_gather_offset(clump_size: f32, clumping: f32) -> f32 {
    2.25 * clump_size * 0.25 * GATHER_STRENGTH * clumping
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::SaturatingFrom;
    use alloc::vec::Vec;

    // Blades on an even jittered grid, as the kernel scatters them.
    fn blades(n: i32, cell: f32) -> Vec<[f32; 2]> {
        let mut out = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let h = grass_hash(hash2([x, z]) ^ 5);
                out.push([
                    (x as f32 + unit(h, 3)) * cell,
                    (z as f32 + unit(h, 4)) * cell,
                ]);
            }
        }
        out
    }

    #[test]
    fn a_point_lies_in_the_cell_of_its_nearest_center() {
        for p in blades(40, 0.11) {
            let c = clump_at(p, 0.6, 3, 0);
            assert!((0.0..=1.0).contains(&c.spot));
            let d = ((p[0] - c.center[0]).powi(2) + (p[1] - c.center[1]).powi(2)).sqrt();
            assert!(d <= 1.5 * 0.6, "{d}");
            assert_eq!(c, clump_at(p, 0.6, 3, 0));
        }
        // Salts lay their own lattices.
        let p = [1.3, -2.7];
        assert_ne!(clump_at(p, 0.6, 3, 0).hash, clump_at(p, 0.6, 3, 9).hash);
    }

    #[test]
    fn no_clumping_moves_nothing() {
        for p in blades(10, 0.2) {
            let c = clump_at(p, 0.7, 1, 0);
            let g = gathered(p, &c, 0.0);
            assert!((g[0] - p[0]).abs() < 1e-6 && (g[1] - p[1]).abs() < 1e-6);
            assert_eq!(gathered_density(c.spot, 0.0), 1.0);
        }
    }

    // A gathered blade stays in its own clump, never moves further than the
    // bound the tile cull grows its boxes by, and sits nearer the center.
    #[test]
    fn gathered_blades_stay_in_their_clump_and_crowd_its_middle() {
        let (size, clumping) = (0.6, 1.0);
        let (mut before, mut after) = (0.0, 0.0);
        let points = blades(60, 0.09);
        for &p in &points {
            let c = clump_at(p, size, 7, 0);
            let g = gathered(p, &c, clumping);
            assert_eq!(clump_at(g, size, 7, 0).hash, c.hash);
            let moved = ((g[0] - p[0]).powi(2) + (g[1] - p[1]).powi(2)).sqrt();
            assert!(moved <= max_gather_offset(size, clumping) + 1e-5, "{moved}");
            before += clump_at(p, size, 7, 0).spot;
            after += clump_at(g, size, 7, 0).spot;
        }
        let n = points.len() as f32;
        assert!(
            after / n < 0.85 * before / n,
            "{} {}",
            before / n,
            after / n
        );
    }

    // Every blade is kept, so the density the far field reads averages to the
    // layer's own over the ground: the clumps' middles are as much denser as
    // the gaps between them are sparser, and no gap is bare.
    #[test]
    fn the_gathered_density_keeps_the_fields_coverage() {
        for clumping in [0.3, 0.6, 1.0] {
            let points = blades(200, 0.013);
            let mean = points
                .iter()
                .map(|&p| gathered_density(clump_at(p, 0.6, 2, 0).spot, clumping))
                .sum::<f32>()
                / points.len() as f32;
            assert!((mean - 1.0).abs() < 0.08, "{clumping}: {mean}");
            assert!(gathered_density(0.0, clumping) > 1.5);
            let edge = gathered_density(1.0, clumping);
            assert!(edge < 1.0 && edge > 0.5, "{edge}");
        }
    }

    // What the far field reads agrees with where the kernel's blades land:
    // blade counts by spot band, against the density integrated over the
    // same bands.
    #[test]
    fn the_gathered_density_matches_the_gathered_blades() {
        let (size, clumping) = (0.6, 0.8);
        let bands = 4;
        let mut blades_in = [0.0f32; 4];
        let mut ground_in = [0.0f32; 4];
        let mut predicted = [0.0f32; 4];
        let band = |s: f32| usize::saturating_from(s * bands as f32).min(bands - 1);
        for &p in &blades(260, 0.01) {
            let c = clump_at(p, size, 4, 0);
            blades_in[band(clump_at(gathered(p, &c, clumping), size, 4, 0).spot)] += 1.0;
            ground_in[band(c.spot)] += 1.0;
            predicted[band(c.spot)] += gathered_density(c.spot, clumping);
        }
        for b in 0..bands {
            let seen = blades_in[b] / ground_in[b];
            let model = predicted[b] / ground_in[b];
            assert!(
                (seen / model - 1.0).abs() < 0.12,
                "band {b}: {seen} {model}"
            );
        }
    }

    #[test]
    fn the_strength_is_the_shaders() {
        let src = crate::render::shader::embedded::GRASS_FIELD;
        assert_eq!(
            crate::render::shader::consts::float(src, "GRASS_GATHER"),
            GATHER_STRENGTH
        );
    }

    #[test]
    fn the_hashes_are_the_kernels() {
        assert_eq!(hash2([0, 0]), grass_hash(grass_hash(0)));
        assert!((unit(0, 0) - (grass_hash(0) >> 8) as f32 / 16_777_216.0).abs() < 1e-9);
    }
}
