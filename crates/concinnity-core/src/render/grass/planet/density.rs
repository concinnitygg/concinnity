//! How densely a planet's grass grows from place to place: a meadow of soft
//! edged bare patches, a noise field laid over the sphere itself, so it has no
//! seams between faces and is the same wherever the world is simulated from.
//!
//! [`meadow_density`] is the kernel's `grass_meadow_density` line for line, so
//! the far field the tiles are tinted with thins exactly where the blades do.

use crate::math::floor;

/// Width of the bare and grassy patches a planet's meadow is made of, in
/// meters.
pub const MEADOW_PATCH_SIZE: f32 = 24.0;

// The noise values the density ramps up over: below the first the ground is
// bare, above the second fully grown.
const BARE_BELOW: f32 = 0.30;
const FULL_ABOVE: f32 = 0.42;

/// One 32-bit hash step (PCG), the kernel's `grass_hash`.
pub fn grass_hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

// A unit value at a 3D lattice point, salted.
fn lattice(c: [i32; 3], salt: u32) -> f32 {
    let h = grass_hash((c[0] as u32).wrapping_add(grass_hash(
        (c[1] as u32).wrapping_add(grass_hash((c[2] as u32) ^ salt)),
    )));
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

// The turn from a position into the noise lattice, by rows, so no lattice
// plane (where the noise is flat across, a crease) lies along a world axis.
const LATTICE_TURN: [[f32; 3]; 3] = [
    [0.811_838_4, 0.534_144_4, -0.235_813_9],
    [-0.459_433_17, 0.833_629_2, 0.306_567_43],
    [0.360_332_63, -0.140_542_48, 0.922_175_8],
];

// Where position `p` lies in the lattice salted by `salt`: turned, then
// shifted by a salted fraction of a cell, so lattices of different salts and
// sizes do not all meet at the origin.
fn lattice_point(p: [f32; 3], salt: u32) -> [f32; 3] {
    core::array::from_fn(|i| {
        let row = LATTICE_TURN[i];
        let shift = ((salt >> (10 * i)) & 0x3ff) as f32 / 1024.0;
        row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + shift
    })
}

/// Trilinear value noise in [0, 1] at position `p` (in lattice cells),
/// salted: the kernel's `grass_value_noise3`. The lattice is turned off the
/// axes, so the noise has no creases along them.
pub fn value_noise(p: [f32; 3], salt: u32) -> f32 {
    let p = lattice_point(p, salt);
    let cell = p.map(floor);
    let f: [f32; 3] = core::array::from_fn(|i| {
        let t = p[i] - cell[i];
        t * t * (3.0 - 2.0 * t)
    });
    let c = cell.map(|v| v as i32);
    let at = |dx: i32, dy: i32, dz: i32| lattice([c[0] + dx, c[1] + dy, c[2] + dz], salt);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(at(0, 0, 0), at(1, 0, 0), f[0]);
    let x10 = lerp(at(0, 1, 0), at(1, 1, 0), f[0]);
    let x01 = lerp(at(0, 0, 1), at(1, 0, 1), f[0]);
    let x11 = lerp(at(0, 1, 1), at(1, 1, 1), f[0]);
    lerp(lerp(x00, x10, f[1]), lerp(x01, x11, f[1]), f[2])
}

/// Lattice cells of the meadow noise per unit of direction on a planet of
/// `radius` meters.
pub fn meadow_scale(radius: f64) -> f32 {
    (radius / f64::from(MEADOW_PATCH_SIZE)) as f32
}

/// The share of a layer's density that grows in unit direction `dir` from
/// the planet's center, in [0, 1], for the meadow noise at `scale` lattice
/// cells per unit of direction, salted by the layer's `seed`.
pub fn meadow_density(dir: [f32; 3], scale: f32, seed: u32) -> f32 {
    let p = dir.map(|v| v * scale);
    let q = p.map(|v| v * 2.07 + 13.1);
    let n = value_noise(p, seed) * 0.65 + value_noise(q, seed ^ 0x68e3_1da4) * 0.35;
    let t = ((n - BARE_BELOW) / (FULL_ABOVE - BARE_BELOW)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::CubeFace;
    use crate::render::grass::planet::cells::face_direction;

    const SCALE: f32 = 50_000.0 / MEADOW_PATCH_SIZE;

    fn dir_at(i: u32) -> [f32; 3] {
        let face = CubeFace((i % 6) as u8);
        let s = ((i * 7919) % 1000) as f32 / 500.0 - 1.0;
        let t = ((i * 104_729) % 1000) as f32 / 500.0 - 1.0;
        face_direction(face, s, t)
    }

    // A meadow, not a lawn and not a desert: most ground grows, some patches
    // are bare, and the field is the same every time it is asked.
    #[test]
    fn most_ground_grows_and_some_lies_bare() {
        let n = 20_000;
        let (mut bare, mut full) = (0, 0);
        for i in 0..n {
            let d = meadow_density(dir_at(i), SCALE, 11);
            assert!((0.0..=1.0).contains(&d));
            assert_eq!(d, meadow_density(dir_at(i), SCALE, 11));
            bare += u32::from(d < 0.05);
            full += u32::from(d > 0.95);
        }
        let (bare, full) = (bare as f32 / n as f32, full as f32 / n as f32);
        assert!(bare > 0.04 && bare < 0.25, "bare {bare}");
        assert!(full > 0.6, "full {full}");
    }

    // Layers with their own seeds lay their own patches.
    #[test]
    fn seeds_lay_different_patches() {
        let differ = (0..500)
            .filter(|&i| {
                (meadow_density(dir_at(i), SCALE, 1) - meadow_density(dir_at(i), SCALE, 2)).abs()
                    > 0.5
            })
            .count();
        assert!(differ > 20, "{differ}");
    }

    // Walking scale: a few centimeters apart the density barely changes, so
    // patch edges are soft and a frame move's rounding moves nothing.
    #[test]
    fn the_field_is_smooth_at_walking_scale() {
        let step = 0.05 / 50_000.0;
        for i in 0..400 {
            let d = dir_at(i);
            let e = crate::math::vec3::normalize_or([d[0] + step, d[1], d[2] - step], 1e-9, d);
            let a = meadow_density(d, SCALE, 3);
            let b = meadow_density(e, SCALE, 3);
            assert!((a - b).abs() < 0.05, "{a} {b}");
        }
    }

    // The turn keeps cells their size and shape.
    #[test]
    fn the_lattice_turn_is_a_rotation() {
        let r = LATTICE_TURN;
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot(r[i], r[j]) - want).abs() < 1e-5, "{i} {j}");
            }
        }
        let det = dot(
            r[0],
            [
                r[1][1] * r[2][2] - r[1][2] * r[2][1],
                r[1][2] * r[2][0] - r[1][0] * r[2][2],
                r[1][0] * r[2][1] - r[1][1] * r[2][0],
            ],
        );
        assert!((det - 1.0).abs() < 1e-5, "{det}");
    }

    // Across the plane x = 0 the noise changes as fast as it does anywhere:
    // an axis-aligned lattice would be flat across it, a crease along the
    // axis.
    #[test]
    fn the_noise_has_no_crease_along_an_axis_plane() {
        let h = 1e-3;
        let slope_across = |x: f32, i: u32| {
            let y = (i * 7919 % 1000) as f32 * 0.037;
            let z = (i * 104_729 % 1000) as f32 * 0.041;
            (value_noise([x + h, y, z], 5) - value_noise([x - h, y, z], 5)).abs() / (2.0 * h)
        };
        let n = 2_000;
        let on_plane: f32 = (0..n).map(|i| slope_across(0.0, i)).sum::<f32>() / n as f32;
        let anywhere: f32 = (0..n)
            .map(|i| slope_across((i * 31 % 997) as f32 * 0.013, i))
            .sum::<f32>()
            / n as f32;
        assert!(on_plane > anywhere * 0.5, "{on_plane} vs {anywhere}");
    }

    #[test]
    fn the_hash_is_the_kernels() {
        // PCG output of 0 and 1, as the HLSL `grass_hash` computes them.
        assert_eq!(grass_hash(0), 129_708_002);
        assert_ne!(grass_hash(1), grass_hash(0));
    }
}
