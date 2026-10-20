//! The ground under grass: dead grass strewn over the terrain's own surface,
//! the more of it the denser the grass grows, broken up by patches of bare
//! earth and of lighter, drier litter at a few scales. The patches are a noise
//! field over fixed ground points (a terrain's world `x`, `z`; a planet's
//! point in the planet's own frame), so they hold still under the camera and
//! every move of the simulated frame.
//!
//! `ground_variation` is the shaders' `grass_ground_variation` line for
//! line.

#[cfg(test)]
use super::planet::density::grass_hash;
#[cfg(test)]
use concinnity_core::math::{SaturatingFrom, floor, lerp, smoothstep};

/// The hue dead grass browns toward, at unit luminance.
const LITTER_HUE: [f32; 3] = [1.38, 0.94, 0.52];

// How far a derived litter color moves from the root's hue to brown, and how
// much darker than the root it lies.
const LITTER_BROWNING: f32 = 0.8;
const LITTER_SHADE: f32 = 0.8;

/// Sizes of the ground's patches, finest first, in meters. Matches
/// `GRASS_LITTER_SCALES` in the shaders.
#[cfg(test)]
const LITTER_SCALES: [f32; 3] = [0.45, 1.8, 7.0];

// Salts of the patch noises, one per scale.
#[cfg(test)]
const SALTS: [u32; 3] = [0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372];

/// A unit value at a 3D lattice point, salted, from a single hash step. The
/// shaders' `grass_patch_lattice`.
#[cfg(test)]
fn patch_lattice(c: [i32; 3], salt: u32) -> f32 {
    let h = grass_hash(
        c[0].cast_unsigned().wrapping_mul(0x8da6_b343)
            ^ c[1].cast_unsigned().wrapping_mul(0xd816_3841)
            ^ c[2].cast_unsigned().wrapping_mul(0xcb1a_b31f)
            ^ salt,
    );
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Trilinear value noise in [0, 1] over [`patch_lattice`]: the shaders'
/// `grass_patch_value_noise3`.
#[cfg(test)]
fn patch_noise(p: [f32; 3], salt: u32) -> f32 {
    let cell = p.map(floor);
    let f: [f32; 3] = core::array::from_fn(|i| {
        let t = p[i] - cell[i];
        t * t * (3.0 - 2.0 * t)
    });
    let c = cell.map(i32::saturating_from);
    let at = |dx: i32, dy: i32, dz: i32| patch_lattice([c[0] + dx, c[1] + dy, c[2] + dz], salt);
    let x00 = lerp(at(0, 0, 0), at(1, 0, 0), f[0]);
    let x10 = lerp(at(0, 1, 0), at(1, 1, 0), f[0]);
    let x01 = lerp(at(0, 0, 1), at(1, 0, 1), f[0]);
    let x11 = lerp(at(0, 1, 1), at(1, 1, 1), f[0]);
    lerp(lerp(x00, x10, f[1]), lerp(x01, x11, f[1]), f[2])
}

fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// The litter color a grass of `root_color` sheds when it names none: the
/// root's color turned toward brown at its own brightness, then darkened.
pub fn litter_color(root_color: [f32; 3]) -> [f32; 3] {
    let luma = luminance(root_color);
    core::array::from_fn(|i| {
        let brown = LITTER_HUE[i] * luma;
        (root_color[i] + (brown - root_color[i]) * LITTER_BROWNING) * LITTER_SHADE
    })
}

/// What lies on the ground at fixed point `p` (meters) under a pixel covering
/// `footprint` meters: how much of it is litter rather than bare earth
/// (`[0]`, in [0.35, 1]), how dark the earth is (`[1]`, a multiplier in
/// [0.65, 1]), and how dry and light the litter is (`[2]`, in [0, 1]).
/// Patches a pixel cannot hold fade to their mean.
#[cfg(test)]
fn ground_variation(p: [f32; 3], footprint: f32) -> [f32; 3] {
    let [fine, mid, broad] = core::array::from_fn(|i| {
        let scale = LITTER_SCALES[i];
        let detail = 1.0 - smoothstep(scale * 0.5, scale * 2.0, footprint);
        let n = patch_noise(p.map(|c| c / scale), SALTS[i]);
        0.5 + (n - 0.5) * detail
    });
    let patch = fine * 0.25 + mid * 0.35 + broad * 0.4;
    let dry = fine * 0.6 + (1.0 - mid) * 0.4;
    [
        0.35 + 0.65 * smoothstep(0.36, 0.52, patch),
        0.65 + 0.35 * smoothstep(0.3, 0.46, patch),
        smoothstep(0.55, 0.7, dry),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn point(i: u32) -> [f32; 3] {
        let a = i as f32 * 0.618_034;
        let r = 0.37 * i as f32;
        [r * a.cos(), 0.0, r * a.sin()]
    }

    #[test]
    fn the_scales_are_the_shaders() {
        let src = crate::render::shader::embedded::GRASS_FIELD;
        let text = crate::render::shader::consts::literal(src, "GRASS_LITTER_SCALES");
        let inner = text.trim_start_matches("float3(").trim_end_matches(')');
        let scales: Vec<f32> = inner
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        assert_eq!(scales, LITTER_SCALES);
    }

    // Browner and darker than the root, from any green.
    #[test]
    fn derived_litter_is_a_darker_brown_than_the_root() {
        for root in [[0.025, 0.06, 0.012], [0.05, 0.09, 0.02], [0.02, 0.04, 0.03]] {
            let l = litter_color(root);
            assert!(luminance(l) < luminance(root));
            assert!(l[0] / l[1] > root[0] / root[1], "{l:?}");
            assert!(l[2] / l[1] < 1.0, "{l:?}");
            assert!(l.iter().all(|&c| c >= 0.0));
        }
        assert_eq!(litter_color([0.0; 3]), [0.0; 3]);
    }

    // The ground is mostly litter with some bare earth and some dry patches,
    // inside the documented ranges, the same answer every time.
    #[test]
    fn the_ground_mixes_litter_earth_and_dry_patches() {
        let n = 20_000;
        let (mut bare, mut full, mut dry) = (0, 0, 0);
        for i in 0..n {
            let v = ground_variation(point(i), 0.0);
            assert_eq!(v, ground_variation(point(i), 0.0));
            assert!((0.35..=1.0).contains(&v[0]));
            assert!((0.65..=1.0).contains(&v[1]));
            assert!((0.0..=1.0).contains(&v[2]));
            bare += u32::from(v[0] < 0.5);
            full += u32::from(v[0] > 0.95);
            dry += u32::from(v[2] > 0.5);
        }
        let share = |c: u32| c as f32 / n as f32;
        assert!(
            share(bare) > 0.03 && share(bare) < 0.3,
            "bare {}",
            share(bare)
        );
        assert!(share(full) > 0.25, "full {}", share(full));
        assert!(share(dry) > 0.03 && share(dry) < 0.35, "dry {}", share(dry));
    }

    // Lattice corners are independent: neighbors and salts give unrelated
    // values, spread over the unit range.
    #[test]
    fn the_patch_lattice_spreads_over_the_unit_range() {
        let n = 4096;
        let mut sum = 0.0;
        let mut low = 0;
        for i in 0..n {
            let c = [i % 16 - 8, i / 256 - 8, (i / 16) % 16 - 8];
            let v = patch_lattice(c, 7);
            assert!((0.0..1.0).contains(&v));
            sum += v;
            low += i32::from(v < 0.25);
            assert_ne!(v, patch_lattice(c, 8));
        }
        let mean = sum / n as f32;
        assert!((mean - 0.5).abs() < 0.03, "{mean}");
        assert!((low as f32 / n as f32 - 0.25).abs() < 0.03);
        assert_eq!(patch_noise([0.0, 0.0, 0.0], 3), patch_lattice([0, 0, 0], 3));
    }

    // A pixel too coarse for any patch sees their mean everywhere, so the far
    // ground neither aliases nor shimmers.
    #[test]
    fn coarse_pixels_see_the_mean() {
        let mean = ground_variation(point(1), 100.0);
        for i in 0..200 {
            assert_eq!(ground_variation(point(i), 100.0), mean);
        }
    }

    // The patches change over decimeters, not from pixel to pixel.
    #[test]
    fn the_ground_is_smooth_at_pixel_scale() {
        for i in 0..400 {
            let p = point(i);
            let a = ground_variation(p, 0.0);
            let b = ground_variation([p[0] + 0.005, p[1], p[2] - 0.005], 0.0);
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 0.1, "{a:?} {b:?}");
            }
        }
    }
}
