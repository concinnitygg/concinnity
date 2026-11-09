//! How a field's color drifts across hundreds of meters to kilometers: deeper
//! and lighter green patches, drier stretches, and drier grass on slopes. A
//! noise field over fixed ground points (a terrain's world `x`, `z`; a
//! planet's surface direction times its radius), so the blades and the far
//! field read the same drift wherever they meet and it never moves with the
//! simulated frame.
//!
//! [`macro_variation`] and `macro_tint` are the shaders' `grass_macro` and
//! `grass_macro_tint` line for line.

use super::planet::density::value_noise;

/// Size of the smaller patches the drift is made of, in meters.
const MACRO_NEAR_SIZE: f32 = 150.0;

/// Size of the larger stretches, in meters.
const MACRO_FAR_SIZE: f32 = 1200.0;

/// The tint a dry clump or a dry stretch of field leans toward.
#[cfg(test)]
const DRY_TINT: [f32; 3] = [1.35, 1.1, 0.55];

// How much lighter or deeper the green of a lush or sparse patch reads.
#[cfg(test)]
const LUSH_SHIFT: f32 = 0.3;
// How much drier a slope grows per unit of `1 - cos` of its angle.
const SLOPE_DRYNESS: f32 = 4.0;

/// The drift at ground point `p` (meters) for a layer salted by `seed`: how
/// dry the grass is there, in [0, 1], and how lush, in [-1, 1], both scaled
/// by the layer's `amount`. `slope` is `1 - cos` of the ground's angle from
/// the up the blades stand along. `near_detail` in [0, 1] fades the smaller
/// patches toward their mean where a pixel covers too much ground to show
/// them.
pub fn macro_variation(
    p: [f32; 3],
    seed: u32,
    slope: f32,
    amount: f32,
    near_detail: f32,
) -> [f32; 2] {
    let far = value_noise(p.map(|c| c * (1.0 / MACRO_FAR_SIZE)), seed ^ 0x2c1b_3c6d);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let near = lerp(
        0.5,
        value_noise(p.map(|c| c * (1.0 / MACRO_NEAR_SIZE)), seed ^ 0x297a_2d39),
        near_detail,
    );
    let dry = ((far - 0.55) * 2.5 + (near - 0.5) * 1.2 + slope * SLOPE_DRYNESS).clamp(0.0, 1.0);
    let lush = ((0.5 - far) * 1.4 + (near - 0.5) * 2.0).clamp(-1.0, 1.0);
    [dry * amount, lush * amount]
}

/// The multiplier a drift `m` (from [`macro_variation`]) puts on a blade's
/// color.
#[cfg(test)]
fn macro_tint(m: [f32; 2]) -> [f32; 3] {
    core::array::from_fn(|i| (1.0 + (DRY_TINT[i] - 1.0) * m[0]) * (1.0 + LUSH_SHIFT * m[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(i: u32) -> [f32; 3] {
        let a = i as f32 * 0.618_033_9;
        let r = 37.0 * i as f32;
        [r * a.cos(), 0.0, r * a.sin()]
    }

    // A meadow, not a patchwork: most ground is green, a fair share of it
    // dry, and none of it beyond the bounds; the same answer every time.
    #[test]
    fn the_drift_stays_in_range_and_covers_some_ground_dry() {
        let n = 20_000;
        let mut dry = 0;
        let mut lush = 0;
        let mut sparse = 0;
        for i in 0..n {
            let m = macro_variation(point(i), 7, 0.0, 1.0, 1.0);
            assert!((0.0..=1.0).contains(&m[0]) && (-1.0..=1.0).contains(&m[1]));
            assert_eq!(m, macro_variation(point(i), 7, 0.0, 1.0, 1.0));
            dry += u32::from(m[0] > 0.3);
            lush += u32::from(m[1] > 0.3);
            sparse += u32::from(m[1] < -0.3);
        }
        let share = |c: u32| c as f32 / n as f32;
        assert!(share(dry) > 0.08 && share(dry) < 0.45, "dry {}", share(dry));
        assert!(share(lush) > 0.08, "lush {}", share(lush));
        assert!(share(sparse) > 0.08, "sparse {}", share(sparse));
    }

    #[test]
    fn no_amount_is_no_drift_and_slopes_dry() {
        for i in 0..200 {
            assert_eq!(macro_variation(point(i), 3, 0.2, 0.0, 1.0), [0.0, 0.0]);
            let flat = macro_variation(point(i), 3, 0.0, 1.0, 1.0);
            let steep = macro_variation(point(i), 3, 0.1, 1.0, 1.0);
            assert!(steep[0] >= flat[0]);
        }
        assert_eq!(macro_tint([0.0, 0.0]), [1.0; 3]);
        let t = macro_tint([1.0, 0.0]);
        assert_eq!(t, DRY_TINT);
    }

    // The patches change over tens of meters, not from blade to blade, so the
    // drift the kernel stores per blade and the one the far field evaluates per
    // pixel agree wherever the two meet.
    #[test]
    fn the_drift_is_smooth_at_blade_scale() {
        for i in 0..400 {
            let p = point(i);
            let q = [p[0] + 0.5, p[1], p[2] - 0.5];
            let a = macro_variation(p, 9, 0.0, 1.0, 1.0);
            let b = macro_variation(q, 9, 0.0, 1.0, 1.0);
            assert!((a[0] - b[0]).abs() < 0.05 && (a[1] - b[1]).abs() < 0.05);
        }
    }

    // With no detail left, the smaller patches stand at their mean.
    #[test]
    fn fading_the_near_detail_leaves_the_far_stretches() {
        for i in 0..50 {
            let p = point(i);
            let faded = macro_variation(p, 1, 0.0, 1.0, 0.0);
            let far = value_noise(p.map(|c| c * (1.0 / MACRO_FAR_SIZE)), 1 ^ 0x2c1b_3c6d);
            assert!((faded[1] - ((0.5 - far) * 1.4).clamp(-1.0, 1.0)).abs() < 1e-6);
        }
    }
}
