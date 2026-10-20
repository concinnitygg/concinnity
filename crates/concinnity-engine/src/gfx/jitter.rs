//! Sub-pixel projection jitter for temporal accumulation.
//!
//! Successive frames offset the projection by a low-discrepancy sample so the
//! TAA pass and any upscaler see slightly different sample positions and can
//! accumulate detail. Pure integer math, shared by every backend so the
//! rasterized scene and the upscale agree on the same offset.

/// Van der Corput radical inverse of `index` in `base`, in `[0, 1)`.
///
/// The Halton sequence is the radical inverse taken in a different prime base
/// per dimension.
///
/// ```
/// # use concinnity_engine::gfx::jitter::radical_inverse;
/// assert_eq!(radical_inverse(1, 2), 0.5);
/// assert_eq!(radical_inverse(2, 2), 0.25);
/// assert_eq!(radical_inverse(0, 2), 0.0);
/// ```
pub fn radical_inverse(mut index: u32, base: u32) -> f32 {
    let inv_base = 1.0 / base as f32;
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    while index > 0 {
        f *= inv_base;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Frames in the jitter cycle the TAA resolve accumulates under, on every
/// backend. The resolve's thin-feature lock lasts this long, so a feature the
/// cycle hits once stays locked until it is hit again.
pub const TAA_CYCLE: u32 = 8;

/// The frame's jitter offset in `[-0.5, 0.5]` render-pixel units, from a
/// 16-frame Halton (2, 3) cycle.
///
/// ```
/// # use concinnity_engine::gfx::jitter::offset;
/// let [x, y] = offset(0);
/// assert!((-0.5..0.5).contains(&x) && (-0.5..0.5).contains(&y));
/// assert_eq!(offset(0), offset(16), "the cycle repeats every 16 frames");
/// ```
pub fn offset(frame_index: u32) -> [f32; 2] {
    offset_in_cycle(frame_index, 16)
}

/// The frame's jitter offset in `[-0.5, 0.5]` render-pixel units, from a
/// Halton (2, 3) cycle of `cycle` frames. The sequence skips index 0, which
/// would sit on the pixel corner.
///
/// ```
/// # use concinnity_engine::gfx::jitter::offset_in_cycle;
/// assert_eq!(offset_in_cycle(0, 8), offset_in_cycle(8, 8));
/// ```
pub fn offset_in_cycle(frame_index: u32, cycle: u32) -> [f32; 2] {
    let idx = (frame_index % cycle) + 1;
    [radical_inverse(idx, 2) - 0.5, radical_inverse(idx, 3) - 0.5]
}

/// `proj` offset by a jitter of `jitter_px` render pixels on a `width` x
/// `height` target.
///
/// The shift lands in the z-coefficients of clip x/y, so after the divide by
/// `clip.w == -view_z` NDC moves by exactly the jitter and depth is untouched.
///
/// ```
/// # use concinnity_engine::gfx::jitter::jitter_projection;
/// let identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
/// let p = jitter_projection(identity, [0.5, -0.25], 100, 50);
/// assert_eq!(p[2][0], -0.01);
/// assert_eq!(p[2][1], 0.01);
/// ```
pub fn jitter_projection(
    proj: [[f32; 4]; 4],
    jitter_px: [f32; 2],
    width: u32,
    height: u32,
) -> [[f32; 4]; 4] {
    let mut p = proj;
    p[2][0] -= jitter_px[0] * 2.0 / width.max(1) as f32;
    p[2][1] -= jitter_px[1] * 2.0 / height.max(1) as f32;
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    // Base 2 halves the interval at each digit, so the first terms are the
    // dyadic rationals in bit-reversed order.
    #[test]
    fn base_two_walks_the_dyadic_rationals() {
        assert_eq!(radical_inverse(1, 2), 0.5);
        assert_eq!(radical_inverse(2, 2), 0.25);
        assert_eq!(radical_inverse(3, 2), 0.75);
        assert_eq!(radical_inverse(4, 2), 0.125);
    }

    // A zero-sized target must not divide the projection by zero.
    #[test]
    fn jitter_on_an_empty_target_stays_finite() {
        let p = jitter_projection([[0.0; 4]; 4], [0.5, 0.5], 0, 0);
        assert!(p[2][0].is_finite() && p[2][1].is_finite());
    }

    #[test]
    fn base_three_walks_thirds() {
        assert_eq!(radical_inverse(1, 3), 1.0 / 3.0);
        assert_eq!(radical_inverse(2, 3), 2.0 / 3.0);
    }

    // Index zero has no digits, so the sequence starts at the origin.
    #[test]
    fn index_zero_is_the_origin() {
        assert_eq!(radical_inverse(0, 2), 0.0);
        assert_eq!(radical_inverse(0, 3), 0.0);
    }

    // Every offset stays inside the pixel, and the cycle is 16 frames long.
    #[test]
    fn offsets_stay_within_the_pixel_and_repeat_every_sixteen_frames() {
        for frame in 0..64 {
            let [x, y] = offset(frame);
            assert!((-0.5..0.5).contains(&x), "frame {frame} x = {x}");
            assert!((-0.5..0.5).contains(&y), "frame {frame} y = {y}");
        }
        assert_eq!(offset(0), offset(16));
        assert_eq!(offset(7), offset(23));
    }

    #[test]
    fn an_eight_frame_cycle_stays_within_the_pixel_and_repeats() {
        for frame in 0..32 {
            let [x, y] = offset_in_cycle(frame, 8);
            assert!((-0.5..0.5).contains(&x), "frame {frame} x = {x}");
            assert!((-0.5..0.5).contains(&y), "frame {frame} y = {y}");
            assert_eq!(offset_in_cycle(frame, 8), offset_in_cycle(frame + 8, 8));
        }
        // Index 1: radical_inverse(1, 2) = 0.5 and radical_inverse(1, 3) = 1/3.
        let [x, y] = offset(0);
        assert!(x.abs() < 1e-6, "{x}");
        assert!((y - (1.0 / 3.0 - 0.5)).abs() < 1e-6, "{y}");
    }
}
