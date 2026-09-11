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
/// # use concinnity_core::gfx::jitter::radical_inverse;
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

/// The frame's jitter offset in `[-0.5, 0.5]` render-pixel units, from a
/// 16-frame Halton (2, 3) cycle.
///
/// ```
/// # use concinnity_core::gfx::jitter::offset;
/// let [x, y] = offset(0);
/// assert!((-0.5..0.5).contains(&x) && (-0.5..0.5).contains(&y));
/// assert_eq!(offset(0), offset(16), "the cycle repeats every 16 frames");
/// ```
pub fn offset(frame_index: u32) -> [f32; 2] {
    let idx = (frame_index % 16) + 1;
    [radical_inverse(idx, 2) - 0.5, radical_inverse(idx, 3) - 0.5]
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
}
