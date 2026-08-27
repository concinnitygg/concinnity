// Signed distance field conversion for rasterised glyph cells.
//
// Each atlas texel stores a normalised SDF value in [0, 1] where 0.5 = the glyph
// outline. Values > 0.5 are inside; values < 0.5 are outside. The fragment
// shader uses smoothstep + fwidth to reconstruct crisp, scale-independent alpha.

use crate::math::{round, sqrt};
use alloc::vec::Vec;

// Reusable working buffers for the separable Euclidean distance transform,
// sized once for the largest cell so the per-glyph SDF pass allocates nothing.
pub(crate) struct EdtScratch {
    pub(crate) v: Vec<usize>,
    pub(crate) z: Vec<f32>,
    pub(crate) row_tmp: Vec<f32>,
    pub(crate) col_src: Vec<f32>,
    pub(crate) col_dst: Vec<f32>,
}

// Reusable working buffers for `cell_coverage_to_sdf`: the two per-pixel
// distance grids plus the distance-transform scratch. Caller-owned and reused
// across glyphs to avoid per-call allocation.
pub(crate) struct SdfScratch {
    pub(crate) inside_dist2: Vec<f32>,
    pub(crate) outside_dist2: Vec<f32>,
    pub(crate) edt: EdtScratch,
}

// 1-D squared Euclidean distance transform (Felzenszwalb-Huttenlocher).
// `f` is either 0.0 (foreground) or a large value (background).
// `d` receives the squared distance to the nearest foreground sample.
// `v` and `z` are caller-supplied scratch buffers of length >= n and >= n+1.
fn edt_1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    debug_assert_eq!(n, d.len());
    debug_assert!(v.len() >= n);
    debug_assert!(z.len() > n);
    if n == 0 {
        return;
    }

    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    let mut k = 0usize;

    for q in 1..n {
        loop {
            let r = v[k];
            let s = ((f[q] + (q * q) as f32) - (f[r] + (r * r) as f32))
                / (2.0 * q as f32 - 2.0 * r as f32);
            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
            // z[0] = -INF so s > z[0] is always true; k==0 branch never reached
            if k == 0 {
                break;
            }
            k -= 1;
        }
    }

    k = 0;
    for (q, dq) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let r = v[k];
        let diff = q as f32 - r as f32;
        *dq = diff * diff + f[r];
    }
}

// 2-D squared Euclidean distance transform via two separable 1-D passes.
// `out` must be pre-initialised by the caller: 0.0 for foreground, INF for background.
// The scratch buffers are caller-provided to avoid per-call allocation.
fn edt_2d(w: usize, h: usize, out: &mut [f32], edt: &mut EdtScratch) {
    let EdtScratch {
        v,
        z,
        row_tmp,
        col_src,
        col_dst,
    } = edt;
    // Row pass
    for y in 0..h {
        edt_1d(
            &out[y * w..(y + 1) * w],
            row_tmp,
            &mut v[..w],
            &mut z[..w + 1],
        );
        out[y * w..(y + 1) * w].copy_from_slice(&row_tmp[..w]);
    }

    // Column pass
    for x in 0..w {
        for y in 0..h {
            col_src[y] = out[y * w + x];
        }
        edt_1d(col_src, col_dst, &mut v[..h], &mut z[..h + 1]);
        for y in 0..h {
            out[y * w + x] = col_dst[y];
        }
    }
}

// Convert the R channel of an RGBA cell buffer from raw coverage (0-255) to SDF.
// All buffers are caller-provided so no heap allocation happens per call.
// After conversion every channel holds the normalised distance in [0, 255]:
//   128 ≈ glyph outline, >128 = inside, <128 = outside.
pub(crate) fn cell_coverage_to_sdf(
    cell: &mut [u8],
    w: usize,
    h: usize,
    spread: f32,
    scratch: &mut SdfScratch,
) {
    const INF: f32 = 1e9;
    let n = w * h;
    let SdfScratch {
        inside_dist2,
        outside_dist2,
        edt,
    } = scratch;

    // Initialise EDT grids directly from coverage, skipping the bool_buf pass.
    for i in 0..n {
        let fg = cell[i * 4] > 127;
        inside_dist2[i] = if fg { 0.0 } else { INF };
        outside_dist2[i] = if fg { INF } else { 0.0 };
    }

    edt_2d(w, h, &mut inside_dist2[..n], edt);
    edt_2d(w, h, &mut outside_dist2[..n], edt);

    let spread2 = spread * spread;
    for i in 0..n {
        // For every pixel exactly one of inside_dist2/outside_dist2 is 0.0
        // (foreground pixels have inside_dist2=0; background have outside_dist2=0).
        // Avoid one sqrt unconditionally, and both sqrts for clamped pixels.
        let v = if inside_dist2[i] == 0.0 {
            let d2 = outside_dist2[i];
            if d2 >= spread2 {
                255
            } else {
                round(255.0 * (0.5 + 0.5 * sqrt(d2) / spread)) as u8
            }
        } else {
            let d2 = inside_dist2[i];
            if d2 >= spread2 {
                0
            } else {
                round(255.0 * (0.5 - 0.5 * sqrt(d2) / spread).max(0.0)) as u8
            }
        };
        cell[i * 4] = v;
        cell[i * 4 + 1] = v;
        cell[i * 4 + 2] = v;
        cell[i * 4 + 3] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn edt_1d_leaves_an_empty_row_untouched() {
        let mut v = [0usize; 1];
        let mut z = [0.0f32; 2];
        let mut d: [f32; 0] = [];
        edt_1d(&[], &mut d, &mut v, &mut z);
        // The scratch buffers are untouched: nothing was seeded.
        assert_eq!(z[0], 0.0);
    }

    #[test]
    fn edt_1d_measures_squared_distance_to_the_nearest_seed() {
        const INF: f32 = 1e9;
        // Seeds at index 0 and 4; interior samples take the nearer of the two.
        let f = [0.0, INF, INF, INF, 0.0];
        let mut d = [0.0f32; 5];
        let mut v = [0usize; 5];
        let mut z = [0.0f32; 6];
        edt_1d(&f, &mut d, &mut v, &mut z);
        assert_eq!(d, [0.0, 1.0, 4.0, 1.0, 0.0]);
    }

    #[test]
    fn cell_coverage_to_sdf_puts_the_outline_at_mid_grey() {
        // An 8x8 cell whose left half is covered: the field must fall across
        // the vertical edge, saturating away from it on both sides.
        let (w, h) = (8usize, 8usize);
        let mut cell = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..4 {
                cell[(y * w + x) * 4] = 255;
            }
        }
        let mut scratch = SdfScratch {
            inside_dist2: vec![0.0; w * h],
            outside_dist2: vec![0.0; w * h],
            edt: EdtScratch {
                v: vec![0; w.max(h)],
                z: vec![0.0; w.max(h) + 1],
                row_tmp: vec![0.0; w],
                col_src: vec![0.0; h],
                col_dst: vec![0.0; h],
            },
        };
        cell_coverage_to_sdf(&mut cell, w, h, 4.0, &mut scratch);

        let at = |x: usize, y: usize| cell[(y * w + x) * 4];
        // Inside stays above mid-grey, outside below, and the value decreases
        // monotonically left to right across the edge.
        assert!(at(0, 4) > 128, "deep inside: {}", at(0, 4));
        assert!(at(7, 4) < 128, "far outside: {}", at(7, 4));
        for x in 1..w {
            assert!(at(x, 4) <= at(x - 1, 4), "field rises at x={x}");
        }
        // All four channels carry the same value.
        let i = (4 * w + 4) * 4;
        assert_eq!(cell[i], cell[i + 1]);
        assert_eq!(cell[i], cell[i + 3]);
    }
}
