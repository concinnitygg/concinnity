//! An unnormalized 2D inverse FFT over a square power-of-two grid of complex
//! values, the transform `water_fft.hlsl` runs on the GPU: point `n` of a row
//! becomes `sum_f X[f] e^(2 pi i f n / N)`.

use alloc::vec::Vec;
use concinnity_core::math::{SaturatingFrom, narrow_f32};
use core::f64::consts::TAU;

/// A square grid of complex values, row-major, its real and imaginary parts
/// in separate planes so a pass along whole rows runs as plain vector
/// arithmetic.
#[derive(Debug, Clone)]
pub(super) struct Plane {
    /// The real parts.
    pub(super) re: Vec<f32>,
    /// The imaginary parts.
    pub(super) im: Vec<f32>,
}

impl Plane {
    /// A zeroed grid of `size` x `size`.
    pub(super) fn new(size: usize) -> Self {
        Self {
            re: alloc::vec![0.0; size * size],
            im: alloc::vec![0.0; size * size],
        }
    }

    /// Every value back to zero.
    pub(super) fn clear(&mut self) {
        self.re.fill(0.0);
        self.im.fill(0.0);
    }
}

/// The tables one grid size transforms with.
#[derive(Debug, Clone)]
pub(super) struct Ifft2 {
    size: usize,
    twiddles: Vec<[f32; 2]>,
    reversed: Vec<u32>,
}

impl Ifft2 {
    /// The transform of a `size` x `size` grid; `size` is rounded up to a
    /// power of two.
    pub(super) fn new(size: usize) -> Self {
        let size = size.max(2).next_power_of_two();
        let bits = size.trailing_zeros();
        let twiddles = (0..size / 2)
            .map(|m| {
                let a = TAU * m as f64 / size as f64;
                [narrow_f32(libm::cos(a)), narrow_f32(libm::sin(a))]
            })
            .collect();
        let reversed = (0..size)
            .map(|i| u32::saturating_from(i).reverse_bits() >> (32 - bits))
            .collect();
        Self {
            size,
            twiddles,
            reversed,
        }
    }

    /// Points along each side.
    #[cfg(test)]
    pub(super) fn size(&self) -> usize {
        self.size
    }

    /// Transform `grid` in place.
    #[cfg(test)]
    pub(super) fn inverse(&self, grid: &mut Plane) {
        self.inverse_columns(grid, self.size);
        self.transpose(grid);
        self.inverse_columns(grid, self.size);
        self.transpose(grid);
    }

    /// Transform the columns of `grid` in place, of which only the first
    /// and last `reach - 1` past the first hold anything (`size` for all).
    /// Each butterfly runs along whole rows, every column at once.
    pub(super) fn inverse_columns(&self, grid: &mut Plane, reach: usize) {
        let n = self.size;
        let spans = if 2 * reach >= n {
            [0..n, 0..0]
        } else {
            [0..reach, n + 1 - reach.max(1)..n]
        };
        for i in 0..n {
            let j = self.reversed[i] as usize;
            if j > i {
                for plane in [&mut grid.re, &mut grid.im] {
                    let (low, high) = plane.split_at_mut(j * n);
                    low[i * n..(i + 1) * n].swap_with_slice(&mut high[..n]);
                }
            }
        }
        let mut span = 1;
        while span < n {
            let step = n / (2 * span);
            for start in (0..n).step_by(2 * span) {
                for j in 0..span {
                    let w = self.twiddles[j * step];
                    let (a, b) = ((start + j) * n, (start + j + span) * n);
                    for columns in &spans {
                        butterflies(grid, a, b, columns.clone(), w);
                    }
                }
            }
            span *= 2;
        }
    }

    /// Swap `grid`'s rows and columns in place, in square blocks that stay
    /// in cache.
    pub(super) fn transpose(&self, grid: &mut Plane) {
        const BLOCK: usize = 8;
        let n = self.size;
        for plane in [&mut grid.re, &mut grid.im] {
            for by in (0..n).step_by(BLOCK) {
                for bx in (by..n).step_by(BLOCK) {
                    for y in by..(by + BLOCK).min(n) {
                        let from = if bx == by { y + 1 } else { bx };
                        for x in from..(bx + BLOCK).min(n) {
                            plane.swap(y * n + x, x * n + y);
                        }
                    }
                }
            }
        }
    }
}

// One butterfly between the rows starting at `a` and `b` (`a` < `b`) over
// `columns`, twiddle `w` on the second.
fn butterflies(
    grid: &mut Plane,
    a: usize,
    b: usize,
    columns: core::ops::Range<usize>,
    w: [f32; 2],
) {
    let (re_lo, re_hi) = grid.re.split_at_mut(b);
    let (im_lo, im_hi) = grid.im.split_at_mut(b);
    let ar = &mut re_lo[a + columns.start..a + columns.end];
    let ai = &mut im_lo[a + columns.start..a + columns.end];
    let br = &mut re_hi[columns.clone()];
    let bi = &mut im_hi[columns];
    for (((ar, ai), br), bi) in ar
        .iter_mut()
        .zip(ai.iter_mut())
        .zip(br.iter_mut())
        .zip(bi.iter_mut())
    {
        let tr = w[0] * *br - w[1] * *bi;
        let ti = w[0] * *bi + w[1] * *br;
        *br = *ar - tr;
        *bi = *ai - ti;
        *ar += tr;
        *ai += ti;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The transform agrees with the sum it stands for at every point.
    #[test]
    fn the_transform_is_the_inverse_sum() {
        let n = 8;
        let input: Vec<[f32; 2]> = (0..n * n)
            .map(|i| {
                let i = i as f32;
                [libm::sinf(1.3 * i + 0.2), libm::cosf(0.7 * i * i)]
            })
            .collect();
        let mut grid = Plane::new(n);
        for (i, v) in input.iter().enumerate() {
            grid.re[i] = v[0];
            grid.im[i] = v[1];
        }
        Ifft2::new(n).inverse(&mut grid);
        for y in 0..n {
            for x in 0..n {
                let mut want = [0.0f64; 2];
                for fy in 0..n {
                    for fx in 0..n {
                        let v = input[fy * n + fx];
                        let a = TAU * ((fx * x + fy * y) % n) as f64 / n as f64;
                        let (s, c) = (libm::sin(a), libm::cos(a));
                        want[0] += f64::from(v[0]) * c - f64::from(v[1]) * s;
                        want[1] += f64::from(v[0]) * s + f64::from(v[1]) * c;
                    }
                }
                let got = [grid.re[y * n + x], grid.im[y * n + x]];
                for i in 0..2 {
                    assert!((f64::from(got[i]) - want[i]).abs() < 1e-4, "{x} {y}");
                }
            }
        }
    }

    // Columns past the reach are left alone and the rest transform as if
    // the whole grid had.
    #[test]
    fn a_partial_reach_matches_the_whole() {
        let n = 16;
        let mut full = Plane::new(n);
        for y in 0..n {
            for x in (0..4).chain(n - 3..n) {
                full.re[y * n + x] = libm::sinf((y * 7 + x) as f32);
                full.im[y * n + x] = libm::cosf((y * 3 + x * x) as f32);
            }
        }
        let mut part = full.clone();
        let ifft = Ifft2::new(n);
        ifft.inverse_columns(&mut full, n);
        ifft.inverse_columns(&mut part, 4);
        for i in 0..n * n {
            assert!(
                (full.re[i] - part.re[i]).abs() < 1e-5 && (full.im[i] - part.im[i]).abs() < 1e-5
            );
        }
    }

    #[test]
    fn sizes_round_up_to_a_power_of_two() {
        assert_eq!(Ifft2::new(48).size(), 64);
        assert_eq!(Ifft2::new(64).size(), 64);
        assert_eq!(Ifft2::new(0).size(), 2);
    }
}
