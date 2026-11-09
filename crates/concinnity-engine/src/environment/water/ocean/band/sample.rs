//! Reading the band's grid between its points: the height by bicubic Hermite
//! from the exact gradient (smooth, and its slope the derivative of the
//! height it gives), everything else bilinearly.

use super::BandPoint;
use concinnity_core::math::SaturatingFrom;

/// The band's waves at one place and instant, along the tile's two axes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct BandSample {
    /// The rise above rest, in meters.
    pub height: f32,
    /// The rise per meter along each axis.
    pub slope: [f32; 2],
    /// How far the water at rest here moves along each axis, in meters.
    pub shift: [f32; 2],
    /// The height's rate of change, in meters per second.
    pub rise_rate: f32,
    /// The shift's rate of change along each axis, in meters per second.
    pub shift_rate: [f32; 2],
}

impl BandSample {
    /// `self` blended toward `other` by `t`.
    #[must_use]
    pub(crate) fn lerp(&self, other: &Self, t: f32) -> Self {
        let l = |a: f32, b: f32| a + (b - a) * t;
        Self {
            height: l(self.height, other.height),
            slope: [
                l(self.slope[0], other.slope[0]),
                l(self.slope[1], other.slope[1]),
            ],
            shift: [
                l(self.shift[0], other.shift[0]),
                l(self.shift[1], other.shift[1]),
            ],
            rise_rate: l(self.rise_rate, other.rise_rate),
            shift_rate: [
                l(self.shift_rate[0], other.shift_rate[0]),
                l(self.shift_rate[1], other.shift_rate[1]),
            ],
        }
    }
}

// The cubic Hermite basis at `t` and its derivative: value at 0, slope at 0,
// value at 1, slope at 1.
fn hermite(t: f32) -> ([f32; 4], [f32; 4]) {
    let t2 = t * t;
    let t3 = t2 * t;
    (
        [
            2.0 * t3 - 3.0 * t2 + 1.0,
            t3 - 2.0 * t2 + t,
            3.0 * t2 - 2.0 * t3,
            t3 - t2,
        ],
        [
            6.0 * t2 - 6.0 * t,
            3.0 * t2 - 4.0 * t + 1.0,
            6.0 * t - 6.0 * t2,
            3.0 * t2 - 2.0 * t,
        ],
    )
}

// The four points around `p` texels along a `size`-point grid, wrapping,
// and `p`'s place between them.
fn cell(size: usize, p: [f64; 2]) -> ([usize; 4], [f32; 2]) {
    let n = size as f64;
    let mut i = [0usize; 2];
    let mut f = [0.0f32; 2];
    for a in 0..2 {
        let x = p[a] - n * libm::floor(p[a] / n);
        let c = libm::floor(x);
        f[a] = narrow(x - c);
        i[a] = usize::saturating_from(c).min(size - 1);
    }
    // Point (x, y) sits at `x * size + y`.
    let (x1, y1) = ((i[0] + 1) % size, (i[1] + 1) % size);
    (
        [
            i[0] * size + i[1],
            x1 * size + i[1],
            i[0] * size + y1,
            x1 * size + y1,
        ],
        f,
    )
}

fn bilinear(f: [f32; 2]) -> [f32; 4] {
    [
        (1.0 - f[0]) * (1.0 - f[1]),
        f[0] * (1.0 - f[1]),
        (1.0 - f[0]) * f[1],
        f[0] * f[1],
    ]
}

// The grid of `size` points a side at `p` texels along its axes, wrapping,
// whose texels are `texel` meters across.
pub(super) fn read(points: &[BandPoint], size: usize, p: [f64; 2], texel: f64) -> BandSample {
    let (corners, f) = cell(size, p);
    let (hx, dhx) = hermite(f[0]);
    let (hy, dhy) = hermite(f[1]);
    let w = bilinear(f);
    let t = narrow(texel);
    let mut s = BandSample::default();
    let (mut du, mut dv) = (0.0, 0.0);
    for (i, &at) in corners.iter().enumerate() {
        let p = &points[at];
        let (ix, iy) = (2 * (i & 1), 2 * (i >> 1));
        let (gx, gy) = (p.gradient[0] * t, p.gradient[1] * t);
        s.height +=
            p.height * hx[ix] * hy[iy] + gx * hx[ix + 1] * hy[iy] + gy * hx[ix] * hy[iy + 1];
        du += p.height * dhx[ix] * hy[iy] + gx * dhx[ix + 1] * hy[iy] + gy * dhx[ix] * hy[iy + 1];
        dv += p.height * hx[ix] * dhy[iy] + gx * hx[ix + 1] * dhy[iy] + gy * hx[ix] * dhy[iy + 1];
        s.shift[0] += p.shift[0] * w[i];
        s.shift[1] += p.shift[1] * w[i];
        s.rise_rate += p.rise_rate * w[i];
        s.shift_rate[0] += p.shift_rate[0] * w[i];
        s.shift_rate[1] += p.shift_rate[1] * w[i];
    }
    s.slope = [du / t, dv / t];
    s
}

// Only the height of `read`.
pub(super) fn height(points: &[BandPoint], size: usize, p: [f64; 2], texel: f64) -> f32 {
    let (corners, f) = cell(size, p);
    let (hx, _) = hermite(f[0]);
    let (hy, _) = hermite(f[1]);
    let t = narrow(texel);
    let mut h = 0.0;
    for (i, &at) in corners.iter().enumerate() {
        let p = &points[at];
        let (ix, iy) = (2 * (i & 1), 2 * (i >> 1));
        h += p.height * hx[ix] * hy[iy]
            + p.gradient[0] * t * hx[ix + 1] * hy[iy]
            + p.gradient[1] * t * hx[ix] * hy[iy + 1];
    }
    h
}

// Only the horizontal displacement of `read`.
pub(super) fn shift(points: &[BandPoint], size: usize, p: [f64; 2]) -> [f32; 2] {
    let (corners, f) = cell(size, p);
    let w = bilinear(f);
    let mut out = [0.0; 2];
    for (i, &at) in corners.iter().enumerate() {
        out[0] += points[at].shift[0] * w[i];
        out[1] += points[at].shift[1] * w[i];
    }
    out
}

fn narrow(x: f64) -> f32 {
    concinnity_core::math::narrow_f32(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hermite interpolation through exact gradients reproduces a gentle
    // wave closely between the points, slope included.
    #[test]
    fn the_height_follows_a_wave_between_the_points() {
        let size = 16;
        let texel = 2.0;
        let k = core::f32::consts::TAU / (8.0 * narrow(texel));
        let points: alloc::vec::Vec<BandPoint> = (0..size * size)
            .map(|i| {
                let x = (i / size) as f32 * narrow(texel);
                BandPoint {
                    height: libm::sinf(k * x),
                    gradient: [k * libm::cosf(k * x), 0.0],
                    ..Default::default()
                }
            })
            .collect();
        for step in 0..40 {
            let x = f64::from(step) * 0.37;
            let s = read(&points, size, [x, 3.3], texel);
            let xm = narrow(x * texel);
            assert!((s.height - libm::sinf(k * xm)).abs() < 2e-3, "{x}");
            assert!((s.slope[0] - k * libm::cosf(k * xm)).abs() < 5e-3, "{x}");
            assert!(s.slope[1].abs() < 1e-6);
        }
    }
}
