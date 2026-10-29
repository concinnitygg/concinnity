// One page of a planet's stored edits: a square grid of samples over one
// quadtree tile, each a height change and a detail weight, read bilinearly.

use alloc::vec;
use alloc::vec::Vec;
use concinnity_core::math::{SaturatingFrom, narrow_f32};

use super::super::tile::TileId;

/// Cells along each side of an edit page.
pub const PAGE_CELLS: usize = 32;

/// Samples along each side of an edit page: one more than its cells, so the
/// samples on a side are its neighbor's too and each page reads on its own.
pub const PAGE_SIDE: usize = PAGE_CELLS + 1;

/// Samples an edit page holds.
pub const PAGE_SAMPLES: usize = PAGE_SIDE * PAGE_SIDE;

// The largest quantized height change, either way.
const E_STEPS: f64 = 32_767.0;

/// The edits stored over one tile: per sample, a height change `e` in meters
/// and a detail weight `w` in `[0, 1]` the generated hills are scaled by,
/// row-major along the tile's `t`.
///
/// `e` is held as 16-bit steps of the page's own `scale` around its `offset`;
/// `w` in 8 bits, left out where it is 1 everywhere.
#[derive(Debug, Clone, PartialEq)]
pub struct EditPage {
    offset: f32,
    scale: f32,
    e: Vec<i16>,
    w: Option<Vec<u8>>,
}

/// An edit page read at one point: the height change and the detail weight,
/// and how each changes per unit of the page's coordinates along `s` and `t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageSample {
    /// The height change, in meters.
    pub e: f64,
    /// `e`'s change across the page along `s` and `t`.
    pub e_slope: [f64; 2],
    /// The detail weight, in `[0, 1]`.
    pub w: f64,
    /// `w`'s change across the page along `s` and `t`.
    pub w_slope: [f64; 2],
}

/// What a page holds at most and how steeply it changes: the figures a page
/// index carries, so a planet bounds its ground without reading the samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageFigures {
    /// The lowest height change, in meters.
    pub e_min: f32,
    /// The highest height change, in meters.
    pub e_max: f32,
    /// The largest change of `e` between neighboring samples, in meters.
    pub e_step: f32,
    /// The largest change of `w` between neighboring samples.
    pub w_step: f32,
}

impl EditPage {
    /// A page of height change `e` and detail weight `w` per sample, row-major
    /// along `t`, each [`PAGE_SAMPLES`] long, `e` quantized to the page's range
    /// and `w` clamped to `[0, 1]`. `None` when a list has the wrong length or
    /// a sample is not finite.
    pub fn from_samples(e: &[f64], w: &[f64]) -> Option<Self> {
        let finite = e.iter().chain(w).all(|v| v.is_finite());
        if e.len() != PAGE_SAMPLES || w.len() != PAGE_SAMPLES || !finite {
            return None;
        }
        let (lo, hi) = e
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let offset = narrow_f32((lo + hi) * 0.5);
        let scale = narrow_f32((hi - lo) * 0.5 / E_STEPS);
        let step = |v: f64| {
            if scale > 0.0 {
                let q = libm::round((v - f64::from(offset)) / f64::from(scale));
                i16::saturating_from(q.clamp(-E_STEPS, E_STEPS))
            } else {
                0
            }
        };
        let w: Vec<u8> = w
            .iter()
            .map(|&v| u8::saturating_from(libm::round(v.clamp(0.0, 1.0) * 255.0)))
            .collect();
        Some(Self {
            offset,
            scale,
            e: e.iter().map(|&v| step(v)).collect(),
            w: w.iter().any(|&v| v != u8::MAX).then_some(w),
        })
    }

    /// A page that changes nothing: no height change, full detail.
    pub fn neutral() -> Self {
        Self {
            offset: 0.0,
            scale: 0.0,
            e: vec![0; PAGE_SAMPLES],
            w: None,
        }
    }

    /// The page with this storage: `e`'s steps of `scale` around `offset`,
    /// and `w`'s if any. `None` when a list has the wrong length or the
    /// offset or scale is not finite.
    pub(super) fn from_parts(
        offset: f32,
        scale: f32,
        e: Vec<i16>,
        w: Option<Vec<u8>>,
    ) -> Option<Self> {
        let sized = e.len() == PAGE_SAMPLES && w.as_ref().is_none_or(|w| w.len() == PAGE_SAMPLES);
        (sized && offset.is_finite() && scale.is_finite() && scale >= 0.0).then_some(Self {
            offset,
            scale,
            e,
            w,
        })
    }

    /// The storage: `(offset, scale, e steps, w)`.
    pub(super) fn parts(&self) -> (f32, f32, &[i16], Option<&[u8]>) {
        (self.offset, self.scale, &self.e, self.w.as_deref())
    }

    /// Whether the page stores a detail weight.
    pub fn has_w(&self) -> bool {
        self.w.is_some()
    }

    /// The height change at sample `(i, j)`, in meters.
    pub fn e_at(&self, i: usize, j: usize) -> f64 {
        let q = self.e.get(j * PAGE_SIDE + i).copied().unwrap_or(0);
        f64::from(self.offset) + f64::from(self.scale) * f64::from(q)
    }

    /// The detail weight at sample `(i, j)`.
    pub fn w_at(&self, i: usize, j: usize) -> f64 {
        self.w
            .as_ref()
            .and_then(|w| w.get(j * PAGE_SIDE + i))
            .map_or(1.0, |&v| f64::from(v) / 255.0)
    }

    /// The page at `(fs, ft)` across it, each in `[0, 1]`: bilinear between
    /// the four samples around the point.
    pub fn sample(&self, fs: f64, ft: f64) -> PageSample {
        let n = PAGE_CELLS as f64;
        let (gs, gt) = (fs.clamp(0.0, 1.0) * n, ft.clamp(0.0, 1.0) * n);
        let (cs, ct) = (libm::floor(gs).min(n - 1.0), libm::floor(gt).min(n - 1.0));
        let (rs, rt) = (gs - cs, gt - ct);
        let (i, j) = (usize::saturating_from(cs), usize::saturating_from(ct));
        let bilinear = |at: &dyn Fn(usize, usize) -> f64| {
            let (a, b) = (at(i, j), at(i + 1, j));
            let (c, d) = (at(i, j + 1), at(i + 1, j + 1));
            let value = a + (b - a) * rs + (c - a) * rt + (a - b - c + d) * rs * rt;
            let along_s = ((b - a) + (a - b - c + d) * rt) * n;
            let along_t = ((c - a) + (a - b - c + d) * rs) * n;
            (value, [along_s, along_t])
        };
        let (e, e_slope) = bilinear(&|i, j| self.e_at(i, j));
        let (w, w_slope) = if self.w.is_some() {
            bilinear(&|i, j| self.w_at(i, j))
        } else {
            (1.0, [0.0; 2])
        };
        PageSample {
            e,
            e_slope,
            w,
            w_slope,
        }
    }

    /// The page's figures, from its samples as they read back.
    pub fn figures(&self) -> PageFigures {
        let mut f = PageFigures {
            e_min: f32::INFINITY,
            e_max: f32::NEG_INFINITY,
            e_step: 0.0,
            w_step: 0.0,
        };
        let step = |a: f64, b: f64| narrow_f32((a - b).abs());
        for j in 0..PAGE_SIDE {
            for i in 0..PAGE_SIDE {
                let e = self.e_at(i, j);
                f.e_min = f.e_min.min(narrow_f32(e));
                f.e_max = f.e_max.max(narrow_f32(e));
                for (a, b) in [(i + 1, j), (i, j + 1)] {
                    if a < PAGE_SIDE && b < PAGE_SIDE {
                        f.e_step = f.e_step.max(step(self.e_at(a, b), e));
                        f.w_step = f.w_step.max(step(self.w_at(a, b), self.w_at(i, j)));
                    }
                }
            }
        }
        f
    }

    /// The page over `fine`, a tile under `coarse`, sampled from this page,
    /// which lies over `coarse`. `None` unless `fine` lies under `coarse`.
    pub fn refine(&self, coarse: TileId, fine: TileId) -> Option<Self> {
        let depth = fine.level.checked_sub(coarse.level)?;
        if fine.ancestor_at(coarse.level) != Some(coarse) {
            return None;
        }
        let span = 1.0 / f64::from(1u32 << depth);
        let s0 = f64::from(fine.x - (coarse.x << depth)) * span;
        let t0 = f64::from(fine.y - (coarse.y << depth)) * span;
        let mut e = vec![0.0; PAGE_SAMPLES];
        let mut w = vec![1.0; PAGE_SAMPLES];
        let n = PAGE_CELLS as f64;
        for j in 0..PAGE_SIDE {
            for i in 0..PAGE_SIDE {
                let at = self.sample(s0 + i as f64 / n * span, t0 + j as f64 / n * span);
                e[j * PAGE_SIDE + i] = at.e;
                w[j * PAGE_SIDE + i] = at.w;
            }
        }
        Self::from_samples(&e, &w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::CubeFace;

    // A page whose height change is `e(fs, ft)` and detail weight `w(fs, ft)`.
    fn page(e: impl Fn(f64, f64) -> f64, w: impl Fn(f64, f64) -> f64) -> EditPage {
        let n = PAGE_CELLS as f64;
        let at = |k: usize| ((k % PAGE_SIDE) as f64 / n, (k / PAGE_SIDE) as f64 / n);
        let es: Vec<f64> = (0..PAGE_SAMPLES).map(|k| e(at(k).0, at(k).1)).collect();
        let ws: Vec<f64> = (0..PAGE_SAMPLES).map(|k| w(at(k).0, at(k).1)).collect();
        EditPage::from_samples(&es, &ws).unwrap()
    }

    #[test]
    fn samples_quantize_within_half_a_step() {
        let p = page(|s, t| 12.0 * s - 3.0 * t * t, |s, _| 1.0 - 0.5 * s);
        let step = 15.0 / 65_534.0;
        for j in 0..PAGE_SIDE {
            for i in 0..PAGE_SIDE {
                let (s, t) = (i as f64 / 32.0, j as f64 / 32.0);
                assert!((p.e_at(i, j) - (12.0 * s - 3.0 * t * t)).abs() <= step);
                assert!((p.w_at(i, j) - (1.0 - 0.5 * s)).abs() <= 0.5 / 255.0 + 1e-12);
            }
        }
        assert!(p.has_w());
        let full = page(|_, _| 2.0, |_, _| 1.0);
        assert!(!full.has_w());
        assert_eq!(full.e_at(5, 7), 2.0);
        assert!(EditPage::from_samples(&[0.0; 3], &[1.0; 3]).is_none());
    }

    // Between samples a page reads bilinearly, and its slopes are its own.
    #[test]
    fn a_page_reads_bilinearly_with_its_slopes() {
        let p = page(
            |s, t| 8.0 * s + 2.0 * t + 4.0 * s * t,
            |s, t| 0.5 + 0.25 * (s - t),
        );
        let h = 1e-6;
        for (fs, ft) in [
            (0.0, 0.0),
            (0.31, 0.77),
            (0.5, 0.5),
            (1.0, 1.0),
            (0.999, 0.01),
        ] {
            let a = p.sample(fs, ft);
            let want = 8.0 * fs + 2.0 * ft + 4.0 * fs * ft;
            assert!((a.e - want).abs() < 1e-3, "{fs} {ft}");
            let ds = (p.sample(fs + h, ft).e - p.sample(fs - h, ft).e) / (2.0 * h);
            let dt = (p.sample(fs, ft + h).e - p.sample(fs, ft - h).e) / (2.0 * h);
            if fs > h && fs < 1.0 - h && ft > h && ft < 1.0 - h {
                assert!((ds - a.e_slope[0]).abs() < 1e-3 && (dt - a.e_slope[1]).abs() < 1e-3);
            }
            assert!((a.w - (0.5 + 0.25 * (fs - ft))).abs() < 3e-3);
        }
    }

    #[test]
    fn the_figures_bound_the_samples() {
        let p = page(|s, t| 10.0 * s - 4.0 * t, |s, _| s);
        let f = p.figures();
        assert!((f.e_min + 4.0).abs() < 1e-3 && (f.e_max - 10.0).abs() < 1e-3);
        assert!((f.e_step - 10.0 / 32.0).abs() < 1e-3, "{}", f.e_step);
        assert!((f.w_step - 1.0 / 32.0).abs() < 2.0 / 255.0, "{}", f.w_step);
        let flat = EditPage::neutral().figures();
        assert_eq!(
            (flat.e_min, flat.e_max, flat.e_step, flat.w_step),
            (0.0, 0.0, 0.0, 0.0)
        );
    }

    // A page refined from a coarser one reads what the coarse page reads,
    // wherever on the fine tile, a level down or several.
    #[test]
    fn a_refined_page_reads_the_coarse_one() {
        let coarse = TileId {
            face: CubeFace(3),
            level: 5,
            x: 11,
            y: 20,
        };
        let p = page(|s, t| 30.0 * s * s - 7.0 * t, |s, t| 0.2 + 0.6 * s * t);
        for fine in [coarse.children()[2], coarse.children()[1].children()[3]] {
            let r = p.refine(coarse, fine).unwrap();
            let depth = fine.level - coarse.level;
            let span = 1.0 / f64::from(1u32 << depth);
            let (s0, t0) = (
                f64::from(fine.x - (coarse.x << depth)) * span,
                f64::from(fine.y - (coarse.y << depth)) * span,
            );
            for (fs, ft) in [(0.0, 0.0), (0.37, 0.61), (1.0, 0.5), (0.03, 0.97)] {
                let a = r.sample(fs, ft);
                let b = p.sample(s0 + fs * span, t0 + ft * span);
                assert!(
                    (a.e - b.e).abs() < 2e-3,
                    "{fine:?} {fs} {ft}: {} {}",
                    a.e,
                    b.e
                );
                assert!((a.w - b.w).abs() < 4e-3, "{fine:?} {fs} {ft}");
            }
        }
        let elsewhere = TileId {
            x: 3,
            ..coarse.children()[0]
        };
        assert!(p.refine(coarse, elsewhere).is_none());
        assert!(p.refine(coarse.children()[0], coarse).is_none());
    }
}
