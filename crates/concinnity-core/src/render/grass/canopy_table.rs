//! A grass layer's canopy view, tabulated: where view rays stop in the field
//! ([`canopy_view`]) over the view's angle and how much of the view the blades
//! stop, so the main pass reads it with one bilinear fetch rather than walking
//! the blades per pixel. Matches `canopy_add_view` in `shaders/canopy.hlsl`.
//!
//! The angle axis is even in the view's angle from the blades' up. The depth
//! axis is the optical depth the view crosses, spaced logarithmically: how
//! deep into the blades a ray gets keeps changing long after nearly every ray
//! is stopped.

use crate::math::{SaturatingFrom, acos, cos, exp, floor, ln};
use alloc::vec::Vec;

use super::canopy::{CanopyBand, CanopyView, canopy_view, projection};
use crate::render::uniforms::canopy::MAX_CANOPY_LAYERS;

/// View angles a layer's table holds, from straight down to grazing.
pub const CANOPY_TABLE_ANGLES: usize = 16;

/// Optical depths a layer's table holds, from bare ground to closed canopy.
pub const CANOPY_TABLE_DEPTHS: usize = 16;

/// `float4` rows in each cell of a layer's table.
pub const CANOPY_TABLE_ROWS: usize = 3;

/// `float4` rows in one layer's table.
pub const CANOPY_TABLE_LEN: usize = CANOPY_TABLE_ANGLES * CANOPY_TABLE_DEPTHS * CANOPY_TABLE_ROWS;

/// `float4` rows in every layer's table together, at the most layers.
pub const MAX_CANOPY_TABLE_ROWS: usize = MAX_CANOPY_LAYERS * CANOPY_TABLE_LEN;

/// The shallowest view the canopy is seen along, as the cosine from the
/// blades' up: grazing views are held there.
pub const CANOPY_MIN_MU: f32 = 0.02;

/// The optical depth the depth axis is spaced around: rows are even in
/// `ln(1 + depth / CANOPY_DEPTH_KNEE)`.
pub const CANOPY_DEPTH_KNEE: f32 = 0.25;

/// The deepest view a layer's table holds; deeper views read its last row.
pub const CANOPY_MAX_DEPTH: f32 = 400.0;

// The view angle from the blades' up at grazing.
fn max_angle() -> f32 {
    acos(CANOPY_MIN_MU)
}

// The view cosine at angle row `i`.
fn angle_mu(i: usize) -> f32 {
    cos(max_angle() * i as f32 / (CANOPY_TABLE_ANGLES - 1) as f32)
}

// The fractional angle row of view cosine `mu`.
fn angle_coord(mu: f32) -> f32 {
    (acos(mu.clamp(CANOPY_MIN_MU, 1.0)) / max_angle()).clamp(0.0, 1.0)
        * (CANOPY_TABLE_ANGLES - 1) as f32
}

// The log-spaced coordinate of optical depth `depth`, in [0, 1].
fn depth_unit(depth: f32) -> f32 {
    (ln(1.0 + depth.max(0.0) / CANOPY_DEPTH_KNEE) / ln(1.0 + CANOPY_MAX_DEPTH / CANOPY_DEPTH_KNEE))
        .clamp(0.0, 1.0)
}

// The optical depth at depth row `j`, kept off zero where the band means are
// undefined.
fn row_depth(j: usize) -> f32 {
    let u = j as f32 / (CANOPY_TABLE_DEPTHS - 1) as f32;
    let depth = CANOPY_DEPTH_KNEE * (exp(u * ln(1.0 + CANOPY_MAX_DEPTH / CANOPY_DEPTH_KNEE)) - 1.0);
    depth.max(1e-3)
}

// One cell: the upper band's share of the stopped rays and the projection,
// then both bands' means.
fn cell(view: &CanopyView, projection: f32) -> [[f32; 4]; CANOPY_TABLE_ROWS] {
    let stopped = (1.0 - view.ground).max(1e-9);
    let (u, l) = (&view.upper, &view.lower);
    [
        [u.share / stopped, projection, u.height, u.ramp],
        [u.occlusion, u.ramp_occlusion, u.above, l.above],
        [l.height, l.ramp, l.occlusion, l.ramp_occlusion],
    ]
}

/// The table of a layer bent by `bend` whose blades vary in height by
/// `height_variance`: angle rows of depth columns of [`CANOPY_TABLE_ROWS`]
/// rows each.
pub fn canopy_table(bend: f32, height_variance: f32) -> Vec<[f32; 4]> {
    let mut rows = Vec::with_capacity(CANOPY_TABLE_LEN);
    for i in 0..CANOPY_TABLE_ANGLES {
        let mu = angle_mu(i);
        let p = projection(bend, mu);
        for j in 0..CANOPY_TABLE_DEPTHS {
            let depth = row_depth(j);
            let view = canopy_view(bend, height_variance, mu, depth * mu / p.max(1e-9));
            rows.extend_from_slice(&cell(&view, p));
        }
    }
    rows
}

// Row `r` of the cell at fractional angle `x` and depth `y`, interpolated.
fn fetch(table: &[[f32; 4]], x: f32, y: f32, r: usize) -> [f32; 4] {
    let (i0, j0) = (
        usize::saturating_from(floor(x)),
        usize::saturating_from(floor(y)),
    );
    let (i1, j1) = (
        (i0 + 1).min(CANOPY_TABLE_ANGLES - 1),
        (j0 + 1).min(CANOPY_TABLE_DEPTHS - 1),
    );
    let (fx, fy) = (x - i0 as f32, y - j0 as f32);
    let at = |i: usize, j: usize| table[(i * CANOPY_TABLE_DEPTHS + j) * CANOPY_TABLE_ROWS + r];
    let lerp =
        |a: [f32; 4], b: [f32; 4], t: f32| core::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
    lerp(
        lerp(at(i0, j0), at(i1, j0), fx),
        lerp(at(i0, j1), at(i1, j1), fx),
        fy,
    )
}

/// The projection of `table`'s layer along view cosine `mu`, as
/// [`projection`] gives it.
pub fn table_projection(table: &[[f32; 4]], mu: f32) -> f32 {
    fetch(table, angle_coord(mu), 0.0, 0)[1]
}

/// Where rays seen along `mu` stop in a field of `leaf_area` of `table`'s
/// layer, as [`canopy_view`] gives it.
pub fn table_view(table: &[[f32; 4]], mu: f32, leaf_area: f32) -> CanopyView {
    let mu = mu.max(CANOPY_MIN_MU);
    let x = angle_coord(mu);
    let depth = fetch(table, x, 0.0, 0)[1] * leaf_area / mu;
    let stopped = 1.0 - exp(-depth);
    let y = depth_unit(depth) * (CANOPY_TABLE_DEPTHS - 1) as f32;
    let [r0, r1, r2] = core::array::from_fn(|r| fetch(table, x, y, r));
    let upper = r0[0].clamp(0.0, 1.0) * stopped;
    CanopyView {
        ground: 1.0 - stopped,
        upper: CanopyBand {
            share: upper,
            height: r0[2],
            ramp: r0[3],
            occlusion: r1[0],
            ramp_occlusion: r1[1],
            above: r1[2],
        },
        lower: CanopyBand {
            share: stopped - upper,
            height: r2[0],
            ramp: r2[1],
            occlusion: r2[2],
            ramp_occlusion: r2[3],
            above: r1[3],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band_terms(b: &CanopyBand) -> [f32; 5] {
        [b.height, b.ramp, b.occlusion, b.ramp_occlusion, b.above]
    }

    #[test]
    fn the_table_shape_is_the_shaders() {
        let src = crate::render::shaders::CANOPY;
        let uint = |name| crate::render::shader_consts::uint(src, name);
        assert_eq!(uint("CANOPY_TABLE_ANGLES"), CANOPY_TABLE_ANGLES);
        assert_eq!(uint("CANOPY_TABLE_DEPTHS"), CANOPY_TABLE_DEPTHS);
        assert_eq!(uint("CANOPY_TABLE_ROWS"), CANOPY_TABLE_ROWS);
    }

    #[test]
    fn axes_round_trip() {
        for i in 0..CANOPY_TABLE_ANGLES {
            assert!((angle_coord(angle_mu(i)) - i as f32).abs() < 1e-3);
        }
        assert_eq!(angle_mu(0), 1.0);
        assert!((angle_mu(CANOPY_TABLE_ANGLES - 1) - CANOPY_MIN_MU).abs() < 1e-6);
        assert_eq!(angle_coord(1.0), 0.0);
        for j in 1..CANOPY_TABLE_DEPTHS {
            let y = depth_unit(row_depth(j)) * (CANOPY_TABLE_DEPTHS - 1) as f32;
            assert!((y - j as f32).abs() < 1e-3);
        }
    }

    #[test]
    fn table_has_one_cell_per_angle_and_depth() {
        assert_eq!(canopy_table(0.4, 0.45).len(), CANOPY_TABLE_LEN);
    }

    #[test]
    fn table_reproduces_the_view_between_its_rows() {
        // Worst ground, band share and band mean errors.
        let mut worst = [0.0f32; 3];
        for (bend, variance) in [(0.325, 0.0), (0.4, 0.45), (0.7, 0.9)] {
            let table = canopy_table(bend, variance);
            for k in 0..23 {
                let mu = CANOPY_MIN_MU + (1.0 - CANOPY_MIN_MU) * (k as f32 + 0.37) / 23.0;
                let p = projection(bend, mu);
                assert!((table_projection(&table, mu) - p).abs() < 0.01 * p);
                for leaf_area in [0.05, 0.4, 1.3, 3.0, 9.0] {
                    let want = canopy_view(bend, variance, mu, leaf_area);
                    let got = table_view(&table, mu, leaf_area);
                    worst[0] = worst[0].max((got.ground - want.ground).abs());
                    for (g, w) in [(&got.upper, &want.upper), (&got.lower, &want.lower)] {
                        worst[1] = worst[1].max((g.share - w.share).abs());
                        // A band holding next to no rays adds next to nothing.
                        if w.share > 0.02 {
                            for (a, b) in band_terms(g).into_iter().zip(band_terms(w)) {
                                worst[2] = worst[2].max((a - b).abs());
                            }
                        }
                    }
                }
            }
        }
        assert!(worst[0] < 0.002, "{worst:?}");
        assert!(worst[1] < 0.01, "{worst:?}");
        assert!(worst[2] < 0.01, "{worst:?}");
    }

    #[test]
    fn bare_and_closed_ends() {
        let table = canopy_table(0.4, 0.45);
        let bare = table_view(&table, 0.5, 0.0);
        assert_eq!(bare.ground, 1.0);
        assert_eq!(bare.upper.share + bare.lower.share, 0.0);
        let closed = table_view(&table, 0.5, 1e4);
        assert!(closed.ground < 1e-6);
        assert!((closed.upper.share + closed.lower.share - 1.0).abs() < 1e-5);
    }
}
