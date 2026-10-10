//! How the field simplifies with distance from the camera: the strip each
//! blade draws with, how many blades survive, how a blade on its way out
//! shrinks into the ground instead of popping, how much wider the survivors
//! grow so the field keeps its coverage, and how many blades each strip's draw
//! can hold. The kernel and the draws in `grass.hlsl` read the same numbers
//! from [`GrassParams`](crate::render::uniforms::grass::GrassParams).
//!
//! Every blade carries a stable hash `u` in [0, 1). A blade survives while the
//! field's keep fraction at its distance exceeds `u * (1 - GRASS_SHRINK_BAND)`,
//! and is full height once the keep fraction clears that by the shrink band, so
//! as the camera backs away each blade sinks out of view over the band before
//! the kernel drops it.

use super::tiles::{GrassGrid, GroundRect, MAX_GRASS_BLADES};
use crate::math::ceil;

/// Strip detail levels: full, half and quarter segment counts.
pub const GRASS_LOD_COUNT: usize = 3;

/// Vertices in each level's strip: pairs up the blade plus the tip.
pub const GRASS_LOD_VERTICES: [u32; GRASS_LOD_COUNT] = [15, 9, 5];

/// Spacing of the levels' vertex ids: level `k`'s draw starts at vertex
/// `k * GRASS_LOD_VERTEX_STRIDE`, so the vertex stage reads its level back from
/// the id.
pub const GRASS_LOD_VERTEX_STRIDE: u32 = 16;

/// Distance from the camera, in meters, at which each coarser level starts.
pub const GRASS_LOD_DISTANCES: [f32; GRASS_LOD_COUNT - 1] = [12.0, 30.0];

/// The share of each level's reach, measured back from where the next level
/// starts, over which its extra vertices fold onto the next level's strip.
pub const GRASS_MORPH_FRACTION: f32 = 0.35;

/// Distance within which every blade is kept, in meters. Past it the field
/// thins with the reciprocal of distance.
pub const GRASS_FULL_DENSITY_DISTANCE: f32 = 10.0;

/// The fewest blades distance thinning keeps, as a fraction of the field.
pub const GRASS_MIN_KEEP: f32 = 0.25;

/// The span of the hash, as a fraction of it, over which a thinned-out blade
/// shrinks from full height to nothing.
pub const GRASS_SHRINK_BAND: f32 = 0.3;

/// The last share of the draw distance over which the field fades out and the
/// terrain's far-field tint fades in.
pub const GRASS_FADE_FRACTION: f32 = 0.25;

/// The detail level of a blade `distance` meters from the camera.
pub fn lod_at(distance: f32) -> usize {
    GRASS_LOD_DISTANCES
        .iter()
        .take_while(|&&start| distance >= start)
        .count()
}

/// How far toward the next level's strip level `lod`'s folding vertices have
/// moved at `distance`: 0 within the level, 1 where the next level starts. The
/// coarsest level never folds.
pub fn morph(lod: usize, distance: f32) -> f32 {
    let Some(&end) = GRASS_LOD_DISTANCES.get(lod) else {
        return 0.0;
    };
    let band = end * GRASS_MORPH_FRACTION;
    ((distance - (end - band)) / band).clamp(0.0, 1.0)
}

/// The fraction of the field distance thinning keeps at `distance`.
pub fn thinning(distance: f32) -> f32 {
    if distance <= GRASS_FULL_DENSITY_DISTANCE {
        return 1.0;
    }
    (GRASS_FULL_DENSITY_DISTANCE / distance).clamp(GRASS_MIN_KEEP, 1.0)
}

/// Where the field starts fading out for a field drawn to `draw_distance`.
pub fn fade_start(draw_distance: f32) -> f32 {
    draw_distance * (1.0 - GRASS_FADE_FRACTION)
}

/// How much of the field is left at `distance` as it fades out toward
/// `draw_distance`: 1 before the fade, 0 at its end. The terrain's far-field
/// tint is its complement.
pub fn fade(distance: f32, draw_distance: f32) -> f32 {
    let span = draw_distance - fade_start(draw_distance);
    ((draw_distance - distance) / span).clamp(0.0, 1.0)
}

/// The keep fraction at `distance`: thinning, then the fade at the end.
pub fn keep(distance: f32, draw_distance: f32) -> f32 {
    thinning(distance) * fade(distance, draw_distance)
}

/// The hash threshold under which a blade survives a keep fraction.
fn hash_span() -> f32 {
    1.0 - GRASS_SHRINK_BAND
}

/// Whether the blade with hash `u` survives keep fraction `keep`.
pub fn survives(keep: f32, u: f32) -> bool {
    u * hash_span() < keep
}

/// The height scale of the blade with hash `u` at keep fraction `keep`: 1
/// once the keep fraction clears the blade by the shrink band, falling to 0
/// where the blade is dropped.
pub fn shrink(keep: f32, u: f32) -> f32 {
    ((keep - u * hash_span()) / GRASS_SHRINK_BAND).clamp(0.0, 1.0)
}

/// The field's mean blade height scale at keep fraction `keep`, over every
/// candidate blade, dropped ones counted as 0: the coverage left relative to a
/// full field.
pub fn coverage(keep: f32) -> f32 {
    // The integral of shrink over the hash: with x = u * span uniform on
    // [0, span), it is the ramp clamp((keep - x) / band, 0, 1) integrated, as
    // the difference of its antiderivative at the two ends.
    if keep >= 1.0 {
        return 1.0;
    }
    let band = GRASS_SHRINK_BAND;
    let ramp = |t: f32| {
        if t <= 0.0 {
            0.0
        } else if t <= band {
            t * t / (2.0 * band)
        } else {
            t - 0.5 * band
        }
    };
    let span = hash_span();
    (ramp(keep) - ramp(keep - span)) / span
}

/// How much wider every surviving blade grows at `distance` to make up for
/// the blades thinning drops there.
pub fn width_scale(distance: f32) -> f32 {
    1.0 / coverage(thinning(distance))
}

/// The widest any blade grows: the scale where thinning bottoms out.
pub fn max_width_scale() -> f32 {
    1.0 / coverage(GRASS_MIN_KEEP)
}

/// The fraction of candidate blades alive at `distance`.
fn alive(distance: f32, draw_distance: f32) -> f32 {
    (keep(distance, draw_distance) / hash_span()).min(1.0)
}

/// Upper bounds, per level, on the blades one frame keeps on `ground` with a
/// field drawn to `draw_distance`, wherever the camera stands.
///
/// A blade lands in level `k` when its distance from the camera falls within
/// that level's reach. It then stands within the reach's outer radius of the
/// camera on the ground plane, and its distance is at least the reach's inner
/// radius, so the level's count is bounded by the alive fraction at
/// `max(r, inner)` integrated over the disc of the outer radius, and by the
/// whole ground at the inner radius's alive fraction. Each level adds a tile
/// of slack for the cells a disc boundary cuts.
pub fn lod_capacities(
    ground: &GroundRect,
    grid: &GrassGrid,
    draw_distance: f32,
) -> [u32; GRASS_LOD_COUNT] {
    // Ring width the disc is integrated in; each ring takes the alive fraction
    // at its inner edge, the largest within it.
    const RING: f32 = 0.25;
    let density = grid.density();
    core::array::from_fn(|lod| {
        let inner = if lod == 0 {
            0.0
        } else {
            GRASS_LOD_DISTANCES[lod - 1].min(draw_distance)
        };
        let outer = GRASS_LOD_DISTANCES
            .get(lod)
            .copied()
            .unwrap_or(draw_distance)
            .min(draw_distance);
        let mut disc = 0.0f32;
        let mut r = 0.0f32;
        while r < outer {
            let next = (r + RING).min(outer);
            let ring = core::f32::consts::PI * (next * next - r * r);
            disc += ring * alive(r.max(inner), draw_distance);
            r = next;
        }
        let whole = ground.area() * alive(inner, draw_distance);
        let blades = ceil(disc.min(whole) * density) + grid.blades_per_tile() as f32;
        if blades >= MAX_GRASS_BLADES as f32 {
            MAX_GRASS_BLADES
        } else {
            blades as u32
        }
    })
}

/// How many blades each level's region of the visible-blade buffer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GrassCapacity {
    /// Blades per level.
    pub lods: [u32; GRASS_LOD_COUNT],
}

impl GrassCapacity {
    /// The sum of `per_layer`'s bounds, scaled down together so the buffer
    /// never holds more than [`MAX_GRASS_BLADES`].
    pub fn sum(per_layer: impl IntoIterator<Item = [u32; GRASS_LOD_COUNT]>) -> Self {
        let mut sums = [0u64; GRASS_LOD_COUNT];
        for layer in per_layer {
            for (sum, n) in sums.iter_mut().zip(layer) {
                *sum += u64::from(n);
            }
        }
        let total: u64 = sums.iter().sum();
        let cap = u64::from(MAX_GRASS_BLADES);
        let lods = sums.map(|n| {
            let n = if total > cap { n * cap / total } else { n };
            n.max(1) as u32
        });
        Self { lods }
    }

    /// Blades in the whole buffer.
    pub fn total(&self) -> u32 {
        self.lods.iter().sum()
    }

    /// The first blade of each level's region, the regions laid end to end.
    pub fn bases(&self) -> [u32; GRASS_LOD_COUNT] {
        let mut base = 0;
        self.lods.map(|n| {
            let b = base;
            base += n;
            b
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::sqrt;
    use crate::render::grass::tiles::GRASS_DRAW_DISTANCE;
    use crate::render::{shader_consts, shaders};

    const D: f32 = GRASS_DRAW_DISTANCE;

    #[test]
    fn levels_coarsen_at_their_distances() {
        assert_eq!(lod_at(0.0), 0);
        assert_eq!(lod_at(GRASS_LOD_DISTANCES[0] - 0.01), 0);
        assert_eq!(lod_at(GRASS_LOD_DISTANCES[0]), 1);
        assert_eq!(lod_at(GRASS_LOD_DISTANCES[1] - 0.01), 1);
        assert_eq!(lod_at(GRASS_LOD_DISTANCES[1]), 2);
        assert_eq!(lod_at(1.0e6), GRASS_LOD_COUNT - 1);
    }

    // A level's folding vertices reach the next level's strip exactly where
    // the kernel hands the blade to that level, so the switch moves nothing.
    #[test]
    fn a_level_has_fully_folded_where_the_next_begins() {
        for (lod, &end) in GRASS_LOD_DISTANCES.iter().enumerate() {
            assert_eq!(morph(lod, 0.0), 0.0);
            assert_eq!(morph(lod, end * (1.0 - GRASS_MORPH_FRACTION)), 0.0);
            assert_eq!(morph(lod, end), 1.0);
            let mid = end * (1.0 - 0.5 * GRASS_MORPH_FRACTION);
            assert!((morph(lod, mid) - 0.5).abs() < 1e-5);
        }
        assert_eq!(morph(GRASS_LOD_COUNT - 1, 1.0e6), 0.0);
    }

    #[test]
    fn each_level_halves_the_segments() {
        // Pairs up the blade: every level keeps every other pair of the last.
        let pairs = GRASS_LOD_VERTICES.map(|v| (v - 1) / 2);
        assert_eq!(pairs, [7, 4, 2]);
        assert!(GRASS_LOD_VERTICES.iter().all(|&v| v % 2 == 1));
        assert!(
            GRASS_LOD_VERTICES
                .iter()
                .all(|&v| v <= GRASS_LOD_VERTEX_STRIDE)
        );
    }

    #[test]
    fn thinning_keeps_everything_near_and_bottoms_out_far() {
        assert_eq!(thinning(0.0), 1.0);
        assert_eq!(thinning(GRASS_FULL_DENSITY_DISTANCE), 1.0);
        assert!((thinning(2.0 * GRASS_FULL_DENSITY_DISTANCE) - 0.5).abs() < 1e-6);
        assert_eq!(thinning(1.0e5), GRASS_MIN_KEEP);
        let mut last = 1.0;
        for i in 0..200 {
            let k = thinning(i as f32 * 0.5);
            assert!(k <= last);
            last = k;
        }
    }

    #[test]
    fn the_field_fades_over_the_last_quarter() {
        assert_eq!(fade_start(D), 0.75 * D);
        assert_eq!(fade(0.0, D), 1.0);
        assert_eq!(fade(fade_start(D), D), 1.0);
        assert!((fade(0.875 * D, D) - 0.5).abs() < 1e-5);
        assert_eq!(fade(D, D), 0.0);
        assert_eq!(fade(2.0 * D, D), 0.0);
        assert_eq!(keep(D, D), 0.0);
    }

    // At full keep no blade shrinks or drops; as the keep fraction falls a
    // blade shrinks to nothing exactly where it stops surviving.
    #[test]
    fn a_blade_shrinks_away_before_it_is_dropped() {
        for i in 0..100 {
            let u = i as f32 / 100.0;
            assert!(survives(1.0, u));
            assert_eq!(shrink(1.0, u), 1.0);
            let gone = u * (1.0 - GRASS_SHRINK_BAND);
            assert!(!survives(gone, u));
            assert_eq!(shrink(gone, u), 0.0);
            if gone > 1e-3 {
                assert!(survives(gone + 1e-3, u));
                assert!(shrink(gone + 1e-3, u) < 0.01);
            }
        }
    }

    #[test]
    fn coverage_matches_the_mean_shrink_over_the_hash() {
        for &k in &[0.0, 0.05, 0.25, 0.3, 0.5, 0.69, 0.7, 0.9, 1.0] {
            let n = 20_000;
            let mean: f32 = (0..n)
                .map(|i| shrink(k, (i as f32 + 0.5) / n as f32))
                .sum::<f32>()
                / n as f32;
            assert!(
                (coverage(k) - mean).abs() < 1e-3,
                "keep {k}: {} vs {mean}",
                coverage(k)
            );
        }
        assert!((coverage(1.0) - 1.0).abs() < 1e-6);
        assert_eq!(coverage(0.0), 0.0);
    }

    // Widening the survivors by the coverage lost keeps the field's covered
    // area, blades times width times height, the same at every distance.
    #[test]
    fn widened_survivors_keep_the_field_coverage() {
        for i in 0..120 {
            let d = i as f32 * 0.5;
            let k = thinning(d);
            assert!((coverage(k) * width_scale(d) - 1.0).abs() < 1e-5);
        }
        assert_eq!(width_scale(0.0), 1.0);
        assert_eq!(coverage(1.0), 1.0);
        assert!(width_scale(1.0e5) <= max_width_scale());
        assert!((width_scale(1.0e5) - max_width_scale()).abs() < 1e-6);
    }

    fn field() -> (GroundRect, GrassGrid) {
        (
            GroundRect::centered([0.0, 0.0], [1000.0, 1000.0]),
            GrassGrid::for_density(100.0).unwrap(),
        )
    }

    // Each level's bound covers what a camera at the ground's height keeps in
    // that level's ring, counted cell by cell with the kernel's own rules.
    #[test]
    fn level_capacities_bound_what_the_kernel_keeps() {
        let (ground, grid) = field();
        let caps = lod_capacities(&ground, &grid, D);
        let mut kept = [0u32; GRASS_LOD_COUNT];
        let cell = grid.cell_size();
        let n = (D / cell) as i32 + 1;
        let mut h = 0x1234_5678u32;
        for z in -n..=n {
            for x in -n..=n {
                h = h.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
                let u = (h >> 8) as f32 / 16_777_216.0;
                let p = [(x as f32 + 0.5) * cell, (z as f32 + 0.5) * cell];
                let d = sqrt(p[0] * p[0] + p[1] * p[1]);
                if d < D && survives(keep(d, D), u) {
                    kept[lod_at(d)] += 1;
                }
            }
        }
        for lod in 0..GRASS_LOD_COUNT {
            assert!(
                kept[lod] <= caps[lod],
                "level {lod}: {} > {}",
                kept[lod],
                caps[lod]
            );
            // Loose, but not absurdly so for a camera on the ground.
            assert!(caps[lod] < 4 * kept[lod] + grid.blades_per_tile());
        }
    }

    // Seen from high above, every blade is at least the camera's height away,
    // so the coarse levels hold blades that stand right under the camera.
    #[test]
    fn level_capacities_bound_a_camera_high_above_the_field() {
        let (ground, grid) = field();
        let caps = lod_capacities(&ground, &grid, D);
        let cell = grid.cell_size();
        let n = (D / cell) as i32 + 1;
        for height in [5.0f32, 15.0, 25.0, 40.0] {
            // The blades each level keeps on average: every candidate in reach
            // weighted by the fraction alive at its distance.
            let mut kept = [0.0f32; GRASS_LOD_COUNT];
            for z in -n..=n {
                for x in -n..=n {
                    let p = [(x as f32 + 0.5) * cell, (z as f32 + 0.5) * cell];
                    let d = sqrt(p[0] * p[0] + p[1] * p[1] + height * height);
                    if d < D {
                        kept[lod_at(d)] += alive(d, D);
                    }
                }
            }
            for lod in 0..GRASS_LOD_COUNT {
                assert!(
                    kept[lod] <= caps[lod] as f32,
                    "height {height}, level {lod}: {} > {}",
                    kept[lod],
                    caps[lod]
                );
            }
        }
    }

    #[test]
    fn a_small_ground_bounds_every_level_by_its_area() {
        let grid = GrassGrid::for_density(100.0).unwrap();
        let small = GroundRect::centered([0.0, 0.0], [5.0, 5.0]);
        let caps = lod_capacities(&small, &grid, D);
        assert_eq!(caps[0], 100 * 100 + grid.blades_per_tile());
        for &c in &caps {
            assert!(c <= 100 * 100 + grid.blades_per_tile());
        }
    }

    // LOD'd capacity is well under the full draw disc the buffer used to be
    // sized for.
    #[test]
    fn thinning_shrinks_the_buffer() {
        let (ground, grid) = field();
        let caps = lod_capacities(&ground, &grid, D);
        let full = core::f32::consts::PI * D * D * grid.density();
        let total: u32 = caps.iter().sum();
        assert!((total as f32) < 0.75 * full, "{total} vs {full}");
    }

    #[test]
    fn capacities_sum_per_level_and_lay_regions_end_to_end() {
        let c = GrassCapacity::sum([[10, 20, 30], [1, 2, 3]]);
        assert_eq!(c.lods, [11, 22, 33]);
        assert_eq!(c.total(), 66);
        assert_eq!(c.bases(), [0, 11, 33]);
    }

    #[test]
    fn capacities_scale_down_to_the_buffer_cap() {
        let c = GrassCapacity::sum([[MAX_GRASS_BLADES, MAX_GRASS_BLADES, 2 * MAX_GRASS_BLADES]]);
        assert!(c.total() <= MAX_GRASS_BLADES);
        assert_eq!(c.lods[0], MAX_GRASS_BLADES / 4);
        assert_eq!(c.lods[2], MAX_GRASS_BLADES / 2);
        let empty = GrassCapacity::sum([]);
        assert_eq!(empty.lods, [1, 1, 1]);
    }

    #[test]
    fn the_draws_agree_on_the_vertex_spacing() {
        let src = shaders::GRASS;
        assert_eq!(
            shader_consts::uint(src, "GRASS_LOD_VERTEX_STRIDE"),
            GRASS_LOD_VERTEX_STRIDE as usize
        );
        assert_eq!(shader_consts::uint(src, "GRASS_LOD_COUNT"), GRASS_LOD_COUNT);
    }
}
