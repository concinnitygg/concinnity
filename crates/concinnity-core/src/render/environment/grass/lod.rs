//! How the field simplifies with distance from the camera, by the size its
//! blades take on screen: the strip each blade draws with, how many blades
//! survive, how a blade on its way out narrows away instead of popping, how
//! much wider the survivors grow so the field keeps its coverage, where the
//! blades hand the field to the canopy and end, and how many blades each
//! region of the visible-blade buffer holds. The kernel and the draws in
//! `grass.hlsl` read the same numbers from
//! [`GrassParams`](crate::render::uniforms::grass::GrassParams), and the far
//! field's canopy reads the same band from
//! [`CanopyParams`](crate::render::uniforms::canopy::CanopyParams).
//!
//! Every distance is measured in what the view resolves there: a blade `h`
//! meters tall `d` meters away stands `h * focal / d` pixels tall. Blades end
//! where they shrink to `GRASS_END_PIXELS`, coarser strips start where they
//! stand `GRASS_LOD_PIXELS` tall, and the field thins so its survivors,
//! widened to keep its coverage, stand `GRASS_BLADE_PIXELS` wide: at a
//! grazing view, where the blades stand side by side, about one blade shows
//! per that many pixels across at every distance.
//!
//! Where a view ray meets the ground at a grazing angle it crosses many blades
//! deep, and only the first few show. Past where the blades stand the coarsest
//! strip's height, the field also thins so a ray crosses at most
//! `GRASS_DEPTH_LIMIT` blade layers, without widening the survivors (see
//! [`GrassLodView::keep`]).
//!
//! Every blade carries a stable hash `u` in [0, 1). A blade survives while the
//! field's keep fraction at its distance exceeds `u * (1 - GRASS_SHRINK_BAND)`,
//! and is full width once the keep fraction clears that by the shrink band's
//! share of the keep fraction, so as the camera backs away each blade narrows
//! out of view before the kernel drops it, while where the keep fraction holds
//! steady most survivors stand at full width. A blade's height is never
//! scaled: the heights the field shows, which the canopy past the blades
//! models, are the same at every distance.
//!
//! Over the last stretch of their reach the blades hand the field to the
//! canopy pixel by pixel: a pixel there shows either the whole blade field or
//! the canopy, chosen by a screen-space hash against [`GrassLodView::fade`],
//! so the mean of the two is exact at every distance and the temporal
//! antialiasing resolves the mix. The blades there go to a region of their own
//! (the band, [`GRASS_BAND_REGION`]), drawn by the only pipelines that discard.
//!
//! The visible-blade buffer is sized for the views of a render target (see
//! [`GrassSizing`]): what any of them can keep in each region, inside the cone
//! of its corners, from any height.

use crate::math::{SaturatingFrom, acos, asin, atan2, ceil, cos, powf, sin, sqrt, tan};
use alloc::vec::Vec;

use super::nest::GRASS_NEST_LEVELS;
use super::tiles::{GrassGrid, GroundRect, MAX_GRASS_BLADES};
use crate::render::uniforms::grass::MAX_GRASS_TILE_RECORDS;

/// Strip detail levels: full, half and quarter segment counts.
pub const GRASS_LOD_COUNT: usize = 3;

/// Vertices in each level's strip: pairs up the blade plus the tip.
pub const GRASS_LOD_VERTICES: [u32; GRASS_LOD_COUNT] = [15, 9, 5];

/// Spacing of the regions' vertex ids: region `k`'s draw starts at vertex
/// `k * GRASS_LOD_VERTEX_STRIDE`, so the vertex stage reads its region (and
/// so its level) back from the id.
pub const GRASS_LOD_VERTEX_STRIDE: u32 = 16;

/// Regions of the visible-blade buffer, one draw each: one per detail level,
/// then the band where the blades hand the field to the canopy.
pub const GRASS_REGION_COUNT: usize = GRASS_LOD_COUNT + 1;

/// The region of the blades handing the field to the canopy, drawn with the
/// coarsest strip.
pub const GRASS_BAND_REGION: usize = GRASS_LOD_COUNT;

/// Spans of the visible-blade buffer: one for each finer level's region, then
/// one the coarsest level's region fills from its start and the band's from
/// its end. Those two fill most at different camera heights (the band from
/// above), so sharing a span holds every frame in far fewer blades than a span
/// each.
pub const GRASS_SPAN_COUNT: usize = GRASS_LOD_COUNT;

/// The strip level region `region`'s draw uses: its own, the coarsest for the
/// band.
pub const fn region_level(region: usize) -> usize {
    if region < GRASS_LOD_COUNT {
        region
    } else {
        GRASS_LOD_COUNT - 1
    }
}

/// How tall on screen, in pixels, a blade of the field's height stands where
/// each coarser level starts.
const GRASS_LOD_PIXELS: [f32; GRASS_LOD_COUNT - 1] = [60.0, 24.0];

/// How tall on screen, in pixels, a blade of the field's height stands where
/// the blades end and the canopy alone shows the field.
const GRASS_END_PIXELS: f32 = 1.5;

/// How wide on screen, in pixels, the field's surviving blades stand once
/// widened for the blades thinning dropped.
const GRASS_BLADE_PIXELS: f32 = 4.0;

/// The narrowest vertical field of view, in radians, the visible-blade buffer
/// is sized for at a view's height: a narrower one thins as this one does
/// (see [`GrassSizing`]).
pub const GRASS_SIZING_FOV: f32 = core::f32::consts::FRAC_PI_3;

/// The share of each level's reach, measured back from where the next level
/// starts, over which its extra vertices fold onto the next level's strip.
pub const GRASS_MORPH_FRACTION: f32 = 0.35;

/// The share of the keep fraction over which a thinned-out blade narrows from
/// full width to nothing.
pub const GRASS_SHRINK_BAND: f32 = 0.3;

/// The most blade layers a view ray crossing the field at a grazing angle meets
/// before the field thins further: past it the ray would see only blade tips,
/// so deeper blades add vertices but nothing to the picture.
const GRASS_DEPTH_LIMIT: f32 = 16.0;

/// The last share of the blades' reach over which they hand the field to the
/// canopy, pixel by pixel.
const GRASS_FADE_FRACTION: f32 = 0.25;

/// The vertical focal length, in pixels, of a view `height` pixels tall with
/// vertical field of view `fov_y` radians: how many pixels one unit of height
/// spans one unit away.
pub fn focal_length(height: u32, fov_y: f32) -> f32 {
    0.5 * height as f32 / tan(0.5 * fov_y).max(1e-6)
}

/// How a view of focal length `focal` pixels resolves a field whose tallest
/// layer's blades stand `blade_height` meters tall on average.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassLodView {
    /// The focal length the field thins for, in pixels: the view's, at most
    /// the one the visible-blade buffer is sized for.
    pub focal: f32,
    /// Where the blades end, in meters from the camera.
    pub extent: f32,
    /// Where each coarser strip starts, in meters from the camera.
    pub lod_distances: [f32; GRASS_LOD_COUNT - 1],
}

impl GrassLodView {
    /// The view of focal length `focal` pixels over blades `blade_height`
    /// meters tall.
    pub fn new(focal: f32, blade_height: f32) -> Self {
        let focal = focal.max(1.0);
        let height = blade_height.max(1e-3);
        let extent = height * focal / GRASS_END_PIXELS;
        Self {
            focal,
            extent,
            lod_distances: GRASS_LOD_PIXELS.map(|px| (height * focal / px).min(extent)),
        }
    }

    /// Thinning's scale: the field keeps `width * keep_scale / distance` of
    /// blades `width` meters wide, at most all of them.
    pub fn keep_scale(&self) -> f32 {
        self.focal / GRASS_BLADE_PIXELS
    }

    /// The fraction of a field of blades `width` meters wide that distance
    /// thinning keeps at `distance`.
    pub fn thinning(&self, width: f32, distance: f32) -> f32 {
        (width * self.keep_scale() / distance.max(1e-6)).min(1.0)
    }

    /// The fraction of a layer's blades, `width` meters wide and `depth`
    /// blade layers deep (width x density x height, see [`blade_depth`]),
    /// kept at `distance` where the view ray meets the ground at an angle
    /// whose sine is `facing`: thinning's, lowered where the ray would cross
    /// more than `GRASS_DEPTH_LIMIT` layers. The survivors keep thinning's
    /// widening, so the field stays opaque along the ray and only its deepest
    /// layers go. Nearer than [`Self::depth_onset`] the limit grows with the
    /// square of the shortfall, so the near field keeps its depth.
    pub fn keep(&self, width: f32, depth: f32, distance: f32, facing: f32) -> f32 {
        let thin = self.thinning(width, distance);
        let onset = self.depth_onset() / distance.max(1e-6);
        let limit = GRASS_DEPTH_LIMIT * (onset * onset).max(1.0);
        let layers = depth / facing.abs().max(1e-4);
        thin * (limit / layers.max(1e-6)).min(1.0)
    }

    /// Where the depth limit takes full effect: where the field's blades
    /// stand the coarsest strip's `GRASS_LOD_PIXELS` tall.
    pub fn depth_onset(&self) -> f32 {
        self.lod_distances[GRASS_LOD_COUNT - 2]
    }

    /// Where the blades start handing the field to the canopy.
    pub fn fade_start(&self) -> f32 {
        self.extent * (1.0 - GRASS_FADE_FRACTION)
    }

    /// The share of pixels `distance` meters away that show the blades rather
    /// than the canopy: 1 before the hand-off, falling evenly to 0 where the
    /// blades end.
    pub fn fade(&self, distance: f32) -> f32 {
        let span = (self.extent - self.fade_start()).max(1e-3);
        ((self.extent - distance) / span).clamp(0.0, 1.0)
    }

    /// How much wider every surviving blade `width` meters wide grows at
    /// `distance` to make up for the blades thinning drops there.
    #[cfg(test)]
    fn width_scale(&self, width: f32, distance: f32) -> f32 {
        1.0 / coverage(self.thinning(width, distance))
    }

    /// The detail level of a blade `distance` meters from the camera.
    #[cfg(test)]
    fn lod_at(&self, distance: f32) -> usize {
        self.lod_distances
            .iter()
            .take_while(|&&start| distance >= start)
            .count()
    }

    /// The region of the visible-blade buffer a blade `distance` meters from
    /// the camera lands in: its detail level's, or the band past where the
    /// blades start handing over.
    #[cfg(test)]
    fn region_at(&self, distance: f32) -> usize {
        if distance >= self.fade_start() {
            GRASS_BAND_REGION
        } else {
            self.lod_at(distance)
        }
    }

    // The distances region `region` spans, `[inner, outer)`.
    fn region_span(&self, region: usize) -> [f32; 2] {
        let start = self.fade_start();
        if region == GRASS_BAND_REGION {
            return [start, self.extent];
        }
        let inner = if region == 0 {
            0.0
        } else {
            self.lod_distances[region - 1]
        };
        let outer = self.lod_distances.get(region).copied().unwrap_or(start);
        [inner.min(start), outer.min(start)]
    }

    /// How far toward the next level's strip level `lod`'s folding vertices
    /// have moved at `distance`: 0 within the level, 1 where the next level
    /// starts. The coarsest level never folds.
    pub fn morph(&self, lod: usize, distance: f32) -> f32 {
        let Some(&end) = self.lod_distances.get(lod) else {
            return 0.0;
        };
        let band = (end * GRASS_MORPH_FRACTION).max(1e-6);
        ((distance - (end - band)) / band).clamp(0.0, 1.0)
    }

    // The fraction of candidate blades of `width` meters alive at `distance`.
    fn alive(&self, width: f32, distance: f32) -> f32 {
        if distance > self.extent {
            return 0.0;
        }
        (self.thinning(width, distance) / hash_span()).min(1.0)
    }
}

/// The widest any blade `width` meters wide grows, in any view, in a field
/// whose tallest layer stands `blade_height` meters tall: the scale at the
/// blades' end in a view whose end is set by `GRASS_END_PIXELS`, where
/// thinning keeps the least.
pub fn max_width_scale(width: f32, blade_height: f32) -> f32 {
    let fewest = width * GRASS_END_PIXELS / (GRASS_BLADE_PIXELS * blade_height.max(1e-3));
    1.0 / coverage(fewest.min(1.0))
}

/// The blade layers a view ray straight down crosses in a layer of blades
/// `width` meters wide and `height` tall, `density` per square meter: its
/// leaf area over the ground. A ray meeting the ground at an angle whose sine
/// is `s` crosses `1 / s` times as many.
pub fn blade_depth(width: f32, density: f32, height: f32) -> f32 {
    width * density * height
}

/// The hash threshold under which a blade survives a keep fraction.
fn hash_span() -> f32 {
    1.0 - GRASS_SHRINK_BAND
}

/// Whether the blade with hash `u` survives keep fraction `keep`.
pub fn survives(keep: f32, u: f32) -> bool {
    u * hash_span() < keep
}

// The span of the keep fraction over which a blade shrinks at keep `keep`.
fn shrink_band(keep: f32) -> f32 {
    GRASS_SHRINK_BAND * keep
}

/// The width scale of the blade with hash `u` at keep fraction `keep`: 1
/// once the keep fraction clears the blade by the shrink band, falling to 0
/// where the blade is dropped.
pub fn shrink(keep: f32, u: f32) -> f32 {
    ((keep - u * hash_span()) / shrink_band(keep).max(1e-6)).clamp(0.0, 1.0)
}

/// The field's mean blade width scale at keep fraction `keep`, over every
/// candidate blade, dropped ones counted as 0: the coverage left relative to a
/// full field.
pub fn coverage(keep: f32) -> f32 {
    // The integral of shrink over the hash: with x = u * span uniform on
    // [0, span), it is the ramp clamp((keep - x) / band, 0, 1) integrated, as
    // the difference of its antiderivative at the two ends.
    if keep >= 1.0 {
        return 1.0;
    }
    let band = shrink_band(keep).max(1e-6);
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

/// The field a layer's capacity is sized for: its ground, its cells, its
/// blades' width, and the mean height of the field's tallest layer, which the
/// view's distances follow.
#[derive(Debug, Clone, Copy)]
pub struct CapacityField<'a> {
    /// The ground the layer grows on.
    pub ground: &'a GroundRect,
    /// The layer's cells.
    pub grid: &'a GrassGrid,
    /// The layer's blade width, in meters.
    pub width: f32,
    /// The field's reference blade height, in meters.
    pub blade_height: f32,
}

/// The views a layer's capacity holds: render targets `size` pixels across
/// and down, with any focal length up to `focal`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapacityViews {
    /// Render target width and height, in pixels.
    pub size: [u32; 2],
    /// The longest focal length, in pixels.
    pub focal: f32,
}

impl CapacityViews {
    // The focal lengths capacities are taken over: every view keeps at most
    // what the largest of these does in each region, to within the sampling,
    // from the longest focal length down by halves of a half-octave.
    fn focals(&self) -> impl Iterator<Item = f32> + '_ {
        (0..24).map(|i| self.focal * powf(0.5, i as f32 * 0.5))
    }

    // The half-angle of the cone around the view direction that holds every
    // ray of a view of focal length `focal`: the angle to its corners.
    fn cone(&self, focal: f32) -> f32 {
        let [w, h] = self.size.map(|v| v.max(1) as f32);
        atan2(sqrt(w * w + h * h), 2.0 * focal.max(1.0))
    }
}

/// Upper bounds, per span of the visible-blade buffer (see
/// [`GRASS_SPAN_COUNT`]), on the blades one frame keeps of `field`, from any
/// of `views`, wherever the camera stands and however it turns.
///
/// A blade lands in region `k` when its distance from the camera falls within
/// that region's reach, and a view sees it only when it lies within the cone
/// of the view's corners around the view direction. At a distance `d` from a
/// camera `h` over the ground, a cone of half-angle `a` turned any way covers
/// at most the yaw whose cosine squared reaches
/// `(cos^2 a - sin^2 e) / cos^2 e` (all of it once `sin e >= cos a`), `e` the
/// ground's angle below the horizon. So the region's count is bounded by the
/// alive fraction at `max(d, inner)` times that share of the ring, integrated
/// out to the region's outer radius, and by the whole ground at the inner
/// radius's alive fraction. Each region adds a tile of slack for the cells the
/// boundary cuts. A span takes the most its regions keep together at any one
/// camera height, over focal lengths from the longest down, with a margin for
/// the views and heights between the ones sampled.
pub fn region_capacities(
    field: &CapacityField<'_>,
    views: &CapacityViews,
) -> [u32; GRASS_SPAN_COUNT] {
    let mut most = [0u32; GRASS_SPAN_COUNT];
    for focal in views.focals() {
        let view = GrassLodView::new(focal, field.blade_height);
        let cone = views.cone(focal);
        for h in camera_heights(view.extent) {
            let kept: [u32; GRASS_REGION_COUNT] = core::array::from_fn(|region| {
                band_capacity(field, &view, view.region_span(region), Some((cone, h)))
            });
            for (span, slot) in most.iter_mut().enumerate() {
                let together = if span + 1 == GRASS_SPAN_COUNT {
                    kept[span].saturating_add(kept[GRASS_BAND_REGION])
                } else {
                    kept[span]
                };
                *slot = (*slot).max(together);
            }
        }
    }
    most.map(with_sampling_margin)
}

/// An upper bound on the blades one frame keeps of `field` within
/// `cull_distance` of the camera, from any of `views`, wherever the camera
/// stands: the region the nearest shadow cascade's blades fill, whichever way
/// the light looks.
pub fn shadow_capacity(
    field: &CapacityField<'_>,
    views: &CapacityViews,
    cull_distance: f32,
) -> u32 {
    let most = views
        .focals()
        .map(|focal| {
            let view = GrassLodView::new(focal, field.blade_height);
            band_capacity(field, &view, [0.0, cull_distance.min(view.extent)], None)
        })
        .max()
        .unwrap_or(0);
    with_sampling_margin(most)
}

/// An upper bound on the tiles one run of the kernel visits for `field`, its
/// tiles of level 0 `tile_size` meters wide and its blades reaching `reach`
/// meters from their cells, from any of `views`, wherever the camera stands
/// and however it turns (see [`super::tiles::nested_tiles`]).
///
/// A tile is split while its box is nearer than where thinning lets its level
/// stop, so the ground `d` meters from the camera lies in a tile no finer than
/// the finest level whose parent's box can still be that near, and no coarser
/// than the coarsest level thinning allows at `d`. The bound counts the ground
/// in reach over the finest such tile's area at each distance, within the
/// view's cone widened by the coarsest tile's box, on flat ground from the
/// heights [`region_capacities`] takes, and at most the tiles of every level
/// the ground holds.
pub fn tile_capacity(
    field: &CapacityField<'_>,
    views: &CapacityViews,
    [tile_size, reach]: [f32; 2],
) -> u32 {
    let top = GRASS_NEST_LEVELS;
    let edge = |j: u32| tile_size.max(1e-3) * (1u32 << j) as f32;
    let diameter = |j: u32| {
        let side = edge(j) + 2.0 * reach;
        sqrt(2.0 * side * side + 4.0 * reach * reach)
    };
    let [w, h] = [0, 1].map(|i| (field.ground.max[i] - field.ground.min[i]).max(0.0));
    let whole: f32 = (0..=top)
        .map(|j| {
            let e = edge(j);
            ceil((w + 2.0 * e) / e) * ceil((h + 2.0 * e) / e)
        })
        .sum();
    let mut most = 0.0f32;
    for focal in views.focals() {
        let view = GrassLodView::new(focal, field.blade_height);
        let cone = views.cone(focal);
        // Where tiles of level `j` stop splitting: thinning keeps no more
        // than their cells' representatives.
        let stop =
            |j: u32| field.width * view.keep_scale() * (1u64 << (2 * j)) as f32 / hash_span();
        let finest = |d: f32| {
            let split = (0..top)
                .find(|&j| d < stop(j + 1) + diameter(j + 1))
                .unwrap_or(top);
            (split..=top).find(|&j| d <= view.extent + diameter(j))
        };
        let coarsest = |d: f32| (1..=top).rev().find(|&j| stop(j) <= d).unwrap_or(0);
        let outer = view.extent + diameter(top);
        for height in camera_heights(view.extent) {
            let mut tiles = 0.0f32;
            let mut r = 0.0f32;
            while r < outer {
                let next = (r + (r / 64.0).max(0.25)).min(outer);
                let d = sqrt(r * r + height * height);
                if let Some(j) = finest(d) {
                    let far = coarsest(sqrt(next * next + height * height)).max(j);
                    let widen = asin((diameter(far) / d.max(1e-6)).min(1.0));
                    let e = asin((height / d.max(1e-6)).min(1.0));
                    let ring = core::f32::consts::PI * (next * next - r * r);
                    tiles += ring * cone_share(cone + widen, e) / (edge(j) * edge(j));
                }
                r = next;
            }
            most = most.max(tiles.min(whole));
        }
    }
    let tiles = ceil(most);
    if tiles >= u32::MAX as f32 {
        u32::MAX
    } else {
        margin(u32::saturating_from(tiles))
    }
}

// The camera heights capacities are taken over: the ground, then a fifth of a
// meter up to past the blades' reach, by half-octaves.
fn camera_heights(extent: f32) -> impl Iterator<Item = f32> {
    core::iter::once(0.0).chain(
        (0..)
            .map(|i| 0.2 * powf(2.0, i as f32 * 0.5))
            .take_while(move |&h| h < 2.0 * extent),
    )
}

// A sixteenth over the most sampled, for views and heights between the
// samples (half an octave apart, over which a region's count moves by far
// less).
fn with_sampling_margin(blades: u32) -> u32 {
    margin(blades).min(MAX_GRASS_BLADES)
}

fn margin(n: u32) -> u32 {
    n.saturating_add(n / 16)
}

// The share of the yaw around a camera a cone of half-angle `cone` can cover
// at ground `e` radians below the horizon, turned any way.
fn cone_share(cone: f32, e: f32) -> f32 {
    let (ca, se, ce) = (cos(cone), sin(e), cos(e));
    if se >= ca {
        return 1.0;
    }
    let c2 = ((ca * ca - se * se) / (ce * ce).max(1e-12)).clamp(0.0, 1.0);
    acos(sqrt(c2)) / core::f32::consts::PI
}

// The blades `view` keeps of `field` at distances in `[inner, outer)` of the
// camera: the alive fraction at `max(d, inner)`, times the share of each ring
// a view cone of half-angle `cone` sees from `height` over the ground (or all
// of it), integrated over the disc of the outer radius, bounded by the whole
// ground at the inner radius's alive fraction, plus a tile of slack for the
// cells the boundary cuts.
fn band_capacity(
    field: &CapacityField<'_>,
    view: &GrassLodView,
    [inner, outer]: [f32; 2],
    cone: Option<(f32, f32)>,
) -> u32 {
    if outer <= inner {
        return 0;
    }
    let height = cone.map_or(0.0, |(_, h)| h);
    // Rings a quarter meter wide near the camera, a sixty-fourth of their
    // radius farther out; each takes the alive fraction and the cone's share
    // at its inner edge, the largest within it.
    let mut disc = 0.0f32;
    let mut r = 0.0f32;
    while r < outer {
        let next = (r + (r / 64.0).max(0.25)).min(outer);
        let d = sqrt(r * r + height * height);
        if d < outer && next * next + height * height > inner * inner {
            let share = cone.map_or(1.0, |(a, _)| {
                cone_share(a, asin((height / d.max(1e-6)).min(1.0)))
            });
            let ring = core::f32::consts::PI * (next * next - r * r);
            disc += ring * share * view.alive(field.width, d.max(inner));
        }
        r = next;
    }
    let whole = field.ground.area() * view.alive(field.width, inner);
    let blades = ceil(disc.min(whole) * field.grid.density()) + field.grid.blades_per_tile() as f32;
    if blades >= MAX_GRASS_BLADES as f32 {
        MAX_GRASS_BLADES
    } else {
        u32::saturating_from(blades)
    }
}

/// What a field's layers keep at most from some views: each layer's spans,
/// the shadow cascade's blades over every layer, and the tiles the view's run
/// visits over every layer, with how far the views' blades reach.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CapacityBounds {
    /// Each layer's bound per span.
    pub layers: Vec<[u32; GRASS_SPAN_COUNT]>,
    /// The cascade's bound.
    pub shadow: u32,
    /// The view's run's tiles (see [`tile_capacity`]).
    pub tiles: u32,
    /// How far the longest view's blades reach, and the farthest the field
    /// lets them, in meters.
    pub reach: [f32; 2],
}

impl CapacityBounds {
    // How many times longer the focal length is than the longest these
    // views could have and still fit the buffers, the tile records and the
    // reach: at most 1 when they fit. Blades and tiles go with about the
    // square of the focal length, the reach with the focal length.
    fn excess(&self) -> f32 {
        let blades: u64 = self.layers.iter().flatten().map(|&n| u64::from(n)).sum();
        let [extent, limit] = self.reach;
        [
            sqrt(blades as f32 / MAX_GRASS_BLADES as f32),
            sqrt(self.tiles as f32 / MAX_GRASS_TILE_RECORDS as f32),
            extent / limit.max(1e-3),
        ]
        .into_iter()
        .fold(0.0, f32::max)
    }
}

/// What the visible-blade buffers are sized for: the views they hold every
/// frame of, and the blades each region and the shadow cascade's buffer hold.
///
/// The views are a render target's, up to the focal length of
/// [`GRASS_SIZING_FOV`] at its height, or shorter where the buffer could not
/// hold that one's blades within [`MAX_GRASS_BLADES`], a run's tiles within
/// [`MAX_GRASS_TILE_RECORDS`], or its blades within the field's reach. A longer
/// focal length thins and ends as the sized one does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassSizing {
    /// The views the buffers hold.
    pub views: CapacityViews,
    /// Blades each region of the visible-blade buffer holds.
    pub capacity: GrassCapacity,
    /// Blades the shadow cascade's buffer holds.
    pub shadow_capacity: u32,
}

impl GrassSizing {
    /// The sizing for render targets `size` pixels across and down, of a
    /// field whose layers keep at most `bound` from views up to a focal
    /// length. A field that keeps nothing in the cascade sizes its buffer for
    /// none.
    pub fn new(size: [u32; 2], bound: &dyn Fn(&CapacityViews) -> CapacityBounds) -> Self {
        let mut views = CapacityViews {
            size,
            focal: focal_length(size[1], GRASS_SIZING_FOV),
        };
        let mut bounds;
        let mut tries = 0;
        loop {
            bounds = bound(&views);
            let excess = bounds.excess();
            if excess <= 1.0 || tries == 8 || views.focal <= 1.0 {
                break;
            }
            views.focal = (views.focal / excess * 0.98).max(1.0);
            tries += 1;
        }
        Self {
            views,
            capacity: GrassCapacity::sum(bounds.layers),
            shadow_capacity: bounds.shadow.min(MAX_GRASS_BLADES),
        }
    }
}

/// How many blades each span of the visible-blade buffer holds (see
/// [`GRASS_SPAN_COUNT`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GrassCapacity {
    /// Blades per span.
    pub spans: [u32; GRASS_SPAN_COUNT],
}

impl GrassCapacity {
    /// The sum of `per_layer`'s bounds, scaled down together so the buffer
    /// never holds more than [`MAX_GRASS_BLADES`].
    pub fn sum(per_layer: impl IntoIterator<Item = [u32; GRASS_SPAN_COUNT]>) -> Self {
        let mut sums = [0u64; GRASS_SPAN_COUNT];
        for layer in per_layer {
            for (sum, n) in sums.iter_mut().zip(layer) {
                *sum += u64::from(n);
            }
        }
        let total: u64 = sums.iter().sum();
        let cap = u64::from(MAX_GRASS_BLADES);
        let spans = sums.map(|n| {
            let n = if total > cap { n * cap / total } else { n };
            u32::saturating_from(n.max(1))
        });
        Self { spans }
    }

    /// Blades in the whole buffer.
    pub fn total(&self) -> u32 {
        self.spans.iter().sum()
    }

    /// The first blade of each span, the spans laid end to end.
    pub fn bases(&self) -> [u32; GRASS_SPAN_COUNT] {
        let mut base = 0;
        self.spans.map(|n| {
            let b = base;
            base += n;
            b
        })
    }

    /// Where each region's blades start, as the kernel and the draws read
    /// them: the band's is the last blade of the span it shares, and it fills
    /// down from there.
    pub fn region_bases(&self) -> [u32; GRASS_REGION_COUNT] {
        let spans = self.bases();
        let shared = GRASS_SPAN_COUNT - 1;
        core::array::from_fn(|r| {
            if r == GRASS_BAND_REGION {
                (spans[shared] + self.spans[shared]).saturating_sub(1)
            } else {
                spans[r]
            }
        })
    }

    /// Blades each region may hold: the shared span's whole, for both regions
    /// that share it.
    pub fn region_capacities(&self) -> [u32; GRASS_REGION_COUNT] {
        core::array::from_fn(|r| self.spans[r.min(GRASS_SPAN_COUNT - 1)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::sqrt;
    use crate::render::environment::grass::tiles::{
        GRASS_TILE_SIZE, NestView, NestedTiles, terrain_tiles,
    };
    use crate::render::{shader::consts, shader::embedded};

    // The planet example's meadow.
    const HEIGHT: f32 = 0.55;
    const WIDTH: f32 = 0.03;

    // The focal length of a 2160 pixel tall view at 60 degrees.
    const LONG: f32 = 1870.0;

    // A 3200 x 1800 view with a 70 degree vertical field of view.
    fn retina() -> GrassLodView {
        GrassLodView::new(focal_length(1800, 70f32.to_radians()), HEIGHT)
    }

    #[test]
    fn the_focal_length_spans_half_the_view_at_the_half_angle() {
        let f = focal_length(1800, 70f32.to_radians());
        assert!((f - 900.0 / tan(35f32.to_radians())).abs() < 1e-3);
        // A unit tall object one unit away fills the view at a 2 atan(0.5)
        // field of view.
        let fov = 2.0 * atan2(0.5, 1.0);
        assert!((focal_length(1000, fov) - 1000.0).abs() < 0.5);
    }

    // A blade stands GRASS_END_PIXELS tall where the field ends: 471 m for
    // the meadow on a 1800 pixel tall view.
    #[test]
    fn blades_end_at_their_projected_size() {
        for v in [GrassLodView::new(100.0, HEIGHT), retina()] {
            assert!((HEIGHT * v.focal / v.extent - GRASS_END_PIXELS).abs() < 1e-4);
        }
        assert!((retina().extent - 471.2).abs() < 0.5, "{}", retina().extent);
        // Shorter grass ends nearer in the same view.
        let short = GrassLodView::new(100.0, 0.5 * HEIGHT);
        let small = GrassLodView::new(100.0, HEIGHT);
        assert!((short.extent - 0.5 * small.extent).abs() < 1e-3);
    }

    // The levels start where a blade stands GRASS_LOD_PIXELS tall: 11.8 and
    // 29.5 m for the meadow on a 1800 pixel view.
    #[test]
    fn levels_coarsen_at_their_projected_sizes() {
        let v = retina();
        for (d, px) in v.lod_distances.iter().zip(GRASS_LOD_PIXELS) {
            assert!((HEIGHT * v.focal / d - px).abs() < 1e-3);
        }
        assert_eq!(v.lod_at(0.0), 0);
        assert_eq!(v.lod_at(v.lod_distances[0] - 0.01), 0);
        assert_eq!(v.lod_at(v.lod_distances[0]), 1);
        assert_eq!(v.lod_at(v.lod_distances[1] - 0.01), 1);
        assert_eq!(v.lod_at(v.lod_distances[1]), 2);
        assert_eq!(v.lod_at(1.0e6), GRASS_LOD_COUNT - 1);
        // A view so coarse the blades end before a level starts never reaches
        // it.
        let coarse = GrassLodView::new(20.0, HEIGHT);
        assert!(coarse.lod_distances.iter().all(|&d| d <= coarse.extent));
    }

    // A level's folding vertices reach the next level's strip exactly where
    // the kernel hands the blade to that level, so the switch moves nothing.
    #[test]
    fn a_level_has_fully_folded_where_the_next_begins() {
        let v = retina();
        for (lod, &end) in v.lod_distances.iter().enumerate() {
            assert_eq!(v.morph(lod, 0.0), 0.0);
            assert!(v.morph(lod, end * (1.0 - GRASS_MORPH_FRACTION)) < 1e-5);
            assert!((v.morph(lod, end) - 1.0).abs() < 1e-5);
            let mid = end * (1.0 - 0.5 * GRASS_MORPH_FRACTION);
            assert!((v.morph(lod, mid) - 0.5).abs() < 1e-4);
        }
        assert_eq!(v.morph(GRASS_LOD_COUNT - 1, 1.0e6), 0.0);
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

    // Survivors widened for the thinning stand GRASS_BLADE_PIXELS wide on
    // screen wherever the field thins, at any focal length.
    #[test]
    fn widened_survivors_stand_a_fixed_width_on_screen() {
        for focal in [300.0, 900.0, 1285.0, LONG] {
            let v = GrassLodView::new(focal, HEIGHT);
            let full = WIDTH * v.keep_scale();
            for i in 1..60 {
                let d = full + i as f32 * (v.extent - full).max(0.0) / 60.0;
                let keep = v.thinning(WIDTH, d);
                if keep >= 1.0 {
                    continue;
                }
                // The mean widening over the survivors, against the keep
                // fraction alone: the shrink band holds coverage, not width.
                let widened_px = WIDTH * keep.recip() * v.focal / d;
                assert!(
                    (widened_px - GRASS_BLADE_PIXELS).abs() < 1e-3,
                    "{widened_px}"
                );
            }
            assert_eq!(v.thinning(WIDTH, 0.5 * full), 1.0);
        }
    }

    #[test]
    fn thinning_keeps_everything_near_and_falls_with_distance() {
        let v = retina();
        let full = WIDTH * v.keep_scale();
        assert_eq!(v.thinning(WIDTH, 0.0), 1.0);
        assert_eq!(v.thinning(WIDTH, full), 1.0);
        assert!((v.thinning(WIDTH, 2.0 * full) - 0.5).abs() < 1e-6);
        let mut last = 1.0;
        for i in 0..200 {
            let k = v.thinning(WIDTH, i as f32 * 0.5);
            assert!(k <= last);
            last = k;
        }
        // Wider blades thin out later.
        assert!(v.thinning(2.0 * WIDTH, 60.0) > v.thinning(WIDTH, 60.0));
    }

    #[test]
    fn the_field_fades_over_the_last_quarter_of_its_reach() {
        let v = retina();
        let d = v.extent;
        assert_eq!(v.fade_start(), 0.75 * d);
        assert_eq!(v.fade(0.0), 1.0);
        assert_eq!(v.fade(v.fade_start()), 1.0);
        assert!((v.fade(0.875 * d) - 0.5).abs() < 1e-5);
        assert_eq!(v.fade(d), 0.0);
        assert_eq!(v.fade(2.0 * d), 0.0);
    }

    // Blades land in their level's region up to the hand-off, then in the
    // band, whatever level their distance would give them.
    #[test]
    fn blades_past_the_hand_off_land_in_the_band() {
        let v = retina();
        assert_eq!(v.region_at(0.0), 0);
        assert_eq!(v.region_at(v.lod_distances[0]), 1);
        assert_eq!(v.region_at(v.lod_distances[1]), 2);
        assert_eq!(v.region_at(v.fade_start() - 0.01), 2);
        assert_eq!(v.region_at(v.fade_start()), GRASS_BAND_REGION);
        assert_eq!(v.region_at(v.extent), GRASS_BAND_REGION);
        // Spans tile the reach: each region starts where the last ends.
        let mut last = 0.0;
        for r in 0..GRASS_REGION_COUNT {
            let [inner, outer] = v.region_span(r);
            assert_eq!(inner, last);
            assert!(outer >= inner);
            last = outer;
        }
        assert_eq!(last, v.extent);
        // A view so coarse the hand-off starts before the coarser levels
        // leaves them empty.
        let coarse = GrassLodView::new(20.0, HEIGHT);
        let [inner, outer] = coarse.region_span(2);
        assert!(outer <= coarse.fade_start() && inner <= outer);
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
                let just = gone * 1.001;
                assert!(survives(just, u));
                assert!(shrink(just, u) < 0.01);
            }
        }
    }

    // Where the keep fraction holds steady, most survivors stand at full
    // height: only those near the threshold shrink.
    #[test]
    fn survivors_mostly_stand_full_height() {
        let keep = 0.2;
        let n = 20_000;
        let survivors: Vec<f32> = (0..n)
            .map(|i| (i as f32 + 0.5) / n as f32)
            .filter(|&u| survives(keep, u))
            .map(|u| shrink(keep, u))
            .collect();
        let full = survivors.iter().filter(|&&s| s >= 1.0).count();
        let share = full as f32 / survivors.len() as f32;
        assert!((share - (1.0 - GRASS_SHRINK_BAND)).abs() < 0.01, "{share}");
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
    // area, blades times width times height, the same at every distance up to
    // the fade, and never past the widest the layer's reach allows for.
    #[test]
    fn widened_survivors_keep_the_field_coverage() {
        let widest = max_width_scale(WIDTH, HEIGHT);
        for focal in [60.0, 300.0, 1285.0, LONG] {
            let v = GrassLodView::new(focal, HEIGHT);
            for i in 0..=120 {
                let d = v.extent * i as f32 / 120.0;
                let k = v.thinning(WIDTH, d);
                assert!((coverage(k) * v.width_scale(WIDTH, d) - 1.0).abs() < 1e-4);
                assert!(v.width_scale(WIDTH, d) <= widest * (1.0 + 1e-4));
            }
        }
        // The widest is reached at the end of a view whose end is its own.
        let v = GrassLodView::new(100.0, HEIGHT);
        assert!((v.width_scale(WIDTH, v.extent) - widest).abs() / widest < 1e-3);
    }

    fn ground() -> GroundRect {
        GroundRect::centered([0.0, 0.0], [2000.0, 2000.0])
    }

    fn field<'a>(ground: &'a GroundRect, grid: &'a GrassGrid) -> CapacityField<'a> {
        CapacityField {
            ground,
            grid,
            width: WIDTH,
            blade_height: HEIGHT,
        }
    }

    fn views(size: [u32; 2], focal: f32) -> CapacityViews {
        CapacityViews { size, focal }
    }

    // A camera `height` over flat ground, looking `pitch` radians down and
    // rolled `roll` radians, through a view `size` pixels across and down.
    struct Shot {
        size: [u32; 2],
        height: f32,
        pitch: f32,
        roll: f32,
    }

    // The mean blades `view` keeps per region within `shot`'s frustum: the
    // ground in polar rings around the camera's foot, each patch weighted by
    // the fraction alive at its distance.
    fn kept(view: &GrassLodView, grid: GrassGrid, shot: &Shot) -> [f32; GRASS_REGION_COUNT] {
        let (sp, cp) = (sin(shot.pitch), cos(shot.pitch));
        let forward = [0.0, -sp, cp];
        let up0 = [0.0, cp, sp];
        let (sr, cr) = (sin(shot.roll), cos(shot.roll));
        let right: [f32; 3] = core::array::from_fn(|i| cr * [1.0, 0.0, 0.0][i] + sr * up0[i]);
        let up: [f32; 3] = core::array::from_fn(|i| -sr * [1.0, 0.0, 0.0][i] + cr * up0[i]);
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let half = shot.size.map(|v| 0.5 * v as f32 / view.focal);
        let mut kept = [0.0f32; GRASS_REGION_COUNT];
        const YAWS: usize = 512;
        let mut r = 0.0f32;
        while r < view.extent {
            let dr = (r / 128.0).max(0.125);
            let rm = r + 0.5 * dr;
            let d = sqrt(rm * rm + shot.height * shot.height);
            if d < view.extent {
                let area = core::f32::consts::TAU * rm * dr / YAWS as f32;
                let alive = view.alive(WIDTH, d) * grid.density() * area;
                for k in 0..YAWS {
                    let a = (k as f32 + 0.5) / YAWS as f32 * core::f32::consts::TAU;
                    let v = [rm * sin(a), -shot.height, rm * cos(a)];
                    let z = dot(v, forward);
                    if z > 0.0
                        && dot(v, right).abs() <= half[0] * z
                        && dot(v, up).abs() <= half[1] * z
                    {
                        kept[view.region_at(d)] += alive;
                    }
                }
            }
            r += dr;
        }
        kept
    }

    // Each region's bound covers what views of any focal length up to the
    // sized one keep in its ring, from the ground and above it, however the
    // camera turns.
    #[test]
    fn region_capacities_bound_what_the_kernel_keeps() {
        let ground = ground();
        let grid = GrassGrid::for_density(100.0).unwrap();
        for (size, sized) in [([1600, 900], 643.0), ([3200, 1800], 1285.0)] {
            let caps = region_capacities(&field(&ground, &grid), &views(size, sized));
            let mut most = 0.0f32;
            for focal in [sized, 0.6 * sized, 0.2 * sized] {
                let view = GrassLodView::new(focal, HEIGHT);
                for height in [0.0, 2.5, 30.0, 150.0] {
                    for pitch in [0.0f32, 0.3, 1.2] {
                        for roll in [0.0f32, 0.5] {
                            let shot = Shot {
                                size,
                                height,
                                pitch,
                                roll,
                            };
                            let kept = kept(&view, grid, &shot);
                            let spans = [kept[0], kept[1], kept[2] + kept[GRASS_BAND_REGION]];
                            for r in 0..GRASS_SPAN_COUNT {
                                assert!(
                                    spans[r] <= caps[r] as f32,
                                    "{size:?} focal {focal}, height {height}, pitch {pitch}, \
                                     roll {roll}, span {r}: {} > {}",
                                    spans[r],
                                    caps[r]
                                );
                            }
                            most = most.max(kept.iter().sum());
                        }
                    }
                }
            }
            // Loose, but within a few times the most any of these keeps.
            let total: u32 = caps.iter().sum();
            assert!((total as f32) < 3.0 * most, "{size:?}: {total} vs {most}");
        }
    }

    // The cascade's region covers every blade the field keeps within its cull
    // distance, from any view, and is far smaller than the view's.
    #[test]
    fn the_shadow_capacity_bounds_the_blades_near_the_camera() {
        let ground = ground();
        let grid = GrassGrid::for_density(100.0).unwrap();
        let f = field(&ground, &grid);
        let v = views([3200, 1800], LONG);
        let r = 24.0;
        let cap = shadow_capacity(&f, &v, r);
        let cell = grid.cell_size();
        let n = i32::saturating_from(r / cell) + 1;
        for focal in [100.0, 600.0, 1285.0, LONG] {
            let view = GrassLodView::new(focal, HEIGHT);
            let mut kept = 0.0f32;
            for z in -n..=n {
                for x in -n..=n {
                    let p = [(x as f32 + 0.5) * cell, (z as f32 + 0.5) * cell];
                    let d = sqrt(p[0] * p[0] + p[1] * p[1]);
                    if d < r.min(view.extent) {
                        kept += view.alive(WIDTH, d);
                    }
                }
            }
            assert!(kept <= cap as f32, "focal {focal}: {kept} > {cap}");
        }
        let view: u32 = region_capacities(&f, &v).iter().sum();
        assert!(cap < view / 2, "{cap} vs {view}");
    }

    #[test]
    fn a_small_ground_bounds_every_level_by_its_area() {
        let grid = GrassGrid::for_density(100.0).unwrap();
        let small = GroundRect::centered([0.0, 0.0], [5.0, 5.0]);
        let caps = region_capacities(&field(&small, &grid), &views([3200, 1800], LONG));
        let whole = 100 * 100 + grid.blades_per_tile();
        for &c in &caps {
            assert!(c <= whole + whole / 16);
        }
    }

    fn sized(size: [u32; 2]) -> GrassSizing {
        let ground = ground();
        let grid = GrassGrid::for_density(100.0).unwrap();
        let f = field(&ground, &grid);
        GrassSizing::new(size, &|v| CapacityBounds {
            layers: alloc::vec![region_capacities(&f, v)],
            shadow: shadow_capacity(&f, v, 24.0),
            tiles: tile_capacity(&f, v, [GRASS_TILE_SIZE, REACH]),
            reach: [GrassLodView::new(v.focal, HEIGHT).extent, f32::INFINITY],
        })
    }

    // The meadow's blade reach from its cell.
    const REACH: f32 = 0.7;

    // The frustum planes of `shot`'s camera, `(normal, d)` with the inside
    // non-negative, open at the far end.
    fn frustum(shot: &Shot, focal: f32) -> [[f32; 4]; 6] {
        let (sp, cp) = (sin(shot.pitch), cos(shot.pitch));
        let forward = [0.0, -sp, cp];
        let up0 = [0.0, cp, sp];
        let (sr, cr) = (sin(shot.roll), cos(shot.roll));
        let right: [f32; 3] = core::array::from_fn(|i| cr * [1.0, 0.0, 0.0][i] + sr * up0[i]);
        let up: [f32; 3] = core::array::from_fn(|i| -sr * [1.0, 0.0, 0.0][i] + cr * up0[i]);
        let half = shot.size.map(|v| 0.5 * v as f32 / focal);
        let eye = [0.0, shot.height, 0.0];
        let plane = |n: [f32; 3]| {
            let d = -(n[0] * eye[0] + n[1] * eye[1] + n[2] * eye[2]);
            [n[0], n[1], n[2], d]
        };
        let side = |axis: [f32; 3], half: f32, sign: f32| -> [f32; 3] {
            core::array::from_fn(|i| forward[i] * half - sign * axis[i])
        };
        [
            plane(forward),
            plane(side(right, half[0], 1.0)),
            plane(side(right, half[0], -1.0)),
            plane(side(up, half[1], 1.0)),
            plane(side(up, half[1], -1.0)),
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    // The tile bound covers the tiles the quadtree visits for views of any
    // focal length up to the sized one, from the ground and above it,
    // however the camera turns, and stays within a few times the most.
    #[test]
    fn the_tile_capacity_bounds_the_tiles_a_run_visits() {
        let ground = ground();
        let grid = GrassGrid::for_density(100.0).unwrap();
        for (size, sized) in [([1600, 900], 643.0), ([3200, 1800], 1559.0)] {
            let cap = tile_capacity(
                &field(&ground, &grid),
                &views(size, sized),
                [GRASS_TILE_SIZE, REACH],
            );
            let mut most = 0usize;
            for focal in [sized, 0.4 * sized] {
                let view = GrassLodView::new(focal, HEIGHT);
                for height in [0.0, 2.5, 30.0, 300.0] {
                    for pitch in [0.0f32, 0.3, 1.2, 1.57] {
                        for roll in [0.0f32, 0.7] {
                            let shot = Shot {
                                size,
                                height,
                                pitch,
                                roll,
                            };
                            let planes = frustum(&shot, view.focal);
                            let nest = NestView {
                                camera: [0.0, height, 0.0],
                                distance: view.extent,
                                planes: &planes,
                                keep_scale: view.keep_scale(),
                            };
                            let mut out = NestedTiles::default();
                            terrain_tiles(&nest, &ground, [[0.0, 0.0], [REACH, WIDTH]], &mut out);
                            let n = out.tiles.len();
                            assert!(
                                n <= cap as usize,
                                "{size:?} focal {focal}, height {height}, pitch {pitch}, \
                                 roll {roll}: {n} > {cap}"
                            );
                            most = most.max(n);
                        }
                    }
                }
            }
            assert!(
                most > 0 && (cap as usize) < 4 * most,
                "{size:?}: {cap} vs {most}"
            );
        }
    }

    // A field whose tiles outgrow the records a run holds is sized for a
    // shorter focal length, as is one whose blades would reach past the
    // field's limit; one keeping nothing in the cascade sizes it for none.
    #[test]
    fn the_sizing_shortens_the_view_to_fit_the_tiles_and_the_reach() {
        let views_at = |v: &CapacityViews| GrassLodView::new(v.focal, HEIGHT).extent;
        let full = focal_length(1800, GRASS_SIZING_FOV);
        let many = GrassSizing::new([3200, 1800], &|v| CapacityBounds {
            layers: alloc::vec![[1000; GRASS_SPAN_COUNT]],
            shadow: 0,
            tiles: u32::saturating_from(
                4.0 * MAX_GRASS_TILE_RECORDS as f32 * (v.focal / full).powi(2),
            ),
            reach: [views_at(v), f32::INFINITY],
        });
        assert!(many.views.focal < 0.5 * full, "{}", many.views.focal);
        assert!(many.views.focal > 0.4 * full, "{}", many.views.focal);
        assert_eq!(many.shadow_capacity, 0);
        let limit = 300.0;
        let far = GrassSizing::new([3200, 1800], &|v| CapacityBounds {
            layers: alloc::vec![[1000; GRASS_SPAN_COUNT]],
            shadow: 10,
            tiles: 10,
            reach: [views_at(v), limit],
        });
        let extent = GrassLodView::new(far.views.focal, HEIGHT).extent;
        assert!(extent <= limit && extent > 0.95 * limit, "{extent}");
        assert_eq!(far.shadow_capacity, 10);
    }

    // A small view keeps the focal length of the sizing field of view and a
    // buffer far under the cap; a large one shortens it until the buffer
    // fits.
    #[test]
    fn the_buffer_is_sized_for_the_view() {
        let small = sized([1600, 900]);
        assert_eq!(small.views.focal, focal_length(900, GRASS_SIZING_FOV));
        assert!(small.capacity.total() < MAX_GRASS_BLADES / 2);
        let large = sized([3840, 2160]);
        assert!(large.views.focal < focal_length(2160, GRASS_SIZING_FOV));
        assert!(large.views.focal > 1000.0, "{}", large.views.focal);
        assert!(large.capacity.total() <= MAX_GRASS_BLADES);
        assert!(small.capacity.total() * 3 < large.capacity.total());
    }

    #[test]
    fn capacities_sum_per_span_and_lay_spans_end_to_end() {
        let c = GrassCapacity::sum([[10, 20, 30], [1, 2, 3]]);
        assert_eq!(c.spans, [11, 22, 33]);
        assert_eq!(c.total(), 66);
        assert_eq!(c.bases(), [0, 11, 33]);
        // The band fills the coarsest level's span down from its last blade.
        assert_eq!(c.region_bases(), [0, 11, 33, 65]);
        assert_eq!(c.region_capacities(), [11, 22, 33, 33]);
    }

    #[test]
    fn capacities_scale_down_to_the_buffer_cap() {
        let c = GrassCapacity::sum([[MAX_GRASS_BLADES; GRASS_SPAN_COUNT]]);
        assert!(c.total() <= MAX_GRASS_BLADES);
        assert_eq!(c.spans, [MAX_GRASS_BLADES / 3; GRASS_SPAN_COUNT]);
        let empty = GrassCapacity::sum([]);
        assert_eq!(empty.spans, [1, 1, 1]);
    }

    // The meadow's blade layers.
    const DEPTH: f32 = WIDTH * 100.0 * HEIGHT;

    // A ray seen from above keeps thinning's blades; a grazing one past the
    // onset keeps what crosses the limit's layers, at the same width.
    #[test]
    fn the_depth_limit_thins_only_grazing_far_rays() {
        let v = retina();
        let far = 2.0 * v.depth_onset();
        assert_eq!(v.keep(WIDTH, DEPTH, far, 1.0), v.thinning(WIDTH, far));
        let grazing = 0.01;
        let kept = v.keep(WIDTH, DEPTH, far, grazing);
        let layers = DEPTH / grazing;
        assert!((kept / v.thinning(WIDTH, far) - GRASS_DEPTH_LIMIT / layers).abs() < 1e-5);
        assert_eq!(
            v.width_scale(WIDTH, far),
            1.0 / coverage(v.thinning(WIDTH, far))
        );
        // A sparser layer is crossed by fewer layers and keeps more.
        assert!(v.keep(WIDTH, 0.25 * DEPTH, far, grazing) > kept);
    }

    // Nearer than the onset the limit grows, so a grazing ray from standing
    // height keeps the near field whole; the keep fraction is continuous
    // across the onset, so no line follows the camera.
    #[test]
    fn the_near_field_keeps_its_depth() {
        let v = retina();
        let onset = v.depth_onset();
        assert_eq!(onset, v.lod_distances[1]);
        for d in [2.0f32, 5.0, 10.0] {
            let facing = 2.5 / d;
            assert_eq!(v.keep(WIDTH, DEPTH, d, facing), v.thinning(WIDTH, d), "{d}");
        }
        let at = |d: f32| v.keep(WIDTH, DEPTH, d, 2.5 / d);
        let (inside, outside) = (at(onset * 0.9999), at(onset * 1.0001));
        assert!(
            (inside - outside).abs() < 1e-3 * inside,
            "{inside} {outside}"
        );
        // Past the onset, the grazing field from standing height thins far
        // faster than thinning alone.
        let d = 8.0 * onset;
        assert!(
            at(d) < 0.15 * v.thinning(WIDTH, d),
            "{}",
            at(d) / v.thinning(WIDTH, d)
        );
    }

    #[test]
    fn the_draws_agree_on_the_vertex_spacing() {
        let src = embedded::GRASS;
        assert_eq!(
            consts::uint(src, "GRASS_LOD_VERTEX_STRIDE"),
            GRASS_LOD_VERTEX_STRIDE as usize
        );
        assert_eq!(consts::uint(src, "GRASS_LOD_COUNT"), GRASS_LOD_COUNT);
        assert_eq!(consts::uint(src, "GRASS_BAND_REGION"), GRASS_BAND_REGION);
        assert_eq!(consts::uint(src, "GRASS_REGION_COUNT"), GRASS_REGION_COUNT);
        assert_eq!(consts::uint(src, "GRASS_SPAN_COUNT"), GRASS_SPAN_COUNT);
        assert_eq!(region_level(GRASS_BAND_REGION), GRASS_LOD_COUNT - 1);
        assert_eq!(consts::float(src, "GRASS_DEPTH_LIMIT"), GRASS_DEPTH_LIMIT);
    }
}
