//! The far field's canopy: past where the blades end, the ground is shaded as
//! the field of blades it stands for, seen from afar.
//!
//! The canopy is the blade shading averaged over the field. A view ray enters
//! the field from above and is stopped by blade area at a rate set by the
//! layer's leaf area and how much of a blade faces along the ray, so it meets
//! the upper blades along the ground and, looking down, often reaches the
//! shaded ground between them ([`canopy_view`]). Where it stops, the blades it
//! sees lean every way around the view, each lit by the blade model itself at
//! that height. Every term here is derived from a layer's look and the
//! kernel's own blade shapes; nothing is tuned to one light.
//!
//! The far field also carries the blades' own variation: their clumps (hue
//! and height, from the kernel's clump lattice), their density (a terrain's
//! masks, a planet's meadow) and the field's slow drift
//! ([`super::variation`]), all on ground-fixed coordinates.

use alloc::vec::Vec;
use concinnity_core::math::{SaturatingFrom, atan2, cos, exp, narrow_f32, powf, sin, sqrt};

use super::canopy_table::canopy_table;
use super::planet::PlanetGrass;
use super::planet::density::meadow_density;
use super::planet::tiles::{DrawnPlanetTile, ground_reference, planet_center};
use super::{GrassBladeLook, GrassLodView, GrassTerrain, grown_layers};
use crate::planet::{CONTINENT_OCTAVES, OCTAVE_TURN, PlanetShape, octave_weight_sum};
use crate::render::environment::ground::GroundCovers;
use crate::render::uniforms::canopy::{
    CanopyLayerGpu, CanopyParams, MAX_CANOPY_LAYERS, ShoreSwashGpu,
};
use crate::render::uniforms::ground::GroundRuleGpu;
use crate::water::sea::shore::ShoreWaves;

/// Most layers one terrain's far field shades: its vertex colors carry one
/// layer's density per channel.
const MAX_CANOPY_LAYERS_PER_TERRAIN: usize = 3;

/// A blade's mean lean: the kernel's `(0.15 + 0.35 u) (0.6 + 0.8 v)` over its
/// uniform hashes.
const CANOPY_BEND: f32 = 0.325;

// A blade's width tapers as `1 - t^WIDTH_TAPER` from root to tip.
const WIDTH_TAPER: f32 = 1.4;

/// How many facings around the view the canopy lights its blades at: the
/// shaders' `CANOPY_FACINGS`.
#[cfg(test)]
const CANOPY_FACINGS: u32 = 8;

/// How many height segments a blade is cut into for what a view sees of it.
/// The upper half of them makes the upper band, the lower half the lower.
const CANOPY_SEGMENTS: usize = 8;

/// How wide a blade is at height `t` along it, as a share of its root width.
fn blade_width(t: f32) -> f32 {
    1.0 - powf(t.clamp(0.0, 1.0), WIDTH_TAPER)
}

/// One-sided blade area per square meter of ground, for blades of `look`
/// grown at `density` per square meter.
pub fn leaf_area(look: &GrassBladeLook, density: f32) -> f32 {
    let mean_width = 1.0 - 1.0 / (1.0 + WIDTH_TAPER);
    density * look.width * look.height * mean_width
}

/// How far a blade leaning by `bend` tilts from upright at height `t` along
/// it, in radians: the slope of the kernel's curve there, upright at the root
/// and arcing over toward the tip. Mirrors `canopy_tilt`.
fn blade_tilt(bend: f32, t: f32) -> f32 {
    let rise = sqrt((1.0 - bend * bend).max(0.0));
    atan2(t * bend, (1.0 - t) * rise)
}

/// A blade's mean bend in a wind of mean `wind_speed` (m/s): its own lean,
/// and the wind's push on a blade of `stiffness`, in random directions to
/// each other. Mirrors the draws' `grass_wind_bend` at the mean speed.
pub fn canopy_bend(wind_speed: f32, stiffness: f32) -> f32 {
    let give = 3.0 + 25.0 * stiffness.clamp(0.0, 1.0);
    let speed = wind_speed.max(0.0);
    let push = 0.9 * speed / (speed + give);
    sqrt(CANOPY_BEND * CANOPY_BEND + push * push).min(0.95)
}

// The blade area seen along a direction `mu` from the blades' up, per unit of
// area, of a blade part whose normal tilts from level by `tilt`: the part's
// share facing up seen from above, and its upright share seen along the
// ground from every side.
fn part_projection(tilt: f32, mu: f32) -> f32 {
    let down = sin(tilt) * mu;
    let side = core::f32::consts::FRAC_2_PI * cos(tilt) * sqrt((1.0 - mu * mu).max(0.0));
    sqrt(down * down + side * side)
}

// The middle height of segment `j` counted from the tip.
fn segment_height(j: usize) -> f32 {
    1.0 - (j as f32 + 0.5) / CANOPY_SEGMENTS as f32
}

/// The blade area seen along a direction `mu` (the cosine of its angle from
/// the blades' up) per unit of leaf area, for blades bent by `bend`.
pub fn projection(bend: f32, mu: f32) -> f32 {
    let (mut seen, mut total) = (0.0, 0.0);
    for j in 0..CANOPY_SEGMENTS {
        let t = segment_height(j);
        let w = blade_width(t);
        seen += w * part_projection(blade_tilt(bend, t), mu);
        total += w;
    }
    seen / total
}

/// What a band of blade height shows a view, given a ray stopped in it: its
/// share of the rays, their mean height, and the means of the blade model's
/// height terms and their products, so the band shades as the average of its
/// heights rather than as its average height.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CanopyBand {
    /// The share of view rays stopped in the band.
    pub share: f32,
    /// Their mean height (0 root, 1 tip).
    pub height: f32,
    /// The mean root-to-tip ramp the colors blend by.
    pub ramp: f32,
    /// The mean sky occlusion.
    pub occlusion: f32,
    /// The mean of the ramp times the occlusion.
    pub ramp_occlusion: f32,
    /// The mean share of the field's leaf area standing above it, which a
    /// direct light crosses on its way down (`leaf_above`).
    pub above: f32,
}

/// Where a view ray into the field stops: on the ground between the blades,
/// or in the upper or lower half of the blades' height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanopyView {
    /// The share that reaches the ground.
    pub ground: f32,
    /// The upper half of the blades' height.
    pub upper: CanopyBand,
    /// The lower half.
    pub lower: CanopyBand,
}

/// The blade's root-to-tip color ramp at height `t`. Mirrors the shaders'
/// `smoothstep(0, 1, t)`.
pub fn ramp(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How much of the sky reaches height `t` inside the field. Mirrors
/// `grass_root_occlusion`.
fn root_occlusion(t: f32) -> f32 {
    let s = (t / 0.6).clamp(0.0, 1.0);
    0.25 + 0.75 * s * s * (3.0 - 2.0 * s)
}

// The share of one blade's area above height `t` along it.
fn blade_area_above(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    let k = 1.0 + WIDTH_TAPER;
    ((1.0 - t) - (1.0 - powf(t, k)) / k) / (1.0 - 1.0 / k)
}

/// The share of a field's leaf area standing above height `z`, in units of
/// its mean blade height, for blades varying in height by `height_variance`:
/// what a direct light crosses on its way down to there. Mirrors
/// `grass_leaf_above`.
fn leaf_above(z: f32, height_variance: f32) -> f32 {
    let heights = class_heights(height_variance);
    let total: f32 = heights.iter().sum();
    heights
        .iter()
        .map(|&h| h * blade_area_above(z / h))
        .sum::<f32>()
        / total
}

/// How many view cosines [`projection_axes`] fits its ellipse over.
const PROJECTION_FIT_STEPS: usize = 64;

/// The side-on and top-down axes of the ellipse that stands for
/// [`projection`] where no canopy table is read (the blade draws), scaled so
/// it errs as far over as under. See `ellipse_projection`.
pub fn projection_axes(bend: f32) -> [f32; 2] {
    let axes = [projection(bend, 0.0), projection(bend, 1.0)];
    let (mut low, mut high) = (f32::MAX, 0.0f32);
    for i in 0..=PROJECTION_FIT_STEPS {
        let mu = i as f32 / PROJECTION_FIT_STEPS as f32;
        let fit = ellipse_projection(axes, mu);
        if fit > 1e-6 {
            let r = projection(bend, mu) / fit;
            low = low.min(r);
            high = high.max(r);
        }
    }
    let scale = if high > 0.0 { 0.5 * (low + high) } else { 1.0 };
    axes.map(|a| a * scale)
}

/// The projection along view cosine `mu` of blades whose ellipse has `axes`
/// (side-on, top-down). Mirrors `grass_light_depth`.
fn ellipse_projection(axes: [f32; 2], mu: f32) -> f32 {
    let mu = mu.clamp(0.0, 1.0);
    sqrt(axes[1] * axes[1] * mu * mu + axes[0] * axes[0] * (1.0 - mu * mu))
}

/// The blade heights a field's height variation spreads its blades over, as
/// offsets per unit of variation from the mean: the means of the three equally
/// likely terciles of the kernel's clump plus blade spread,
/// `1.2 (u - 0.5) + 0.8 (v - 0.5)`.
const CANOPY_HEIGHT_CLASSES: [f32; 3] = [-0.4665, 0.0, 0.4665];

// The three class heights, relative to the mean, for `height_variance`; the
// kernel floors a blade at a tenth of the mean.
fn class_heights(height_variance: f32) -> [f32; 3] {
    CANOPY_HEIGHT_CLASSES.map(|q| (1.0 + height_variance * q).max(0.1))
}

/// Where rays seen along `mu` stop in a field of `leaf_area` bent by `bend`
/// whose blades vary in height by `height_variance`: blade area met at a rate
/// set by how much of it faces along the ray, down from the tallest tips,
/// segment by segment of height above the ground. The tallest blades stand
/// above a thinner layer, so a ray reaches further down them. The main pass
/// reads it tabulated ([`super::canopy_table`]).
pub fn canopy_view(bend: f32, height_variance: f32, mu: f32, leaf_area: f32) -> CanopyView {
    let mu = mu.max(0.02);
    let heights = class_heights(height_variance);
    let top = heights.iter().fold(0.0f32, |a, &h| a.max(h));
    // Per segment: the area seen there, and its area-weighted terms.
    let mut seen = [0.0f32; CANOPY_SEGMENTS];
    let mut terms = [[0.0f32; 5]; CANOPY_SEGMENTS];
    for j in 0..CANOPY_SEGMENTS {
        let z = top * segment_height(j);
        let above = leaf_above(z, height_variance);
        for &h in &heights {
            if z >= h {
                continue;
            }
            let t = z / h;
            let a = blade_width(t) * part_projection(blade_tilt(bend, t), mu);
            let (r, o) = (ramp(t), root_occlusion(t));
            seen[j] += a;
            for (acc, f) in terms[j].iter_mut().zip([t, r, o, r * o, above]) {
                *acc += a * f;
            }
        }
    }
    let total: f32 = seen.iter().sum();
    let depth = projection(bend, mu) * leaf_area / mu;
    let mut bands = [CanopyBand::default(); 2];
    let mut passed = 0.0;
    let mut left = 1.0;
    for j in 0..CANOPY_SEGMENTS {
        passed += seen[j];
        let after = exp(-depth * passed / total.max(1e-9));
        let stopped = left - after;
        left = after;
        // The segment's terms are sums over its seen area; a ray stopped there
        // stops on each part in proportion to it.
        let p = stopped / seen[j].max(1e-9);
        let b = &mut bands[j * 2 / CANOPY_SEGMENTS];
        b.share += stopped;
        let [t, r, o, ro, above] = terms[j];
        b.height += p * t;
        b.ramp += p * r;
        b.occlusion += p * o;
        b.ramp_occlusion += p * ro;
        b.above += p * above;
    }
    let [upper, lower] = bands.map(|mut b| {
        let n = b.share.max(1e-9);
        b.height /= n;
        b.ramp /= n;
        b.occlusion /= n;
        b.ramp_occlusion /= n;
        b.above /= n;
        b
    });
    CanopyView {
        ground: left,
        upper,
        lower,
    }
}

impl CanopyLayerGpu {
    /// The record of a layer of `look` grown at `density`, placed with the
    /// kernel's `seed`, swaying in a wind of mean `wind_speed`.
    pub fn of(look: &GrassBladeLook, density: f32, seed: u32, wind_speed: f32) -> Self {
        let bend = canopy_bend(wind_speed, look.stiffness);
        Self {
            root_color: look.root_color,
            color_variation: look.color_variation,
            tip_color: look.tip_color,
            macro_variation: look.macro_variation,
            clump_size: look.clump_size,
            height_variance: look.height_variance,
            leaf_area: leaf_area(look, density),
            bend,
            seed,
            sky_projection: projection(bend, SKY_MU),
            _pad: [0.0; 2],
            litter_color: look.litter_color,
            clumping: look.clumping,
            rule: GroundRuleGpu::EVERYWHERE,
        }
    }
}

/// Which run of the canopy block's layers a far field shades with, and how it
/// finds their density.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanopyRange {
    /// The run's first layer.
    pub first: u32,
    /// Layers in the run.
    pub count: u32,
    /// A planet's ground (density from the meadow) rather than a terrain's
    /// (density from its vertex colors).
    pub planet: bool,
}

impl CanopyRange {
    /// The range as the object record carries it: first layer in the low
    /// byte, count in the next, the planet flag above them.
    pub fn packed(self) -> u32 {
        (self.first & 0xff) | ((self.count & 0xff) << 8) | (u32::from(self.planet) << 16)
    }

    /// The range an object record's `packed` word carries.
    pub fn unpacked(packed: u32) -> Self {
        Self {
            first: packed & 0xff,
            count: (packed >> 8) & 0xff,
            planet: (packed >> 16) & 1 != 0,
        }
    }
}

/// A terrain's far field in the canopy block: its range, and which of the
/// terrain's own layers each layer of the range is.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainCanopy {
    /// The run of the block's layers.
    pub range: CanopyRange,
    /// Each layer's index among the terrain's layers.
    pub layers: Vec<usize>,
}

// A planet's ground as the block lays it out.
#[derive(Debug, Clone, PartialEq)]
struct PlanetCanopy {
    shape: PlanetShape,
    meadow_scale: f32,
    meadow_mean: f32,
    covers: GroundCovers,
}

/// The cosine from the blades' up the sky is taken to reach the ground along,
/// on average. Matches `sky_mu` in the shaders' `canopy_at`.
const SKY_MU: f32 = 2.0 / 3.0;

/// Every far field's layers in the world, and the planet's ground, with or
/// without grass on it: what each frame's canopy block and view tables are
/// filled from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CanopyField {
    layers: Vec<CanopyLayerGpu>,
    tables: Vec<[f32; 4]>,
    planet: Option<PlanetCanopy>,
    shore: Option<ShoreWaves>,
}

impl CanopyField {
    /// The far fields of `terrains` in a wind of mean `wind_speed`, and each
    /// terrain's run of layers (`None` for a terrain growing no blades, or past
    /// the block's room).
    pub fn for_terrains(
        terrains: &[GrassTerrain],
        wind_speed: f32,
    ) -> (Self, Vec<Option<TerrainCanopy>>) {
        let mut layers = Vec::new();
        let mut runs: Vec<Option<TerrainCanopy>> = alloc::vec![None; terrains.len()];
        for grown in grown_layers(terrains) {
            let run = runs[grown.terrain].get_or_insert_with(|| TerrainCanopy {
                range: CanopyRange {
                    first: u32::saturating_from(layers.len()),
                    count: 0,
                    planet: false,
                },
                layers: Vec::new(),
            });
            if run.layers.len() >= MAX_CANOPY_LAYERS_PER_TERRAIN
                || layers.len() >= MAX_CANOPY_LAYERS
            {
                continue;
            }
            let look = GrassBladeLook::of(&grown.grass);
            layers.push(CanopyLayerGpu::of(
                &look,
                grown.grass.density,
                grown.seed,
                wind_speed,
            ));
            run.layers.push(grown.layer);
            run.range.count += 1;
        }
        for run in &mut runs {
            if run.as_ref().is_some_and(|r| r.range.count == 0) {
                *run = None;
            }
        }
        (Self::new(layers, None), runs)
    }

    /// The ground of the planet `shape`, laid with `covers`, and the far field
    /// of the `grass` it grows, if any, in a wind of mean `wind_speed`; and
    /// the planet's run of layers, empty without grass.
    pub fn for_planet(
        shape: PlanetShape,
        grass: Option<&PlanetGrass>,
        covers: GroundCovers,
        wind_speed: f32,
    ) -> (Self, CanopyRange) {
        let layers: Vec<_> = grass
            .map_or(&[][..], |g| &g.layers)
            .iter()
            .take(MAX_CANOPY_LAYERS)
            .map(|l| CanopyLayerGpu {
                rule: l.rule,
                ..CanopyLayerGpu::of(&l.look, l.density, l.seed, wind_speed)
            })
            .collect();
        let seeds: Vec<u32> = layers.iter().map(|l| l.seed).collect();
        let range = CanopyRange {
            first: 0,
            count: u32::saturating_from(layers.len()),
            planet: true,
        };
        let meadow_scale = grass.map_or_else(
            || super::planet::density::meadow_scale(shape.radius),
            |g| g.meadow_scale,
        );
        let planet = PlanetCanopy {
            shape,
            meadow_scale,
            meadow_mean: meadow_mean(meadow_scale, &seeds),
            covers,
        };
        (Self::new(layers, Some(planet)), range)
    }

    /// The run of layers a planet's ground shades with, when the block lays
    /// one.
    pub fn planet_range(&self) -> Option<CanopyRange> {
        self.planet.as_ref().map(|_| CanopyRange {
            first: 0,
            count: u32::saturating_from(self.layers.len()),
            planet: true,
        })
    }

    fn new(layers: Vec<CanopyLayerGpu>, planet: Option<PlanetCanopy>) -> Self {
        let tables = layers
            .iter()
            .flat_map(|l| canopy_table(l.bend, l.height_variance))
            .collect();
        Self {
            layers,
            tables,
            planet,
            shore: None,
        }
    }

    /// The planet's ground reshaped to `shape`, keeping its center and
    /// radius, its beaches wet by `shore`.
    pub fn reshape(&mut self, shape: &PlanetShape, shore: Option<ShoreWaves>) {
        if let Some(planet) = self.planet.as_mut() {
            planet.shape.clone_from(shape);
        }
        self.shore = shore;
    }

    /// Wet the planet's beaches with the wash of `shore`, the waves of its
    /// sea.
    pub fn set_shore(&mut self, shore: Option<ShoreWaves>) {
        self.shore = shore;
    }

    /// Every layer's view table in layer order,
    /// [`CANOPY_TABLE_LEN`](super::canopy_table::CANOPY_TABLE_LEN) rows each,
    /// never more than [`MAX_CANOPY_TABLE_ROWS`](super::canopy_table::MAX_CANOPY_TABLE_ROWS).
    pub fn tables(&self) -> &[[f32; 4]] {
        &self.tables
    }

    /// This frame's block `time` seconds into the world, the blades handing
    /// the field over as `lod` has them, at hand-off phase `phase`; with no
    /// blades drawn (`hand_off` `None`) no layer is laid and the ground is
    /// bare under its covers. On a planet, `tile` is any tile drawn this
    /// frame, whose model places the planet's axes in the simulated frame.
    pub fn params(
        &self,
        hand_off: Option<(&GrassLodView, f32)>,
        tile: Option<&DrawnPlanetTile>,
        time: f64,
    ) -> CanopyParams {
        let mut params = CanopyParams::default();
        if let Some((lod, phase)) = hand_off {
            params.layer_count = u32::saturating_from(self.layers.len());
            params.band = [lod.fade_start(), lod.extent, phase, 0.0];
            params.layers[..self.layers.len()].copy_from_slice(&self.layers);
        }
        if let (Some(planet), Some(tile)) = (&self.planet, tile) {
            let center = planet_center(&planet.shape, tile);
            // The model turns the planet's axes into the frame's, so its
            // rotation's columns are the planet's axes seen from the frame.
            params.planet_rows = core::array::from_fn(|i| {
                let axis = [tile.model[i][0], tile.model[i][1], tile.model[i][2]];
                let d = axis[0] * center[0] + axis[1] * center[1] + axis[2] * center[2];
                [axis[0], axis[1], axis[2], -d]
            });
            params.planet_radius = narrow_f32(planet.shape.radius);
            params.meadow_scale = planet.meadow_scale;
            params.meadow_mean = planet.meadow_mean;
            params.covers = planet.covers.covers;
            params.cover_count = planet.covers.count;
            [params.ground_origin, params.ground_up] = ground_reference(&planet.shape, tile);
            set_ground_noise(&mut params, &planet.shape);
            if let Some(shore) = &self.shore {
                params.swash = ShoreSwashGpu::new(shore, time);
            }
        }
        params
    }
}

// The noise `shape`'s height is summed from, in `params`, for the ground's
// normals per pixel.
fn set_ground_noise(params: &mut CanopyParams, shape: &PlanetShape) {
    params.ground_turn = core::array::from_fn(|r| {
        core::array::from_fn(|c| {
            if c < 3 {
                narrow_f32(OCTAVE_TURN[c][r])
            } else {
                0.0
            }
        })
    });
    let octaves = shape.octave_count();
    for octave in 0..octaves {
        params.ground_salts[octave as usize / 4][octave as usize % 4] = shape.octave_salt(octave);
    }
    params.ground_frequency = narrow_f32(shape.base_frequency());
    params.ground_height_scale = narrow_f32(shape.noise_scale());
    params.ground_cell = narrow_f32(shape.radius / shape.base_frequency());
    params.ground_octaves = octaves;
    let c = &shape.continents;
    if !c.divides() {
        params.continent_octaves = 0;
        return;
    }
    const { assert!(CONTINENT_OCTAVES <= 8) };
    for octave in 0..CONTINENT_OCTAVES {
        params.continent_salts[octave as usize / 4][octave as usize % 4] = c.octave_salt(octave);
    }
    let per_sum = c.coast_scale() / octave_weight_sum(CONTINENT_OCTAVES);
    params.continent_frequency = narrow_f32(c.base_frequency(shape.radius));
    params.continent_scale = narrow_f32(per_sum);
    params.continent_offset = narrow_f32(c.threshold() * c.coast_scale());
    params.continent_octaves = CONTINENT_OCTAVES;
    params.beach_width = narrow_f32(c.beach_width());
    params.beach_rise = narrow_f32(c.beach_rise());
    params.hill_rise = narrow_f32(shape.hill_rise());
    params.ocean_depth = narrow_f32(c.ocean_depth());
    params.shelf_length = narrow_f32(c.shelf_length());
    params.seabed_lift = narrow_f32(c.seabed_lift(shape.hill_height()));
}

// The meadow's mean density over the sphere for `seeds`, sampled on a
// Fibonacci lattice: what the far field fades toward where a pixel spans many
// of its patches.
fn meadow_mean(scale: f32, seeds: &[u32]) -> f32 {
    const SAMPLES: u32 = 4096;
    if seeds.is_empty() {
        return 1.0;
    }
    let golden = core::f32::consts::PI * (3.0 - sqrt(5.0));
    let mut sum = 0.0;
    for i in 0..SAMPLES {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / SAMPLES as f32;
        let r = sqrt((1.0 - y * y).max(0.0));
        let a = golden * i as f32;
        let dir = [r * cos(a), y, r * sin(a)];
        for &seed in seeds {
            sum += meadow_density(dir, scale, seed);
        }
    }
    sum / (SAMPLES as usize * seeds.len()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Grass;
    use crate::planet::{CubeFace, TileId};
    use crate::render::environment::grass::litter::litter_color;
    use crate::terrain::TerrainGrid;
    use alloc::vec;
    use concinnity_core::math::vec3::{cross, dot, normalize_or, scale};

    const PI: f32 = core::f32::consts::PI;
    const UP: [f32; 3] = [0.0, 1.0, 0.0];

    fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    fn unit(v: [f32; 3]) -> [f32; 3] {
        normalize_or(v, 1e-9, UP)
    }

    // `grass_light`'s two parts for a blade of unit albedo: what multiplies the
    // albedo (diffuse plus translucency, its tint folded in), and the sheen.
    // Mirrors `grass_light_terms`.
    fn light_terms(n: [f32; 3], v: [f32; 3], l: [f32; 3]) -> ([f32; 3], f32) {
        let ndl = dot(n, l);
        let diffuse = ((ndl + 0.35) / 1.35).clamp(0.0, 1.0) / PI;
        let behind = (-ndl).clamp(0.0, 1.0);
        let through = powf((-dot(v, l)).clamp(0.0, 1.0), 4.0);
        let carried = (0.35 * behind + 0.9 * through) / PI;
        let tint = [1.1, 1.25, 0.6];
        let h = unit(add(v, l));
        let nl = ndl.clamp(0.0, 1.0);
        let nv = dot(n, v).max(1e-3);
        let a2 = 0.0625;
        let ndh = dot(n, h).max(0.0);
        let denom = ndh * ndh * (a2 - 1.0) + 1.0;
        let d = a2 / (PI * denom * denom + 0.0001);
        let k = 1.5 * 1.5 / 8.0;
        let g1 = |x: f32| x / (x * (1.0 - k) + k);
        let g = g1(dot(n, v).max(0.0)) * g1(nl);
        let spec = d * g / (4.0 * nv * nl).max(1e-3);
        let f = 0.04 + 0.96 * powf((1.0 - dot(h, v).clamp(0.0, 1.0)).clamp(0.0, 1.0), 5.0);
        (tint.map(|c| diffuse + carried * c), f * spec * 0.5 * nl)
    }

    // The canopy's blade light at height `t`: eight facings around the view,
    // each turned toward it and weighted by how much of it faces the view.
    // Mirrors `canopy_add_light`.
    fn canopy_terms(v: [f32; 3], l: [f32; 3], bend: f32, t: f32) -> ([f32; 3], f32) {
        let flat = add(v, scale(UP, -dot(v, UP)));
        let h = normalize_or(flat, 1e-4, [1.0, 0.0, 0.0]);
        let s = cross(UP, h);
        let tilt = blade_tilt(bend, t);
        let (mut k, mut sheen, mut total) = ([0.0; 3], 0.0, 0.0);
        for j in 0..CANOPY_FACINGS {
            let a = 2.0 * PI * (j as f32 + 0.5) / CANOPY_FACINGS as f32;
            let front = add(scale(h, cos(a)), scale(s, sin(a)));
            let mut n = add(scale(front, cos(tilt)), scale(UP, -sin(tilt)));
            if dot(n, v) < 0.0 {
                n = scale(n, -1.0);
            }
            let w = dot(n, v);
            let (kj, sj) = light_terms(n, v, l);
            for c in 0..3 {
                k[c] += w * kj[c];
            }
            sheen += w * sj;
            total += w;
        }
        (k.map(|c| c / total), sheen / total)
    }

    // The blade model averaged by brute force over the kernel's blades at
    // height `t`: every facing, the lean's spread, the roll across the width,
    // each fragment weighted by the screen area it covers.
    fn blades_terms(v: [f32; 3], l: [f32; 3], t: f32) -> ([f32; 3], f32) {
        let (mut k, mut sheen, mut total) = ([0.0; 3], 0.0, 0.0);
        let n = 48;
        for i in 0..n {
            let facing = 2.0 * PI * (i as f32 + 0.5) / n as f32;
            for a in 0..4 {
                for b in 0..4 {
                    let lean = (0.15 + 0.35 * (a as f32 + 0.5) / 4.0)
                        * (0.6 + 0.8 * (b as f32 + 0.5) / 4.0);
                    let side = [cos(facing), 0.0, sin(facing)];
                    let front = [-sin(facing), 0.0, cos(facing)];
                    let rise = sqrt(1.0 - lean * lean);
                    let tangent = unit(add(scale(front, t * lean), scale(UP, (1.0 - t) * rise)));
                    let normal = unit(cross(side, tangent));
                    let facing_view = if dot(normal, v) < 0.0 {
                        scale(normal, -1.0)
                    } else {
                        normal
                    };
                    for r in 0..4 {
                        let u = -0.75 + 0.5 * r as f32;
                        let rolled = unit(add(facing_view, scale(side, u * 0.45)));
                        let w = dot(facing_view, v).abs();
                        let (kj, sj) = light_terms(rolled, v, l);
                        for c in 0..3 {
                            k[c] += w * kj[c];
                        }
                        sheen += w * sj;
                        total += w;
                    }
                }
            }
        }
        (k.map(|c| c / total), sheen / total)
    }

    fn dir(elevation_deg: f32, azimuth_deg: f32) -> [f32; 3] {
        let (e, a) = (elevation_deg.to_radians(), azimuth_deg.to_radians());
        [cos(e) * cos(a), sin(e), cos(e) * sin(a)]
    }

    // The canopy's facings stand for the whole field of blades under any
    // light: a high sun, a low one ahead, behind or beside the view, a moon off
    // to one side, and a view along the ground, at a slant, or straight down.
    #[test]
    fn the_canopy_light_is_the_blades_averaged() {
        let lights = [
            ("high sun", dir(60.0, 30.0)),
            ("low sun ahead", dir(8.0, 180.0)),
            ("low sun behind", dir(8.0, 10.0)),
            ("low sun aside", dir(3.0, 90.0)),
            ("moon aside", dir(35.0, 95.0)),
            ("moon behind", dir(25.0, -120.0)),
        ];
        let views = [dir(2.0, 0.0), dir(35.0, 0.0), dir(85.0, 0.0)];
        for (name, l) in lights {
            for v in views {
                for t in [0.25, 0.6, 0.9] {
                    let (ck, cs) = canopy_terms(v, l, CANOPY_BEND, t);
                    let (bk, bs) = blades_terms(v, l, t);
                    // Green carries most of what the eye sees.
                    let got = ck[1] + cs;
                    let want = bk[1] + bs;
                    assert!(
                        (got - want).abs() <= 0.12 * want + 0.004,
                        "{name} v {v:?} t {t}: canopy {got} blades {want}"
                    );
                }
            }
        }
    }

    fn mean_height(v: &CanopyView) -> f32 {
        (v.upper.share * v.upper.height + v.lower.share * v.lower.height)
            / (v.upper.share + v.lower.share)
    }

    #[test]
    fn a_view_ray_stops_on_the_ground_or_a_blade() {
        for hv in [0.0, 0.45, 1.0] {
            for mu in [0.02, 0.1, 0.5, 1.0] {
                for lai in [0.0, 0.3, 1.0, 4.0] {
                    let v = canopy_view(CANOPY_BEND, hv, mu, lai);
                    let sum = v.ground + v.upper.share + v.lower.share;
                    assert!((sum - 1.0).abs() < 1e-5, "{hv} {mu} {lai}: {sum}");
                    let depth = projection(CANOPY_BEND, mu) * lai / mu;
                    assert!((v.ground - exp(-depth)).abs() < 1e-5);
                    for b in [v.upper, v.lower] {
                        if b.share > 1e-4 {
                            assert!((0.0..=1.0).contains(&b.height), "{hv} {mu} {lai} {b:?}");
                        }
                    }
                }
            }
        }
        // An empty field shows the ground; a dense one seen along the ground,
        // its upper half.
        assert_eq!(canopy_view(CANOPY_BEND, 0.45, 0.5, 0.0).ground, 1.0);
        let dense = canopy_view(CANOPY_BEND, 0.45, 0.02, 4.0);
        assert!(dense.upper.share > 0.99, "{dense:?}");
    }

    // The same rays followed through finely sampled heights of the same three
    // blade heights: what the segments give is the field's own average, band
    // by band.
    #[test]
    fn the_segments_match_a_fine_integration() {
        let n = 4000;
        for bend in [CANOPY_BEND, 0.45] {
            for hv in [0.0, 0.45] {
                for mu in [0.03, 0.3, 0.9] {
                    for lai in [0.5, 1.0, 3.0] {
                        let heights = class_heights(hv);
                        let top = heights.iter().fold(0.0f32, |a, &h| a.max(h));
                        // Per fine step: the seen area, and its sums of t and of
                        // the leaf above.
                        let steps: Vec<[f32; 3]> = (0..n)
                            .map(|i| {
                                let z = top * (1.0 - (i as f32 + 0.5) / n as f32);
                                let mut acc = [0.0f32; 3];
                                let above = leaf_above(z, hv);
                                for &h in &heights {
                                    if z < h {
                                        let t = z / h;
                                        let a = blade_width(t)
                                            * part_projection(blade_tilt(bend, t), mu);
                                        acc[0] += a;
                                        acc[1] += a * t;
                                        acc[2] += a * above;
                                    }
                                }
                                acc
                            })
                            .collect();
                        let total: f32 = steps.iter().map(|s| s[0]).sum();
                        let depth = projection(bend, mu) * lai / mu;
                        let (mut passed, mut left) = (0.0, 1.0f32);
                        let mut want = [[0.0f32; 3]; 2];
                        for (i, st) in steps.iter().enumerate() {
                            passed += st[0];
                            let after = exp(-depth * passed / total);
                            let stopped = left - after;
                            left = after;
                            let q = stopped / st[0].max(1e-12);
                            let band = &mut want[usize::from(i >= n / 2)];
                            band[0] += stopped;
                            band[1] += q * st[1];
                            band[2] += q * st[2];
                        }
                        let got = canopy_view(bend, hv, mu, lai);
                        for (b, w) in [got.upper, got.lower].iter().zip(want) {
                            let case = alloc::format!("{bend} {hv} {mu} {lai}: {b:?} {w:?}");
                            assert!((b.share - w[0]).abs() < 0.04, "{case}");
                            if w[0] > 0.05 {
                                assert!((b.height - w[1] / w[0]).abs() < 0.05, "{case}");
                                assert!((b.above - w[2] / w[0]).abs() < 0.05, "{case}");
                            }
                        }
                    }
                }
            }
        }
    }

    // Seen along the ground, the arched-over tips face up and hide little, so
    // a ray reaches further down the blades than their area alone suggests; a
    // wind bends them further still, and the tallest blades of an uneven
    // field are seen further down too.
    #[test]
    fn grazing_rays_pass_the_tips() {
        let level = canopy_view(CANOPY_BEND, 0.0, 0.03, 1.0);
        let n = 4000;
        let (mut passed, mut left, mut height, mut stopped) = (0.0, 1.0f32, 0.0, 0.0);
        let total: f32 = (0..n)
            .map(|i| blade_width((i as f32 + 0.5) / n as f32))
            .sum();
        let depth = projection(CANOPY_BEND, 0.03) / 0.03;
        for i in 0..n {
            let t = 1.0 - (i as f32 + 0.5) / n as f32;
            passed += blade_width(t);
            let after = exp(-depth * passed / total);
            height += (left - after) * t;
            stopped += left - after;
            left = after;
        }
        let seen = mean_height(&level);
        assert!(
            seen < height / stopped - 0.02,
            "{seen} {}",
            height / stopped
        );
        let windy = canopy_view(canopy_bend(8.0, 0.5), 0.0, 0.03, 1.0);
        assert!(mean_height(&windy) < seen);
        let uneven = canopy_view(CANOPY_BEND, 0.45, 0.03, 1.0);
        assert!(
            mean_height(&uneven) < seen - 0.01,
            "{}",
            mean_height(&uneven)
        );
    }

    // The height classes stand for the kernel's spread: equally likely, their
    // mean the field's mean, each the mean of its tercile.
    #[test]
    fn the_height_classes_are_the_kernels_terciles() {
        let n = 600;
        let mut spread: Vec<f32> = (0..n * n)
            .map(|i| {
                let (u, v) = ((i % n) as f32 + 0.5, (i / n) as f32 + 0.5);
                1.2 * (u / n as f32 - 0.5) + 0.8 * (v / n as f32 - 0.5)
            })
            .collect();
        spread.sort_by(f32::total_cmp);
        let third = spread.len() / 3;
        for (k, q) in CANOPY_HEIGHT_CLASSES.iter().enumerate() {
            let part = &spread[k * third..(k + 1) * third];
            let mean = part.iter().sum::<f32>() / part.len() as f32;
            assert!((mean - q).abs() < 2e-3, "{k}: {mean} {q}");
        }
        assert_eq!(CANOPY_HEIGHT_CLASSES.iter().sum::<f32>(), 0.0);
    }

    // Leaf area, projection and bend follow from the look and the wind alone:
    // a default meadow grows around one square meter of blade per square
    // meter of ground, and its blades show more of themselves along the
    // ground than from above.
    #[test]
    fn canopy_terms_follow_from_the_look() {
        let g = Grass {
            height: 0.55,
            width: 0.03,
            density: 100.0,
            ..Grass::default()
        };
        let look = GrassBladeLook::of(&g);
        let lai = leaf_area(&look, g.density);
        assert!((lai - 100.0 * 0.03 * 0.55 * (1.0 - 1.0 / 2.4)).abs() < 1e-5);
        assert!(lai > 0.8 && lai < 1.2, "{lai}");
        assert!(projection(CANOPY_BEND, 0.0) > projection(CANOPY_BEND, 1.0));
        // The segments against the projection averaged over every facing and
        // height, at a slant.
        let mu: f32 = 0.5;
        let (mut want, mut total) = (0.0, 0.0);
        for i in 0..64 {
            let t = (i as f32 + 0.5) / 64.0;
            let tilt = blade_tilt(CANOPY_BEND, t);
            let w = blade_width(t);
            for j in 0..64 {
                let a = 2.0 * PI * (j as f32 + 0.5) / 64.0;
                let n = [cos(tilt) * cos(a), -sin(tilt), cos(tilt) * sin(a)];
                let view = [sqrt(1.0 - mu * mu), mu, 0.0];
                want += w * dot(n, view).abs();
                total += w;
            }
        }
        let want = want / total;
        let got = projection(CANOPY_BEND, mu);
        assert!((got - want).abs() < 0.06 * want, "{got} {want}");
        // Still air leaves the blade's own lean; a wind bends a soft blade
        // further than a stiff one, never past flat.
        assert_eq!(canopy_bend(0.0, 0.5), CANOPY_BEND);
        assert!(canopy_bend(4.0, 0.2) > canopy_bend(4.0, 0.8));
        assert!(canopy_bend(4.0, 0.8) > CANOPY_BEND);
        assert!(canopy_bend(1e6, 0.0) <= 0.95);
        let rec = CanopyLayerGpu::of(&look, g.density, 77, 4.0);
        assert_eq!(rec.leaf_area, lai);
        assert_eq!(rec.seed, 77);
        assert_eq!(rec.bend, canopy_bend(4.0, look.stiffness));
        assert_eq!(rec.macro_variation, g.macro_variation);
        assert_eq!(rec.sky_projection, projection(rec.bend, SKY_MU));
        assert_eq!(rec.clumping, g.clumping);
        assert_eq!(rec.litter_color, litter_color(g.root_color));
        // An authored litter color is taken as it is.
        let named = Grass {
            litter_color: Some([0.1, 0.07, 0.03]),
            ..g
        };
        let rec = CanopyLayerGpu::of(&GrassBladeLook::of(&named), 100.0, 1, 0.0);
        assert_eq!(rec.litter_color, [0.1, 0.07, 0.03]);
    }

    #[test]
    fn the_tilt_rises_from_upright_to_level() {
        assert_eq!(blade_tilt(CANOPY_BEND, 0.0), 0.0);
        assert!((blade_tilt(CANOPY_BEND, 1.0) - PI / 2.0).abs() < 1e-6);
        assert_eq!(blade_tilt(0.0, 0.7), 0.0);
        assert!(blade_tilt(CANOPY_BEND, 0.5) < blade_tilt(CANOPY_BEND, 0.8));
    }

    // All of the leaf stands above the ground and none above the tallest
    // tips, falling in between; a blade's narrow tip holds little of it.
    #[test]
    fn the_leaf_above_falls_from_the_ground_to_the_tips() {
        for hv in [0.0, 0.45, 1.0] {
            assert!((leaf_above(0.0, hv) - 1.0).abs() < 1e-6);
            let top = class_heights(hv).iter().fold(0.0f32, |a, &h| a.max(h));
            assert!(leaf_above(top, hv).abs() < 1e-6);
            let mut last = 1.0;
            for i in 1..=40 {
                let z = top * i as f32 / 40.0;
                let a = leaf_above(z, hv);
                assert!(a <= last + 1e-6, "{hv} {z}: {a} after {last}");
                last = a;
            }
        }
        // Even blades: the share above the middle, integrated finely.
        let n = 4000;
        let (mut above, mut total) = (0.0, 0.0);
        for i in 0..n {
            let t = (i as f32 + 0.5) / n as f32;
            total += blade_width(t);
            if t > 0.5 {
                above += blade_width(t);
            }
        }
        assert!((leaf_above(0.5, 0.0) - above / total).abs() < 1e-3);
        assert!(leaf_above(0.5, 0.0) < 0.35);
    }

    #[test]
    fn the_leaf_constants_are_the_shaders() {
        let src = crate::render::shader::embedded::GRASS_FIELD;
        let float = |name| crate::render::shader::consts::float(src, name);
        assert_eq!(float("GRASS_WIDTH_TAPER"), WIDTH_TAPER);
        assert_eq!(float("GRASS_HEIGHT_CLASS"), CANOPY_HEIGHT_CLASSES[2]);
        assert_eq!(CANOPY_HEIGHT_CLASSES[0], -CANOPY_HEIGHT_CLASSES[2]);
    }

    // The ellipse the blade draws use stays within a few percent of the
    // projection from straight down to grazing, for any bend.
    #[test]
    fn the_projection_ellipse_fits_the_blades() {
        for bend in [0.0, CANOPY_BEND, 0.5, canopy_bend(8.0, 0.2), 0.95] {
            let axes = projection_axes(bend);
            for i in 0..=100 {
                let mu = i as f32 / 100.0;
                let want = projection(bend, mu);
                let got = ellipse_projection(axes, mu);
                assert!(
                    (got - want).abs() <= 0.06 * want + 1e-6,
                    "{bend} {mu}: {got} {want}"
                );
            }
        }
    }

    fn terrain(layers: Vec<Grass>) -> GrassTerrain {
        GrassTerrain {
            center: [0.0; 3],
            grid: TerrainGrid::new(4, [10.0, 10.0], vec![0.0; 25]).unwrap(),
            layers: layers.into_iter().map(|g| (g, None)).collect(),
        }
    }

    // Each terrain shades its own run of the block, in the order the kernel
    // draws them and with the kernel's seeds, at most three to a terrain.
    #[test]
    fn terrains_take_runs_with_the_kernels_seeds() {
        let hidden = Grass {
            visible: false,
            ..Grass::default()
        };
        let bare = Grass {
            density: 0.0,
            ..Grass::default()
        };
        let terrains = vec![
            terrain(vec![Grass::default(), hidden, Grass::default()]),
            terrain(vec![bare]),
            terrain(vec![Grass::default(); 5]),
        ];
        let (field, runs) = CanopyField::for_terrains(&terrains, 0.0);
        let first = runs[0].as_ref().unwrap();
        assert_eq!(
            first.range,
            CanopyRange {
                first: 0,
                count: 2,
                planet: false
            }
        );
        assert_eq!(first.layers, vec![0, 2]);
        assert_eq!(runs[1], None);
        let third = runs[2].as_ref().unwrap();
        assert_eq!(third.range.first, 2);
        assert_eq!(third.range.count, 3);
        assert_eq!(third.layers, vec![0, 1, 2]);
        let grass = super::super::GrassField::resolve(&terrains, None).unwrap();
        assert_eq!(field.layers[0].seed, grass.layers[0].seed);
        assert_eq!(field.layers[1].seed, grass.layers[1].seed);
        assert_eq!(field.layers[2].seed, grass.layers[2].seed);
        let lod = grass.lod_view(1000.0);
        let p = field.params(Some((&lod, 5.0)), None, 0.0);
        assert_eq!(p.layer_count, 5);
        assert_eq!(p.planet_rows, [[0.0; 4]; 3]);
        // The band is the blades' hand-off, at the frame's phase.
        assert_eq!(p.band, [lod.fade_start(), lod.extent, 5.0, 0.0]);
        // One view table per layer, in layer order.
        use super::super::canopy_table::{CANOPY_TABLE_LEN, canopy_table};
        assert_eq!(field.tables().len(), 5 * CANOPY_TABLE_LEN);
        let l = &field.layers[3];
        assert_eq!(
            field.tables()[3 * CANOPY_TABLE_LEN..4 * CANOPY_TABLE_LEN],
            canopy_table(l.bend, l.height_variance)
        );
    }

    #[test]
    fn a_range_packs_into_one_word() {
        let r = CanopyRange {
            first: 5,
            count: 3,
            planet: true,
        };
        assert_eq!(r.packed(), 5 | (3 << 8) | (1 << 16));
        assert_eq!(CanopyRange::unpacked(r.packed()), r);
        let terrain = CanopyRange { planet: false, ..r };
        assert_eq!(CanopyRange::unpacked(terrain.packed()), terrain);
    }

    #[test]
    fn a_planet_past_the_block_keeps_the_tables_in_bounds() {
        use super::super::canopy_table::MAX_CANOPY_TABLE_ROWS;
        let shape = PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 6,
            seed: 1,
            continents: crate::planet::Continents::NONE,
            edits: None,
        };
        let grass = PlanetGrass::everywhere(
            shape.clone(),
            &vec![Grass::default(); MAX_CANOPY_LAYERS + 4],
        )
        .unwrap();
        assert!(grass.layers.len() > MAX_CANOPY_LAYERS);
        let (field, range) =
            CanopyField::for_planet(shape, Some(&grass), GroundCovers::default(), 0.0);
        assert_eq!(range.count as usize, MAX_CANOPY_LAYERS);
        assert_eq!(field.tables().len(), MAX_CANOPY_TABLE_ROWS);
    }

    // On a planet the block's rows take a point of the simulated frame to the
    // planet's own axes about its center, whichever way the frame is turned.
    #[test]
    fn the_planet_rows_undo_the_frame() {
        let shape = PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 6,
            seed: 1,
            continents: crate::planet::Continents::NONE,
            edits: None,
        };
        let grass = PlanetGrass::everywhere(shape.clone(), &[Grass::default()]).unwrap();
        let (field, range) =
            CanopyField::for_planet(shape, Some(&grass), GroundCovers::default(), 0.0);
        assert_eq!(range.count, 1);
        assert!(range.planet);
        let id = TileId {
            face: CubeFace(2),
            level: 10,
            x: 512,
            y: 511,
        };
        // A frame turned 30 degrees about x and moved, as a rebase leaves it.
        let (s, c) = (
            sin(core::f32::consts::FRAC_PI_6),
            cos(core::f32::consts::FRAC_PI_6),
        );
        let model = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, s, 0.0],
            [0.0, -s, c, 0.0],
            [3.0, -2.0, 7.0, 1.0],
        ];
        let tile = DrawnPlanetTile {
            id,
            model,
            vertex_base: 0,
            radial: [0.0, 1.0],
            origin_altitude: 0.0,
            morph: crate::gfx::render_types::GpuTileMorph::default(),
            morph_extent: 0.0,
        };
        let p = field.params(
            Some((&GrassLodView::new(1000.0, 0.5), 0.0)),
            Some(&tile),
            0.0,
        );
        assert_eq!(p.planet_radius, 50_000.0);
        assert!(
            p.meadow_mean > 0.5 && p.meadow_mean < 1.0,
            "{}",
            p.meadow_mean
        );
        // The tile's origin, drawn at the model's translation, sits along the
        // tile's own direction from the center.
        let at = [model[3][0], model[3][1], model[3][2]];
        let rel: [f32; 3] = core::array::from_fn(|i| {
            let r = p.planet_rows[i];
            dot([r[0], r[1], r[2]], at) + r[3]
        });
        let want = concinnity_core::math::dvec3::to_f32(id.direction(0.5, 0.5));
        let got = unit(rel);
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() < 1e-5, "{got:?} {want:?}");
        }
    }

    // Each planet layer's rule and the planet's covers ride the block, so the
    // ground under the grass and its far field place them as the kernel does.
    #[test]
    fn the_block_carries_the_shores_wash() {
        use crate::components::Planet;
        use crate::render::uniforms::canopy::ShoreSwashGpu;
        use crate::water::spectrum::SeaShape;
        let planet = Planet {
            land_share: 0.4,
            ..Planet::default()
        };
        let shape = planet.shape();
        let grass = PlanetGrass::resolve(
            shape.clone(),
            &[(Grass::default(), crate::components::GroundRule::default())],
        )
        .unwrap();
        let covers = GroundCovers::of(&planet, |_| None, 1);
        let (mut field, _) = CanopyField::for_planet(shape.clone(), Some(&grass), covers, 0.0);
        let id = TileId {
            face: CubeFace(2),
            level: 10,
            x: 512,
            y: 511,
        };
        let mesh = crate::planet::tile_mesh(&shape, id);
        let tile = DrawnPlanetTile {
            id,
            model: crate::planet::LocalFrame::AUTHORED.model_at(mesh.origin),
            vertex_base: 0,
            radial: mesh.radial,
            origin_altitude: mesh.origin_altitude,
            morph: crate::gfx::render_types::GpuTileMorph::default(),
            morph_extent: 0.0,
        };
        let lod = GrassLodView::new(1000.0, 0.5);
        assert_eq!(
            field.params(Some((&lod, 0.0)), Some(&tile), 7.0).swash,
            ShoreSwashGpu::default()
        );
        let sea = SeaShape::new(7.0, 50_000.0).unwrap();
        let shore = ShoreWaves::new(&shape, 0.0, &sea, None).unwrap();
        field.set_shore(Some(shore.clone()));
        let p = field.params(Some((&lod, 0.0)), Some(&tile), 7.0);
        assert_eq!(p.swash, ShoreSwashGpu::new(&shore, 7.0));
        assert!(p.swash.runup > 0.0);
        assert_ne!(
            p.swash.phases,
            field
                .params(Some((&lod, 0.0)), Some(&tile), 8.0)
                .swash
                .phases
        );
        // Off the planet, or before a tile is drawn, nothing washes.
        assert_eq!(
            field.params(Some((&lod, 0.0)), None, 7.0).swash,
            ShoreSwashGpu::default()
        );
    }

    #[test]
    fn the_block_carries_layer_rules_and_covers() {
        use crate::components::{GroundCover, GroundRule, Planet};
        let shape = PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 6,
            seed: 1,
            continents: crate::planet::Continents::NONE,
            edits: None,
        };
        let above = GroundRule {
            altitude: [Some(2.0), None],
            ..GroundRule::default()
        };
        let sand = GroundCover {
            rule: GroundRule {
                altitude: [None, Some(2.0)],
                ..GroundRule::default()
            },
            ..GroundCover::default()
        };
        let planet = Planet {
            covers: vec![sand.clone()],
            ..Planet::default()
        };
        let covers = GroundCovers::of(&planet, |_| None, 1);
        let grass = PlanetGrass::resolve(
            shape.clone(),
            &[
                (Grass::default(), GroundRule::default()),
                (Grass::default(), above),
            ],
        )
        .unwrap();
        let (field, _) = CanopyField::for_planet(shape.clone(), Some(&grass), covers, 0.0);
        let id = TileId {
            face: CubeFace(2),
            level: 10,
            x: 512,
            y: 511,
        };
        let tile_mesh = crate::planet::tile_mesh(&shape, id);
        let tile = DrawnPlanetTile {
            id,
            model: crate::planet::LocalFrame::AUTHORED.model_at(tile_mesh.origin),
            vertex_base: 0,
            radial: tile_mesh.radial,
            origin_altitude: tile_mesh.origin_altitude,
            morph: crate::gfx::render_types::GpuTileMorph::default(),
            morph_extent: 0.0,
        };
        let p = field.params(
            Some((&GrassLodView::new(1000.0, 0.5), 0.0)),
            Some(&tile),
            0.0,
        );
        assert_eq!(p.layers[0].rule, GroundRuleGpu::EVERYWHERE);
        assert_eq!(p.layers[1].rule, GroundRuleGpu::of(&above));
        assert_eq!(p.cover_count, 1);
        assert_eq!(p.covers[0].rule, GroundRuleGpu::of(&sand.rule));
        // The block's reference gives each corner of the tile its height.
        let mesh = tile_mesh;
        let center = planet_center(&shape, &tile);
        for v in mesh.vertices.iter().step_by(97) {
            let at: [f32; 3] = core::array::from_fn(|i| p.ground_origin[i] + v.pos[i]);
            let off: [f32; 3] = core::array::from_fn(|i| at[i] - p.ground_origin[i]);
            let rel: [f32; 3] = core::array::from_fn(|i| at[i] - center[i]);
            let up = [p.ground_up[0], p.ground_up[1], p.ground_up[2]];
            let got = crate::render::environment::ground::ground_altitude(
                p.ground_origin[3],
                p.ground_up[3],
                dot(up, off),
                dot(off, off),
                concinnity_core::math::vec3::length(rel),
            );
            let world = concinnity_core::math::dvec3::add(
                mesh.origin,
                concinnity_core::math::dvec3::from_f32(v.pos),
            );
            let dir = concinnity_core::math::dvec3::normalize_or(
                concinnity_core::math::dvec3::sub(world, shape.center),
                [0.0, 1.0, 0.0],
            );
            let want = shape.height(dir);
            assert!((f64::from(got) - want).abs() < 0.01, "{got} {want}");
        }
        // Off a planet nothing is laid.
        let (terrain, _) = CanopyField::for_terrains(&[], 0.0);
        let t = terrain.params(Some((&GrassLodView::new(1000.0, 0.5), 0.0)), None, 0.0);
        assert_eq!(t.cover_count, 0);
    }

    // A planet growing no grass still lays its ground in the block: its
    // frame, noise, covers and the shore's wash, with no layers.
    #[test]
    fn a_grassless_planet_lays_its_ground() {
        use crate::components::{GroundCover, GroundRule, Planet};
        use crate::render::uniforms::canopy::ShoreSwashGpu;
        use crate::water::spectrum::SeaShape;
        let sand = GroundCover {
            rule: GroundRule {
                altitude: [None, Some(2.0)],
                ..GroundRule::default()
            },
            ..GroundCover::default()
        };
        let planet = Planet {
            land_share: 0.4,
            covers: vec![sand.clone()],
            ..Planet::default()
        };
        let shape = planet.shape();
        let covers = GroundCovers::of(&planet, |_| None, 1);
        let (mut field, range) = CanopyField::for_planet(shape.clone(), None, covers, 0.0);
        assert_eq!(
            range,
            CanopyRange {
                first: 0,
                count: 0,
                planet: true
            }
        );
        assert_ne!(range.packed(), 0);
        assert_eq!(field.planet_range(), Some(range));
        assert!(field.tables().is_empty());
        let sea = SeaShape::new(7.0, 50_000.0).unwrap();
        let shore = ShoreWaves::new(&shape, 0.0, &sea, None).unwrap();
        field.set_shore(Some(shore.clone()));
        let id = TileId {
            face: CubeFace(2),
            level: 10,
            x: 512,
            y: 511,
        };
        let mesh = crate::planet::tile_mesh(&shape, id);
        let tile = DrawnPlanetTile {
            id,
            model: crate::planet::LocalFrame::AUTHORED.model_at(mesh.origin),
            vertex_base: 0,
            radial: mesh.radial,
            origin_altitude: mesh.origin_altitude,
            morph: crate::gfx::render_types::GpuTileMorph::default(),
            morph_extent: 0.0,
        };
        let p = field.params(None, Some(&tile), 7.0);
        assert_eq!(p.layer_count, 0);
        assert_eq!(p.band, [0.0; 4]);
        assert_eq!(p.planet_radius, narrow_f32(shape.radius));
        assert_eq!(p.ground_octaves, shape.octave_count());
        assert!(p.continent_octaves > 0);
        assert_eq!(p.cover_count, 1);
        assert_eq!(p.covers[0].rule, GroundRuleGpu::of(&sand.rule));
        assert_eq!(p.swash, ShoreSwashGpu::new(&shore, 7.0));
        assert_eq!(
            [p.ground_origin, p.ground_up],
            ground_reference(&shape, &tile)
        );
        // The same ground as under grass, but for the layers.
        let grass = PlanetGrass::everywhere(shape.clone(), &[Grass::default()]).unwrap();
        let (mut grassy, _) = CanopyField::for_planet(shape, Some(&grass), covers, 0.0);
        grassy.set_shore(Some(shore));
        let lod = GrassLodView::new(1000.0, 0.5);
        let under = grassy.params(Some((&lod, 0.0)), Some(&tile), 7.0);
        assert_eq!(under.layer_count, 1);
        let bare = CanopyParams {
            layer_count: 0,
            band: [0.0; 4],
            layers: CanopyParams::default().layers,
            meadow_mean: p.meadow_mean,
            ..under
        };
        assert_eq!(bytemuck::bytes_of(&bare), bytemuck::bytes_of(&p));
        // With no blades drawn this frame, grass lays no layer either.
        assert_eq!(grassy.params(None, Some(&tile), 7.0).layer_count, 0);
    }

    // The normal the shader's `ground_normal_at` returns, over the block's noise.
    fn shader_ground_normal(p: &CanopyParams, dir: [f32; 3], footprint: f32) -> [f32; 3] {
        shader_ground(p, dir, footprint).0
    }

    // The shader's `ground_normal_at` where no edits lie, normal and height.
    fn shader_ground(p: &CanopyParams, dir: [f32; 3], footprint: f32) -> ([f32; 3], f32) {
        shader_edited_ground(p, dir, footprint, [1.0, 0.0, 0.0], [0.0; 3])
    }

    // The shader's `ground_normal_at`, normal and height, line for line, the
    // tile's corners carrying `edit` with tangent `tangent`.
    fn shader_edited_ground(
        p: &CanopyParams,
        dir: [f32; 3],
        footprint: f32,
        edit: [f32; 3],
        tangent: [f32; 3],
    ) -> ([f32; 3], f32) {
        let octaves = p.ground_octaves.min(16);
        let weight = if edit[0] > 0.999 { 1.0 } else { edit[0] };
        let hills = shader_octaves(
            p,
            dir,
            p.ground_frequency,
            p.ground_cell,
            octaves,
            &p.ground_salts,
            footprint,
        )
        .map(|v| v * (p.ground_height_scale * weight));
        let mut gradient = [hills[0], hills[1], hills[2]];
        let mut height = hills[3];
        if p.continent_octaves > 0 {
            let salts = [p.continent_salts[0], p.continent_salts[1], [0; 4], [0; 4]];
            let field = shader_octaves(
                p,
                dir,
                p.continent_frequency,
                p.planet_radius / p.continent_frequency,
                p.continent_octaves.min(8),
                &salts,
                footprint,
            )
            .map(|v| v * p.continent_scale);
            let shore = shader_shore(p, field[3] - p.continent_offset, hills[3]);
            gradient = core::array::from_fn(|i| field[i] * shore[0] + hills[i] * shore[1]);
            height = shore[2];
        }
        if edit[1] != 0.0 || edit[2] != 0.0 {
            let t = dot(dir, tangent);
            let across = unit(core::array::from_fn(|i| tangent[i] - dir[i] * t));
            let side = cross(dir, across);
            gradient =
                core::array::from_fn(|i| gradient[i] + across[i] * edit[1] + side[i] * edit[2]);
        }
        let d = dot(dir, gradient);
        let normal = unit(core::array::from_fn(|i| {
            dir[i] - (gradient[i] - dir[i] * d) / p.planet_radius
        }));
        (normal, height)
    }

    // The shader's `ground_octaves`, line for line.
    fn shader_octaves(
        p: &CanopyParams,
        dir: [f32; 3],
        frequency: f32,
        mut cell: f32,
        octaves: u32,
        salts: &[[u32; 4]; 4],
        footprint: f32,
    ) -> [f32; 4] {
        let row = |r: usize| {
            [
                p.ground_turn[r][0],
                p.ground_turn[r][1],
                p.ground_turn[r][2],
            ]
        };
        let turn = [row(0), row(1), row(2)];
        let mul_mv = |m: &[[f32; 3]; 3], v: [f32; 3]| m.map(|r| dot(r, v));
        let mul_mm = |a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]| {
            core::array::from_fn(|r| {
                core::array::from_fn(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum())
            })
        };
        let mut to_lattice: [[f32; 3]; 3] = turn.map(|r| r.map(|v| v * frequency));
        let mut weight = 1.0;
        let mut sum = [0.0f32; 4];
        for o in 0..octaves as usize {
            let fade = ((cell / (2.0 * footprint).max(1e-4)).log2()).clamp(0.0, 1.0);
            if fade <= 0.0 {
                let left = (octaves as usize - o) as f32;
                sum[3] += weight * (2.0 - 2f32.powf(1.0 - left)) * 0.5;
                break;
            }
            let salt = salts[o / 4][o % 4];
            let offset = [salt, salt >> 10, salt >> 20].map(|b| (b & 0x3ff) as f32 / 1024.0);
            let q = mul_mv(&to_lattice, dir);
            let n = shader_noise(core::array::from_fn(|i| q[i] + offset[i]), salt);
            for c in 0..3 {
                let back: f32 = (0..3).map(|r| n[r] * to_lattice[r][c]).sum();
                sum[c] += back * weight * fade;
            }
            sum[3] += (n[3] * fade + 0.5 * (1.0 - fade)) * weight;
            to_lattice = mul_mm(&turn, &to_lattice).map(|r: [f32; 3]| r.map(|v| v * 2.0));
            weight *= 0.5;
            cell *= 0.5;
        }
        sum
    }

    // The shader's `ground_shore`, line for line.
    fn shader_shore(p: &CanopyParams, inland: f32, hill: f32) -> [f32; 3] {
        let rise = p.hill_rise.max(1e-3);
        if inland >= 0.0 {
            let w = p.beach_width;
            let t = (inland / w).min(1.0);
            let x = ((inland - w) / rise).clamp(0.0, 1.0);
            let fade = x * x * (3.0 - 2.0 * x);
            let fade_d = 6.0 * x * (1.0 - x);
            return [
                p.beach_rise * 2.0 * (1.0 - t) / w + hill * fade_d / rise,
                fade,
                p.beach_rise * t * (2.0 - t) + hill * fade,
            ];
        }
        if p.ocean_depth <= 0.0 {
            return [0.0; 3];
        }
        let out = -inland;
        let e = exp(-out / p.shelf_length);
        let (fall, fall_d) = (1.0 - e, e / p.shelf_length);
        let x = (out / rise).clamp(0.0, 1.0);
        let fade = x * x * (3.0 - 2.0 * x);
        let fade_d = 6.0 * x * (1.0 - x);
        let lift = p.seabed_lift;
        let d_out = -p.ocean_depth * fall_d + lift * hill * (fall_d * fade + fall * fade_d / rise);
        [
            -d_out,
            lift * fall * fade,
            -p.ocean_depth * fall + lift * hill * fall * fade,
        ]
    }

    // The shader's `ground_noise`, line for line.
    fn shader_noise(p: [f32; 3], salt: u32) -> [f32; 4] {
        use concinnity_core::math::noise::lcg_hash;
        let lattice = |c: [i32; 3]| {
            let h = lcg_hash(
                lcg_hash(
                    lcg_hash(c[0].cast_unsigned() ^ salt)
                        .wrapping_add(c[1].cast_unsigned().wrapping_mul(0x85eb_ca6b)),
                )
                .wrapping_add(c[2].cast_unsigned().wrapping_mul(0xc2b2_ae35)),
            );
            (h & 0xffff) as f32 / 65535.0
        };
        let cell = p.map(concinnity_core::math::floor);
        let t: [f32; 3] = core::array::from_fn(|i| p[i] - cell[i]);
        let s = t.map(|t| t * t * (3.0 - 2.0 * t));
        let ds = t.map(|t| 6.0 * t * (1.0 - t));
        let c = cell.map(i32::saturating_from);
        let at = |x: i32, y: i32, z: i32| lattice([c[0] + x, c[1] + y, c[2] + z]);
        let (a, b, d0, d1) = (at(0, 0, 0), at(1, 0, 0), at(0, 1, 0), at(1, 1, 0));
        let (e, f, g, h) = (at(0, 0, 1), at(1, 0, 1), at(0, 1, 1), at(1, 1, 1));
        let k1 = b - a;
        let k2 = d0 - a;
        let k3 = e - a;
        let k4 = a - b - d0 + d1;
        let k5 = a - d0 - e + g;
        let k6 = a - b - e + f;
        let k7 = -a + b + d0 - d1 + e - f - g + h;
        let value = a
            + k1 * s[0]
            + k2 * s[1]
            + k3 * s[2]
            + k4 * s[0] * s[1]
            + k5 * s[1] * s[2]
            + k6 * s[2] * s[0]
            + k7 * s[0] * s[1] * s[2];
        [
            ds[0] * (k1 + k4 * s[1] + k6 * s[2] + k7 * s[1] * s[2]),
            ds[1] * (k2 + k5 * s[2] + k4 * s[0] + k7 * s[2] * s[0]),
            ds[2] * (k3 + k6 * s[0] + k5 * s[1] + k7 * s[0] * s[1]),
            value,
        ]
    }

    fn ground_shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 8,
            seed: 4,
            continents: crate::planet::Continents::NONE,
            edits: None,
        }
    }

    fn ground_params() -> CanopyParams {
        let mut p = CanopyParams {
            planet_radius: 50_000.0,
            ..CanopyParams::default()
        };
        set_ground_noise(&mut p, &ground_shape());
        p
    }

    // A pixel fine enough to hold every octave shades with the ground's exact
    // normal, the one its tiles' corners carry.
    #[test]
    fn a_fine_pixel_shades_with_the_exact_ground_normal() {
        let shape = ground_shape();
        let p = ground_params();
        assert_eq!(p.ground_octaves, 8);
        for i in 0..32 {
            let dir = CubeFace(2).direction(0.001 * f64::from(i), -0.04 + 0.0007 * f64::from(i));
            let want = concinnity_core::math::dvec3::to_f32(shape.normal(dir));
            let got = shader_ground_normal(&p, concinnity_core::math::dvec3::to_f32(dir), 0.01);
            assert!(dot(got, want) > 1.0 - 1e-6, "{got:?} {want:?}");
        }
    }

    // On a planet of seas, a fine pixel near a coast shades with the
    // ground's exact normal too: the beach's, the shelf's, the hills rising
    // in behind.
    #[test]
    fn a_fine_pixel_near_a_coast_shades_with_the_exact_ground_normal() {
        let shape = PlanetShape {
            continents: crate::planet::Continents::new(50_000.0, 4, 20_000.0, 0.4, 60.0, 60.0),
            ..ground_shape()
        };
        let mut p = CanopyParams {
            planet_radius: 50_000.0,
            ..CanopyParams::default()
        };
        set_ground_noise(&mut p, &shape);
        assert_eq!(p.continent_octaves, CONTINENT_OCTAVES);
        let face = CubeFace(2);
        let mut checked = 0;
        for i in 0..20_000 {
            let dir = face.direction(-0.9 + 9e-5 * f64::from(i), 0.31);
            if shape.height(dir).abs() > 20.0 {
                continue;
            }
            let want = concinnity_core::math::dvec3::to_f32(shape.normal(dir));
            let (got, height) = shader_ground(&p, concinnity_core::math::dvec3::to_f32(dir), 0.01);
            assert!(dot(got, want) > 1.0 - 1e-5, "{got:?} {want:?}");
            // The height the wash is measured on, within a few centimeters.
            let exact = shape.height(dir);
            assert!((f64::from(height) - exact).abs() < 0.03, "{height} {exact}");
            checked += 1;
        }
        assert!(checked > 100, "{checked}");
        set_ground_noise(&mut p, &ground_shape());
        assert_eq!(p.continent_octaves, 0);
    }

    // A planet tile's corners carry what the shader needs of the edits:
    // with their detail weight and correction, a fine pixel at a corner
    // shades with the ground's exact edited normal, inland and on the coast.
    #[test]
    fn edited_corners_shade_with_the_exact_ground_normal() {
        use crate::planet::{EditPage, EditPages, PAGE_SAMPLES, PAGE_SIDE, TILE_CELLS, TileId};
        let coast = crate::planet::Continents::new(50_000.0, 4, 20_000.0, 0.4, 60.0, 60.0);
        for continents in [crate::planet::Continents::NONE, coast] {
            let generated = PlanetShape {
                continents,
                ..ground_shape()
            };
            let mut p = CanopyParams {
                planet_radius: 50_000.0,
                ..CanopyParams::default()
            };
            set_ground_noise(&mut p, &generated);
            let page_tile = TileId {
                face: CubeFace(2),
                level: 9,
                x: 260,
                y: 250,
            };
            let at = |k: usize| ((k % PAGE_SIDE) as f64 / 32.0, (k / PAGE_SIDE) as f64 / 32.0);
            let e: Vec<f64> = (0..PAGE_SAMPLES)
                .map(|k| 6.0 * libm::sin(at(k).0 * 7.0) * at(k).1)
                .collect();
            let w: Vec<f64> = (0..PAGE_SAMPLES)
                .map(|k| 0.2 + 0.7 * at(k).0 * (1.0 - at(k).1))
                .collect();
            let pages = EditPages::new([(page_tile, EditPage::from_samples(&e, &w).unwrap())]);
            let shape = generated.with_edits(Some(alloc::sync::Arc::new(pages)));
            let tile = page_tile.children()[1].children()[2];
            let mesh = crate::planet::tile_mesh(&shape, tile);
            let side = TILE_CELLS as usize + 1;
            let mut corrected = 0;
            for j in 0..side {
                for i in 0..side {
                    let v = mesh.vertices[j * side + i];
                    let dir = tile.direction(i as f64 / 16.0, j as f64 / 16.0);
                    let (got, _) = shader_edited_ground(
                        &p,
                        concinnity_core::math::dvec3::to_f32(dir),
                        0.01,
                        v.color,
                        v.tangent,
                    );
                    assert!(
                        dot(got, v.normal) > 1.0 - 2e-6,
                        "{i} {j}: {got:?} {:?}",
                        v.normal
                    );
                    corrected += usize::from(v.color[1] != 0.0 || v.color[2] != 0.0);
                    assert!(v.color[0] < 1.0);
                }
            }
            assert_eq!(corrected, side * side);
        }
    }

    // As a pixel widens, octaves fade out finest first, by the meters it
    // covers alone: a pixel wider than half the broadest octave's cells sees
    // the bare sphere.
    #[test]
    fn octaves_fade_out_as_the_footprint_grows() {
        let p = ground_params();
        let dir = concinnity_core::math::dvec3::to_f32(CubeFace(2).direction(0.013, -0.021));
        let tilt = |footprint: f32| 1.0 - dot(shader_ground_normal(&p, dir, footprint), dir);
        assert!(tilt(p.ground_cell) < 1e-6);
        assert!(tilt(p.ground_cell / 1024.0) > 1e-6);
        let finest = p.ground_cell / 128.0;
        // Below the finest octave's half cell, nothing has faded yet.
        let full = shader_ground_normal(&p, dir, finest / 4.0);
        assert_eq!(full, shader_ground_normal(&p, dir, finest / 8.0));
        // Each octave fades over a doubling of the footprint, continuously.
        let mut last = full;
        for k in 0..64 {
            let f = finest / 4.0 * 2f32.powf(k as f32 / 8.0);
            let n = shader_ground_normal(&p, dir, f);
            assert!(dot(n, last) > 1.0 - 2e-4, "a step at footprint {f}");
            last = n;
        }
    }
}
