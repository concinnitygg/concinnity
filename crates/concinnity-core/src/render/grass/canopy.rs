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

use super::canopy_table::canopy_table;
use super::planet::PlanetGrass;
use super::planet::density::meadow_density;
use super::planet::tiles::{DrawnPlanetTile, planet_center};
use super::{GrassBladeLook, GrassTerrain, grown_layers};
use crate::math::{atan2, cos, exp, powf, sin, sqrt};
use crate::planet::PlanetShape;
use crate::render::uniforms::canopy::{CanopyLayerGpu, CanopyParams, MAX_CANOPY_LAYERS};

/// Most layers one terrain's far field shades: its vertex colors carry one
/// layer's density per channel.
pub const MAX_CANOPY_LAYERS_PER_TERRAIN: usize = 3;

/// A blade's mean lean: the kernel's `(0.15 + 0.35 u) (0.6 + 0.8 v)` over its
/// uniform hashes.
pub const CANOPY_BEND: f32 = 0.325;

// A blade's width tapers as `1 - t^WIDTH_TAPER` from root to tip.
const WIDTH_TAPER: f32 = 1.4;

/// How many facings around the view the canopy lights its blades at: the
/// shaders' `CANOPY_FACINGS`.
pub const CANOPY_FACINGS: u32 = 8;

/// How many height segments a blade is cut into for what a view sees of it.
/// The upper half of them makes the upper band, the lower half the lower.
pub const CANOPY_SEGMENTS: usize = 8;

/// How wide a blade is at height `t` along it, as a share of its root width.
pub fn blade_width(t: f32) -> f32 {
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
pub fn blade_tilt(bend: f32, t: f32) -> f32 {
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
    /// The mean of the direct share times the occlusion.
    pub direct_occlusion: f32,
    /// The mean of the ramp, the direct share and the occlusion together.
    pub ramp_direct_occlusion: f32,
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
pub fn root_occlusion(t: f32) -> f32 {
    let s = (t / 0.6).clamp(0.0, 1.0);
    0.25 + 0.75 * s * s * (3.0 - 2.0 * s)
}

/// How much of a direct light reaches height `t` inside the field. Mirrors
/// `grass_direct_share`.
pub fn direct_share(t: f32) -> f32 {
    0.55 + 0.45 * t
}

/// The blade heights a field's height variation spreads its blades over, as
/// offsets per unit of variation from the mean: the means of the three equally
/// likely terciles of the kernel's clump plus blade spread,
/// `1.2 (u - 0.5) + 0.8 (v - 0.5)`.
pub const CANOPY_HEIGHT_CLASSES: [f32; 3] = [-0.4665, 0.0, 0.4665];

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
    let mut terms = [[0.0f32; 6]; CANOPY_SEGMENTS];
    for j in 0..CANOPY_SEGMENTS {
        let z = top * segment_height(j);
        for &h in &heights {
            if z >= h {
                continue;
            }
            let t = z / h;
            let a = blade_width(t) * part_projection(blade_tilt(bend, t), mu);
            let (r, o, d) = (ramp(t), root_occlusion(t), direct_share(t));
            seen[j] += a;
            for (acc, f) in terms[j].iter_mut().zip([t, r, o, r * o, d * o, r * d * o]) {
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
        let [t, r, o, ro, dout, rdo] = terms[j];
        b.height += p * t;
        b.ramp += p * r;
        b.occlusion += p * o;
        b.ramp_occlusion += p * ro;
        b.direct_occlusion += p * dout;
        b.ramp_direct_occlusion += p * rdo;
    }
    let [upper, lower] = bands.map(|mut b| {
        let n = b.share.max(1e-9);
        b.height /= n;
        b.ramp /= n;
        b.occlusion /= n;
        b.ramp_occlusion /= n;
        b.direct_occlusion /= n;
        b.ramp_direct_occlusion /= n;
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
}

/// The cosine from the blades' up the sky is taken to reach the ground along,
/// on average. Matches `sky_mu` in the shaders' `canopy_at`.
pub const SKY_MU: f32 = 2.0 / 3.0;

/// Every far field's layers in the world, and the planet they may grow on:
/// what each frame's canopy block and view tables are filled from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CanopyField {
    layers: Vec<CanopyLayerGpu>,
    tables: Vec<[f32; 4]>,
    planet: Option<PlanetCanopy>,
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
                    first: layers.len() as u32,
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

    /// The far field of a planet's `grass` in a wind of mean `wind_speed`, and
    /// its run of layers.
    pub fn for_planet(grass: &PlanetGrass, wind_speed: f32) -> (Self, CanopyRange) {
        let layers: Vec<_> = grass
            .layers
            .iter()
            .take(MAX_CANOPY_LAYERS)
            .map(|l| CanopyLayerGpu::of(&l.look, l.density, l.seed, wind_speed))
            .collect();
        let seeds: Vec<u32> = layers.iter().map(|l| l.seed).collect();
        let range = CanopyRange {
            first: 0,
            count: layers.len() as u32,
            planet: true,
        };
        let planet = PlanetCanopy {
            shape: grass.shape,
            meadow_scale: grass.meadow_scale,
            meadow_mean: meadow_mean(grass.meadow_scale, &seeds),
        };
        (Self::new(layers, Some(planet)), range)
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
        }
    }

    /// Every layer's view table in layer order,
    /// [`CANOPY_TABLE_LEN`](super::canopy_table::CANOPY_TABLE_LEN) rows each.
    pub fn tables(&self) -> &[[f32; 4]] {
        &self.tables
    }

    /// This frame's block. On a planet, `tile` is any tile drawn this frame,
    /// whose model places the planet's axes in the simulated frame.
    pub fn params(&self, tile: Option<&DrawnPlanetTile>) -> CanopyParams {
        let mut params = CanopyParams {
            layer_count: self.layers.len() as u32,
            ..CanopyParams::default()
        };
        params.layers[..self.layers.len()].copy_from_slice(&self.layers);
        if let (Some(planet), Some(tile)) = (&self.planet, tile) {
            let center = planet_center(&planet.shape, tile);
            // The model turns the planet's axes into the frame's, so its
            // rotation's columns are the planet's axes seen from the frame.
            params.planet_rows = core::array::from_fn(|i| {
                let axis = [tile.model[i][0], tile.model[i][1], tile.model[i][2]];
                let d = axis[0] * center[0] + axis[1] * center[1] + axis[2] * center[2];
                [axis[0], axis[1], axis[2], -d]
            });
            params.planet_radius = planet.shape.radius as f32;
            params.meadow_scale = planet.meadow_scale;
            params.meadow_mean = planet.meadow_mean;
        }
        params
    }
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
    use crate::math::vec3::{cross, dot, normalize_or, scale};
    use crate::planet::{CubeFace, TileId};
    use crate::render::grass::litter::litter_color;
    use crate::terrain::TerrainGrid;
    use alloc::vec;

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
                        // the ramp, direct share and occlusion product.
                        let steps: alloc::vec::Vec<[f32; 3]> = (0..n)
                            .map(|i| {
                                let z = top * (1.0 - (i as f32 + 0.5) / n as f32);
                                let mut acc = [0.0f32; 3];
                                for &h in &heights {
                                    if z < h {
                                        let t = z / h;
                                        let a = blade_width(t)
                                            * part_projection(blade_tilt(bend, t), mu);
                                        let rdo = ramp(t) * direct_share(t) * root_occlusion(t);
                                        acc[0] += a;
                                        acc[1] += a * t;
                                        acc[2] += a * rdo;
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
                                let rdo = w[2] / w[0];
                                assert!((b.ramp_direct_occlusion - rdo).abs() < 0.05, "{case}");
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
        let mut spread: alloc::vec::Vec<f32> = (0..n * n)
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
        assert_eq!(grass.canopy, field);
        let p = field.params(None);
        assert_eq!(p.layer_count, 5);
        assert_eq!(p.planet_rows, [[0.0; 4]; 3]);
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
        };
        let grass = PlanetGrass::resolve(shape, &[Grass::default()]).unwrap();
        let (field, range) = CanopyField::for_planet(&grass, 0.0);
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
        };
        let p = field.params(Some(&tile));
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
        let want = crate::planet::dvec::to_f32(id.direction(0.5, 0.5));
        let got = unit(rel);
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() < 1e-5, "{got:?} {want:?}");
        }
    }
}
