//! A water surface resolved from its asset: its authored fields, the waves the
//! world's wind raises on it plus its authored swell, and which of those move
//! the grid rather than only its shading. A body large enough takes its wind
//! sea from a simulated ocean instead of the sum ([`ocean`]).

pub mod ocean;
pub(crate) mod query;
pub mod ripples;
pub mod sea;
pub mod spectrum;

use crate::components::Vocabulary;
use crate::planet::PlanetShape;
use crate::wind::WindField;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;
use ocean::grid::OceanGrid;
use ocean::{BodyOcean, OCEAN_CHOPPINESS, OceanPlan, OceanQuality, OceanSea};
use ripples::{RIPPLES, Ripple};
use sea::shore::ShoreWaves;
use sea::{SeaBody, SeaSphere};
use spectrum::{GRAVITY, WIND_SEA_WAVES, Wave, unit_hash, wind_sea};

/// The most waves one surface sums: the wind sea plus the authored swell.
pub const MAX_WAVES: usize = WIND_SEA_WAVES + MAX_WATER_WAVES;

/// The most waves that displace the grid; the rest shade it only.
const MAX_DISPLACING_WAVES: usize = 16;

/// The shortest wave that displaces the grid, in grid cells: anything finer
/// would alias between vertices, so it moves the shading normal only.
const DISPLACE_MIN_CELLS: f32 = 2.0;

/// The longest ripple the shader's procedural ripples carry, in meters, when
/// the wind raises no sea for them to continue.
const STILL_RIPPLE_WAVELENGTH: f32 = 0.06;

// The share of its own self-intersection limit the wind sea's displacing
// waves pinch their crests by together, and the ceiling every displacing wave
// together is held under so the summed crests never fold over.
const WIND_SEA_PINCH: f32 = 0.6;
const MAX_TOTAL_PINCH: f32 = 0.95;

// How many standard deviations of its height a simulated sea's crests are
// taken to reach.
const OCEAN_CREST_SIGMAS: f32 = 5.0;

/// One wave of a body's sum with its horizontal pinch: a point at rest moves
/// `pinch * amplitude` along the wave's direction at the crest's flank, which
/// sharpens crests and broadens troughs (Gerstner). 0 for a wave that only
/// shades.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyWave {
    /// The wave.
    pub wave: Wave,
    /// Horizontal displacement per meter of amplitude.
    pub pinch: f32,
}

/// One water surface as the backends draw it.
#[derive(Debug, Clone)]
pub struct WaterBody {
    /// The surface as authored, clamped the way the asset validator clamps it.
    pub surface: WaterSurface,
    /// The wind moving it.
    pub wind: WindField,
    /// Its waves, longest first; the first `displacing` move the grid.
    pub waves: Vec<BodyWave>,
    /// How many of `waves` displace the grid.
    pub displacing: usize,
    /// The wavelength, in meters, the shader's ripples start below: the
    /// shortest wave of the wind sea, or `STILL_RIPPLE_WAVELENGTH`.
    pub ripple_wavelength: f32,
    /// The ripple octaves, longest first, starting at `ripple_wavelength`.
    pub ripples: [Ripple; RIPPLES],
    /// The simulated ocean its wind sea comes from, in place of the summed
    /// sea; `None` for a body whose sea is summed.
    pub ocean: Option<BodyOcean>,
    /// The planet sea it is, drawn around the planet rather than over a
    /// rectangle; `None` for a rectangle.
    pub sea: Option<SeaBody>,
}

impl WaterBody {
    /// `surface` moved by `wind`: the sea the wind raises over the surface's
    /// fetch, at its `wind_response`, summed, and its authored swell.
    pub fn resolve(surface: &WaterSurface, wind: &WindField) -> Self {
        let surface = crate::components::validate::water_surface(surface.clone());
        let sea = wind_sea(
            wind.strength * surface.wind_response,
            wind.direction,
            surface.fetch_meters,
        );
        let ripple_wavelength = sea.last().map_or(STILL_RIPPLE_WAVELENGTH, |w| w.wavelength);
        let cell = cell_size(&surface);
        Self::with_sea(surface, wind, &sea, ripple_wavelength, cell, None)
    }

    /// Every surface moved by `wind`, each large one taking its wind sea from
    /// a simulated ocean at `quality` while there is room for its sea, and
    /// the simulations they share. A planet sea lies on `planet`; with no
    /// planet it is left out.
    pub fn resolve_all<'a>(
        surfaces: impl IntoIterator<Item = &'a WaterSurface>,
        wind: &WindField,
        quality: OceanQuality,
        planet: Option<&PlanetShape>,
    ) -> (Vec<Self>, OceanPlan) {
        let mut plan = OceanPlan::new(quality);
        let bodies = surfaces
            .into_iter()
            .filter(|s| s.shape != WaterShape::PlanetSea || planet.is_some())
            .map(|s| {
                let surface = crate::components::validate::water_surface(s.clone());
                if let (WaterShape::PlanetSea, Some(planet)) = (surface.shape, planet) {
                    return Self::resolve_sea(surface, wind, &mut plan, planet);
                }
                let simulated = OceanSea::of(&surface, wind.strength, wind.direction)
                    .and_then(|sea| Some((sea, plan.share(sea, surface.whitecaps > 0.0)?)));
                match simulated {
                    Some((sea, index)) => Self::resolve_ocean(surface, wind, &sea, &plan, index),
                    None => Self::resolve(&surface, wind),
                }
            })
            .collect();
        (bodies, plan)
    }

    // `surface` as the sea of `planet`: simulated whenever the wind raises a
    // sea at all and there is room for it, however large its peak, and drawn
    // on a grid reaching the horizon.
    fn resolve_sea(
        surface: WaterSurface,
        wind: &WindField,
        plan: &mut OceanPlan,
        planet: &PlanetShape,
    ) -> Self {
        let sea = OceanSea::open(&surface, wind.strength, wind.direction);
        let peak = sea
            .and_then(|s| s.shape())
            .map_or(1.0, |s| s.peak_wavelength());
        let mut body = SeaBody::new(SeaSphere::new(planet, surface.sea_level), peak);
        let shared = sea.and_then(|sea| Some((sea, plan.share(sea, surface.whitecaps > 0.0)?)));
        let mut water = if let Some((sea, index)) = shared {
            Self::resolve_ocean(surface, wind, &sea, plan, index)
        } else {
            let cell = body.grid.inner_cell;
            Self::with_sea(surface, wind, &[], STILL_RIPPLE_WAVELENGTH, cell, None)
        };
        if let Some(ocean) = water.ocean.as_mut() {
            ocean.grid = body.grid;
        }
        body.shore = sea.and_then(|s| s.shape()).and_then(|shape| {
            let cascades = water.ocean.as_ref().map(|o| &o.cascades);
            ShoreWaves::new(planet, f64::from(water.surface.sea_level), &shape, cascades)
        });
        water.sea = Some(body);
        water
    }

    // `surface` with its wind sea simulated by `plan`'s simulation `index`:
    // only its swell is summed, and its ripples start where the last cascade
    // ends.
    fn resolve_ocean(
        surface: WaterSurface,
        wind: &WindField,
        sea: &OceanSea,
        plan: &OceanPlan,
        index: usize,
    ) -> Self {
        let sim = &plan.simulations[index];
        let peak = sea.shape().map_or(1.0, |s| s.peak_wavelength());
        let ocean = BodyOcean {
            simulation: index,
            cascades: sim.cascades.clone(),
            first_layer: sim.first_layer,
            grid: OceanGrid::new(peak, surface.extent),
        };
        let ripple_wavelength = ocean.cascades.ripple_wavelength();
        let cell = ocean.grid.inner_cell;
        Self::with_sea(surface, wind, &[], ripple_wavelength, cell, Some(ocean))
    }

    // A body summing `sea` and the surface's swell on a grid of `cell`-meter
    // cells, its ripples starting at `ripple_wavelength`.
    fn with_sea(
        surface: WaterSurface,
        wind: &WindField,
        sea: &[Wave],
        ripple_wavelength: f32,
        cell: f32,
        ocean: Option<BodyOcean>,
    ) -> Self {
        let swell = surface
            .waves
            .iter()
            .enumerate()
            .filter(|(_, w)| w.amplitude > 0.0)
            .map(|(i, w)| Wave {
                direction: w.direction,
                amplitude: w.amplitude,
                wavelength: w.wavelength,
                phase: unit_hash(u32::saturating_from(i), 11) * core::f32::consts::TAU,
                steepness: w.steepness,
            });

        // Longest first, wind sea before swell on a tie, so the displacing
        // waves are a prefix.
        let mut tagged: Vec<(Wave, bool)> = sea
            .iter()
            .map(|w| (*w, true))
            .chain(swell.map(|w| (w, false)))
            .collect();
        tagged.sort_by(|a, b| b.0.wavelength.total_cmp(&a.0.wavelength));
        tagged.truncate(MAX_WAVES);

        let min_displacing = DISPLACE_MIN_CELLS * cell;
        let displacing = tagged
            .iter()
            .take(MAX_DISPLACING_WAVES)
            .take_while(|(w, _)| w.wavelength >= min_displacing)
            .count();
        let waves = pinch_waves(&tagged, displacing);
        Self {
            surface,
            wind: *wind,
            waves,
            displacing,
            ripple_wavelength,
            ripples: ripples::ripples(wind.direction, ripple_wavelength),
            ocean,
            sea: None,
        }
    }

    /// How far the displaced grid reaches from its flat rest: `[rise, sway]`,
    /// the most a vertex moves vertically and horizontally.
    pub fn reach(&self) -> [f32; 2] {
        let [rise, sway] = self.waves[..self.displacing]
            .iter()
            .fold([0.0, 0.0], |[rise, sway], w| {
                [rise + w.wave.amplitude, sway + w.pinch * w.wave.amplitude]
            });
        let crest = self.ocean_crest();
        [rise + crest, sway + OCEAN_CHOPPINESS * crest]
    }

    // How high a simulated sea's crests reach: a few standard deviations of
    // the height its cascades' band raises. 0 for a summed sea.
    fn ocean_crest(&self) -> f32 {
        let Some(ocean) = &self.ocean else {
            return 0.0;
        };
        let wind = self.wind.strength * self.surface.wind_response;
        let Some(shape) = spectrum::SeaShape::new(wind, self.surface.fetch_meters) else {
            return 0.0;
        };
        let omega = |k: f32| concinnity_core::math::sqrt(GRAVITY * k);
        let c = &ocean.cascades;
        let lo = core::f32::consts::TAU / c.lengths()[0];
        let hi = c.band(c.count() - 1)[1];
        OCEAN_CREST_SIGMAS
            * concinnity_core::math::sqrt(shape.variance_between(omega(lo), omega(hi)))
    }

    /// The world-space XZ, in f64, a body's motion is measured from for a
    /// camera at `camera`: the surface's center for a summed body, the
    /// camera snapped to the grid for a simulated one.
    pub fn anchor(&self, camera: [f32; 3]) -> [f64; 2] {
        match &self.ocean {
            Some(ocean) => ocean
                .grid
                .snap([f64::from(camera[0]), f64::from(camera[2])]),
            None => [
                f64::from(self.surface.center[0]),
                f64::from(self.surface.center[2]),
            ],
        }
    }
}

/// The widest grid cell of `surface`, in meters (see `geometry::water_grid`).
pub(crate) fn cell_size(surface: &WaterSurface) -> f32 {
    let n = surface.subdivisions.clamp(8, 255) as f32;
    2.0 * surface.extent[0].max(surface.extent[1]).max(0.0) / n
}

// The pinch of each of `tagged` (wave, is wind sea), of which the first
// `displacing` move the grid. The wind sea's displacing waves share one pinch
// so together they reach `WIND_SEA_PINCH`; each swell wave takes its authored
// steepness split over the displacing swell, as a lone Gerstner wave would.
// The whole is scaled down if it would pass `MAX_TOTAL_PINCH`.
fn pinch_waves(tagged: &[(Wave, bool)], displacing: usize) -> Vec<BodyWave> {
    let moving = &tagged[..displacing];
    let slope = |w: &Wave| w.wavenumber() * w.amplitude;
    let sea_slope: f32 = moving.iter().filter(|t| t.1).map(|t| slope(&t.0)).sum();
    let swell_count = moving.iter().filter(|t| !t.1).count().max(1) as f32;
    let mut waves: Vec<BodyWave> = tagged
        .iter()
        .enumerate()
        .map(|(i, (wave, is_sea))| {
            let pinch = if i >= displacing || slope(wave) <= 0.0 {
                0.0
            } else if *is_sea {
                WIND_SEA_PINCH / sea_slope
            } else {
                wave.steepness / (slope(wave) * swell_count)
            };
            BodyWave { wave: *wave, pinch }
        })
        .collect();
    let total: f32 = waves.iter().map(|w| w.pinch * slope(&w.wave)).sum();
    if total > MAX_TOTAL_PINCH {
        let scale = MAX_TOTAL_PINCH / total;
        for w in &mut waves {
            w.pinch *= scale;
        }
    }
    waves
}

/// Maximum number of authored swell waves per water surface; the wind's own
/// waves come on top of these.
pub const MAX_WATER_WAVES: usize = 4;

/// One authored swell wave on a water surface, on top of the waves the wind
/// raises. Swell is the long, regular motion left by distant weather: it rolls
/// along `direction`, rising and falling `amplitude` around the rest height,
/// `wavelength` between crests, at the speed deep water carries a wave that
/// long (about 1.25 m/s times the square root of the wavelength in meters).
/// `steepness` in [0, 1] pinches the crests and broadens the troughs.
#[derive(
    Debug,
    Clone,
    Copy,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct WaterWave {
    /// Peak height of the wave, in world units.
    #[asset(default = 0.15)]
    pub amplitude: f32,
    /// Distance between successive crests, in world units.
    #[asset(default = 4.0)]
    pub wavelength: f32,
    /// Horizontal travel direction `[x, z]`. Does not need to be normalized.
    #[asset(default = [1.0, 0.0])]
    pub direction: [f32; 2],
    /// Crest sharpness in [0, 1]. 0 is a smooth sine; higher pinches crests and
    /// broadens troughs.
    #[asset(default = 0.4)]
    pub steepness: f32,
}

/// What a [`WaterSurface`](#watersurface) covers.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "snake_case")]
pub enum WaterShape {
    /// A horizontal rectangle `extent` around `center`.
    #[default]
    #[vocab("rectangle")]
    Rectangle,
    /// The sea of the world's [Planet](#planet): water all the way around
    /// it, `sea_level` meters above its radius, filling every hollow of the
    /// ground below that height.
    #[vocab("planet_sea")]
    PlanetSea,
}

/// A translucent animated water surface.
///
/// A flat, subdivided horizontal surface moved by the world's `Wind`. The wind
/// raises a field of waves of many lengths, mostly running downwind with some
/// crossing it, each traveling at the speed deep water carries a wave of its
/// length, plus fine ripples that roughen the surface. Gusts roll across it in
/// step with the grass: rough dark patches where a gust touches down, glassier
/// water in the lulls. A stronger wind and a longer `fetch_meters` (the open
/// water the wind blows across) build longer, taller waves; with no `Wind`
/// the water lies almost mirror-still. Authored `waves` add swell on top.
///
/// Large open water, a surface hundreds of meters to kilometers across, is a
/// simulated ocean: its wind waves rise, cross and fade continuously over the
/// whole surface with no repeating pattern, crests sharpen and troughs
/// broaden, and the water stays as detailed beside the camera as anywhere,
/// however large the surface or far from the world origin. Pools, ponds and
/// small lakes keep a lighter sum of waves. The graphics quality setting
/// trades the ocean's finest detail for speed.
///
/// A simulated ocean can whitecap, as strongly as `whitecaps` asks (none by
/// default): where a crest grows steep enough to break, a patch of white foam
/// appears on it and stays behind as the wave runs on, thinning into lacy
/// streaks over a few seconds. How much of the sea is white follows the wind
/// as at sea: almost none below about 4 m/s, about one percent at 10 m/s, a
/// few percent at 14 m/s and more in a gale. Whitecaps hide the reflection
/// and the sun's glint beneath them. Smaller water never whitecaps.
///
/// Far off, waves too fine to draw fold into a rougher surface, so the sun or
/// moon spreads into a glittering path instead of shimmering.
///
/// It refracts and reflects the scene, and its body is lit like any
/// other surface: the sun or moon (in shadow where the scene casts one) and the
/// sky's light scatter back out of the water, off a reflectance that blends
/// from `shallow_color` to `deep_color` as the water deepens. Shoreline foam
/// and whitecaps are a lit white surface.
///
/// A [`PropBody`](#propbody) with `buoyancy` floats in the water on the waves
/// it draws, buoyed by the water's `density`.
///
/// A `rectangle` surface is positioned by `center` and sized by `extent` (XZ
/// half-widths). The mesh itself is flat; all height variation comes from the
/// animated waves.
///
/// A `planet_sea` is the ocean of the world's [Planet](#planet): a sphere of
/// water `sea_level` meters above the planet's radius, so every valley whose
/// ground lies lower floods, and the coast is wherever the hills rise out of
/// it. It is always a simulated ocean, its waves fixed to the planet's ground
/// wherever on it the camera roams, and it curves away to the horizon with
/// the ground. Its gusts follow the wind in step with the planet's grass. It
/// reflects the scene by ray tracing, reflection probes or the sky, never by
/// a planar mirror, and it takes no authored swell: `waves` must be empty.
/// `center`, `extent` and `subdivisions` do not apply. A world without a
/// planet draws no planet sea.
///
/// Where the planet has continents and the sea meets their beaches, its waves
/// come ashore as at a real coast. Over the shelf the open sea's longest
/// waves give way to long swell lines running in toward the land; they slow
/// and bunch up as the water shallows, turn to meet the beach nearly square,
/// a little askew so the breakers peel along it, and grow taller until they
/// break, about where a wave stands four fifths as tall as the water under
/// it. Bigger waves come in sets and break farther out. A breaking crest
/// pitches forward into a bore of white water that trails foam, and the
/// wash of each wave runs on up the sand and drains back, leaving the beach
/// darker and glossy where it reached, drying back between sets. The height
/// and rhythm of the surf follow the wind sea: a stronger wind or a higher
/// `wind_response` brings bigger, slower waves ashore. A sea standing over
/// the planet's beaches, or around a planet without seas, has no surf.
///
/// Its sun glint rides the waves, and its reflection and refraction move with
/// neither the surface nor what they show, so temporal anti-aliasing and
/// upscaling favor the current frame where the glint shows and, within about
/// 16 meters of the camera, where the reflection is strong. Farther off they
/// keep their history, which holds distant waves still.
///
/// ```rust
/// # use concinnity_engine::components::WaterSurface;
/// WaterSurface {
///     center: [0.0, 0.4, 0.0],
///     extent: [12.0, 8.0],
///     subdivisions: 96,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct WaterSurface {
    /// What the surface covers: `rectangle` (the default) or `planet_sea`.
    pub shape: WaterShape,
    /// Height of a `planet_sea` above the [Planet](#planet)'s radius, in
    /// meters, at least 0.
    pub sea_level: f32,
    /// World-space position of a rectangle's center.
    pub center: [f32; 3],
    /// Half-width and half-depth of a rectangle `[x, z]`, in world units.
    #[asset(default = [10.0, 10.0])]
    pub extent: [f32; 2],
    /// Grid subdivisions across a rectangle. Higher gives smoother waves.
    /// Clamped to [8, 255].
    #[asset(default = 64)]
    pub subdivisions: u32,
    /// The open water the wind blows across, in meters: how large a sea it can
    /// build. A pond is tens of meters, a lake one to a few thousand, an open
    /// sea a hundred thousand or more. At least 1.
    #[asset(default = 1000.0)]
    pub fetch_meters: f32,
    /// How strongly the wind moves this surface, at least 0. 0 is sheltered
    /// water the wind never touches, 1 the full wind; above 1 exaggerates it.
    #[asset(default = 1.0)]
    pub wind_response: f32,
    /// The water's density, in kilograms per cubic meter, at least 1: how
    /// strongly it buoys what floats in it. Seawater by default; fresh water
    /// is 1000.
    #[asset(default = 1025.0)]
    pub density: f32,
    /// How deep a rectangle's water is, in meters below its rest, at least 0:
    /// anything lower is out of the water, so a cave under a pond stays dry.
    #[asset(default = 10.0)]
    pub depth_meters: f32,
    /// Extra swell rolling over the wind's waves, up to 4. None by default,
    /// and none on a planet sea.
    pub waves: Vec<WaterWave>,
    /// Linear-space RGB reflectance of deep water's body, each channel in
    /// [0, 1]: the share of the light reaching the surface that a deep column
    /// scatters back out. Values outside [0, 1] are clamped.
    #[asset(default = [0.02, 0.05, 0.15])]
    pub deep_color: [f32; 3],
    /// Linear-space RGB reflectance of shallow water's body near the shore,
    /// each channel in [0, 1]. The seabed seen through the shallows is
    /// filtered by it too. Values outside [0, 1] are clamped.
    #[asset(default = [0.20, 0.50, 0.55])]
    pub shallow_color: [f32; 3],
    /// Depth over which the reflectance blends from shallow to deep, in meters.
    #[asset(default = 4.0)]
    pub depth_falloff_meters: f32,
    /// Width of the shoreline foam band, in meters.
    #[asset(default = 0.30)]
    pub foam_width_meters: f32,
    /// Strength of the shoreline foam, in [0, 1].
    #[asset(default = 0.8)]
    pub foam_intensity: f32,
    /// Strength of the foam on a simulated ocean's breaking crests, in [0, 1].
    /// 0 is none; smaller water never whitecaps. Values outside [0, 1] are
    /// clamped.
    pub whitecaps: f32,
    /// Sharpness of the grazing-angle reflection. Higher confines reflections to
    /// steeper viewing angles.
    #[asset(default = 5.0)]
    pub fresnel_power: f32,
    /// Surface roughness in [0, 1]. Higher gives blurrier reflections, and
    /// pushes a mirrored reflection further off its line with each wave: a
    /// near-mirror surface keeps its reflection almost still.
    #[asset(default = 0.05)]
    pub roughness: f32,
    /// How strongly the surface bends the view of what's beneath it.
    #[asset(default = 0.15)]
    pub refraction_strength: f32,
    /// When false the surface is skipped each frame.
    #[asset(default = true)]
    pub visible: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{WaterWave, Wind};
    use alloc::vec;

    fn breeze() -> WindField {
        WindField::new(Some(&Wind::default()))
    }

    fn total_pinch(body: &WaterBody) -> f32 {
        body.waves
            .iter()
            .map(|w| w.pinch * w.wave.wavenumber() * w.wave.amplitude)
            .sum()
    }

    // A default surface in a default wind is a lake in a breeze: a full wind
    // sea, its long waves moving the grid and its short ones shading it.
    #[test]
    fn a_default_surface_in_a_default_wind_raises_a_wind_sea() {
        let body = WaterBody::resolve(&WaterSurface::default(), &breeze());
        assert_eq!(body.waves.len(), WIND_SEA_WAVES);
        assert!(body.displacing > 0 && body.displacing < body.waves.len());
        let cell = cell_size(&body.surface);
        for (i, w) in body.waves.iter().enumerate() {
            assert_eq!(
                i < body.displacing,
                w.wave.wavelength >= DISPLACE_MIN_CELLS * cell,
                "{i}"
            );
            assert_eq!(w.pinch > 0.0, i < body.displacing, "{i}");
        }
        assert!((total_pinch(&body) - WIND_SEA_PINCH).abs() < 1e-4);
        assert_eq!(
            body.ripple_wavelength,
            body.waves.last().unwrap().wave.wavelength
        );
    }

    // Still air and sheltered water keep only the ripples.
    #[test]
    fn still_or_sheltered_water_has_no_waves() {
        let still = WaterBody::resolve(&WaterSurface::default(), &WindField::STILL);
        assert!(still.waves.is_empty());
        assert_eq!(still.displacing, 0);
        assert_eq!(still.reach(), [0.0, 0.0]);
        assert_eq!(still.ripple_wavelength, STILL_RIPPLE_WAVELENGTH);
        let sheltered = WaterSurface {
            wind_response: 0.0,
            ..Default::default()
        };
        assert!(WaterBody::resolve(&sheltered, &breeze()).waves.is_empty());
    }

    // A finer grid displaces shorter waves, never more than the cap.
    #[test]
    fn a_finer_grid_displaces_more_waves() {
        let coarse = WaterSurface {
            subdivisions: 16,
            ..Default::default()
        };
        let fine = WaterSurface {
            subdivisions: 255,
            extent: [2.0, 2.0],
            ..Default::default()
        };
        let a = WaterBody::resolve(&coarse, &breeze()).displacing;
        let b = WaterBody::resolve(&fine, &breeze()).displacing;
        assert!(a < b && b <= MAX_DISPLACING_WAVES, "{a} {b}");
    }

    // Swell joins the wind sea in wavelength order, and a steep swell cannot
    // push the summed crests past folding over.
    #[test]
    fn swell_joins_the_sea_and_the_pinch_stays_bounded() {
        let surface = WaterSurface {
            waves: vec![
                WaterWave {
                    amplitude: 0.3,
                    wavelength: 12.0,
                    direction: [0.0, 1.0],
                    steepness: 1.0,
                },
                WaterWave {
                    amplitude: 0.0,
                    ..WaterWave::default()
                },
            ],
            ..Default::default()
        };
        let body = WaterBody::resolve(&surface, &breeze());
        assert_eq!(body.waves.len(), WIND_SEA_WAVES + 1);
        assert_eq!(body.waves[0].wave.wavelength, 12.0);
        assert!(
            body.waves
                .windows(2)
                .all(|w| w[0].wave.wavelength >= w[1].wave.wavelength)
        );
        assert!(total_pinch(&body) <= MAX_TOTAL_PINCH + 1e-4);
        let [rise, sway] = body.reach();
        assert!(rise >= 0.3 && sway > 0.0);
        let still = WaterBody::resolve(&surface, &WindField::STILL);
        assert_eq!(still.waves.len(), 1);
        assert!((total_pinch(&still) - MAX_TOTAL_PINCH).abs() < 1e-4);
    }

    fn open_sea(center: [f32; 3]) -> WaterSurface {
        WaterSurface {
            center,
            extent: [4000.0, 3000.0],
            fetch_meters: 1.0e5,
            waves: vec![WaterWave {
                amplitude: 0.4,
                wavelength: 80.0,
                ..WaterWave::default()
            }],
            ..Default::default()
        }
    }

    fn fresh_breeze() -> WindField {
        WindField::new(Some(&Wind {
            strength: 9.0,
            ..Wind::default()
        }))
    }

    // A large sea takes its wind sea from a simulation and sums only its
    // swell; a pond in the same wind keeps the sum; two seas alike share one
    // simulation.
    #[test]
    fn large_seas_are_simulated_and_ponds_summed() {
        let surfaces = [
            open_sea([0.0; 3]),
            WaterSurface::default(),
            open_sea([2.0e4, 0.0, 0.0]),
        ];
        let (bodies, plan) =
            WaterBody::resolve_all(&surfaces, &fresh_breeze(), OceanQuality::FULL, None);
        assert_eq!(plan.simulations.len(), 1);
        let sea = &bodies[0];
        let ocean = sea.ocean.as_ref().expect("simulated");
        assert_eq!(ocean.simulation, 0);
        assert_eq!(sea.waves.len(), 1, "only the swell is summed");
        assert_eq!(sea.waves[0].wave.wavelength, 80.0);
        assert_eq!(sea.ripple_wavelength, ocean.cascades.ripple_wavelength());
        assert_eq!(sea.grid_lane(), ocean.grid.gpu_lane());
        assert_eq!(bodies[2].ocean.as_ref().map(|o| o.simulation), Some(0));
        let pond = &bodies[1];
        assert!(pond.ocean.is_none());
        assert_eq!(pond.waves.len(), WIND_SEA_WAVES);
        let summed = WaterBody::resolve(&WaterSurface::default(), &fresh_breeze());
        assert_eq!(pond.waves, summed.waves);
        assert_eq!(pond.grid_lane(), [cell_size(&pond.surface), 0.0, 0.0, 0.0]);
    }

    // A sea foams only once a body sharing it asks for whitecaps; a pond's
    // request raises no simulation.
    #[test]
    fn a_shared_sea_foams_when_any_body_whitecaps() {
        let calm = [open_sea([0.0; 3]), open_sea([2.0e4, 0.0, 0.0])];
        let (_, plan) = WaterBody::resolve_all(&calm, &fresh_breeze(), OceanQuality::FULL, None);
        assert!(plan.simulations[0].foam.is_none());

        let whitecapping = WaterSurface {
            whitecaps: 0.5,
            ..open_sea([2.0e4, 0.0, 0.0])
        };
        let pond = WaterSurface {
            whitecaps: 1.0,
            ..WaterSurface::default()
        };
        let surfaces = [open_sea([0.0; 3]), whitecapping, pond];
        let (bodies, plan) =
            WaterBody::resolve_all(&surfaces, &fresh_breeze(), OceanQuality::FULL, None);
        assert_eq!(plan.simulations.len(), 1);
        assert!(plan.simulations[0].foam.is_some());
        assert!(bodies[2].ocean.is_none());
    }

    // A simulated sea's crests reach a little past its significant height,
    // and sway by the choppiness; swell adds its own reach on top.
    #[test]
    fn a_simulated_sea_reaches_past_its_crests() {
        let calm_swell = WaterSurface {
            waves: Vec::new(),
            ..open_sea([0.0; 3])
        };
        let (bodies, _) =
            WaterBody::resolve_all([&calm_swell], &fresh_breeze(), OceanQuality::FULL, None);
        let [rise, sway] = bodies[0].reach();
        assert!((1.0..4.0).contains(&rise), "{rise}");
        assert!((sway - OCEAN_CHOPPINESS * rise).abs() < 1e-5, "{sway}");
        let (swell, _) = WaterBody::resolve_all(
            [&open_sea([0.0; 3])],
            &fresh_breeze(),
            OceanQuality::FULL,
            None,
        );
        let [r, s] = swell[0].reach();
        assert!(r > rise + 0.39 && s > sway, "{r} {s}");
    }

    #[test]
    fn resolving_twice_gives_the_same_body() {
        let a = WaterBody::resolve(&WaterSurface::default(), &breeze());
        let b = WaterBody::resolve(&WaterSurface::default(), &breeze());
        assert_eq!(a.waves, b.waves);
    }
}
