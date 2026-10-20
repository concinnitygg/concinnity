//! A water surface resolved for drawing: its authored fields, the waves the
//! world's wind raises on it plus its authored swell, and which of those move
//! the grid rather than only its shading. A body large enough takes its wind
//! sea from a simulated ocean instead of the sum ([`ocean`]).

pub mod frame;
pub mod motion;
pub mod ocean;
pub mod ripples;
pub mod sea;
pub mod spectrum;

#[cfg(test)]
mod shader_tests;

use crate::components::{MAX_WATER_WAVES, WaterShape, WaterSurface};
use crate::gfx::mesh_payload::Vertex;
use crate::math::{SaturatingFrom, narrow_f32};
use crate::planet::PlanetShape;
use crate::render::wind::WindField;
use alloc::string::String;
use alloc::vec::Vec;
use ocean::grid::OceanGrid;
use ocean::{BodyOcean, OCEAN_CHOPPINESS, OceanPlan, OceanQuality, OceanSea};
use ripples::{RIPPLES, Ripple};
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
        let body = SeaBody::new(SeaSphere::new(planet, surface.sea_level), peak);
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
        let omega = |k: f32| crate::math::sqrt(GRAVITY * k);
        let c = &ocean.cascades;
        let lo = core::f32::consts::TAU / c.lengths()[0];
        let hi = c.band(c.count() - 1)[1];
        OCEAN_CREST_SIGMAS * crate::math::sqrt(shape.variance_between(omega(lo), omega(hi)))
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

    /// Where the body sorts among the transparent draws, farthest first: a
    /// rectangle's center, and a planet sea's under everything else, at the
    /// planet's center.
    pub fn sort_center(&self) -> [f32; 3] {
        match &self.sea {
            Some(sea) => sea.sphere.center.map(narrow_f32),
            None => self.surface.center,
        }
    }

    /// `[inner cell, inner half-width, widening, _]` of the grid the body is
    /// drawn on: an ocean's widening grid, or the surface's uniform one.
    pub fn grid_lane(&self) -> [f32; 4] {
        if let Some(sea) = &self.sea {
            return sea.grid.gpu_lane();
        }
        match &self.ocean {
            Some(ocean) => ocean.grid.gpu_lane(),
            None => [cell_size(&self.surface), 0.0, 0.0, 0.0],
        }
    }

    /// The grid the body is drawn on, relative to its anchor: the surface's
    /// own tessellation for a summed body, the camera-centered grid for a
    /// simulated one.
    pub fn mesh(&self) -> Result<(Vec<Vertex>, Vec<u16>), String> {
        if let Some(sea) = &self.sea {
            return Ok(sea.grid.mesh());
        }
        if let Some(ocean) = &self.ocean {
            return Ok(ocean.grid.mesh());
        }
        let s = &self.surface;
        let (verts, indices) = crate::geometry::water_grid::build_water_grid(
            s.extent[0],
            s.extent[1],
            s.subdivisions,
        )?;
        let vertices = verts
            .into_iter()
            .map(|(pos, normal, color, uv)| Vertex {
                pos,
                normal,
                tangent: [1.0, 0.0, 0.0],
                color,
                uv,
            })
            .collect();
        Ok((vertices, indices))
    }
}

// The widest grid cell of `surface`, in meters (see geometry::water_grid).
fn cell_size(surface: &WaterSurface) -> f32 {
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

    // A pond's mesh is its own tessellation; an ocean's is the
    // camera-centered grid.
    #[test]
    fn each_path_draws_its_own_grid() {
        let surfaces = [open_sea([0.0; 3]), WaterSurface::default()];
        let (bodies, _) =
            WaterBody::resolve_all(&surfaces, &fresh_breeze(), OceanQuality::FULL, None);
        let (sea_v, sea_i) = bodies[0].mesh().unwrap();
        let ocean = bodies[0].ocean.as_ref().unwrap();
        assert_eq!((sea_v.len(), sea_i.len()), {
            let (v, i) = ocean.grid.mesh();
            (v.len(), i.len())
        });
        let (pond_v, _) = bodies[1].mesh().unwrap();
        assert_eq!(pond_v.len(), 65 * 65);
        let x = pond_v.iter().map(|v| v.pos[0]).fold(f32::MIN, f32::max);
        assert_eq!(x, 10.0);
    }

    #[test]
    fn resolving_twice_gives_the_same_body() {
        let a = WaterBody::resolve(&WaterSurface::default(), &breeze());
        let b = WaterBody::resolve(&WaterSurface::default(), &breeze());
        assert_eq!(a.waves, b.waves);
    }
}
