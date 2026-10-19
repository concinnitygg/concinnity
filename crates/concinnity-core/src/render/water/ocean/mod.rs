//! Large water as a simulated ocean: the wind sea of a body too large for a
//! sum of waves to stay convincing is synthesized on the GPU from its spectrum
//! instead (Tessendorf, "Simulating Ocean Water"). Each simulation evolves a
//! few square tiles of the sea, the cascades, each covering its own band of
//! wavelengths, and the water shader samples their heights, slopes and slope
//! variance by world position. The procedural ripples carry on below the
//! shortest wave the last cascade holds.
//!
//! The host prepares everything once per simulation: the band each cascade
//! holds, the initial spectrum h0(k), and how far each texel's wave turns per
//! repeat period. Per frame it only wraps time and the camera's position into
//! small numbers ([`clock`]).

pub mod cascades;
pub mod clock;
#[cfg(test)]
mod fields;
pub mod foam;
pub mod grid;
pub mod initial;

use crate::components::WaterSurface;
use crate::math::{SaturatingFrom, atan2};
use alloc::vec::Vec;
use cascades::Cascades;

use super::spectrum::SeaShape;

/// A body takes the simulated ocean once its longer side spans at least this
/// many of its sea's peak wavelengths: past that a sum of a few dozen waves
/// shows its repeating interference.
pub const OCEAN_MIN_PEAK_WAVELENGTHS: f32 = 16.0;

/// A body takes the simulated ocean only once its longer side reaches this
/// many meters: below it the sum costs less than any simulation could.
pub const OCEAN_MIN_SIDE_METERS: f32 = 256.0;

/// The most distinct seas one world simulates. Bodies past it whose sea
/// differs from every simulated one keep the summed waves.
pub const MAX_OCEAN_SIMULATIONS: usize = 4;

/// The most cascades one simulation runs. Mirrors `WATER_OCEAN_MAX_CASCADES`
/// in `shaders/water.hlsl`.
pub const MAX_OCEAN_CASCADES: usize = 4;

/// How far the simulated sea displaces horizontally, per meter it displaces
/// vertically at the same wavelength: sharper crests, broader troughs.
pub const OCEAN_CHOPPINESS: f32 = 0.85;

/// How finely and how widely an ocean is simulated: the texels along each
/// side of a cascade and the number of cascades.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OceanQuality {
    /// Texels along each side of a cascade, a power of two in
    /// [`OceanQuality::MIN_SIZE`, `OceanQuality::MAX_SIZE`].
    pub size: u32,
    /// Cascades per simulation, in `1..=MAX_OCEAN_CASCADES`.
    pub cascades: u32,
}

impl OceanQuality {
    /// The finest simulation: three cascades of 256 x 256.
    pub const FULL: Self = Self {
        size: 256,
        cascades: 3,
    };
    /// The smallest cascade side.
    pub const MIN_SIZE: u32 = 64;
    /// The largest cascade side. Mirrors `WATER_FFT_MAX_SIZE` in
    /// `shaders/water_fft.hlsl`.
    pub const MAX_SIZE: u32 = 256;

    /// `self` held to what the simulation supports: the size rounded down to
    /// a power of two within its range, the cascades within theirs.
    #[must_use]
    pub fn clamped(self) -> Self {
        let size = self.size.clamp(Self::MIN_SIZE, Self::MAX_SIZE);
        Self {
            size: 1 << size.ilog2(),
            cascades: self
                .cascades
                .clamp(1, u32::saturating_from(MAX_OCEAN_CASCADES)),
        }
    }

    /// `log2(size)`, the FFT's stage count.
    pub fn log2_size(&self) -> u32 {
        self.size.trailing_zeros()
    }
}

impl Default for OceanQuality {
    fn default() -> Self {
        Self::FULL
    }
}

/// The sea one simulation evolves: everything its spectrum depends on, so
/// bodies that agree on it share one simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OceanSea {
    /// The wind speed the sea is raised by, in meters per second, after the
    /// surface's `wind_response`.
    pub wind_speed: f32,
    /// Unit direction the wind blows toward, `[x, z]`.
    pub direction: [f32; 2],
    /// The open water the wind blows across, in meters.
    pub fetch: f32,
}

impl OceanSea {
    /// The sea `surface` takes under a wind of `wind_speed` m/s toward
    /// `direction`, when it is large enough to simulate: a wind sea whose
    /// peak wavelength its longer side spans [`OCEAN_MIN_PEAK_WAVELENGTHS`]
    /// times, on a side of at least [`OCEAN_MIN_SIDE_METERS`].
    pub fn of(surface: &WaterSurface, wind_speed: f32, direction: [f32; 2]) -> Option<Self> {
        let wind_speed = wind_speed * surface.wind_response;
        let sea = SeaShape::new(wind_speed, surface.fetch_meters)?;
        let side = 2.0 * surface.extent[0].max(surface.extent[1]);
        let large = side >= OCEAN_MIN_SIDE_METERS
            && side >= OCEAN_MIN_PEAK_WAVELENGTHS * sea.peak_wavelength();
        large.then_some(Self {
            wind_speed,
            direction,
            fetch: surface.fetch_meters,
        })
    }

    /// The sea `surface` takes under a wind of `wind_speed` m/s toward
    /// `direction` whatever its size, when the wind raises one at all: a
    /// planet's sea is always open water.
    pub fn open(surface: &WaterSurface, wind_speed: f32, direction: [f32; 2]) -> Option<Self> {
        let wind_speed = wind_speed * surface.wind_response;
        SeaShape::new(wind_speed, surface.fetch_meters)?;
        Some(Self {
            wind_speed,
            direction,
            fetch: surface.fetch_meters,
        })
    }

    /// The spectrum's shape; a sea built by [`OceanSea::of`] always has one.
    pub fn shape(&self) -> Option<SeaShape> {
        SeaShape::new(self.wind_speed, self.fetch)
    }

    /// The angle the wind blows toward, in radians from +X toward +Z.
    pub fn wind_angle(&self) -> f32 {
        atan2(self.direction[1], self.direction[0])
    }
}

/// One simulated sea: its spectrum, its cascades, and where its cascades sit
/// in the shared texture arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct OceanSimulation {
    /// The sea it evolves.
    pub sea: OceanSea,
    /// Its cascades.
    pub cascades: Cascades,
    /// Its first cascade's layer in the shared arrays; the rest follow.
    pub first_layer: u32,
    /// Where its cascades whitecap; `None` while no body sharing it asks for
    /// whitecaps, which skips the foam step.
    pub foam: Option<foam::FoamCalibration>,
}

/// A body's share of a simulated ocean, and the grid it is drawn on.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyOcean {
    /// Index into the world's simulations.
    pub simulation: usize,
    /// That simulation's cascades.
    pub cascades: Cascades,
    /// Its first cascade's layer in the shared arrays.
    pub first_layer: u32,
    /// The camera-centered grid the body is drawn on.
    pub grid: grid::OceanGrid,
}

/// Every simulation a world's bodies need, in the order their layers are
/// laid out, all at one quality.
#[derive(Debug, Clone, PartialEq)]
pub struct OceanPlan {
    /// The distinct seas, at most [`MAX_OCEAN_SIMULATIONS`].
    pub simulations: Vec<OceanSimulation>,
    /// The quality every simulation runs at.
    pub quality: OceanQuality,
}

impl OceanPlan {
    /// No simulations yet, each to run at `quality`.
    pub fn new(quality: OceanQuality) -> Self {
        Self {
            simulations: Vec::new(),
            quality: quality.clamped(),
        }
    }

    /// The index of the simulation `sea` shares, added if it is new and
    /// there is room; `None` when every slot holds another sea. A body that
    /// `whitecaps` makes its simulation foam for every body sharing it.
    pub fn share(&mut self, sea: OceanSea, whitecaps: bool) -> Option<usize> {
        let index = match self.simulations.iter().position(|s| s.sea == sea) {
            Some(i) => i,
            None if self.simulations.len() < MAX_OCEAN_SIMULATIONS => {
                let shape = sea.shape()?;
                let first_layer = self.layer_count();
                self.simulations.push(OceanSimulation {
                    sea,
                    cascades: Cascades::new(shape.peak_wavelength(), self.quality),
                    first_layer,
                    foam: None,
                });
                self.simulations.len() - 1
            }
            None => return None,
        };
        let sim = &mut self.simulations[index];
        if whitecaps && sim.foam.is_none() {
            sim.foam = Some(foam::FoamCalibration::new(&sim.sea, &sim.cascades));
        }
        Some(index)
    }

    /// Layers the shared arrays hold: every simulation's cascades.
    pub fn layer_count(&self) -> u32 {
        self.simulations
            .iter()
            .map(|s| u32::saturating_from(s.cascades.count()))
            .sum()
    }

    /// Whether no body simulates an ocean.
    pub fn is_empty(&self) -> bool {
        self.simulations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface(half: f32, fetch: f32) -> WaterSurface {
        WaterSurface {
            extent: [half, half],
            fetch_meters: fetch,
            ..Default::default()
        }
    }

    // Pools, ponds and lakes keep the summed waves; a sea across kilometers
    // of open water is simulated.
    #[test]
    fn only_large_water_under_wind_is_simulated() {
        // The default pond in a breeze.
        assert!(OceanSea::of(&WaterSurface::default(), 3.0, [1.0, 0.0]).is_none());
        // A 120 m bay in a breeze spans many peaks but is small.
        assert!(OceanSea::of(&surface(60.0, 1000.0), 3.0, [1.0, 0.0]).is_none());
        // Five kilometers of open sea in a fresh breeze.
        let sea = OceanSea::of(&surface(2500.0, 1.0e5), 9.0, [1.0, 0.0]).expect("simulated");
        assert_eq!(sea.wind_speed, 9.0);
        // The same sea in still air raises nothing to simulate.
        assert!(OceanSea::of(&surface(2500.0, 1.0e5), 0.0, [1.0, 0.0]).is_none());
        // A long fetch builds a peak too long for a 300 m surface to hold
        // sixteen of.
        assert!(OceanSea::of(&surface(150.0, 1.0e6), 15.0, [1.0, 0.0]).is_none());
        // Sheltered water raises no sea however large it is.
        let sheltered = WaterSurface {
            wind_response: 0.0,
            ..surface(2500.0, 1.0e5)
        };
        assert!(OceanSea::of(&sheltered, 9.0, [1.0, 0.0]).is_none());
    }

    // The threshold sits at the larger of the two floors.
    #[test]
    fn the_threshold_is_the_larger_floor() {
        for (wind, fetch) in [(9.0, 1.0e5), (3.0, 1000.0)] {
            let sea = SeaShape::new(wind, fetch).unwrap();
            let need =
                (OCEAN_MIN_PEAK_WAVELENGTHS * sea.peak_wavelength()).max(OCEAN_MIN_SIDE_METERS);
            let half = 0.5 * need;
            let at = |h: f32| OceanSea::of(&surface(h, fetch), wind, [1.0, 0.0]);
            assert!(at(half * 1.01).is_some(), "{wind} {fetch}");
            assert!(at(half * 0.99).is_none(), "{wind} {fetch}");
        }
    }

    #[test]
    fn quality_is_held_to_what_the_simulation_supports() {
        let q = OceanQuality {
            size: 200,
            cascades: 9,
        }
        .clamped();
        assert_eq!(
            q,
            OceanQuality {
                size: 128,
                cascades: 4
            }
        );
        let q = OceanQuality {
            size: 4096,
            cascades: 0,
        }
        .clamped();
        assert_eq!(
            q,
            OceanQuality {
                size: 256,
                cascades: 1
            }
        );
        assert_eq!(OceanQuality::FULL.clamped(), OceanQuality::FULL);
        assert_eq!(OceanQuality::FULL.log2_size(), 8);
    }

    // Bodies with the same sea share a simulation; a new sea takes the next
    // layers until the plan is full.
    #[test]
    fn identical_seas_share_one_simulation() {
        let mut plan = OceanPlan::new(OceanQuality::FULL);
        let sea = |u: f32| OceanSea {
            wind_speed: u,
            direction: [1.0, 0.0],
            fetch: 1.0e5,
        };
        let a = plan.share(sea(8.0), false).unwrap();
        let b = plan.share(sea(8.0), false).unwrap();
        assert_eq!(a, b);
        assert_eq!(plan.simulations.len(), 1);
        let c = plan.share(sea(9.0), false).unwrap();
        assert_eq!(c, 1);
        assert_eq!(plan.simulations[c].first_layer, 3);
        assert_eq!(plan.layer_count(), 6);
        for u in 10..20 {
            let _ = plan.share(sea(u as f32), false);
        }
        assert_eq!(plan.simulations.len(), MAX_OCEAN_SIMULATIONS);
        assert!(plan.share(sea(30.0), false).is_none());
        // A simulated sea still shares once the plan is full.
        assert_eq!(plan.share(sea(8.0), false), Some(a));
    }

    // A shared simulation foams once any body sharing it asks for
    // whitecaps, whichever body that is, and for every body after it.
    #[test]
    fn a_simulation_foams_when_any_member_whitecaps() {
        let sea = OceanSea {
            wind_speed: 12.0,
            direction: [1.0, 0.0],
            fetch: 1.0e5,
        };
        let mut plan = OceanPlan::new(OceanQuality::FULL);
        assert_eq!(plan.share(sea, false), Some(0));
        assert_eq!(plan.share(sea, false), Some(0));
        assert!(plan.simulations[0].foam.is_none());
        assert_eq!(plan.share(sea, true), Some(0));
        let calibrated = foam::FoamCalibration::new(&sea, &plan.simulations[0].cascades);
        assert_eq!(plan.simulations[0].foam, Some(calibrated));
        assert_eq!(plan.share(sea, false), Some(0));
        assert_eq!(plan.simulations[0].foam, Some(calibrated));

        let mut first = OceanPlan::new(OceanQuality::FULL);
        assert_eq!(first.share(sea, true), Some(0));
        assert_eq!(first.share(sea, false), Some(0));
        assert_eq!(first.simulations, plan.simulations);
    }

    #[test]
    fn the_caps_match_the_shaders() {
        let src = crate::render::shaders::WATER;
        assert_eq!(
            crate::render::shader_consts::uint(src, "WATER_OCEAN_MAX_CASCADES"),
            MAX_OCEAN_CASCADES
        );
        let fft = crate::render::shaders::WATER_FFT;
        assert_eq!(
            crate::render::shader_consts::uint(fft, "WATER_FFT_MAX_SIZE"),
            OceanQuality::MAX_SIZE as usize
        );
    }
}
