//! A water surface resolved for drawing: its authored fields, the waves the
//! world's wind raises on it plus its authored swell, and which of those move
//! the grid rather than only its shading.

pub mod motion;
pub mod ripples;
pub mod spectrum;

use crate::components::{MAX_WATER_WAVES, WaterSurface};
use crate::render::wind::WindField;
use alloc::vec::Vec;
use ripples::{RIPPLES, Ripple};
use spectrum::{WIND_SEA_WAVES, Wave, unit_hash, wind_sea};

/// The most waves one surface sums: the wind sea plus the authored swell.
pub const MAX_WAVES: usize = WIND_SEA_WAVES + MAX_WATER_WAVES;

/// The most waves that displace the grid; the rest shade it only.
pub const MAX_DISPLACING_WAVES: usize = 16;

/// The shortest wave that displaces the grid, in grid cells: anything finer
/// would alias between vertices, so it moves the shading normal only.
pub const DISPLACE_MIN_CELLS: f32 = 2.0;

/// The longest ripple the shader's procedural ripples carry, in meters, when
/// the wind raises no sea for them to continue.
pub const STILL_RIPPLE_WAVELENGTH: f32 = 0.06;

// The share of its own self-intersection limit the wind sea's displacing
// waves pinch their crests by together, and the ceiling every displacing wave
// together is held under so the summed crests never fold over.
const WIND_SEA_PINCH: f32 = 0.6;
const MAX_TOTAL_PINCH: f32 = 0.95;

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
    /// shortest wave of the wind sea, or [`STILL_RIPPLE_WAVELENGTH`].
    pub ripple_wavelength: f32,
    /// The ripple octaves, longest first, starting at `ripple_wavelength`.
    pub ripples: [Ripple; RIPPLES],
}

impl WaterBody {
    /// `surface` moved by `wind`: the sea the wind raises over the surface's
    /// fetch, at its `wind_response`, and its authored swell.
    pub fn resolve(surface: &WaterSurface, wind: &WindField) -> Self {
        let surface = crate::components::validate::water_surface(surface.clone());
        let sea = wind_sea(
            wind.strength * surface.wind_response,
            wind.direction,
            surface.fetch_meters,
        );
        let ripple_wavelength = sea.last().map_or(STILL_RIPPLE_WAVELENGTH, |w| w.wavelength);
        let swell = surface
            .waves
            .iter()
            .enumerate()
            .filter(|(_, w)| w.amplitude > 0.0)
            .map(|(i, w)| Wave {
                direction: w.direction,
                amplitude: w.amplitude,
                wavelength: w.wavelength,
                phase: unit_hash(i as u32, 11) * core::f32::consts::TAU,
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

        let min_displacing = DISPLACE_MIN_CELLS * cell_size(&surface);
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
        }
    }

    /// How far the displaced grid reaches from its flat rest: `[rise, sway]`,
    /// the most a vertex moves vertically and horizontally.
    pub fn reach(&self) -> [f32; 2] {
        self.waves[..self.displacing]
            .iter()
            .fold([0.0, 0.0], |[rise, sway], w| {
                [rise + w.wave.amplitude, sway + w.pinch * w.wave.amplitude]
            })
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

    #[test]
    fn resolving_twice_gives_the_same_body() {
        let a = WaterBody::resolve(&WaterSurface::default(), &breeze());
        let b = WaterBody::resolve(&WaterSurface::default(), &breeze());
        assert_eq!(a.waves, b.waves);
    }
}
