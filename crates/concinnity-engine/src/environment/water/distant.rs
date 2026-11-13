//! How a water surface looks from far enough away that none of its waves or
//! ripples are drawn: its body and Fresnel, and every slope they carry folded
//! into one GGX alpha, the budget the water shader folds a pixel's unresolved
//! slopes into.

use super::WaterBody;
use super::ocean::OceanSimulation;
use super::ripples::ripple_variance;
use super::spectrum::GRAVITY;
use concinnity_core::math::sqrt;

// The least authored roughness a surface is drawn at. Mirrors
// `WATER_MIN_ROUGHNESS`.
const MIN_ROUGHNESS: f32 = 0.02;

// The share of the spectrum's peak frequency below which a cascade's band
// holds no energy worth integrating; the first band starts at zero.
const LOWEST_PEAK_SHARE: f32 = 0.05;

/// A water surface as anything too far away to draw its waves sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistantLook {
    /// The reflectance of its body at full depth.
    pub body: [f32; 3],
    /// Its Fresnel falloff exponent.
    pub fresnel_power: f32,
    /// Its GGX alpha ([`WaterBody::distant_alpha`]).
    pub alpha: f32,
}

impl WaterBody {
    /// How this body looks from afar, its simulated ocean, if any, among
    /// `simulations`.
    pub fn distant_look(&self, simulations: &[OceanSimulation]) -> DistantLook {
        DistantLook {
            body: self.surface.deep_color.map(|c| c.clamp(0.0, 1.0)),
            fresnel_power: self.surface.fresnel_power.max(1e-3),
            alpha: self.distant_alpha(simulations),
        }
    }

    /// The GGX alpha this body shows from afar: its authored roughness, the
    /// slopes of its summed waves, of its simulated ocean's cascades among
    /// `simulations`, and of its ripples under the mean wind, all at once.
    /// The water shader's `spread` for a pixel that resolves none of them.
    pub fn distant_alpha(&self, simulations: &[OceanSimulation]) -> f32 {
        let ocean = self
            .ocean
            .as_ref()
            .and_then(|o| simulations.get(o.simulation))
            .map_or(0.0, ocean_slope_variance);
        sqrt(self.fine_slope_variance() + self.summed_slope_variance() + ocean)
    }

    /// The slope variance of the waves the body sums rather than simulates:
    /// its swell, and the wind sea of a body too small to simulate.
    pub fn summed_slope_variance(&self) -> f32 {
        self.waves
            .iter()
            .map(|w| {
                let slope = w.wave.wavenumber() * w.wave.amplitude;
                0.5 * slope * slope
            })
            .sum()
    }

    /// The slope variance finer than any wave or simulated cascade: the
    /// authored roughness's and the ripples' under the mean wind.
    pub fn fine_slope_variance(&self) -> f32 {
        let base = self.surface.roughness.max(MIN_ROUGHNESS);
        let base = base * base;
        base * base + ripple_variance(self.wind.strength * self.surface.wind_response)
    }
}

// The slope variance a simulated ocean's cascades carry between them: its
// spectrum's, band by band.
fn ocean_slope_variance(sim: &OceanSimulation) -> f32 {
    let Some(shape) = sim.sea.shape() else {
        return 0.0;
    };
    let floor = LOWEST_PEAK_SHARE * shape.peak_omega;
    let omega = |k: f32| sqrt(GRAVITY * k);
    (0..sim.cascades.count())
        .map(|c| {
            let [lo, hi] = sim.cascades.band(c);
            shape.slope_variance_between(omega(lo).max(floor), omega(hi))
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{WaterSurface, Wind};
    use crate::environment::water::ocean::OceanQuality;
    use crate::environment::wind::WindField;
    use alloc::vec::Vec;

    fn wind(strength: f32) -> WindField {
        WindField::new(Some(&Wind {
            strength,
            ..Wind::default()
        }))
    }

    #[test]
    fn the_floor_matches_the_shader() {
        let src = crate::render::shader::embedded::WATER;
        assert_eq!(
            crate::render::shader::consts::float(src, "WATER_MIN_ROUGHNESS"),
            MIN_ROUGHNESS
        );
    }

    #[test]
    fn still_water_keeps_its_floor_and_ripples() {
        let glassy = WaterSurface {
            roughness: 0.0,
            waves: Vec::new(),
            ..WaterSurface::default()
        };
        let body = WaterBody::resolve(&glassy, &WindField::STILL);
        let base = MIN_ROUGHNESS * MIN_ROUGHNESS;
        let want = sqrt(base * base + ripple_variance(0.0));
        assert!((body.distant_alpha(&[]) - want).abs() < 1e-6);
    }

    #[test]
    fn wind_and_authored_roughness_roughen_it() {
        let calm = WaterBody::resolve(&WaterSurface::default(), &wind(2.0));
        let windy = WaterBody::resolve(&WaterSurface::default(), &wind(12.0));
        assert!(windy.distant_alpha(&[]) > calm.distant_alpha(&[]));
        let rough = WaterSurface {
            roughness: 0.6,
            ..WaterSurface::default()
        };
        let rough = WaterBody::resolve(&rough, &wind(2.0));
        assert!(rough.distant_alpha(&[]) > calm.distant_alpha(&[]));
    }

    #[test]
    fn a_simulated_ocean_adds_its_spectrum_slopes() {
        let surface = WaterSurface {
            extent: [5000.0, 5000.0],
            fetch_meters: 1.0e5,
            ..WaterSurface::default()
        };
        let (bodies, plan) = WaterBody::resolve_all(
            core::iter::once(&surface),
            &wind(10.0),
            OceanQuality::default(),
            None,
        );
        let body = &bodies[0];
        assert!(body.ocean.is_some(), "a large windy surface is simulated");
        let sim = &plan.simulations[0];
        let slopes = ocean_slope_variance(sim);
        assert!(slopes > 0.0);
        let without = body.distant_alpha(&[]);
        let with = body.distant_alpha(&plan.simulations);
        assert!((with * with - without * without - slopes).abs() < 1e-5);
    }
}
