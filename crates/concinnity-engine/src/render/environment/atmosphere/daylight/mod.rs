//! The light a point on the ground of a planet receives, by how high the sun
//! (or the moon) stands above its own horizon: the curves the surfaces far
//! from the camera are lit from.
//!
//! The sky tables the GPU builds each frame describe the sky around the
//! camera. Seen from high above a planet, the ground far below lies under
//! other skies: day on one side, night on the other, a sunset between. Over
//! air the same all round the planet, what a point's sky looks like depends
//! only on the light's height above the point's horizon, so it is baked here
//! once, on the CPU, as curves over the cosine of that height: how much of the
//! light reaches the point, how much the sky lights a level surface with, and
//! the sky's radiance overhead and at the horizon. Every value is per unit of
//! light above the air.

mod bake;
#[cfg(test)]
mod tests;

use super::settings::AtmosphereSettings;
use concinnity_core::fanout::FanOut;
#[cfg(test)]
use concinnity_core::math::{SaturatingFrom, sqrt};

/// Knots each curve is sampled at. `DAYLIGHT_KNOTS` in `daylight.hlsl`.
pub const DAYLIGHT_KNOTS: usize = 32;

/// What a point on the ground receives from a light at one height, per unit
/// of that light above the air.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Daylight {
    /// The share of the light that reaches the point.
    transmittance: [f32; 3],
    /// The sky's light on a level surface at the point.
    irradiance: [f32; 3],
    /// The sky's radiance straight up.
    zenith: [f32; 3],
    /// The sky's radiance just above the horizon, averaged around it.
    horizon: [f32; 3],
}

#[cfg(test)]
impl Daylight {
    // `self` and `other` mixed by `t`.
    fn lerp(&self, other: &Self, t: f32) -> Self {
        let mix = |a: [f32; 3], b: [f32; 3]| core::array::from_fn(|i| a[i] + (b[i] - a[i]) * t);
        Self {
            transmittance: mix(self.transmittance, other.transmittance),
            irradiance: mix(self.irradiance, other.irradiance),
            zenith: mix(self.zenith, other.zenith),
            horizon: mix(self.horizon, other.horizon),
        }
    }
}

/// The curves for one atmosphere, sampled at [`DAYLIGHT_KNOTS`] heights of the
/// light, closest together where it crosses the horizon.
#[derive(Debug, Clone, PartialEq)]
pub struct DaylightCurves {
    knots: [Daylight; DAYLIGHT_KNOTS],
}

impl DaylightCurves {
    /// The curves for the ground of `settings`, under the sky drawn from that
    /// ground (the apparent one, where the sky is drawn over a bigger planet),
    /// spread over `rows`.
    pub fn bake<S: FanOut>(settings: &AtmosphereSettings, rows: &S) -> Self {
        let ground = settings.with_ground_radius(settings.sky_ground_radius(0.0));
        Self {
            knots: bake::knots(&ground, rows),
        }
    }

    /// The knots packed as `DaylightBlock.knots` carries them: three rows
    /// per knot, the transmittance, the irradiance and the horizon, with the
    /// zenith's red, green and blue in their fourth lanes.
    pub fn packed(&self) -> [[f32; 4]; DAYLIGHT_KNOTS * 3] {
        core::array::from_fn(|row| {
            let k = &self.knots[row / 3];
            let lane = row % 3;
            let c = match lane {
                0 => k.transmittance,
                1 => k.irradiance,
                _ => k.horizon,
            };
            [c[0], c[1], c[2], k.zenith[lane]]
        })
    }

    // The knots, in the order `knot_mu` places them.
    #[cfg(test)]
    fn knots(&self) -> &[Daylight; DAYLIGHT_KNOTS] {
        &self.knots
    }

    // The curves at a light whose cosine to the point's up is `mu`, linear
    // between knots. Mirrors `daylight_at` in `daylight.hlsl`.
    #[cfg(test)]
    fn sample(&self, mu: f32) -> Daylight {
        let x = knot_coord(mu);
        let i = usize::saturating_from(x).min(DAYLIGHT_KNOTS - 2);
        self.knots[i].lerp(&self.knots[i + 1], x - i as f32)
    }
}

// The cosine to the up of knot `i`: from straight down to straight up, by
// the square of an even step, so the knots crowd round the horizon.
fn knot_mu(i: usize) -> f32 {
    let s = 2.0 * i as f32 / (DAYLIGHT_KNOTS - 1) as f32 - 1.0;
    s * s.abs()
}

// Where a cosine `mu` falls among the knots, as a fractional knot index.
// Mirrors `daylight_coord` in `daylight.hlsl`.
#[cfg(test)]
fn knot_coord(mu: f32) -> f32 {
    let mu = mu.clamp(-1.0, 1.0);
    let s = if mu < 0.0 { -sqrt(-mu) } else { sqrt(mu) };
    (s + 1.0) * 0.5 * (DAYLIGHT_KNOTS - 1) as f32
}
