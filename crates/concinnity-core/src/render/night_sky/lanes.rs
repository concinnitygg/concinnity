//! The night sky's lanes in the atmosphere's constant block.

use super::frame::NightFrame;
use super::galaxies::{MAX_GALAXIES, galaxy_lanes};
use super::settings::{EARTHSHINE, GALAXY_GAIN, NightSkySettings};
use super::stars::luminance;
use crate::math::powf;
use core::f32::consts::PI;

/// The night sky's `float4` lanes, as `AtmosphereParams` carries them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NightLanes {
    /// x = a magnitude-0 star's light, y = the faintest magnitude drawn,
    /// z = the Milky Way's radiance at its heart, w = galaxies drawn.
    pub night: [f32; 4],
    /// xyz = the Milky Way's pole on the celestial sphere, w = the 24-bit
    /// seed.
    pub galactic_pole: [f32; 4],
    /// xyz = the direction of its core, w = stars per steradian.
    pub galactic_center: [f32; 4],
    /// xyz = toward the moon, w = cosine of its disc's radius (2 without a
    /// moon, which no direction reaches).
    pub moon_direction: [f32; 4],
    /// xyz = the radiance of the moon's face lit head-on, through the air,
    /// such that the full moon's face, over its disc, sends the moon's light;
    /// w = the share its dark side keeps.
    pub moon_face: [f32; 4],
    /// xyz = up across the moon's face, w = 1 when the night sky draws.
    pub moon_up: [f32; 4],
    /// Two lanes per galaxy (`galaxies::galaxy_lanes`).
    pub galaxies: [[f32; 4]; 2 * MAX_GALAXIES],
}

// Solid angle of a disc whose angular radius has cosine `cos_radius`.
fn disc_solid_angle(cos_radius: f32) -> f32 {
    (2.0 * PI * (1.0 - cos_radius)).max(1e-12)
}

impl NightLanes {
    /// No night sky: every lane zero and no moon.
    pub(crate) const OFF: Self = Self {
        night: [0.0; 4],
        galactic_pole: [0.0; 4],
        galactic_center: [0.0; 4],
        moon_direction: [0.0, 0.0, 0.0, 2.0],
        moon_face: [0.0; 4],
        moon_up: [0.0; 4],
        galaxies: [[0.0; 4]; 2 * MAX_GALAXIES],
    };

    /// The lanes for `settings` this `frame`, under a sun whose light above
    /// the air is `sun_illuminance`.
    pub(crate) fn new(
        settings: &NightSkySettings,
        frame: &NightFrame,
        sun_illuminance: [f32; 3],
    ) -> Self {
        let sun = luminance(sun_illuminance);
        let star_zero = settings.star_illuminance * sun;
        let v3 = |v: [f32; 3], w: f32| [v[0], v[1], v[2], w];
        let mut galaxies = [[0.0; 4]; 2 * MAX_GALAXIES];
        for (i, g) in settings.galaxies[..settings.galaxy_count]
            .iter()
            .enumerate()
        {
            let light = GALAXY_GAIN * star_zero * powf(10.0, -0.4 * g.magnitude);
            let [a, b] = galaxy_lanes(g, light);
            galaxies[2 * i] = a;
            galaxies[2 * i + 1] = b;
        }
        let (moon_direction, moon_face, moon_up) = match (frame.moon, settings.moon) {
            (Some(m), Some(s)) => {
                let k = s.disc_brightness * s.brightness * 1.5 / disc_solid_angle(s.cos_radius);
                let face = core::array::from_fn(|i| k * sun_illuminance[i] * m.transmittance[i]);
                (
                    v3(m.direction, s.cos_radius),
                    v3(face, EARTHSHINE),
                    v3(m.up, 1.0),
                )
            }
            _ => (Self::OFF.moon_direction, [0.0; 4], [0.0, 0.0, 0.0, 1.0]),
        };
        Self {
            night: [
                star_zero,
                settings.star_limit,
                settings.milky_way_radiance * sun,
                settings.galaxy_count as f32,
            ],
            galactic_pole: v3(frame.galactic_pole, settings.seed as f32),
            galactic_center: v3(frame.galactic_center, settings.star_density),
            moon_direction,
            moon_face,
            moon_up,
            galaxies,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::NightSky;
    use crate::sky::SkyOrientation;

    #[test]
    fn the_lanes_scale_with_the_sun_and_carry_the_moon() {
        let s = NightSkySettings::resolve(&NightSky {
            galaxies: 2,
            ..Default::default()
        });
        let sky = SkyOrientation::new([0.0, 1.0, 0.0], 0.0);
        let mut frame = NightFrame::new(&s, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], &sky);
        if let Some(m) = frame.moon.as_mut() {
            m.transmittance = [0.5; 3];
        }
        let lanes = NightLanes::new(&s, &frame, [2.0; 3]);
        assert!((lanes.night[0] - 2.0 * s.star_illuminance).abs() < 1e-15);
        assert_eq!(lanes.night[3], 2.0);
        assert_eq!(lanes.moon_up[3], 1.0);
        assert!(lanes.moon_direction[3] < 1.0);
        let moon = s.moon.expect("moon");
        let face = moon.disc_brightness * moon.brightness * 1.5 / disc_solid_angle(moon.cos_radius);
        assert!((lanes.moon_face[0] / (face * 2.0 * 0.5) - 1.0).abs() < 1e-5);
        // Two galaxies take four lanes; the rest stay zero.
        assert!(lanes.galaxies[3][3] != 0.0);
        assert_eq!(lanes.galaxies[4], [0.0; 4]);
        assert_eq!(lanes.galactic_pole[3], s.seed as f32);
    }

    // A moonless sky still draws its stars, and no direction falls on its
    // moon.
    #[test]
    fn a_moonless_sky_draws_without_a_moon() {
        let s = NightSkySettings::resolve(&NightSky {
            moon: false,
            ..Default::default()
        });
        let sky = SkyOrientation::default();
        let frame = NightFrame::new(&s, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], &sky);
        let lanes = NightLanes::new(&s, &frame, [1.0; 3]);
        assert_eq!(lanes.moon_direction[3], 2.0);
        assert_eq!(lanes.moon_up[3], 1.0);
        assert_eq!(NightLanes::OFF.moon_up[3], 0.0);
    }
}
