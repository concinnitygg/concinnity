//! Where the night sky is this frame: its galactic axes on the celestial
//! sphere and the moon in the sky, with its phase.

use super::celestial::CelestialBasis;
use super::moon::{face_up, lambert_phase};
use super::settings::NightSkySettings;
use crate::sky::SkyOrientation;
use concinnity_core::math::vec3::dot;

/// The moon this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonFrame {
    /// Unit direction toward it, in the axes the frame renders in.
    pub direction: [f32; 3],
    /// Up across its face, in the same axes.
    pub up: [f32; 3],
    /// The share of the full moon's light its phase sends.
    pub phase: f32,
    /// The fraction of its light that reaches the camera through the air.
    pub transmittance: [f32; 3],
    /// The share of its light that is on (`handover::moon_share`): 0 while
    /// the sun holds the shadows.
    pub share: f32,
}

/// The night sky's per-frame inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NightFrame {
    /// The Milky Way's pole, in the celestial sphere's frame.
    pub galactic_pole: [f32; 3],
    /// The direction of its core, in the same frame.
    pub galactic_center: [f32; 3],
    /// The moon, if the sky has one.
    pub moon: Option<MoonFrame>,
}

impl NightFrame {
    /// The frame for a sky turning about `pole` and standing at `sky`, with
    /// the sun toward `sun` on the celestial sphere (its authored
    /// direction). The moon is seen through no air: the atmosphere fills in
    /// its transmittance and share.
    pub fn new(
        settings: &NightSkySettings,
        pole: [f32; 3],
        sun: [f32; 3],
        sky: &SkyOrientation,
    ) -> Self {
        let basis = CelestialBasis::new(pole, sun);
        let (galactic_pole, galactic_center) = basis.galactic_axes(
            settings.milky_way_tilt,
            settings.milky_way_turn,
            settings.milky_way_core,
        );
        let moon = settings.moon.map(|m| {
            let celestial = m
                .path
                .direction(&basis, m.path.elongation(sky.turns, sky.angle_deg));
            MoonFrame {
                direction: sky.rotate(celestial),
                up: sky.rotate(face_up(basis.pole, celestial)),
                phase: lambert_phase(dot(celestial, basis.sun)),
                transmittance: [1.0; 3],
                share: 0.0,
            }
        });
        Self {
            galactic_pole,
            galactic_center,
            moon,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::NightSky;
    use concinnity_core::math::vec3::length;

    // The moon is carried by the sky like the sun, and a full moon opposite
    // the sun sends its whole light.
    #[test]
    fn the_moon_turns_with_the_sky() {
        let s = NightSkySettings::resolve(&NightSky::default());
        let pole = [0.0, 0.0, -1.0];
        let sun = [1.0, 0.0, 0.0];
        let sky = SkyOrientation::new(pole, 90.0);
        let f = NightFrame::new(&s, pole, sun, &sky);
        let moon = f.moon.expect("moon");
        let sun_now = sky.rotate(sun);
        assert!(
            dot(moon.direction, sun_now) < -0.99,
            "full: opposite the sun"
        );
        assert!(moon.phase > 0.99);
        assert!((length(moon.up) - 1.0).abs() < 1e-5);
        assert!(dot(moon.up, moon.direction).abs() < 1e-5);
        assert!((length(f.galactic_pole) - 1.0).abs() < 1e-5);
        let none = NightSkySettings::resolve(&NightSky {
            moon: false,
            ..Default::default()
        });
        assert!(NightFrame::new(&none, pole, sun, &sky).moon.is_none());
    }
}
