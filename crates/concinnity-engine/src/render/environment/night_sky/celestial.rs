//! Directions on the celestial sphere, in its own frame: the frame the sky's
//! rotation carries into the world, where the authored sun sits still.

use concinnity_core::math::vec3::{cross, dot, normalize_or, scale, sub};
use concinnity_core::math::{cos, sin};

/// The celestial pole and the sun, with the direction due east of the sun
/// along the sky's turn: the axes the Milky Way and the moon's path are set
/// against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CelestialBasis {
    /// The pole the sky turns about.
    pub pole: [f32; 3],
    /// Toward the sun.
    pub sun: [f32; 3],
    /// Square to both, the way a body drifting east of the sun moves: it
    /// rises and sets after the sun.
    pub east: [f32; 3],
    /// The sun's side of the pole: square to the pole, toward the sun.
    pub toward_sun: [f32; 3],
}

impl CelestialBasis {
    /// The basis for a sky turning about `pole` with the sun toward `sun`
    /// (neither need be normalized). A sun on the pole takes an arbitrary
    /// side.
    pub fn new(pole: [f32; 3], sun: [f32; 3]) -> Self {
        let pole = normalize_or(pole, 1e-6, [0.0, 1.0, 0.0]);
        let sun = normalize_or(sun, 1e-6, [0.0, 1.0, 0.0]);
        let fallback = if pole[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        let side = sub(sun, scale(pole, dot(pole, sun)));
        let toward_sun = normalize_or(
            side,
            1e-4,
            normalize_or(cross(fallback, pole), 1e-6, fallback),
        );
        let east = cross(pole, toward_sun);
        Self {
            pole,
            sun,
            east,
            toward_sun,
        }
    }

    /// The Milky Way's pole and the direction of its core: the pole leans
    /// `tilt` from the celestial pole toward the side `turn` east of the
    /// sun, and the core sits `core` along the band from the point nearest
    /// the celestial pole. Radians.
    pub fn galactic_axes(&self, tilt: f32, turn: f32, core: f32) -> ([f32; 3], [f32; 3]) {
        let lean: [f32; 3] =
            core::array::from_fn(|i| cos(turn) * self.toward_sun[i] + sin(turn) * self.east[i]);
        let g: [f32; 3] = core::array::from_fn(|i| cos(tilt) * self.pole[i] + sin(tilt) * lean[i]);
        let g = normalize_or(g, 1e-6, self.pole);
        let up = normalize_or(
            sub(self.pole, scale(g, dot(g, self.pole))),
            1e-4,
            self.toward_sun,
        );
        let side = cross(g, up);
        let center = core::array::from_fn(|i| cos(core) * up[i] + sin(core) * side[i]);
        (g, normalize_or(center, 1e-6, up))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::math::vec3::length;

    #[test]
    fn the_basis_is_square_and_east_trails_the_sun() {
        let b = CelestialBasis::new([0.0, 0.0, -2.0], [1.0, 0.0, 0.35]);
        for v in [b.pole, b.sun, b.east, b.toward_sun] {
            assert!((length(v) - 1.0).abs() < 1e-5);
        }
        assert!(dot(b.east, b.pole).abs() < 1e-5);
        assert!(dot(b.east, b.sun).abs() < 1e-5);
        assert!(dot(b.toward_sun, b.sun) > 0.9);
        // The sky's turn carries the sun away from east, so a body east of
        // it follows it across the sky.
        let sun_then = crate::sky::SkyOrientation::new(b.pole, 10.0).rotate(b.sun);
        assert!(dot(sun_then, b.east) < 0.0, "{sun_then:?}");
        // A sun on the pole still gets a basis.
        let on = CelestialBasis::new([0.0, 1.0, 0.0], [0.0, 3.0, 0.0]);
        assert!((length(on.east) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn the_galactic_pole_leans_by_its_tilt() {
        let b = CelestialBasis::new([0.0, 1.0, 0.0], [1.0, 0.2, 0.0]);
        let tilt = 62.9_f32.to_radians();
        let (g, c) = b.galactic_axes(tilt, 0.0, 0.0);
        assert!((dot(g, b.pole) - cos(tilt)).abs() < 1e-5);
        assert!(dot(g, b.toward_sun) > 0.8, "leans toward the sun at turn 0");
        let (g_east, _) = b.galactic_axes(tilt, core::f32::consts::FRAC_PI_2, 0.0);
        assert!(dot(g_east, b.east) > 0.8);
        // The core lies in the band, at the point nearest the pole for 0.
        assert!(dot(c, g).abs() < 1e-5);
        let (_, turned) = b.galactic_axes(tilt, 0.0, 1.0);
        assert!(dot(turned, g).abs() < 1e-5);
        assert!(dot(c, b.pole) > dot(turned, b.pole));
        // Untilted, the band is the celestial equator.
        let (flat, _) = b.galactic_axes(0.0, 0.0, 0.0);
        assert!((dot(flat, b.pole) - 1.0).abs() < 1e-6);
    }
}
