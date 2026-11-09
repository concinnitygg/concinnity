//! A planet's sea: water all the way around the world's planet, drawn on the
//! ocean's camera-centered grid bent onto the sea's sphere. Its waves are laid
//! out on the planet rather than in the simulated frame, so they stay where
//! they are on the water when the frame moves to follow the camera: each cube
//! face around the planet carries its own copy of the sea ([`SeaFace`]), and
//! where two faces meet their copies blend across a band, so no seam shows.

mod face;
pub mod shore;
mod source;

use concinnity_core::math::narrow_f32;
pub use face::SeaFace;
use shore::ShoreWaves;
pub use source::{PlanetWater, split_water};

use super::ocean::grid::OceanGrid;
use crate::environment::planet::{DVec3, PlanetShape};

/// How far a sea's grid reaches from its anchor, in planet radii along the
/// sea, before the stretch of its face: enough for the horizon from about a
/// fifth of a radius up. A higher camera spreads the grid wider.
const SEA_GRID_REACH_RADII: f32 = 0.6;

// The widest a face's stereographic map stretches, at the cube's corners:
// 2 / (1 + 1 / sqrt(3)).
const MAX_FACE_STRETCH: f32 = 1.268;

// Half the band two faces' seas blend across at their shared edge, in
// meters, and the share of the radius it is held to on a small planet.
const FACE_BLEND_METERS: f64 = 200.0;
const FACE_BLEND_RADII: f64 = 0.02;

/// The sphere a sea lies on: the world's planet radius raised by its sea
/// level, in the planet's authored frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaSphere {
    /// The planet's center, authored.
    pub center: DVec3,
    /// The sea's radius, in meters.
    pub radius: f64,
}

impl SeaSphere {
    /// The sea `level` meters above `planet`'s radius.
    pub fn new(planet: &PlanetShape, level: f32) -> Self {
        Self {
            center: planet.center,
            radius: planet.radius + f64::from(level),
        }
    }

    /// Half the band, in meters, two faces' seas blend across at their edge.
    pub fn blend_half_width(&self) -> f64 {
        FACE_BLEND_METERS.min(FACE_BLEND_RADII * self.radius)
    }

    /// How fast a face's share grows per unit of its margin (see
    /// [`SeaFace::shares`]): from 0 to 1 across the blend band.
    pub fn blend_rate(&self) -> f64 {
        self.radius / (2.0 * core::f64::consts::SQRT_2 * self.blend_half_width())
    }
}

/// A planet sea's share of a body: the sphere it lies on and the grid it is
/// drawn on, laid out in face meters around the anchor.
#[derive(Debug, Clone, PartialEq)]
pub struct SeaBody {
    /// The sphere.
    pub sphere: SeaSphere,
    /// The camera-centered grid, in face meters.
    pub grid: OceanGrid,
    /// The waves at its shores; `None` on a planet without coasts.
    pub shore: Option<ShoreWaves>,
    /// The most its waves carry a vertex sideways, in meters: the sea's own
    /// sway (see [`WaterBody::reach`](crate::environment::water::WaterBody::reach)) and its
    /// shore's.
    pub sway: f32,
}

impl SeaBody {
    /// The sea on `sphere` for a sea peaking at `peak_wavelength` meters.
    pub fn new(sphere: SeaSphere, peak_wavelength: f32) -> Self {
        let reach = SEA_GRID_REACH_RADII * MAX_FACE_STRETCH * narrow_f32(sphere.radius);
        Self {
            sphere,
            grid: OceanGrid::reaching(peak_wavelength, reach),
            shore: None,
            sway: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A sea standing at a coast carries its shore's reach in its sway, beside
    // its own.
    #[test]
    fn a_coasts_sea_sways_by_its_shore_too() {
        use crate::components::WaterSurface;
        use crate::environment::water::WaterBody;
        use crate::environment::wind::WindField;
        let planet = shore::tests::coast_planet();
        let wind = WindField::new(Some(&crate::components::Wind {
            strength: 7.0,
            ..Default::default()
        }));
        let surface = WaterSurface::default();
        let sea = PlanetWater {
            planet,
            level: 0.0,
            surface: &surface,
        };
        let (bodies, _) = WaterBody::resolve_all(
            [],
            &wind,
            crate::environment::water::ocean::OceanQuality::FULL,
            Some(sea),
        );
        let body = &bodies[0];
        let sea = body.sea.as_ref().expect("a planet sea");
        let shore = sea.shore.as_ref().expect("the coast planet has shores");
        let own = body.reach()[1];
        assert!(own > 0.0);
        assert!((sea.sway - own - narrow_f32(shore.max_shift())).abs() < 1e-4);
    }
}
