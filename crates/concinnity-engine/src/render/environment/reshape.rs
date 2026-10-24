//! A planet reshaped while the world runs: what a renderer keeps of its shape,
//! brought to the new one together.

use crate::planet::PlanetShape;
use crate::render::environment::grass::{CanopyField, GrassField};
use crate::render::environment::water::frame::WaterMotion;

/// Bring what a renderer keeps of a planet to `shape`, which keeps the
/// planet's center and radius: its sea's shore, where its grass roots, and
/// its canopy block, whose beaches the sea's new shore wets.
pub fn reshape_planet<'a>(
    shape: &PlanetShape,
    water: impl IntoIterator<Item = &'a mut WaterMotion>,
    grass: Option<&mut GrassField>,
    canopy: Option<&mut CanopyField>,
) {
    let mut shore = None;
    for motion in water {
        motion.reshape(shape);
        if shore.is_none() {
            shore = motion.body().sea.as_ref().and_then(|s| s.shore.clone());
        }
    }
    if let Some(planet) = grass.and_then(|g| g.planet.as_mut()) {
        planet.shape = *shape;
    }
    if let Some(canopy) = canopy {
        canopy.reshape(shape, shore);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Planet, WaterSurface, Wind};
    use crate::render::environment::ground::GroundCovers;
    use crate::water::ocean::OceanQuality;
    use crate::water::{PlanetWater, WaterBody};
    use crate::wind::WindField;
    use alloc::vec::Vec;

    fn coast(seed: u32) -> PlanetShape {
        Planet {
            land_share: 0.4,
            seed,
            ..Planet::default()
        }
        .shape()
    }

    // The sea's new shore reaches the canopy, and a rectangle beside it is
    // left alone.
    #[test]
    fn the_sea_and_the_canopy_follow_the_new_coasts() {
        let surface = WaterSurface {
            fetch_meters: 5.0e4,
            ..WaterSurface::default()
        };
        let sea = PlanetWater {
            planet: coast(1),
            level: 0.0,
            surface: &surface,
        };
        let wind = WindField::new(Some(&Wind {
            strength: 7.0,
            ..Wind::default()
        }));
        let pond = WaterSurface::default();
        let (bodies, _) = WaterBody::resolve_all([&pond], &wind, OceanQuality::FULL, Some(sea));
        let mut water: Vec<WaterMotion> =
            bodies.iter().map(|b| WaterMotion::new(b, false)).collect();
        let (mut canopy, _) = CanopyField::for_planet(coast(1), None, GroundCovers::default(), 7.0);
        let reshaped = coast(2);
        reshape_planet(&reshaped, &mut water, None, Some(&mut canopy));
        let mut expected = bodies[0].clone();
        expected.reshape(&reshaped);
        assert_eq!(water[0].body().sea, expected.sea);
        assert_eq!(water[1].body().sea, None);
        let (mut want, _) = CanopyField::for_planet(reshaped, None, GroundCovers::default(), 7.0);
        want.set_shore(expected.sea.and_then(|s| s.shore));
        assert_eq!(canopy, want);
    }
}
