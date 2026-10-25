//! Which of a world's water surfaces is its planet's sea, and which are
//! rectangles.

use crate::components::{Planet, WaterSurface};
use crate::ecs::asset_id::AssetId;
use crate::planet::PlanetShape;
use alloc::vec::Vec;

/// A planet's sea as the water is resolved from it: the planet it lies on,
/// its height above the radius, and the surface whose look it takes.
#[derive(Debug, Clone)]
pub struct PlanetWater<'a> {
    /// The planet.
    pub planet: PlanetShape,
    /// Height above the planet's radius, in meters, at least 0.
    pub level: f32,
    /// The surface it looks like.
    pub surface: &'a WaterSurface,
}

/// The world's water: `surfaces`, each with its asset id, split into the
/// rectangles and `planet`'s sea, the surface its sea names. A sea naming no
/// surface of `surfaces` is left out.
pub fn split_water<'a>(
    surfaces: impl IntoIterator<Item = (Option<AssetId>, &'a WaterSurface)>,
    planet: Option<&Planet>,
) -> (Vec<&'a WaterSurface>, Option<PlanetWater<'a>>) {
    let sea = planet.and_then(|p| Some((p, p.sea?)));
    let sea_id = sea.map(|(_, s)| s.water.id());
    let mut rectangles = Vec::new();
    let mut water = None;
    for (id, surface) in surfaces {
        match (sea, id.is_some() && id == sea_id) {
            (Some((planet, sea)), true) => {
                water = Some(PlanetWater {
                    planet: planet.shape(),
                    level: crate::components::validate::planet_sea(sea).level,
                    surface,
                });
            }
            _ => rectangles.push(surface),
        }
    }
    (rectangles, water)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::PlanetSea;
    use crate::ecs::Ref;

    fn id(n: u32) -> Option<AssetId> {
        AssetId(n).into()
    }

    fn planet_with_sea(water: u32, level: f32) -> Planet {
        Planet {
            sea: Some(PlanetSea {
                level,
                water: Ref::new(AssetId(water)),
            }),
            ..Planet::default()
        }
    }

    #[test]
    fn the_named_surface_is_the_sea_and_the_rest_are_rectangles() {
        let pond = WaterSurface::default();
        let sea = WaterSurface {
            whitecaps: 0.5,
            ..WaterSurface::default()
        };
        let planet = planet_with_sea(2, 3.0);
        let (rects, water) = split_water([(id(1), &pond), (id(2), &sea)], Some(&planet));
        assert_eq!(rects.len(), 1);
        let water = water.expect("a sea");
        assert_eq!(water.surface.whitecaps, 0.5);
        assert_eq!(water.level, 3.0);
        assert_eq!(water.planet, planet.shape());
    }

    #[test]
    fn without_a_sea_every_surface_is_a_rectangle() {
        let a = WaterSurface::default();
        let (rects, water) = split_water([(id(2), &a), (None, &a)], Some(&Planet::default()));
        assert_eq!(rects.len(), 2);
        assert!(water.is_none());
        let (rects, water) = split_water([(id(2), &a)], None);
        assert_eq!(rects.len(), 1);
        assert!(water.is_none());
    }

    #[test]
    fn a_sea_naming_no_surface_is_left_out_and_its_level_is_clamped() {
        let a = WaterSurface::default();
        let missing = planet_with_sea(9, 1.0);
        let (rects, water) = split_water([(id(1), &a), (None, &a)], Some(&missing));
        assert_eq!(rects.len(), 2);
        assert!(water.is_none());
        let sunk = planet_with_sea(1, -4.0);
        let (_, water) = split_water([(id(1), &a)], Some(&sunk));
        assert_eq!(water.map(|w| w.level), Some(0.0));
    }
}
