//! A planet's grass, resolved once at graphics init: each of the `Planet`'s
//! layers looked up to its `Grass`. The field the backend grows and the far
//! field the planet's tiles are tinted with both come from it.

use hashbrown::HashMap;

use concinnity_core::components::{Grass, Planet};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::grass::PlanetGrass;

/// The grass `planet`'s layers grow, each layer's `Grass` looked up in `grass`
/// by id; `None` when no layer grows anything. A layer naming no known
/// `Grass` grows nothing. A density mask is not applied on a planet, whose
/// density comes from its meadow alone; one is reported and left unused.
pub(crate) fn resolve(planet: &Planet, grass: &HashMap<AssetId, Grass>) -> Option<PlanetGrass> {
    let layers: Vec<Grass> = planet
        .layers
        .iter()
        .enumerate()
        .filter_map(|(i, layer)| {
            if layer.density_mask.is_some() {
                tracing::warn!(
                    "GraphicsSystem: Planet layer {i} sets a density_mask, which a Planet does not \
                     support; the layer grows over the planet's meadow without it"
                );
            }
            grass.get(&layer.grass.id()).cloned()
        })
        .collect();
    PlanetGrass::resolve(planet.shape(), &layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::TerrainLayer;
    use concinnity_core::ecs::Ref;

    #[test]
    fn layers_resolve_through_their_grass() {
        let grass: HashMap<AssetId, Grass> = [(AssetId(0), Grass::default())].into();
        let layer = |id: u32| TerrainLayer {
            grass: Ref::new(AssetId(id)),
            density_mask: None,
        };
        let planet = Planet {
            layers: vec![layer(0), layer(1)],
            ..Planet::default()
        };
        let resolved = resolve(&planet, &grass).expect("one layer grows");
        assert_eq!(resolved.layers.len(), 1);
        assert_eq!(resolved.shape, planet.shape());
        let bare = Planet {
            layers: vec![layer(1)],
            ..Planet::default()
        };
        assert!(resolve(&bare, &grass).is_none());
        assert!(resolve(&Planet::default(), &grass).is_none());
    }
}
