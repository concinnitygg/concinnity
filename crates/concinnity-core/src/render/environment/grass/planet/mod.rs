//! Grass on a planet: the layers a `Planet` declares, grown over the tiles it
//! is drawn with. Blades are placed in cells fixed to the planet's faces
//! ([`cells`]), thinned by a meadow laid over the sphere ([`density`]), and
//! rooted on the drawn tiles' own meshes ([`tiles`]); the far field past them
//! is thinned by the same meadow (see [`super::canopy`]).

pub mod cells;
pub mod density;
pub mod tiles;

use crate::math::SaturatingFrom;
use alloc::vec::Vec;

use super::GrassBladeLook;
use super::canopy::{CanopyField, CanopyRange};
use crate::components::Grass;
use crate::math::noise::lcg_hash;
use crate::planet::PlanetShape;
use cells::PlanetGrassGrid;

/// One look grown over a planet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetGrassLayer {
    /// What each blade looks like.
    pub look: GrassBladeLook,
    /// Blades per square meter where the meadow grows fully.
    pub density: f32,
    /// Cells along each side of a grass tile.
    pub cells: u32,
    /// How wide a cell is across a face's middle, in meters.
    pub cell_size: f32,
    /// Salts the layer's placement and its meadow.
    pub seed: u32,
}

/// A planet's grass: its surface, the cells its faces divide into, and the
/// layers grown on them.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanetGrass {
    /// The planet's surface.
    pub shape: PlanetShape,
    /// How its faces divide into grass tiles.
    pub grid: PlanetGrassGrid,
    /// The layers, in the order the planet declares them.
    pub layers: Vec<PlanetGrassLayer>,
    /// Lattice cells of the meadow noise per unit of direction.
    pub meadow_scale: f32,
}

impl PlanetGrass {
    /// The grass `layers` grow on `shape`; `None` when no visible layer places
    /// a blade.
    pub fn resolve(shape: PlanetShape, layers: &[Grass]) -> Option<Self> {
        let grid = PlanetGrassGrid::new(&shape);
        let mut out = Vec::new();
        for grass in layers.iter().filter(|g| g.visible) {
            let grass = crate::components::validate::grass(grass.clone());
            let Some(cells) = grid.cells_per_tile(grass.density) else {
                continue;
            };
            out.push(PlanetGrassLayer {
                look: GrassBladeLook::of(&grass),
                density: grass.density.min(super::tiles::MAX_GRASS_DENSITY),
                cells,
                cell_size: grid.cell_size(cells),
                seed: lcg_hash(u32::saturating_from(out.len()) ^ 0x5bd1_e995),
            });
        }
        (!out.is_empty()).then(|| Self {
            shape,
            grid,
            layers: out,
            meadow_scale: density::meadow_scale(shape.radius),
        })
    }

    /// The run of the canopy block's layers the planet's far field shades
    /// with.
    pub fn far_field(&self) -> CanopyRange {
        CanopyField::for_planet(self, 0.0).1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 8,
            seed: 4,
        }
    }

    #[test]
    fn visible_layers_with_density_resolve() {
        let hidden = Grass {
            visible: false,
            ..Grass::default()
        };
        let sparse = Grass {
            density: 0.0,
            ..Grass::default()
        };
        assert_eq!(
            PlanetGrass::resolve(shape(), &[hidden.clone(), sparse.clone()]),
            None
        );
        let g = PlanetGrass::resolve(
            shape(),
            &[hidden, Grass::default(), sparse, Grass::default()],
        )
        .unwrap();
        assert_eq!(g.layers.len(), 2);
        assert_ne!(g.layers[0].seed, g.layers[1].seed);
        let l = &g.layers[0];
        assert_eq!(
            Some(l.cells),
            g.grid.cells_per_tile(Grass::default().density)
        );
        assert_eq!(g.far_field().count, 2);
    }
}
