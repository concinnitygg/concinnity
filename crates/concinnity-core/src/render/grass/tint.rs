//! The far field: past where the blades end, a terrain's ground is shaded as
//! the canopy its grass makes (see [`super::canopy`]), blended in over the
//! same band the blades fade out across, so the field's edge cannot be seen.
//!
//! Every corner of the terrain's grid carries, per channel, how densely one of
//! the far field's layers grows there (its mask, or 1 without one), so bare
//! patches in the far field line up with the gaps in the blades.

use alloc::vec::Vec;

use super::GrassTerrain;
use super::canopy::{CanopyField, CanopyRange};
use crate::render::grass::lod;
use crate::render::grass::tiles::GRASS_DRAW_DISTANCE;

/// A terrain's far field: the density each of its far-field layers grows at
/// each grid corner, and the run of the canopy block's layers it shades with.
#[derive(Debug, Clone, PartialEq)]
pub struct FarField {
    /// Per grid corner, row-major like the grid's heights: each far-field
    /// layer's density there in [0, 1], one channel per layer, 0 past the
    /// last.
    pub shares: Vec<[f32; 3]>,
    /// The canopy block's layers the far field shades with.
    pub canopy: CanopyRange,
}

/// The distances the far field blends over: where the blades start to fade
/// out, and where they end.
pub fn far_field_band() -> [f32; 2] {
    [lod::fade_start(GRASS_DRAW_DISTANCE), GRASS_DRAW_DISTANCE]
}

/// Every terrain's far field, in order; `None` for a terrain growing no
/// blades, or one whose layers no longer fit the canopy block.
pub fn far_fields(terrains: &[GrassTerrain]) -> Vec<Option<FarField>> {
    // Which layers each terrain's far field shades does not depend on the wind.
    let (_, runs) = CanopyField::for_terrains(terrains, 0.0);
    terrains
        .iter()
        .zip(runs)
        .map(|(terrain, run)| {
            let run = run?;
            let side = terrain.grid.side();
            let n = (side - 1).max(1) as f32;
            let mut shares = Vec::with_capacity(side * side);
            for row in 0..side {
                for col in 0..side {
                    let (s, t) = (col as f32 / n, row as f32 / n);
                    let mut share = [0.0; 3];
                    for (c, &layer) in share.iter_mut().zip(&run.layers) {
                        *c = terrain.layers[layer]
                            .1
                            .as_ref()
                            .map_or(1.0, |m| m.sample(s, t));
                    }
                    shares.push(share);
                }
            }
            Some(FarField {
                shares,
                canopy: run.range,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Grass;
    use crate::terrain::{DensityMask, TerrainGrid};
    use alloc::vec;

    fn terrain(layers: Vec<(Grass, Option<DensityMask>)>) -> GrassTerrain {
        GrassTerrain {
            center: [0.0; 3],
            grid: TerrainGrid::new(4, [10.0, 10.0], vec![0.0; 25]).unwrap(),
            layers,
        }
    }

    #[test]
    fn an_unmasked_layer_grows_fully_at_every_corner() {
        let f = far_fields(&[terrain(vec![(Grass::default(), None)])]);
        let f = f[0].as_ref().unwrap();
        assert_eq!(f.shares.len(), 25);
        assert!(f.shares.iter().all(|s| *s == [1.0, 0.0, 0.0]));
        assert_eq!(f.canopy.count, 1);
        assert!(!f.canopy.planet);
    }

    // Each channel is its layer's mask, so a bare texel leaves the corner
    // bare in that layer alone.
    #[test]
    fn masks_set_each_layers_share() {
        let mask = DensityMask::new(2, 1, vec![0, 255]).unwrap();
        let t = terrain(vec![
            (Grass::default(), Some(mask)),
            (Grass::default(), None),
        ]);
        let f = far_fields(&[t]);
        let f = f[0].as_ref().unwrap();
        // Row 0: s = 0 (bare) through s = 1 (full mask).
        assert_eq!(f.shares[0], [0.0, 1.0, 0.0]);
        assert!((f.shares[2][0] - 0.5).abs() < 1e-5);
        assert_eq!(f.shares[4], [1.0, 1.0, 0.0]);
        assert_eq!(f.canopy.count, 2);
    }

    // Runs follow the terrains in order, and a terrain growing nothing has no
    // far field.
    #[test]
    fn terrains_take_their_own_runs() {
        let hidden = Grass {
            visible: false,
            ..Grass::default()
        };
        let f = far_fields(&[
            terrain(vec![(Grass::default(), None)]),
            terrain(vec![(hidden, None)]),
            terrain(vec![(Grass::default(), None), (Grass::default(), None)]),
        ]);
        assert_eq!(f[0].as_ref().unwrap().canopy.first, 0);
        assert_eq!(f[1], None);
        let third = f[2].as_ref().unwrap();
        assert_eq!((third.canopy.first, third.canopy.count), (1, 2));
    }

    // The canopy takes over exactly as the blades fade out.
    #[test]
    fn the_band_is_the_blade_fade() {
        let [start, end] = far_field_band();
        assert_eq!(lod::fade(start, GRASS_DRAW_DISTANCE), 1.0);
        assert_eq!(lod::fade(end, GRASS_DRAW_DISTANCE), 0.0);
        assert!(start < end);
    }
}
