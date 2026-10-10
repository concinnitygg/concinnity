//! Backend-agnostic resolution of a world's grass into what the grass pass
//! dispatches and draws: every terrain layer's ground, blade look and cell
//! grid, the buffers of heights and mask texels the kernel reads, and the wind,
//! turned each frame into the [`GrassParams`] every grass stage reads. Pure
//! CPU; the backend owns the buffers and pipelines.
//!
//! One dispatch covers every layer within reach of the camera: its `z` picks
//! the layer, its `y` the tile within that layer's block, its `x` the run of
//! cells within the tile. A tile off every terrain is never dispatched.

mod ground;
pub mod tiles;

pub use ground::{GrassGround, GrassGroundBuffers, GrassMask, GroundSample};

use alloc::vec::Vec;

use crate::components::{Grass, Wind};
use crate::gfx::frustum::Frustum;
use crate::math::noise::lcg_hash;
use crate::render::uniforms::grass::{
    GRASS_ARGS_SLOTS, GrassLayerGpu, GrassParams, MAX_GRASS_LAYERS,
};
use crate::render::wind::WindField;
use crate::terrain::{DensityMask, TerrainGrid};
use tiles::{GRASS_DRAW_DISTANCE, GRASS_TILE_SIZE, GrassGrid, MAX_GRASS_BLADES, TileRange};

/// What every blade of a layer looks like, wherever it grows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassBladeLook {
    /// Mean height, in meters.
    pub height: f32,
    /// Height variation as a fraction of `height`.
    pub height_variance: f32,
    /// Width at the root, in meters.
    pub width: f32,
    /// Clump diameter, in meters.
    pub clump_size: f32,
    /// Resistance to bending, in [0, 1].
    pub stiffness: f32,
    /// Linear RGB at the root.
    pub root_color: [f32; 3],
    /// Linear RGB at the tip.
    pub tip_color: [f32; 3],
    /// Hue and brightness variation, in [0, 1].
    pub color_variation: f32,
}

impl GrassBladeLook {
    fn of(g: &Grass) -> Self {
        Self {
            height: g.height,
            height_variance: g.height_variance,
            width: g.width,
            clump_size: g.clump_size,
            stiffness: g.stiffness,
            root_color: g.root_color,
            tip_color: g.tip_color,
            color_variation: g.color_variation,
        }
    }
}

/// One terrain as grass sees it: where it lies, its height grid, and the
/// grass it grows.
#[derive(Debug, Clone)]
pub struct GrassTerrain {
    /// World-space center; its height is the terrain's base.
    pub center: [f32; 3],
    /// The cooked height grid.
    pub grid: TerrainGrid,
    /// Each layer's grass and optional density mask.
    pub layers: Vec<(Grass, Option<DensityMask>)>,
}

/// One drawable layer: a blade look grown over one terrain's ground.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassLayer {
    /// The ground the blades root in.
    pub ground: GrassGround,
    /// Where the layer's density mask sits, if it has one.
    pub mask: Option<GrassMask>,
    /// What each blade looks like.
    pub look: GrassBladeLook,
    /// The cell grid the density resolves to.
    pub grid: GrassGrid,
    /// Salts the layer's placement, so layers sharing ground differ.
    pub seed: u32,
}

/// The camera a frame's grass is placed and culled for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassCamera {
    /// World-space camera position.
    pub position: [f32; 3],
    /// This frame's unjittered view-projection, column-major.
    pub vp: [[f32; 4]; 4],
}

/// A frame's grass work: the parameters every grass stage reads and the
/// kernel dispatch that covers them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassFrame {
    /// The block the kernel and the draws read.
    pub params: GrassParams,
    /// Kernel groups along x, y and z. Never empty, since the dispatch also
    /// resets the next frame's draw arguments.
    pub dispatch: [u32; 3],
}

/// Every grass layer in the world, resolved once from its terrains.
#[derive(Debug, Clone, PartialEq)]
pub struct GrassField {
    /// The layers, in the order the world declares them.
    pub layers: Vec<GrassLayer>,
    /// The heights and mask texels the layers index into.
    pub buffers: GrassGroundBuffers,
    /// The wind the blades sway in.
    pub wind: WindField,
    /// Blades the visible-blade buffer is sized for.
    pub capacity: u32,
}

impl GrassField {
    /// The grass `terrains` grow, swaying in `wind`. `None` when no visible
    /// layer places a blade.
    pub fn resolve(terrains: &[GrassTerrain], wind: Option<&Wind>) -> Option<Self> {
        let mut buffers = GrassGroundBuffers::default();
        let mut layers = Vec::new();
        let mut capacity: u64 = 0;
        for terrain in terrains {
            let mut heights_offset = None;
            for (grass, mask) in &terrain.layers {
                if !grass.visible {
                    continue;
                }
                let grass = crate::components::validate::grass(grass.clone());
                let Some(grid) = GrassGrid::for_density(grass.density) else {
                    continue;
                };
                let offset =
                    *heights_offset.get_or_insert_with(|| buffers.push_heights(&terrain.grid));
                let ground = GrassGround::new(terrain.center, &terrain.grid, offset);
                capacity += u64::from(tiles::blade_capacity(
                    &ground.rect,
                    &grid,
                    GRASS_DRAW_DISTANCE,
                ));
                layers.push(GrassLayer {
                    ground,
                    mask: mask.as_ref().map(|m| buffers.push_mask(m)),
                    look: GrassBladeLook::of(&grass),
                    grid,
                    seed: lcg_hash(layers.len() as u32 ^ 0x5bd1_e995),
                });
            }
        }
        if layers.is_empty() {
            return None;
        }
        Some(Self {
            layers,
            buffers,
            wind: WindField::new(wind),
            capacity: capacity.min(u64::from(MAX_GRASS_BLADES)) as u32,
        })
    }

    /// The tiles of each layer a camera at `position` can see blades in, for
    /// the layers in reach: at most [`MAX_GRASS_LAYERS`], in declaration
    /// order.
    pub fn layers_in_reach(&self, position: [f32; 3]) -> Vec<(&GrassLayer, TileRange)> {
        self.layers
            .iter()
            .map(|layer| {
                let tiles = tiles::tiles_in_reach(
                    &layer.ground.rect,
                    [position[0], position[2]],
                    GRASS_DRAW_DISTANCE,
                );
                (layer, tiles)
            })
            .filter(|(_, tiles)| !tiles.is_empty())
            .take(MAX_GRASS_LAYERS)
            .collect()
    }

    /// This frame's grass work for `camera`, filling draw-argument slot
    /// `args_slot`.
    pub fn frame(&self, camera: &GrassCamera, args_slot: u32) -> GrassFrame {
        let reach = self.layers_in_reach(camera.position);
        let frustum = Frustum::from_camera(camera.vp, Some(GRASS_DRAW_DISTANCE));
        let [wind, wind_gust] = self.wind.gpu_rows();
        let mut params = GrassParams {
            cam_pos: camera.position,
            draw_distance: GRASS_DRAW_DISTANCE,
            frustum: frustum
                .planes
                .map(|p| [p.normal[0], p.normal[1], p.normal[2], p.d]),
            wind,
            wind_gust,
            layer_count: reach.len() as u32,
            capacity: self.capacity,
            args_slot: args_slot % GRASS_ARGS_SLOTS as u32,
            tile_size: GRASS_TILE_SIZE,
            layers: [GrassLayerGpu::default(); MAX_GRASS_LAYERS],
        };
        let mut dispatch = [1u32; 3];
        for (slot, (layer, tiles)) in reach.iter().enumerate() {
            params.layers[slot] = layer_gpu(layer, tiles);
            dispatch[0] = dispatch[0].max(layer.grid.groups_per_tile());
            dispatch[1] = dispatch[1].max(tiles.len());
        }
        dispatch[2] = (reach.len() as u32).max(1);
        GrassFrame { params, dispatch }
    }
}

fn layer_gpu(layer: &GrassLayer, tiles: &TileRange) -> GrassLayerGpu {
    let rgb = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
    let ground = &layer.ground;
    let look = &layer.look;
    let mask = layer.mask.as_ref();
    GrassLayerGpu {
        rect: [
            ground.rect.min[0],
            ground.rect.min[1],
            ground.rect.max[0],
            ground.rect.max[1],
        ],
        tile_origin: tiles.origin,
        tile_count: tiles.count,
        base_y: ground.base_y,
        min_y: ground.y_range[0],
        max_y: ground.y_range[1],
        grid_resolution: ground.resolution,
        heights_offset: ground.heights_offset,
        mask_offset: mask.map_or(0, |m| m.offset),
        mask_size: GrassMask::packed_size(mask),
        seed: layer.seed,
        height: look.height,
        height_variance: look.height_variance,
        width: look.width,
        clump_size: look.clump_size,
        stiffness: look.stiffness,
        color_variation: look.color_variation,
        cell_size: layer.grid.cell_size(),
        cells_per_side: layer.grid.cells_per_side,
        root_color: rgb(look.root_color),
        tip_color: rgb(look.tip_color),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn identity() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn camera(position: [f32; 3]) -> GrassCamera {
        GrassCamera {
            position,
            vp: identity(),
        }
    }

    fn flat(extent: [f32; 2]) -> TerrainGrid {
        TerrainGrid::new(4, extent, vec![0.0; 25]).unwrap()
    }

    fn terrain(center: [f32; 3], extent: [f32; 2], layers: Vec<Grass>) -> GrassTerrain {
        GrassTerrain {
            center,
            grid: flat(extent),
            layers: layers.into_iter().map(|g| (g, None)).collect(),
        }
    }

    fn grass(density: f32) -> Grass {
        Grass {
            density,
            ..Grass::default()
        }
    }

    #[test]
    fn a_world_without_grass_on_terrain_grows_nothing() {
        assert_eq!(GrassField::resolve(&[], None), None);
        let bare = terrain([0.0; 3], [10.0, 10.0], vec![]);
        assert_eq!(GrassField::resolve(&[bare], None), None);
        let hidden = Grass {
            visible: false,
            ..Grass::default()
        };
        let sparse = terrain([0.0; 3], [10.0, 10.0], vec![hidden, grass(0.0)]);
        assert_eq!(GrassField::resolve(&[sparse], None), None);
    }

    // Every visible layer of every terrain is drawn; layers on one terrain
    // share its heights, and each terrain's heights are packed once.
    #[test]
    fn every_layer_of_every_terrain_resolves() {
        let a = terrain([0.0; 3], [10.0, 10.0], vec![grass(100.0), grass(20.0)]);
        let b = terrain([100.0, 3.0, 0.0], [5.0, 5.0], vec![grass(50.0)]);
        let f = GrassField::resolve(&[a, b], None).unwrap();
        assert_eq!(f.layers.len(), 3);
        assert_eq!(f.buffers.heights.len(), 50);
        assert_eq!(f.layers[0].ground.heights_offset, 0);
        assert_eq!(f.layers[1].ground.heights_offset, 0);
        assert_eq!(f.layers[2].ground.heights_offset, 25);
        assert_eq!(f.layers[2].ground.base_y, 3.0);
        assert_ne!(f.layers[0].seed, f.layers[1].seed);
        let expected: u32 = f
            .layers
            .iter()
            .map(|l| tiles::blade_capacity(&l.ground.rect, &l.grid, GRASS_DRAW_DISTANCE))
            .sum();
        assert_eq!(f.capacity, expected);
    }

    #[test]
    fn the_field_carries_the_world_wind() {
        let t = terrain([0.0; 3], [10.0, 10.0], vec![grass(10.0)]);
        let wind = Wind {
            strength: 6.0,
            ..Wind::default()
        };
        let f = GrassField::resolve(core::slice::from_ref(&t), Some(&wind)).unwrap();
        assert_eq!(f.wind.strength, 6.0);
        assert_eq!(
            GrassField::resolve(&[t], None).unwrap().wind,
            WindField::STILL
        );
    }

    // The frame covers only layers whose terrain is within reach, and each
    // layer only its own tiles: a tile off every terrain is never dispatched.
    #[test]
    fn a_frame_covers_the_layers_in_reach_and_their_tiles_only() {
        let near = terrain([10.0, 2.0, -4.0], [20.0, 8.0], vec![grass(100.0)]);
        let far = terrain([500.0, 0.0, 0.0], [10.0, 10.0], vec![grass(400.0)]);
        let f = GrassField::resolve(&[far, near], None).unwrap();
        let frame = f.frame(&camera([10.0, 3.0, -4.0]), 1);
        let p = &frame.params;
        assert_eq!(p.layer_count, 1);
        let l = &p.layers[0];
        assert_eq!(l.rect, [-10.0, -12.0, 30.0, 4.0]);
        assert_eq!(l.base_y, 2.0);
        // x in [-10, 30) is tiles -3..=7; z in [-12, 4) is tiles -3..=0.
        assert_eq!(l.tile_origin, [-3, -3]);
        assert_eq!(l.tile_count, [11, 4]);
        assert_eq!(l.cells_per_side, 40);
        assert_eq!(l.cell_size, 0.1);
        assert_eq!(frame.dispatch, [25, 44, 1]);
        assert_eq!(p.cam_pos, [10.0, 3.0, -4.0]);
        assert_eq!(p.draw_distance, GRASS_DRAW_DISTANCE);
        assert_eq!(p.capacity, f.capacity);
        assert_eq!(p.args_slot, 1);
        assert_eq!(p.tile_size, GRASS_TILE_SIZE);
        assert_eq!(p.layers[1], GrassLayerGpu::default());
    }

    #[test]
    fn the_dispatch_spans_the_widest_layer() {
        let a = terrain([0.0; 3], [4.0, 4.0], vec![grass(100.0)]);
        let b = terrain([0.0; 3], [12.0, 2.0], vec![grass(400.0)]);
        let f = GrassField::resolve(&[a, b], None).unwrap();
        let frame = f.frame(&camera([0.0; 3]), 0);
        assert_eq!(frame.params.layer_count, 2);
        // Layer a: 2x2 tiles, 25 groups each; layer b: 6x2 tiles, 100 groups.
        assert_eq!(frame.dispatch, [100, 12, 2]);
    }

    #[test]
    fn a_frame_with_nothing_in_reach_still_dispatches_one_group() {
        let t = terrain([0.0; 3], [10.0, 10.0], vec![grass(100.0)]);
        let f = GrassField::resolve(&[t], None).unwrap();
        let frame = f.frame(&camera([1000.0, 0.0, 0.0]), 0);
        assert_eq!(frame.params.layer_count, 0);
        assert_eq!(frame.dispatch, [1, 1, 1]);
    }

    #[test]
    fn at_most_the_layer_limit_grows_in_one_frame() {
        let many = terrain(
            [0.0; 3],
            [10.0, 10.0],
            vec![grass(10.0); MAX_GRASS_LAYERS + 3],
        );
        let f = GrassField::resolve(&[many], None).unwrap();
        assert_eq!(f.layers.len(), MAX_GRASS_LAYERS + 3);
        let frame = f.frame(&camera([0.0; 3]), 0);
        assert_eq!(frame.params.layer_count as usize, MAX_GRASS_LAYERS);
        assert_eq!(frame.dispatch[2] as usize, MAX_GRASS_LAYERS);
    }

    #[test]
    fn a_masked_layer_points_at_its_texels() {
        let mask = DensityMask::new(2, 3, vec![0, 255, 9, 9, 9, 9]).unwrap();
        let t = GrassTerrain {
            center: [0.0; 3],
            grid: flat([10.0, 10.0]),
            layers: vec![(grass(10.0), None), (grass(10.0), Some(mask))],
        };
        let f = GrassField::resolve(&[t], None).unwrap();
        let frame = f.frame(&camera([0.0; 3]), 0);
        assert_eq!(frame.params.layers[0].mask_size, 0);
        assert_eq!(frame.params.layers[1].mask_size, 2 | (3 << 16));
        assert_eq!(frame.params.layers[1].mask_offset, 0);
        assert_eq!(f.buffers.mask_texel(1), 255);
    }

    #[test]
    fn the_args_slot_wraps_onto_the_two_slots() {
        let t = terrain([0.0; 3], [10.0, 10.0], vec![grass(10.0)]);
        let f = GrassField::resolve(&[t], None).unwrap();
        assert_eq!(f.frame(&camera([0.0; 3]), 2).params.args_slot, 0);
        assert_eq!(f.frame(&camera([0.0; 3]), 7).params.args_slot, 1);
    }

    #[test]
    fn the_frustum_planes_are_the_cameras() {
        let t = terrain([0.0; 3], [10.0, 10.0], vec![grass(10.0)]);
        let p = GrassField::resolve(&[t], None)
            .unwrap()
            .frame(&camera([0.0; 3]), 0)
            .params;
        let expected = Frustum::from_camera(identity(), Some(GRASS_DRAW_DISTANCE));
        for (row, plane) in p.frustum.iter().zip(expected.planes.iter()) {
            assert_eq!(&row[..3], &plane.normal[..]);
            assert_eq!(row[3], plane.d);
        }
    }
}
