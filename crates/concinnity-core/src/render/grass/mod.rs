//! Backend-agnostic resolution of a world's grass into what the grass passes
//! dispatch and draw: every terrain layer's ground, blade look and cell grid,
//! the buffers of heights and mask texels the kernel reads, and the wind,
//! turned each frame into the [`GrassParams`] every grass stage reads and the
//! [`GrassBendParams`] the bend pass reads. Pure CPU; the backend owns the
//! buffers and pipelines.
//!
//! One dispatch covers every layer within reach of the camera: its `z` picks
//! the layer, its `y` the tile within that layer's block, its `x` the run of
//! cells within the tile. A tile off every terrain is never dispatched. The
//! nearest shadow cascade gets a second, smaller dispatch of its own (see
//! [`shadow`]), and the bend field one ahead of both (see [`bend`]).

pub mod bend;
mod ground;
pub mod lod;
pub mod shadow;
pub mod tiles;
pub mod tint;

pub use bend::{BendHistory, GrassBender};
pub use ground::{GrassGround, GrassGroundBuffers, GrassMask, GroundSample};
pub use lod::GrassCapacity;
pub use shadow::GrassShadowView;

use alloc::vec::Vec;

use crate::components::{Grass, Wind};
use crate::gfx::frustum::Frustum;
use crate::math::noise::lcg_hash;
use crate::render::uniforms::grass::{
    GRASS_ARGS_SLOTS, GrassBendParams, GrassLayerGpu, GrassParams, MAX_GRASS_LAYERS,
    MAX_GRASS_STAMPS,
};
use crate::render::wind::WindField;
use crate::terrain::{DensityMask, TerrainGrid};
use bend::{BendStep, GRASS_BEND_CELL_SIZE, GRASS_BEND_RESOLUTION};
use lod::{
    GRASS_FULL_DENSITY_DISTANCE, GRASS_LOD_COUNT, GRASS_LOD_DISTANCES, GRASS_MIN_KEEP,
    GRASS_MORPH_FRACTION, GRASS_SHRINK_BAND,
};
use shadow::GRASS_SHADOW_DISTANCE;
use tiles::{GRASS_DRAW_DISTANCE, GRASS_GROUP_SIZE, GRASS_TILE_SIZE, GrassGrid, TileRange};

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
    /// The depth pyramid tiles are occlusion-tested against, when it holds
    /// the previous frame's depth.
    pub hiz: Option<GrassHiz>,
}

/// The previous frame's depth pyramid, as the kernel's tile test reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassHiz {
    /// The view-projection the pyramid's depth was rendered through,
    /// column-major.
    pub prev_vp: [[f32; 4]; 4],
    /// Base size in texels.
    pub size: [f32; 2],
    /// Mip levels.
    pub mip_count: u32,
}

/// What a frame's grass is placed, cast and bent for.
#[derive(Debug, Clone, Copy)]
pub struct GrassFrameInputs<'a> {
    /// The camera the blades are placed and culled for.
    pub camera: GrassCamera,
    /// Seconds since the world started, the clock the field relaxes by and
    /// the shadow draw sways the blades at.
    pub elapsed: f32,
    /// The nearest shadow cascade, when the grass casts into it this frame.
    pub shadow: Option<GrassShadowView>,
    /// Everything that may be trampling the grass.
    pub benders: &'a [GrassBender],
}

/// What a field's frames carry from one to the next.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GrassHistory {
    // Frames placed so far, which picks the draw-argument slot each fills.
    runs: u32,
    bend: BendHistory,
}

/// One run of the blade kernel: the block it and the draws of its blades read,
/// and the groups that cover it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassPass {
    /// The block the kernel and the draws read.
    pub params: GrassParams,
    /// Kernel groups along x, y and z. Never empty, since the dispatch also
    /// resets the next frame's draw arguments.
    pub dispatch: [u32; 3],
}

/// The bend pass's run: its block and the groups that cover the field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassBendPass {
    /// The block the bend pass reads.
    pub params: GrassBendParams,
    /// Groups along x, y and z.
    pub dispatch: [u32; 3],
}

/// A frame's grass work: the bend field's step, the blades placed for the
/// view, and those placed for the nearest shadow cascade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassFrame {
    /// The blades the view draws.
    pub view: GrassPass,
    /// The blades the nearest cascade draws, when the grass casts this frame.
    pub shadow: Option<GrassPass>,
    /// The bend field's step.
    pub bend: GrassBendPass,
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
    /// Blades each detail level's region of the visible-blade buffer holds.
    pub capacity: GrassCapacity,
    /// Blades the shadow cascade's buffer holds.
    pub shadow_capacity: u32,
    /// The tallest blade of any layer, in meters: how high above the ground a
    /// bender still presses the grass.
    pub blade_height: f32,
}

impl GrassField {
    /// The grass `terrains` grow, swaying in `wind`. `None` when no visible
    /// layer places a blade.
    pub fn resolve(terrains: &[GrassTerrain], wind: Option<&Wind>) -> Option<Self> {
        let mut buffers = GrassGroundBuffers::default();
        let mut layers = Vec::new();
        let mut capacities = Vec::new();
        let mut shadow_capacity = 0u32;
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
                capacities.push(lod::lod_capacities(
                    &ground.rect,
                    &grid,
                    GRASS_DRAW_DISTANCE,
                ));
                shadow_capacity = shadow_capacity.saturating_add(lod::shadow_capacity(
                    &ground.rect,
                    &grid,
                    GRASS_SHADOW_DISTANCE,
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
        let blade_height = layers
            .iter()
            .map(|l| l.look.height * (1.0 + l.look.height_variance))
            .fold(0.0f32, f32::max);
        Some(Self {
            layers,
            buffers,
            wind: WindField::new(wind),
            capacity: GrassCapacity::sum(capacities),
            shadow_capacity: shadow_capacity.clamp(1, tiles::MAX_GRASS_BLADES),
            blade_height,
        })
    }

    /// The tiles of each layer within `distance` of a camera at `position`,
    /// for the layers in reach: at most [`MAX_GRASS_LAYERS`], in declaration
    /// order.
    pub fn layers_in_reach(
        &self,
        position: [f32; 3],
        distance: f32,
    ) -> Vec<(&GrassLayer, TileRange)> {
        self.layers
            .iter()
            .map(|layer| {
                let tiles =
                    tiles::tiles_in_reach(&layer.ground.rect, [position[0], position[2]], distance);
                (layer, tiles)
            })
            .filter(|(_, tiles)| !tiles.is_empty())
            .take(MAX_GRASS_LAYERS)
            .collect()
    }

    /// This frame's grass work for `inputs`, advancing `history`.
    pub fn frame(&self, inputs: &GrassFrameInputs<'_>, history: &mut GrassHistory) -> GrassFrame {
        let camera = &inputs.camera;
        let args_slot = history.runs % GRASS_ARGS_SLOTS as u32;
        history.runs = history.runs.wrapping_add(1);
        let step = history.bend.advance(camera.position, inputs.elapsed);
        let params = self.view_params(camera, &step, args_slot);
        let shadow = inputs
            .shadow
            .map(|light| self.shadow_pass(camera.position, &light, inputs.elapsed, params));
        GrassFrame {
            view: self.place(camera.position, GRASS_DRAW_DISTANCE, params),
            shadow,
            bend: self.bend_pass(&step, camera.position, inputs.benders),
        }
    }

    // The view's block, all but its layers.
    fn view_params(&self, camera: &GrassCamera, step: &BendStep, args_slot: u32) -> GrassParams {
        let frustum = Frustum::from_camera(camera.vp, Some(GRASS_DRAW_DISTANCE));
        let [wind, wind_gust] = self.wind.gpu_rows();
        let hiz = camera.hiz;
        let lod_base = self.capacity.bases();
        let lod_capacity = self.capacity.lods;
        let prev = step.prev.unwrap_or(step.window);
        GrassParams {
            cam_pos: camera.position,
            draw_distance: GRASS_DRAW_DISTANCE,
            frustum: planes(&frustum),
            wind,
            wind_gust,
            prev_vp: hiz.map_or(camera.vp, |h| h.prev_vp),
            hiz_size: hiz.map_or([1.0, 1.0], |h| h.size),
            hiz_mip_count: hiz.map_or(1, |h| h.mip_count),
            hiz_enabled: u32::from(hiz.is_some()),
            lod_base: [lod_base[0], lod_base[1], lod_base[2], 0],
            lod_capacity: [lod_capacity[0], lod_capacity[1], lod_capacity[2], 0],
            lod_distances: [
                GRASS_LOD_DISTANCES[0],
                GRASS_LOD_DISTANCES[1],
                GRASS_MORPH_FRACTION,
                GRASS_FULL_DENSITY_DISTANCE,
            ],
            thinning: [
                GRASS_MIN_KEEP,
                GRASS_SHRINK_BAND,
                lod::fade_start(GRASS_DRAW_DISTANCE),
                lod::max_width_scale(),
            ],
            layer_count: 0,
            args_slot,
            tile_size: GRASS_TILE_SIZE,
            cull_distance: GRASS_DRAW_DISTANCE,
            bend_window: [
                step.window.origin[0],
                step.window.origin[1],
                prev.origin[0],
                prev.origin[1],
            ],
            bend_cell_size: GRASS_BEND_CELL_SIZE,
            bend_resolution: GRASS_BEND_RESOLUTION,
            bend_half: step.write_half,
            bend_prev_valid: u32::from(step.prev.is_some()),
            shadow_vp: camera.vp,
            shadow_light: [0.0; 4],
            layers: [GrassLayerGpu::default(); MAX_GRASS_LAYERS],
        }
    }

    // The cascade's run: the view's block culled against the light's frustum
    // and the cast distance instead, with no occlusion test, every blade
    // appended to the coarsest level's region of its own buffer.
    fn shadow_pass(
        &self,
        position: [f32; 3],
        light: &GrassShadowView,
        elapsed: f32,
        view: GrassParams,
    ) -> GrassPass {
        let coarsest = GRASS_LOD_COUNT - 1;
        let mut lod_capacity = [0; 4];
        lod_capacity[coarsest] = self.shadow_capacity;
        let params = GrassParams {
            frustum: planes(&light.frustum()),
            hiz_enabled: 0,
            lod_base: [0; 4],
            lod_capacity,
            lod_distances: [0.0, 0.0, view.lod_distances[2], view.lod_distances[3]],
            cull_distance: GRASS_SHADOW_DISTANCE,
            shadow_vp: light.vp,
            shadow_light: [
                light.to_light[0],
                light.to_light[1],
                light.to_light[2],
                elapsed,
            ],
            ..view
        };
        self.place(position, GRASS_SHADOW_DISTANCE, params)
    }

    // `params` with the layers within `distance` of `position` and the
    // dispatch that covers their tiles.
    fn place(&self, position: [f32; 3], distance: f32, mut params: GrassParams) -> GrassPass {
        let reach = self.layers_in_reach(position, distance);
        let mut dispatch = [1u32; 3];
        for (slot, (layer, tiles)) in reach.iter().enumerate() {
            params.layers[slot] = layer_gpu(layer, tiles);
            dispatch[0] = dispatch[0].max(layer.grid.groups_per_tile());
            dispatch[1] = dispatch[1].max(tiles.len());
        }
        params.layer_count = reach.len() as u32;
        dispatch[2] = (reach.len() as u32).max(1);
        GrassPass { params, dispatch }
    }

    // The bend pass's run for `step`: the footprints of `benders` near the
    // camera at `position`, and one thread per cell of the window.
    fn bend_pass(
        &self,
        step: &BendStep,
        position: [f32; 3],
        benders: &[GrassBender],
    ) -> GrassBendPass {
        let found = bend::gather_stamps(benders, &step.window, position, self.blade_height, |xz| {
            self.ground_under(xz)
        });
        let mut stamps = [[0.0; 4]; MAX_GRASS_STAMPS];
        stamps[..found.len()].copy_from_slice(&found);
        let prev = step.prev.unwrap_or(step.window);
        let cells = GRASS_BEND_RESOLUTION * GRASS_BEND_RESOLUTION;
        GrassBendPass {
            params: GrassBendParams {
                window: [
                    step.window.origin[0],
                    step.window.origin[1],
                    prev.origin[0],
                    prev.origin[1],
                ],
                resolution: GRASS_BEND_RESOLUTION,
                cell_size: GRASS_BEND_CELL_SIZE,
                decay: step.decay,
                stamp_count: found.len() as u32,
                write_half: step.write_half,
                prev_valid: u32::from(step.prev.is_some()),
                _pad: [0; 2],
                stamps,
            },
            dispatch: [cells.div_ceil(GRASS_GROUP_SIZE), 1, 1],
        }
    }

    // The world height of the first layer's ground under `xz`, `None` off
    // every terrain.
    fn ground_under(&self, xz: [f32; 2]) -> Option<f32> {
        self.layers
            .iter()
            .find(|l| {
                let r = &l.ground.rect;
                (0..2).all(|i| xz[i] >= r.min[i] && xz[i] <= r.max[i])
            })
            .map(|l| l.ground.surface_at(&self.buffers, xz).y)
    }
}

// A frustum's planes as the block's `(normal, d)` rows.
fn planes(frustum: &Frustum) -> [[f32; 4]; 6] {
    frustum
        .planes
        .map(|p| [p.normal[0], p.normal[1], p.normal[2], p.d])
}

/// How far a blade of `look` can reach from its root in any direction: its
/// tallest height, since a bent blade keeps its length, plus half its widest
/// width, distance widening and edge-on widening included. Mirrors the
/// kernel's height and width draws.
pub fn blade_reach(look: &GrassBladeLook) -> f32 {
    let tallest = look.height * (1.0 + look.height_variance);
    let widest = look.width * 1.25 * lod::max_width_scale() * 1.3;
    tallest + 0.5 * widest
}

/// The box every blade rooted in tile `tile` of `layer` stays inside, swaying
/// in any wind: the tile's columns over the terrain's heights, grown by the
/// blades' reach. Mirrors the kernel's tile test.
pub fn tile_bounds(layer: &GrassLayerGpu, tile_size: f32, tile: [i32; 2]) -> ([f32; 3], [f32; 3]) {
    let r = layer.reach;
    let lo = [
        tile[0] as f32 * tile_size - r,
        tile[1] as f32 * tile_size - r,
    ];
    let hi = [
        (tile[0] + 1) as f32 * tile_size + r,
        (tile[1] + 1) as f32 * tile_size + r,
    ];
    (
        [lo[0], layer.min_y - r, lo[1]],
        [hi[0], layer.max_y + r, hi[1]],
    )
}

fn layer_gpu(layer: &GrassLayer, tiles: &TileRange) -> GrassLayerGpu {
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
        root_color: look.root_color,
        reach: blade_reach(look),
        tip_color: [look.tip_color[0], look.tip_color[1], look.tip_color[2], 0.0],
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
            hiz: None,
        }
    }

    fn inputs(position: [f32; 3]) -> GrassFrameInputs<'static> {
        GrassFrameInputs {
            camera: camera(position),
            elapsed: 0.0,
            shadow: None,
            benders: &[],
        }
    }

    // The view's run of a first frame with the camera at `position`.
    fn view(f: &GrassField, position: [f32; 3]) -> GrassPass {
        f.frame(&inputs(position), &mut GrassHistory::default())
            .view
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
        let expected = GrassCapacity::sum(
            f.layers
                .iter()
                .map(|l| lod::lod_capacities(&l.ground.rect, &l.grid, GRASS_DRAW_DISTANCE)),
        );
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
        let frame = view(&f, [10.0, 3.0, -4.0]);
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
        assert_eq!(p.lod_capacity[..3], f.capacity.lods);
        assert_eq!(p.lod_base[..3], f.capacity.bases());
        assert_eq!(p.args_slot, 0);
        assert_eq!(p.hiz_enabled, 0);
        assert_eq!(p.cull_distance, GRASS_DRAW_DISTANCE);
        assert_eq!(p.tile_size, GRASS_TILE_SIZE);
        assert_eq!(p.layers[1], GrassLayerGpu::default());
    }

    #[test]
    fn the_dispatch_spans_the_widest_layer() {
        let a = terrain([0.0; 3], [4.0, 4.0], vec![grass(100.0)]);
        let b = terrain([0.0; 3], [12.0, 2.0], vec![grass(400.0)]);
        let f = GrassField::resolve(&[a, b], None).unwrap();
        let frame = view(&f, [0.0; 3]);
        assert_eq!(frame.params.layer_count, 2);
        // Layer a: 2x2 tiles, 25 groups each; layer b: 6x2 tiles, 100 groups.
        assert_eq!(frame.dispatch, [100, 12, 2]);
    }

    #[test]
    fn a_frame_with_nothing_in_reach_still_dispatches_one_group() {
        let t = terrain([0.0; 3], [10.0, 10.0], vec![grass(100.0)]);
        let f = GrassField::resolve(&[t], None).unwrap();
        let frame = view(&f, [1000.0, 0.0, 0.0]);
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
        let frame = view(&f, [0.0; 3]);
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
        let frame = view(&f, [0.0; 3]);
        assert_eq!(frame.params.layers[0].mask_size, 0);
        assert_eq!(frame.params.layers[1].mask_size, 2 | (3 << 16));
        assert_eq!(frame.params.layers[1].mask_offset, 0);
        assert_eq!(f.buffers.mask_texel(1), 255);
    }

    #[test]
    fn the_args_slot_wraps_onto_the_two_slots() {
        let t = terrain([0.0; 3], [10.0, 10.0], vec![grass(10.0)]);
        let f = GrassField::resolve(&[t], None).unwrap();
        let mut history = GrassHistory::default();
        let slots: Vec<u32> = (0..4)
            .map(|_| {
                f.frame(&inputs([0.0; 3]), &mut history)
                    .view
                    .params
                    .args_slot
            })
            .collect();
        assert_eq!(slots, [0, 1, 0, 1]);
    }

    #[test]
    fn the_frustum_planes_are_the_cameras() {
        let t = terrain([0.0; 3], [10.0, 10.0], vec![grass(10.0)]);
        let p = view(&GrassField::resolve(&[t], None).unwrap(), [0.0; 3]).params;
        let expected = Frustum::from_camera(identity(), Some(GRASS_DRAW_DISTANCE));
        for (row, plane) in p.frustum.iter().zip(expected.planes.iter()) {
            assert_eq!(&row[..3], &plane.normal[..]);
            assert_eq!(row[3], plane.d);
        }
    }

    fn light() -> GrassShadowView {
        let mut vp = identity();
        vp[3][0] = 0.25;
        GrassShadowView {
            vp,
            to_light: [0.0, 1.0, 0.0],
        }
    }

    // The cascade's run is the view's block culled against the light, out to
    // the cast distance only, with every blade sent to the coarsest level's
    // region of its own buffer, and the clock the shadow draw sways it at.
    #[test]
    fn the_cascade_places_its_own_blades_near_the_camera() {
        let t = terrain([0.0; 3], [200.0, 200.0], vec![grass(100.0)]);
        let f = GrassField::resolve(&[t], None).unwrap();
        let frame = f.frame(
            &GrassFrameInputs {
                elapsed: 3.5,
                shadow: Some(light()),
                ..inputs([0.0; 3])
            },
            &mut GrassHistory::default(),
        );
        let view = frame.view.params;
        let cast = frame.shadow.expect("the grass casts").params;
        assert_eq!(cast.cull_distance, GRASS_SHADOW_DISTANCE);
        assert_eq!(cast.hiz_enabled, 0);
        assert_eq!(cast.lod_base, [0; 4]);
        assert_eq!(cast.lod_capacity, [0, 0, f.shadow_capacity, 0]);
        assert_eq!(&cast.lod_distances[..2], &[0.0, 0.0]);
        assert_eq!(cast.lod_distances[3], view.lod_distances[3]);
        assert_eq!(cast.thinning, view.thinning);
        assert_eq!(cast.cam_pos, view.cam_pos);
        assert_eq!(cast.draw_distance, view.draw_distance);
        assert_eq!(cast.args_slot, view.args_slot);
        assert_eq!(cast.bend_window, view.bend_window);
        assert_eq!(cast.shadow_vp, light().vp);
        assert_eq!(cast.shadow_light, [0.0, 1.0, 0.0, 3.5]);
        let expected = light().frustum();
        for (row, plane) in cast.frustum.iter().zip(expected.planes.iter()) {
            assert_eq!(&row[..3], &plane.normal[..]);
        }
        // 24 m of 4 m tiles either side of the camera, against 90 m.
        assert_eq!(cast.layers[0].tile_count, [12, 12]);
        assert_eq!(view.layers[0].tile_count, [46, 46]);
        assert!(frame.shadow.unwrap().dispatch[1] < frame.view.dispatch[1]);
        assert!(f.shadow_capacity < f.capacity.total());
        // No cascade, no cast.
        let none = f.frame(&inputs([0.0; 3]), &mut GrassHistory::default());
        assert_eq!(none.shadow, None);
    }

    // The bend field follows the camera, alternating halves, and stamps the
    // benders standing on the terrain, as hard as they are low.
    #[test]
    fn the_bend_pass_follows_the_camera_and_stamps_the_benders() {
        let t = terrain([0.0, 2.0, 0.0], [50.0, 50.0], vec![grass(100.0)]);
        let f = GrassField::resolve(&[t], None).unwrap();
        let look = Grass::default();
        assert_eq!(f.blade_height, look.height * (1.0 + look.height_variance));
        let benders = [
            GrassBender {
                base: [1.0, 2.0, 1.0],
                radius: 0.3,
                height: 1.7,
            },
            GrassBender {
                base: [3.0, 2.0 + 0.5 * f.blade_height, 1.0],
                radius: 0.5,
                height: 0.5,
            },
            // Off the terrain.
            GrassBender {
                base: [80.0, 2.0, 0.0],
                radius: 0.5,
                height: 0.5,
            },
        ];
        let mut history = GrassHistory::default();
        let first = f.frame(
            &GrassFrameInputs {
                benders: &benders,
                ..inputs([0.0, 3.0, 0.0])
            },
            &mut history,
        );
        let b = first.bend.params;
        assert_eq!(b.stamp_count, 2);
        assert_eq!(b.stamps[0], [1.0, 1.0, 0.3, 1.0]);
        assert_eq!(b.stamps[1][..3], [3.0, 1.0, 0.5]);
        assert!((b.stamps[1][3] - 0.5).abs() < 1e-5);
        assert_eq!(b.prev_valid, 0);
        assert_eq!(b.decay, 1.0);
        assert_eq!(b.write_half, 0);
        let cells = GRASS_BEND_RESOLUTION * GRASS_BEND_RESOLUTION;
        assert_eq!(first.bend.dispatch, [cells / GRASS_GROUP_SIZE, 1, 1]);
        assert_eq!(first.view.params.bend_half, 0);
        assert_eq!(first.view.params.bend_prev_valid, 0);

        let second = f.frame(
            &GrassFrameInputs {
                elapsed: 0.1,
                ..inputs([4.0, 3.0, 0.0])
            },
            &mut history,
        );
        let b2 = second.bend.params;
        assert_eq!(b2.stamp_count, 0);
        assert_eq!(b2.write_half, 1);
        assert_eq!(b2.prev_valid, 1);
        assert!((b2.decay - bend::decay(0.1)).abs() < 1e-6);
        assert_eq!(&b2.window[2..], &b.window[..2]);
        assert_eq!(b2.window[0], b.window[0] + 16);
        assert_eq!(second.view.params.bend_window, b2.window);
        assert_eq!(second.view.params.bend_half, 1);
    }
}
