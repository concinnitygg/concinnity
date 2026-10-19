//! Backend-agnostic resolution of a world's grass into what the grass passes
//! dispatch and draw: every terrain layer's ground, blade look and cell grid,
//! the buffers of heights and mask texels the kernel reads, and the wind,
//! turned each frame into the [`GrassParams`] every grass stage reads and the
//! [`GrassBendParams`] the bend pass reads. Pure CPU; the backend owns the
//! buffers and pipelines.
//!
//! One dispatch covers every layer within reach of the camera: its `y` picks
//! one of the frame's tile records, a tile of some layer at some level of the
//! nested cell grid (see [`nest`]), larger the farther it lies, and its `x`
//! the run of cells within the tile. A tile off every terrain, out of the
//! frustum or past the reach is never dispatched. The nearest shadow cascade
//! gets a second, smaller dispatch of its own (see [`shadow`]), and the bend
//! field one ahead of both (see [`bend`]).
//!
//! On a planet the layers grow over the planet's drawn tiles instead (see
//! [`planet`]): the tile records nest within the drawn tiles in reach, and the
//! field's frames carry the wind's and the bend field's coordinates through
//! every move of the simulated frame.

pub mod bend;
pub mod canopy;
pub mod canopy_table;
pub mod clumping;
mod ground;
pub mod litter;
pub mod lod;
pub mod nest;
pub mod planet;
pub mod shadow;
pub mod tiles;
pub mod tint;
pub mod variation;

pub use bend::{BendHistory, GrassBender};
pub use canopy::{CanopyField, CanopyRange};
pub use ground::{GrassGround, GrassGroundBuffers, GrassMask, GroundSample};
pub use lod::{GrassCapacity, GrassLodView, GrassSizing};

/// Frames the hand-off's per-pixel choice between the blades and the canopy
/// cycles through before it repeats.
pub const GRASS_CROSSFADE_PHASES: u32 = 64;
pub use planet::PlanetGrass;
pub use planet::tiles::DrawnPlanetTile;
pub use shadow::GrassShadowView;

use alloc::vec::Vec;

use crate::components::{Grass, Wind};
use crate::gfx::frustum::Frustum;
use crate::math::noise::lcg_hash;
use crate::planet::Rebase;
use crate::render::uniforms::grass::{
    GRASS_ARGS_SLOTS, GRASS_GROUND_PLANET, GRASS_GROUND_TERRAIN, GRASS_IDENTITY_MAP,
    GrassBendParams, GrassLayerGpu, GrassParams, GrassTileGpu, MAX_GRASS_LAYERS, MAX_GRASS_STAMPS,
    MAX_GRASS_TILE_RECORDS, PlanetGrassTileGpu,
};
use crate::render::wind::WindField;
use crate::terrain::{DensityMask, TerrainGrid};
use crate::transform::Mat4;
use bend::{BendStep, GRASS_BEND_CELL_SIZE, GRASS_BEND_RESOLUTION};
use lod::{
    CapacityBounds, CapacityField, CapacityViews, GRASS_LOD_COUNT, GRASS_MORPH_FRACTION,
    GRASS_SHRINK_BAND,
};
use planet::tiles::{MAX_PLANET_GRASS_REACH, PlanetGrassTiles, nested_planet_tiles, planet_center};
use shadow::GRASS_SHADOW_DISTANCE;
use tiles::{GRASS_GROUP_SIZE, GRASS_TILE_SIZE, GrassGrid, NestView, NestedTile, NestedTiles};

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
    /// How strongly blades crowd toward their clump's center, in [0, 1].
    pub clumping: f32,
    /// Resistance to bending, in [0, 1].
    pub stiffness: f32,
    /// Linear RGB at the root.
    pub root_color: [f32; 3],
    /// Linear RGB at the tip.
    pub tip_color: [f32; 3],
    /// Linear RGB of the dead grass on the ground under the blades.
    pub litter_color: [f32; 3],
    /// Hue and brightness variation, in [0, 1].
    pub color_variation: f32,
    /// How far the field's color drifts over hundreds of meters, in [0, 1].
    pub macro_variation: f32,
}

impl GrassBladeLook {
    pub(crate) fn of(g: &Grass) -> Self {
        Self {
            height: g.height,
            height_variance: g.height_variance,
            width: g.width,
            clump_size: g.clump_size,
            clumping: g.clumping,
            stiffness: g.stiffness,
            root_color: g.root_color,
            tip_color: g.tip_color,
            litter_color: g
                .litter_color
                .unwrap_or_else(|| litter::litter_color(g.root_color)),
            color_variation: g.color_variation,
            macro_variation: g.macro_variation,
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
    /// The view's vertical focal length, in pixels (see
    /// [`lod::focal_length`]): how large the blades stand on screen, which
    /// their detail, thinning and reach follow.
    pub focal: f32,
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
    /// The planet tiles drawn this frame, when the grass grows on a planet.
    pub planet_tiles: &'a [DrawnPlanetTile],
}

/// Tile records one frame's runs can hold together: the view's and the
/// shadow's, each up to [`MAX_GRASS_TILE_RECORDS`]. A backend sizes its
/// per-frame tile buffer to it.
pub const FRAME_TILE_RECORDS: usize = 2 * MAX_GRASS_TILE_RECORDS;

/// Planet tile records one frame's runs can hold together, sized like
/// [`FRAME_TILE_RECORDS`].
pub const FRAME_PLANET_TILE_RECORDS: usize =
    2 * crate::render::uniforms::grass::MAX_PLANET_GRASS_TILES;

/// What a frame's grass is placed, cast and bent for, beyond the planet tiles
/// the scene draws.
#[derive(Debug, Clone, Copy)]
pub struct GrassView<'a> {
    /// The camera the blades are placed and culled for.
    pub camera: GrassCamera,
    /// Seconds since the world started.
    pub elapsed: f32,
    /// The nearest shadow cascade, when the grass casts into it this frame.
    pub shadow: Option<GrassShadowView>,
    /// Everything that may be trampling the grass.
    pub benders: &'a [GrassBender],
}

/// A field's history and the drawn planet tiles each frame gathers, kept so a
/// frame allocates nothing once both have grown.
#[derive(Debug, Clone, Default)]
pub struct GrassFrameState {
    /// What the field's frames carry from one to the next.
    pub history: GrassHistory,
    drawn: Vec<DrawnPlanetTile>,
}

impl GrassFrameState {
    /// `field`'s grass work for `view` over `scene`'s drawn planet tiles,
    /// advancing the history.
    pub fn frame<'h>(
        &'h mut self,
        field: &GrassField,
        scene: &crate::render::scene_state::SceneState,
        view: &GrassView<'_>,
    ) -> GrassFrame<'h> {
        self.drawn.clear();
        if field.planet.is_some() {
            scene.drawn_planet_tiles_into(&mut self.drawn);
        }
        let inputs = GrassFrameInputs {
            camera: view.camera,
            elapsed: view.elapsed,
            shadow: view.shadow,
            benders: view.benders,
            planet_tiles: &self.drawn,
        };
        field.frame(&inputs, &mut self.history)
    }
}

/// What a field's frames carry from one to the next, and the working space
/// they place their tiles in.
#[derive(Debug, Clone)]
pub struct GrassHistory {
    // Frames placed so far, which picks the draw-argument slot each fills.
    runs: u32,
    bend: BendHistory,
    // Where the wind's gusts are laid out, as rows over this frame's points.
    wind_map: [[f32; 4]; 2],
    // This frame's points in last frame's, since the last frame placed.
    prev_map: [[f32; 4]; 2],
    // Whether a frame has reported reaching a placement limit.
    limit_reported: bool,
    scratch: GrassScratch,
}

impl Default for GrassHistory {
    fn default() -> Self {
        Self {
            runs: 0,
            bend: BendHistory::default(),
            wind_map: GRASS_IDENTITY_MAP,
            prev_map: GRASS_IDENTITY_MAP,
            limit_reported: false,
            scratch: GrassScratch::default(),
        }
    }
}

// The space a frame's runs place their tiles in, kept from frame to frame so
// placing allocates nothing once it has grown to the largest frame.
#[derive(Debug, Clone, Default)]
struct GrassScratch {
    // The frame's tile records and the planet tiles they lie in, the view's
    // run first.
    tiles: Vec<GrassTileGpu>,
    planet_tiles: Vec<PlanetGrassTileGpu>,
    // One run's tiles with their distances and the order they were found in.
    found: Vec<(f32, u32, GrassTileGpu)>,
    visited: NestedTiles,
    reach: PlanetGrassTiles,
}

impl GrassHistory {
    /// The phase of the next frame's per-pixel hand-off between the blades
    /// and the canopy: the blades' draws and the canopy's ground both read it,
    /// so a pixel shows one or the other, and it steps every frame so the
    /// temporal antialiasing resolves the mix.
    pub fn crossfade_phase(&self) -> f32 {
        (self.runs % GRASS_CROSSFADE_PHASES) as f32
    }

    /// Carry the field into the frame the world moved to: the wind keeps its
    /// gusts where they were on the ground, and the next frame reads the bend
    /// field it inherits where the blades were.
    pub fn rebase(&mut self, rebase: &Rebase) {
        let back = rebase.inverse_matrix();
        self.wind_map = map_then(self.wind_map, &back);
        self.prev_map = map_then(self.prev_map, &back);
    }

    #[cfg(test)]
    pub(crate) fn wind_map(&self) -> [[f32; 4]; 2] {
        self.wind_map
    }
}

/// `rows` (each taking a point `(x, y, z, 1)` to one coordinate) applied after
/// the column-major affine `m`.
pub fn map_then(rows: [[f32; 4]; 2], m: &Mat4) -> [[f32; 4]; 2] {
    rows.map(|row| core::array::from_fn(|c| (0..4).map(|k| row[k] * m[c][k]).sum()))
}

/// Rows over a point `(x, y, z, 1)` applied at fixed height `y`, as rows over
/// `(x, z, 1)`.
pub fn map_at_height(rows: [[f32; 4]; 2], y: f32) -> [[f32; 4]; 2] {
    rows.map(|r| [r[0], r[2], r[1] * y + r[3], 0.0])
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

/// A limit on what one run places, past which the tiles farthest from the
/// camera are dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrassLimit {
    /// The tile records a run holds, [`MAX_GRASS_TILE_RECORDS`].
    TileRecords,
    /// The drawn planet tiles a run holds,
    /// [`MAX_PLANET_GRASS_TILES`](crate::render::uniforms::grass::MAX_PLANET_GRASS_TILES).
    PlanetTiles,
}

// Log the limit a frame's runs reached; the history passes it only once.
fn report_limit(limit: Option<GrassLimit>) {
    match limit {
        Some(GrassLimit::TileRecords) => tracing::warn!(
            "grass: more grass tiles in reach than a run's records hold; the farthest are dropped"
        ),
        Some(GrassLimit::PlanetTiles) => tracing::warn!(
            "grass: more planet tiles in reach than a grass run holds; the farthest are dropped"
        ),
        None => {}
    }
}

/// A frame's grass work: the bend field's step, the blades placed for the
/// view, and those placed for the nearest shadow cascade.
#[derive(Debug, Clone, PartialEq)]
pub struct GrassFrame<'a> {
    /// The blades the view draws.
    pub view: GrassPass,
    /// The blades the nearest cascade draws, when the grass casts this frame.
    pub shadow: Option<GrassPass>,
    /// The bend field's step.
    pub bend: GrassBendPass,
    /// The tiles both runs visit, the view's first.
    pub tiles: &'a [GrassTileGpu],
    /// The planet tiles both runs read, the view's first; empty on terrain.
    pub planet_tiles: &'a [PlanetGrassTileGpu],
    /// The limit this frame's runs reached, on the first frame of the history
    /// that reached one (and logged then); `None` on every other frame.
    pub limit: Option<GrassLimit>,
}

impl GrassFrame<'_> {
    /// The tile records, at most [`FRAME_TILE_RECORDS`].
    pub fn tile_records(&self) -> &[GrassTileGpu] {
        &self.tiles[..self.tiles.len().min(FRAME_TILE_RECORDS)]
    }

    /// The planet tile records, at most [`FRAME_PLANET_TILE_RECORDS`].
    pub fn planet_tile_records(&self) -> &[PlanetGrassTileGpu] {
        &self.planet_tiles[..self.planet_tiles.len().min(FRAME_PLANET_TILE_RECORDS)]
    }
}

/// What the visible-blade buffers become for a new render-target size, from
/// [`GrassField::resize_for`]: nothing changes until [`GrassField::apply`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassResize {
    /// The new sizing.
    pub sizing: GrassSizing,
    /// Whether it holds other blades than the current one, so its buffers
    /// must be made before it is applied.
    pub rebuild: bool,
}

/// One terrain layer that grows blades: where it sits among the terrains, its
/// validated look, the cell grid its density resolves to, and the seed the
/// kernel salts it with.
#[derive(Debug, Clone)]
pub(crate) struct GrownLayer {
    pub terrain: usize,
    pub layer: usize,
    pub grass: Grass,
    pub grid: GrassGrid,
    pub seed: u32,
}

/// Every layer of `terrains` that grows blades, in declaration order: the
/// layers the kernel draws, and the ones the far field shades.
pub(crate) fn grown_layers(terrains: &[GrassTerrain]) -> Vec<GrownLayer> {
    let mut out = Vec::new();
    for (ti, terrain) in terrains.iter().enumerate() {
        for (li, (grass, _)) in terrain.layers.iter().enumerate() {
            if !grass.visible {
                continue;
            }
            let grass = crate::components::validate::grass(grass.clone());
            let Some(grid) = GrassGrid::for_density(grass.density) else {
                continue;
            };
            out.push(GrownLayer {
                terrain: ti,
                layer: li,
                grass,
                grid,
                seed: lcg_hash(out.len() as u32 ^ 0x5bd1_e995),
            });
        }
    }
    out
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
    /// What the visible-blade buffers are sized for, once a render target's
    /// size is known (see [`Self::resize_for`]).
    pub sizing: Option<GrassSizing>,
    /// The tallest blade of any layer, in meters: how high above the ground a
    /// bender still presses the grass.
    pub blade_height: f32,
    /// The mean blade height of the tallest layer, in meters: the blade whose
    /// size on screen sets where the field coarsens, thins and ends.
    pub lod_height: f32,
    /// The planet the layers grow on instead of terrains.
    pub planet: Option<PlanetGrass>,
    /// What the far fields past the blades are shaded with.
    pub canopy: CanopyField,
}

// The frame a planet's tiles are drawn in this frame, from any one of them.
#[derive(Debug, Clone, Copy)]
struct PlanetView<'a> {
    grass: &'a PlanetGrass,
    center: [f32; 3],
    model: Mat4,
    tiles: &'a [DrawnPlanetTile],
}

impl GrassField {
    /// The grass `terrains` grow, swaying in `wind`. `None` when no visible
    /// layer places a blade.
    pub fn resolve(terrains: &[GrassTerrain], wind: Option<&Wind>) -> Option<Self> {
        let mut buffers = GrassGroundBuffers::default();
        let mut layers = Vec::new();
        let mut heights_offset = None;
        let mut last_terrain = None;
        let grown_all = grown_layers(terrains);
        let lod_height = grown_all
            .iter()
            .map(|g| g.grass.height)
            .fold(0.0f32, f32::max);
        for grown in grown_all {
            let terrain = &terrains[grown.terrain];
            if last_terrain != Some(grown.terrain) {
                last_terrain = Some(grown.terrain);
                heights_offset = None;
            }
            let offset = *heights_offset.get_or_insert_with(|| buffers.push_heights(&terrain.grid));
            let ground = GrassGround::new(terrain.center, &terrain.grid, offset);
            let mask = &terrain.layers[grown.layer].1;
            layers.push(GrassLayer {
                ground,
                mask: mask.as_ref().map(|m| buffers.push_mask(m)),
                look: GrassBladeLook::of(&grown.grass),
                grid: grown.grid,
                seed: grown.seed,
            });
        }
        if layers.is_empty() {
            return None;
        }
        let wind_field = WindField::new(wind);
        let blade_height = layers
            .iter()
            .map(|l| l.look.height * (1.0 + l.look.height_variance))
            .fold(0.0f32, f32::max);
        Some(Self {
            layers,
            buffers,
            wind: wind_field,
            sizing: None,
            blade_height,
            lod_height,
            planet: None,
            canopy: CanopyField::for_terrains(terrains, wind_field.strength).0,
        })
    }

    /// The grass `planet` grows, swaying in `wind`.
    pub fn resolve_planet(planet: PlanetGrass, wind: Option<&Wind>) -> Self {
        let wind_field = WindField::new(wind);
        let lod_height = planet
            .layers
            .iter()
            .map(|l| l.look.height)
            .fold(0.0f32, f32::max);
        let blade_height = planet
            .layers
            .iter()
            .map(|l| l.look.height * (1.0 + l.look.height_variance))
            .fold(0.0f32, f32::max);
        Self {
            layers: Vec::new(),
            buffers: GrassGroundBuffers::default(),
            wind: wind_field,
            sizing: None,
            blade_height,
            lod_height,
            canopy: CanopyField::for_planet(&planet, wind_field.strength).0,
            planet: Some(planet),
        }
    }

    /// How a view of vertical focal length `focal` pixels resolves the field:
    /// where its strips coarsen, how it thins and where its blades end. A
    /// focal length longer than the buffers are sized for thins as that one.
    pub fn lod_view(&self, focal: f32) -> GrassLodView {
        let sized = self.sizing.map_or(focal, |s| s.views.focal);
        GrassLodView::new(focal.min(sized), self.lod_height)
    }

    /// The sizing of the visible-blade buffers for render targets `size`
    /// pixels across and down, `None` when the field is already sized for
    /// them. The field keeps its sizing until the change is applied.
    pub fn resize_for(&self, size: [u32; 2]) -> Option<GrassResize> {
        if self.sizing.is_some_and(|s| s.views.size == size) {
            return None;
        }
        let sizing = GrassSizing::new(size, &|views| self.capacity_bounds(views));
        let rebuild = self.sizing.is_none_or(|s| {
            s.capacity != sizing.capacity || s.shadow_capacity != sizing.shadow_capacity
        });
        Some(GrassResize { sizing, rebuild })
    }

    /// Size the field as `resize` does, once the buffers it needs are made.
    pub fn apply(&mut self, resize: GrassResize) {
        self.sizing = Some(resize.sizing);
    }

    /// Blades each region of the visible-blade buffer holds: none before the
    /// field is sized.
    pub fn capacity(&self) -> GrassCapacity {
        self.sizing
            .map_or_else(GrassCapacity::default, |s| s.capacity)
    }

    /// Blades the shadow cascade's buffer holds: none before the field is
    /// sized, or when no layer keeps a blade near the camera.
    pub fn shadow_capacity(&self) -> u32 {
        self.sizing.map_or(0, |s| s.shadow_capacity)
    }

    // Each layer's region bounds, the cascade's, the view's tiles and how far
    // the blades may reach, from `views`.
    fn capacity_bounds(&self, views: &CapacityViews) -> CapacityBounds {
        let mut regions = Vec::new();
        let mut shadow = 0u32;
        let mut tiles = Vec::new();
        let mut add = |field: &CapacityField<'_>, tile: [f32; 2]| {
            regions.push(lod::region_capacities(field, views));
            shadow =
                shadow.saturating_add(lod::shadow_capacity(field, views, GRASS_SHADOW_DISTANCE));
            tiles.push(lod::tile_capacity(field, views, tile));
        };
        let wind = self.wind.strength;
        let limit = if let Some(planet) = &self.planet {
            // A planet's ground is unbounded, and its grass tiles are
            // narrowest where the face is stretched least.
            let ground = tiles::GroundRect::centered([0.0, 0.0], [1.0e7, 1.0e7]);
            let tile = (planet.grid.tile_size() / libm::sqrt(planet.grid.max_stretch)) as f32;
            for layer in &planet.layers {
                // Its blades stand at most at the layer's density, with a
                // margin for slopes holding more ground than their
                // footprint.
                if let Some(grid) = GrassGrid::for_density(layer.density * 1.15) {
                    let reach = planet_layer_gpu(layer, wind, self.lod_height).reach;
                    add(
                        &CapacityField {
                            ground: &ground,
                            grid: &grid,
                            width: layer.look.width,
                            blade_height: self.lod_height,
                        },
                        [tile, reach],
                    );
                }
            }
            MAX_PLANET_GRASS_REACH
        } else {
            for layer in &self.layers {
                add(
                    &CapacityField {
                        ground: &layer.ground.rect,
                        grid: &layer.grid,
                        width: layer.look.width,
                        blade_height: self.lod_height,
                    },
                    [GRASS_TILE_SIZE, blade_reach(&layer.look, self.lod_height)],
                );
            }
            f32::INFINITY
        };
        // A run holds the tiles of at most so many layers, whichever are in
        // reach.
        tiles.sort_unstable_by(|a, b| b.cmp(a));
        let tiles = tiles
            .iter()
            .take(MAX_GRASS_LAYERS)
            .fold(0u32, |sum, &n| sum.saturating_add(n));
        CapacityBounds {
            layers: regions,
            shadow,
            tiles,
            reach: [
                GrassLodView::new(views.focal, self.lod_height).extent,
                limit,
            ],
        }
    }

    /// This frame's grass work for `inputs`, advancing `history`, whose space
    /// the frame's records live in.
    pub fn frame<'h>(
        &self,
        inputs: &GrassFrameInputs<'_>,
        history: &'h mut GrassHistory,
    ) -> GrassFrame<'h> {
        let camera = &inputs.camera;
        let phase = history.crossfade_phase();
        let args_slot = history.runs % GRASS_ARGS_SLOTS as u32;
        history.runs = history.runs.wrapping_add(1);
        let step = history.bend.advance(camera.position, inputs.elapsed);
        let prev_map = core::mem::replace(&mut history.prev_map, GRASS_IDENTITY_MAP);
        let maps = [history.wind_map, prev_map];
        let planet = self.planet_view(inputs.planet_tiles);
        let lod = self.lod_view(camera.focal);
        let params = self.view_params(
            camera,
            (&lod, phase),
            &step,
            args_slot,
            maps,
            planet.as_ref(),
        );
        let scratch = &mut history.scratch;
        scratch.tiles.clear();
        scratch.planet_tiles.clear();
        let mut limit = None;
        let view = self.place(
            planet.as_ref(),
            camera.position,
            params,
            (scratch, &mut limit),
        );
        let shadow = inputs
            .shadow
            .filter(|_| self.shadow_capacity() > 0)
            .map(|light| {
                let params = self.shadow_params(&light, inputs.elapsed, &params);
                self.place(
                    planet.as_ref(),
                    camera.position,
                    params,
                    (scratch, &mut limit),
                )
            });
        let bend = self.bend_pass(
            &step,
            camera.position,
            inputs.benders,
            prev_map,
            planet.as_ref(),
        );
        let limit = limit.filter(|_| !core::mem::replace(&mut history.limit_reported, true));
        report_limit(limit);
        let scratch = &history.scratch;
        GrassFrame {
            view,
            shadow,
            bend,
            tiles: &scratch.tiles,
            planet_tiles: &scratch.planet_tiles,
            limit,
        }
    }

    // The planet's frame this frame, from its finest drawn tile, the one
    // placed nearest the frame's origin; `None` off a planet or before any
    // tile is drawn.
    fn planet_view<'a>(&'a self, tiles: &'a [DrawnPlanetTile]) -> Option<PlanetView<'a>> {
        let grass = self.planet.as_ref()?;
        let finest = tiles.iter().max_by_key(|t| t.id.level)?;
        Some(PlanetView {
            grass,
            center: planet_center(&grass.shape, finest),
            model: finest.model,
            tiles,
        })
    }

    // The view's block, all but its layers.
    fn view_params(
        &self,
        camera: &GrassCamera,
        (lod, phase): (&GrassLodView, f32),
        step: &BendStep,
        args_slot: u32,
        [wind_map, bend_prev_map]: [[[f32; 4]; 2]; 2],
        planet: Option<&PlanetView<'_>>,
    ) -> GrassParams {
        let frustum = Frustum::from_camera(camera.vp, Some(lod.extent));
        let [wind, wind_gust] = self.wind.gpu_rows();
        let hiz = camera.hiz;
        let capacity = self.capacity();
        let lod_base = capacity.region_bases();
        let lod_capacity = capacity.region_capacities();
        let prev = step.prev.unwrap_or(step.window);
        GrassParams {
            cam_pos: camera.position,
            draw_distance: lod.extent,
            frustum: frustum.packed_planes(),
            wind,
            wind_gust,
            prev_vp: hiz.map_or(camera.vp, |h| h.prev_vp),
            hiz_size: hiz.map_or([1.0, 1.0], |h| h.size),
            hiz_mip_count: hiz.map_or(1, |h| h.mip_count),
            hiz_enabled: u32::from(hiz.is_some()),
            lod_base,
            lod_capacity,
            lod_distances: [
                lod.lod_distances[0],
                lod.lod_distances[1],
                GRASS_MORPH_FRACTION,
                phase,
            ],
            thinning: [
                lod.keep_scale(),
                GRASS_SHRINK_BAND,
                lod.fade_start(),
                lod.depth_onset(),
            ],
            tile_base: 0,
            tile_count: 0,
            _pad: [0; 2],
            layer_count: 0,
            args_slot,
            tile_size: GRASS_TILE_SIZE,
            cull_distance: lod.extent,
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
            planet_center: planet.map_or([0.0; 3], |p| p.center),
            planet_density_scale: self.planet.as_ref().map_or(0.0, |p| p.meadow_scale),
            ground: if self.planet.is_some() {
                GRASS_GROUND_PLANET
            } else {
                GRASS_GROUND_TERRAIN
            },
            planet_tile_base: 0,
            planet_tile_count: 0,
            planet_radius: self.planet.as_ref().map_or(0.0, |p| p.shape.radius as f32),
            wind_map,
            bend_prev_map,
            planet_density: [0.0; MAX_GRASS_LAYERS],
            layers: [GrassLayerGpu::default(); MAX_GRASS_LAYERS],
        }
    }

    // The cascade's block: the view's culled against the light's frustum and
    // the cast distance instead, with no occlusion test and no hand-off, every
    // blade appended to the coarsest level's region of its own buffer.
    fn shadow_params(
        &self,
        light: &GrassShadowView,
        elapsed: f32,
        view: &GrassParams,
    ) -> GrassParams {
        let coarsest = GRASS_LOD_COUNT - 1;
        let cap = self.shadow_capacity();
        let mut lod_capacity = [0; 4];
        lod_capacity[coarsest] = cap;
        let mut lod_base = [0; 4];
        lod_base[lod::GRASS_BAND_REGION] = cap - 1;
        GrassParams {
            frustum: light.frustum().packed_planes(),
            hiz_enabled: 0,
            lod_base,
            lod_capacity,
            lod_distances: [0.0, 0.0, view.lod_distances[2], view.lod_distances[3]],
            thinning: [
                view.thinning[0],
                view.thinning[1],
                f32::MAX,
                view.thinning[3],
            ],
            cull_distance: GRASS_SHADOW_DISTANCE,
            shadow_vp: light.vp,
            shadow_light: [
                light.to_light[0],
                light.to_light[1],
                light.to_light[2],
                elapsed,
            ],
            ..*view
        }
    }

    // `params` with the layers in reach of `position` and the dispatch that
    // covers the tiles they visit, whose records are appended to the
    // scratch's, nearest first, as many as a run holds; a limit reached is
    // noted in `limit`.
    fn place(
        &self,
        planet: Option<&PlanetView<'_>>,
        position: [f32; 3],
        mut params: GrassParams,
        (scratch, limit): (&mut GrassScratch, &mut Option<GrassLimit>),
    ) -> GrassPass {
        let planes = params.frustum;
        let view = NestView {
            camera: position,
            distance: params.cull_distance,
            planes: &planes,
            keep_scale: params.thinning[0],
        };
        scratch.found.clear();
        let groups = match (&self.planet, planet) {
            (Some(grass), Some(p)) => {
                scratch.reach.gather(
                    (&grass.shape, &grass.grid),
                    p.tiles,
                    p.center,
                    position,
                    params.cull_distance,
                );
                if scratch.reach.clipped {
                    limit.get_or_insert(GrassLimit::PlanetTiles);
                }
                params.planet_tile_base = scratch.planet_tiles.len() as u32;
                params.planet_tile_count = scratch.reach.records.len() as u32;
                let groups = place_planet(
                    (grass, p),
                    &view,
                    [self.wind.strength, self.lod_height],
                    &mut params,
                    scratch,
                );
                scratch
                    .planet_tiles
                    .extend_from_slice(&scratch.reach.records);
                groups
            }
            (Some(_), None) => 1,
            (None, _) => self.place_terrain(&view, &mut params, scratch),
        };
        let found = &mut scratch.found;
        found.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        if found.len() > MAX_GRASS_TILE_RECORDS {
            limit.get_or_insert(GrassLimit::TileRecords);
            found.truncate(MAX_GRASS_TILE_RECORDS);
        }
        params.tile_base = scratch.tiles.len() as u32;
        params.tile_count = found.len() as u32;
        scratch.tiles.extend(found.iter().map(|&(_, _, t)| t));
        GrassPass {
            params,
            dispatch: [groups, params.tile_count.max(1), 1],
        }
    }

    // The terrain layers' tiles in reach of `view`, each with its distance,
    // into the scratch's found tiles, the layers into `params`; returns the
    // kernel groups across the largest tile.
    fn place_terrain(
        &self,
        view: &NestView<'_>,
        params: &mut GrassParams,
        scratch: &mut GrassScratch,
    ) -> u32 {
        let GrassScratch { found, visited, .. } = scratch;
        let mut groups = 1u32;
        let mut slot = 0usize;
        for layer in &self.layers {
            if slot == MAX_GRASS_LAYERS {
                break;
            }
            let gpu = layer_gpu(layer, self.wind.strength, self.lod_height);
            visited.clear();
            tiles::terrain_tiles(
                view,
                &layer.ground.rect,
                [[gpu.min_y, gpu.max_y], [gpu.reach, gpu.width]],
                visited,
            );
            if visited.tiles.is_empty() {
                continue;
            }
            params.layers[slot] = gpu;
            groups = groups.max(layer.grid.groups_per_tile());
            add_found(found, &visited.tiles, slot, 0);
            slot += 1;
        }
        params.layer_count = slot as u32;
        groups
    }

    // The bend pass's run for `step`: the footprints of `benders` near the
    // camera at `position`, and one thread per cell of the window.
    fn bend_pass(
        &self,
        step: &BendStep,
        position: [f32; 3],
        benders: &[GrassBender],
        prev_map: [[f32; 4]; 2],
        planet: Option<&PlanetView<'_>>,
    ) -> GrassBendPass {
        let found =
            bend::gather_stamps(benders, &step.window, position, self.blade_height, |base| {
                match planet {
                    Some(p) => Some(planet_ground_under(p, base)),
                    None if self.planet.is_some() => None,
                    None => self.ground_under([base[0], base[2]]),
                }
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
                prev_map: map_at_height(prev_map, position[1]),
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

// The local height of the ground under the bender standing at local `base`
// on `planet`: the planet's own surface along the radial through it, set
// below `base` along the frame's vertical.
fn planet_ground_under(planet: &PlanetView<'_>, base: [f32; 3]) -> f32 {
    use crate::math::dvec3;
    let offset = dvec3::sub(dvec3::from_f32(base), dvec3::from_f32(planet.center));
    let m = &planet.model;
    // The model's rotation takes planet directions into the frame; its
    // transpose takes them back.
    let world: [f64; 3] =
        core::array::from_fn(|c| (0..3).map(|r| f64::from(m[c][r]) * offset[r]).sum());
    let dir = dvec3::normalize_or(world, [0.0, 1.0, 0.0]);
    let shape = &planet.grass.shape;
    let gap = dvec3::length(offset) - (shape.radius + shape.height(dir));
    base[1] - gap as f32
}

// `tiles`' records for layer slot `slot`, in planet tile record
// `planet_tile`, each with its distance and the order it was found in, onto
// `found`.
fn add_found(
    found: &mut Vec<(f32, u32, GrassTileGpu)>,
    tiles: &[NestedTile],
    slot: usize,
    planet_tile: u32,
) {
    let first = found.len() as u32;
    found.extend(tiles.iter().zip(first..).map(|(t, order)| {
        let record = GrassTileGpu::new(t.tile, t.level, slot as u32, planet_tile);
        (t.near, order, record)
    }));
}

// `planet`'s layers into `params` and the tiles they visit over the
// scratch's drawn tiles in reach into its found tiles; returns the kernel
// groups across the largest tile.
fn place_planet(
    (planet, frame): (&PlanetGrass, &PlanetView<'_>),
    view: &NestView<'_>,
    [wind_speed, lod_height]: [f32; 2],
    params: &mut GrassParams,
    scratch: &mut GrassScratch,
) -> u32 {
    let GrassScratch {
        found,
        visited,
        reach: run,
        ..
    } = scratch;
    let mut groups = 1u32;
    let layers = if run.records.is_empty() {
        0
    } else {
        planet.layers.len().min(MAX_GRASS_LAYERS)
    };
    for (slot, layer) in planet.layers.iter().take(layers).enumerate() {
        let gpu = planet_layer_gpu(layer, wind_speed, lod_height);
        params.layers[slot] = gpu;
        params.planet_density[slot] = layer.density;
        groups = groups.max((layer.cells * layer.cells).div_ceil(GRASS_GROUP_SIZE));
        for (record, &(index, origin, count)) in run.blocks.iter().enumerate() {
            visited.clear();
            nested_planet_tiles(
                view,
                (&planet.shape, &planet.grid, frame.center),
                &frame.tiles[index],
                (origin, count),
                [gpu.reach, gpu.width],
                visited,
            );
            add_found(found, &visited.tiles, slot, record as u32);
        }
    }
    params.layer_count = layers as u32;
    params.tile_size = planet.grid.tile_size() as f32;
    groups
}

fn planet_layer_gpu(
    layer: &planet::PlanetGrassLayer,
    wind_speed: f32,
    lod_height: f32,
) -> GrassLayerGpu {
    let look = &layer.look;
    GrassLayerGpu {
        seed: layer.seed,
        height: look.height,
        height_variance: look.height_variance,
        width: look.width,
        clump_size: look.clump_size,
        stiffness: look.stiffness,
        color_variation: look.color_variation,
        cell_size: layer.cell_size,
        cells_per_side: layer.cells,
        root_color: look.root_color,
        reach: blade_reach(look, lod_height),
        tip_color: look.tip_color,
        macro_variation: look.macro_variation,
        clumping: look.clumping,
        light_projection: light_projection(look, wind_speed),
        depth: lod::blade_depth(look.width, layer.density, look.height),
        ..GrassLayerGpu::default()
    }
}

// The projection ellipse of blades of `look` bending in a wind of mean
// `wind_speed`, which the blade draws shade direct light through.
fn light_projection(look: &GrassBladeLook, wind_speed: f32) -> [f32; 2] {
    canopy::projection_axes(canopy::canopy_bend(wind_speed, look.stiffness))
}

/// How far a blade of `look` can reach from where its cell places it, in any
/// direction, in a field whose tallest layer stands `lod_height` meters tall:
/// its tallest height, since a bent blade keeps its length, plus half its
/// widest width, distance widening and edge-on widening included, plus the
/// farthest clumping moves it. Mirrors the kernel's height and width draws.
pub fn blade_reach(look: &GrassBladeLook, lod_height: f32) -> f32 {
    let tallest = look.height * (1.0 + look.height_variance);
    let widest = look.width * 1.25 * lod::max_width_scale(look.width, lod_height) * 1.3;
    tallest + 0.5 * widest + clumping::max_gather_offset(look.clump_size, look.clumping)
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

fn layer_gpu(layer: &GrassLayer, wind_speed: f32, lod_height: f32) -> GrassLayerGpu {
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
        reach: blade_reach(look, lod_height),
        tip_color: look.tip_color,
        macro_variation: look.macro_variation,
        clumping: look.clumping,
        light_projection: light_projection(look, wind_speed),
        depth: lod::blade_depth(look.width, layer.grid.density(), look.height),
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

    // A view-projection every point is inside.
    fn open() -> [[f32; 4]; 4] {
        let mut vp = [[0.0; 4]; 4];
        vp[3][3] = 1.0;
        vp
    }

    // A 1800 pixel tall view with a 70 degree vertical field of view.
    const FOCAL: f32 = 1285.0;

    fn camera(position: [f32; 3]) -> GrassCamera {
        GrassCamera {
            position,
            vp: open(),
            focal: FOCAL,
            hiz: None,
        }
    }

    fn inputs(position: [f32; 3]) -> GrassFrameInputs<'static> {
        GrassFrameInputs {
            camera: camera(position),
            elapsed: 0.0,
            shadow: None,
            benders: &[],
            planet_tiles: &[],
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
        assert_eq!(f.lod_height, Grass::default().height);
    }

    // Size `f` for `size` as a backend does, once the buffers are made: true
    // when they had to be.
    fn resize(f: &mut GrassField, size: [u32; 2]) -> bool {
        let Some(resize) = f.resize_for(size) else {
            return false;
        };
        f.apply(resize);
        resize.rebuild
    }

    // The buffers are sized once the render target's size is known, again
    // only when it changes, and the view thins no finer than they hold.
    #[test]
    fn the_field_is_sized_for_the_render_target() {
        let a = terrain([0.0; 3], [100.0, 100.0], vec![grass(100.0), grass(20.0)]);
        let mut f = GrassField::resolve(&[a], None).unwrap();
        assert_eq!(f.sizing, None);
        assert_eq!(f.capacity(), GrassCapacity::default());
        assert_eq!(f.shadow_capacity(), 0);
        // Nothing changes until the resize is applied, so a failed rebuild
        // leaves the field sized for the buffers it has.
        let pending = f.resize_for([1600, 900]).unwrap();
        assert!(pending.rebuild);
        assert_eq!(f.sizing, None);
        assert_eq!(f.resize_for([1600, 900]), Some(pending));
        assert!(resize(&mut f, [1600, 900]));
        assert!(!resize(&mut f, [1600, 900]));
        let views = f.sizing.unwrap().views;
        assert_eq!(views.size, [1600, 900]);
        let expected = GrassCapacity::sum(f.layers.iter().map(|l| {
            lod::region_capacities(
                &CapacityField {
                    ground: &l.ground.rect,
                    grid: &l.grid,
                    width: l.look.width,
                    blade_height: f.lod_height,
                },
                &views,
            )
        }));
        assert_eq!(f.capacity(), expected);
        assert_eq!(f.lod_view(10.0 * views.focal), f.lod_view(views.focal));
        assert!(f.lod_view(0.5 * views.focal).focal < views.focal);
        let sized = f.sizing;
        let smaller = f.resize_for([800, 450]).unwrap();
        assert_eq!(f.sizing, sized);
        f.apply(smaller);
        assert!(smaller.rebuild);
        assert!(f.capacity().total() < expected.total());
    }

    // A field sized for no blade in the cascade, or not yet sized, places no
    // cascade run, even when the cascade asks for one.
    #[test]
    fn a_cascade_with_no_blades_places_no_run() {
        let t = terrain([0.0; 3], [200.0, 200.0], vec![grass(100.0)]);
        let mut f = GrassField::resolve(&[t], None).unwrap();
        let cast = GrassFrameInputs {
            shadow: Some(light()),
            ..inputs([0.0; 3])
        };
        let mut history = GrassHistory::default();
        assert_eq!(f.frame(&cast, &mut history).shadow, None);
        resize(&mut f, [1600, 900]);
        let mut empty = f.sizing.unwrap();
        empty.shadow_capacity = 0;
        f.sizing = Some(empty);
        let frame = f.frame(&cast, &mut history);
        assert_eq!(frame.shadow, None);
        assert!(frame.view.params.tile_count > 0);
    }

    // Frames reuse the history's space: once it has grown to a frame, the
    // next places its records without growing it.
    #[test]
    fn frames_place_in_the_historys_space() {
        let t = terrain([0.0; 3], [400.0, 400.0], vec![grass(100.0), grass(30.0)]);
        let mut f = GrassField::resolve(&[t], None).unwrap();
        resize(&mut f, [1600, 900]);
        let cast = GrassFrameInputs {
            shadow: Some(light()),
            ..inputs([0.0, 2.0, 0.0])
        };
        let mut history = GrassHistory::default();
        let first = f.frame(&cast, &mut history).tiles.to_vec();
        let space = |h: &GrassHistory| {
            let s = &h.scratch;
            (
                s.tiles.capacity(),
                s.found.capacity(),
                s.visited.tiles.capacity(),
            )
        };
        let grown = space(&history);
        let again = f.frame(&cast, &mut history).tiles.to_vec();
        assert_eq!(first, again);
        assert_eq!(space(&history), grown);
    }

    // The frame state gathers no planet tiles for a terrain field and runs the
    // same frame the field does from the same inputs.
    #[test]
    fn the_frame_state_runs_the_fields_frame() {
        use crate::render::scene_state::{DrawList, SceneState};
        let t = terrain([0.0; 3], [400.0, 400.0], vec![grass(100.0)]);
        let mut f = GrassField::resolve(&[t], None).unwrap();
        resize(&mut f, [1600, 900]);
        let scene = SceneState::new(DrawList::unreserved(Vec::new(), 0), [0.0; 4]);
        let mut state = GrassFrameState::default();
        let view = GrassView {
            camera: camera([0.0, 2.0, 0.0]),
            elapsed: 0.0,
            shadow: Some(light()),
            benders: &[],
        };
        let ours = state.frame(&f, &scene, &view);
        let tiles = ours.tile_records().to_vec();
        assert!(ours.planet_tile_records().is_empty());
        let mut history = GrassHistory::default();
        let expected = f.frame(
            &GrassFrameInputs {
                shadow: Some(light()),
                ..inputs([0.0, 2.0, 0.0])
            },
            &mut history,
        );
        assert_eq!(tiles, expected.tile_records());
        assert!(tiles.len() <= FRAME_TILE_RECORDS);
    }

    // A run holding more tiles than its records keeps the nearest and
    // reports the limit on the first frame that reaches it only.
    #[test]
    fn a_run_past_its_records_keeps_the_nearest_and_reports_once() {
        let many = terrain(
            [0.0; 3],
            [2000.0, 2000.0],
            vec![grass(100.0); MAX_GRASS_LAYERS],
        );
        let f = GrassField::resolve(&[many], None).unwrap();
        let mut history = GrassHistory::default();
        let frame = f.frame(&inputs([0.0, 1.0, 0.0]), &mut history);
        let view = frame.view.params;
        assert_eq!(view.tile_count as usize, MAX_GRASS_TILE_RECORDS);
        assert_eq!(frame.limit, Some(GrassLimit::TileRecords));
        // The nearest tiles, the full-density ones under the camera, of
        // every layer are kept.
        let under: hashbrown::HashSet<u32> = frame
            .tiles
            .iter()
            .filter(|t| t.level_layer.trailing_zeros() >= 8 && t.tile == [0, 0])
            .map(|t| t.level_layer >> 8)
            .collect();
        assert_eq!(under.len(), MAX_GRASS_LAYERS);
        let next = f.frame(&inputs([0.0, 1.0, 0.0]), &mut history);
        assert_eq!(next.view.params.tile_count, view.tile_count);
        assert_eq!(next.limit, None);
    }

    // Sizing keeps a run's tiles within its records: a field whose tiles
    // would outgrow them is sized for a shorter view, which then fits.
    #[test]
    fn the_sizing_keeps_a_run_within_its_records() {
        let wide = Grass {
            width: 0.3,
            ..grass(100.0)
        };
        let many = terrain([0.0; 3], [4000.0, 4000.0], vec![wide; MAX_GRASS_LAYERS]);
        let mut f = GrassField::resolve(&[many], None).unwrap();
        resize(&mut f, [3200, 1800]);
        let sized = f.sizing.unwrap().views.focal;
        assert!(sized < lod::focal_length(1800, lod::GRASS_SIZING_FOV));
        let mut history = GrassHistory::default();
        let frame = f.frame(&inputs([0.0, 1.0, 0.0]), &mut history);
        assert_eq!(frame.limit, None);
        assert!(frame.view.params.tile_count as usize <= MAX_GRASS_TILE_RECORDS);
    }

    // Every level-0 tile a run's records span, asserting none twice.
    fn spanned(tiles: &[GrassTileGpu]) -> hashbrown::HashSet<(u32, [i32; 2])> {
        let mut seen = hashbrown::HashSet::new();
        for t in tiles {
            let n = 1i32 << (t.level_layer & 0xff);
            for z in 0..n {
                for x in 0..n {
                    let fine = [t.tile[0] * n + x, t.tile[1] * n + z];
                    assert!(seen.insert((t.level_layer >> 8, fine)), "{fine:?}");
                }
            }
        }
        seen
    }

    fn run_tiles<'a>(frame: &'a GrassFrame<'_>, pass: &GrassPass) -> &'a [GrassTileGpu] {
        let p = &pass.params;
        &frame.tiles[p.tile_base as usize..(p.tile_base + p.tile_count) as usize]
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
        let mut history = GrassHistory::default();
        let whole = f.frame(&inputs([10.0, 3.0, -4.0]), &mut history);
        let frame = whole.view;
        let p = &frame.params;
        assert_eq!(p.layer_count, 1);
        let l = &p.layers[0];
        assert_eq!(l.rect, [-10.0, -12.0, 30.0, 4.0]);
        assert_eq!(l.base_y, 2.0);
        assert_eq!(l.cells_per_side, 40);
        assert_eq!(l.cell_size, 0.1);
        // x in [-10, 30) is tiles -3..=7; z in [-12, 4) is tiles -3..=0, each
        // visited once.
        let tiles = run_tiles(&whole, &frame);

        assert_eq!(
            spanned(tiles)
                .into_iter()
                .filter(|(_, t)| (-3..=7).contains(&t[0]) && (-3..=0).contains(&t[1]))
                .count(),
            44
        );
        assert!(tiles.iter().all(|t| t.level_layer >> 8 == 0));
        assert_eq!(p.tile_base, 0);
        assert!(p.tile_count > 0 && p.tile_count <= 44);
        assert_eq!(frame.dispatch, [25, p.tile_count, 1]);
        assert_eq!(p.cam_pos, [10.0, 3.0, -4.0]);
        let lod = f.lod_view(FOCAL);
        assert_eq!(p.draw_distance, lod.extent);
        assert_eq!(p.lod_capacity, f.capacity().region_capacities());
        assert_eq!(p.lod_base, f.capacity().region_bases());
        assert_eq!(p.args_slot, 0);
        assert_eq!(p.hiz_enabled, 0);
        assert_eq!(p.cull_distance, lod.extent);
        assert_eq!(p.lod_distances[..2], lod.lod_distances);
        assert_eq!(p.thinning[0], lod.keep_scale());
        assert_eq!(p.thinning[2], lod.fade_start());
        assert_eq!(p.thinning[3], lod.depth_onset());
        let look = GrassBladeLook::of(&grass(100.0));
        assert_eq!(l.depth, lod::blade_depth(look.width, 100.0, look.height));
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
        assert_eq!(frame.dispatch, [100, 16, 1]);
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
        let mut history = GrassHistory::default();
        let whole = f.frame(&inputs([0.0; 3]), &mut history);
        assert_eq!(whole.view.params.layer_count as usize, MAX_GRASS_LAYERS);
        let slots: hashbrown::HashSet<u32> =
            whole.tiles.iter().map(|t| t.level_layer >> 8).collect();
        assert_eq!(slots.len(), MAX_GRASS_LAYERS);
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
        let f = GrassField::resolve(&[t], None).unwrap();
        let p = view(&f, [0.0; 3]).params;
        let expected = Frustum::from_camera(open(), Some(f.lod_view(FOCAL).extent));
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
        let mut f = GrassField::resolve(&[t], None).unwrap();
        resize(&mut f, [1600, 900]);
        assert!(f.shadow_capacity() < f.capacity().total());
        let mut history = GrassHistory::default();
        let frame = f.frame(
            &GrassFrameInputs {
                elapsed: 3.5,
                shadow: Some(light()),
                ..inputs([0.0; 3])
            },
            &mut history,
        );
        let view = frame.view.params;
        let cast = frame.shadow.expect("the grass casts").params;
        assert_eq!(cast.cull_distance, GRASS_SHADOW_DISTANCE);
        assert_eq!(cast.hiz_enabled, 0);
        let cap = f.shadow_capacity();
        assert_eq!(cast.lod_base, [0, 0, 0, cap - 1]);
        assert_eq!(cast.lod_capacity, [0, 0, cap, 0]);
        assert_eq!(&cast.lod_distances[..2], &[0.0, 0.0]);
        assert_eq!(cast.lod_distances[3], view.lod_distances[3]);
        // The same thinning, and no hand-off: every blade it keeps casts.
        assert_eq!(cast.thinning[..2], view.thinning[..2]);
        assert_eq!(cast.thinning[2], f32::MAX);
        // The depth limit's onset is the view's, so the cascade keeps the
        // blades the view keeps.
        assert_eq!(cast.thinning[3], view.thinning[3]);
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
        // 24 m around the camera, after the view's tiles.
        let cast_pass = frame.shadow.as_ref().unwrap();
        assert_eq!(cast.tile_base, view.tile_count);
        for t in spanned(run_tiles(&frame, cast_pass)) {
            let fine = t.1.map(|v| (v as f32 + 0.5) * GRASS_TILE_SIZE);
            assert!(fine[0].abs() < 40.0 && fine[1].abs() < 40.0, "{t:?}");
        }
        assert!(cast_pass.dispatch[1] < frame.view.dispatch[1]);
        // No cascade, no cast.
        let mut history = GrassHistory::default();
        let none = f.frame(&inputs([0.0; 3]), &mut history);
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

    fn apply_rows(rows: [[f32; 4]; 2], p: [f32; 3]) -> [f32; 2] {
        rows.map(|r| r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3])
    }

    fn turn() -> Rebase {
        Rebase {
            rotation: crate::math::quat_from_axis_angle([1.0, 0.0, 0.4], 0.02),
            translation: [12.0, -19.0, 998.0],
        }
    }

    // A move leaves the wind's gusts where they were on the ground, and the
    // frame after it reads last frame's bend field where each point was, then
    // back to reading it in place.
    #[test]
    fn a_move_carries_the_wind_and_the_bend_field() {
        let t = terrain([0.0; 3], [100.0, 100.0], vec![grass(10.0)]);
        let f = GrassField::resolve(&[t], None).unwrap();
        let mut history = GrassHistory::default();
        let p = [3.0, 1.0, -998.0];
        let before = f.frame(&inputs([0.0, 2.0, -990.0]), &mut history).view;
        assert_eq!(before.params.wind_map, GRASS_IDENTITY_MAP);
        assert_eq!(before.params.bend_prev_map, GRASS_IDENTITY_MAP);
        let wind_before = apply_rows(before.params.wind_map, p);
        let r = turn();
        history.rebase(&r);
        let q = r.apply_point(p);
        let after = f.frame(&inputs(r.apply_point([0.0, 2.0, -990.0])), &mut history);
        let (after_view, after_bend) = (after.view, after.bend);
        let wind_after = apply_rows(after_view.params.wind_map, q);
        let prev = apply_rows(after_view.params.bend_prev_map, q);
        for i in 0..2 {
            assert!((wind_before[i] - wind_after[i]).abs() < 2e-3);
        }
        assert!((prev[0] - p[0]).abs() < 2e-3 && (prev[1] - p[2]).abs() < 2e-3);
        // The bend pass reads at the camera's height through the same map.
        let cam_y = r.apply_point([0.0, 2.0, -990.0])[1];
        let rows = after_bend.params.prev_map;
        let at = rows.map(|row| row[0] * q[0] + row[1] * q[2] + row[2]);
        let expected = apply_rows(after_view.params.bend_prev_map, [q[0], cam_y, q[2]]);
        assert!((at[0] - expected[0]).abs() < 1e-3 && (at[1] - expected[1]).abs() < 1e-3);
        let next = f.frame(&inputs(r.apply_point([0.0, 2.0, -990.0])), &mut history);
        assert_eq!(next.view.params.bend_prev_map, GRASS_IDENTITY_MAP);
        assert_eq!(next.view.params.wind_map, after_view.params.wind_map);
    }

    mod on_a_planet {
        use super::*;
        use crate::planet::{LocalFrame, PlanetShape, TileLod, tile_mesh};

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

        fn field() -> GrassField {
            let grass = PlanetGrass::resolve(shape(), &[Grass::default(), grass(30.0)]).unwrap();
            GrassField::resolve_planet(grass, None)
        }

        // Every tile a standing camera draws, and the camera, in the
        // authored frame.
        fn standing() -> (Vec<DrawnPlanetTile>, [f32; 3]) {
            let s = shape();
            let lod = TileLod::new(s, crate::planet::FINEST_CELL_SIZE);
            let cam = [0.0, s.height([0.0, 1.0, 0.0]) as f32 + 1.7, 0.0];
            let cam_world = LocalFrame::AUTHORED.to_world(cam);
            let tiles = lod
                .select(
                    &crate::planet::TileView::nearest(cam_world),
                    &|_| true,
                    &mut crate::planet::TileBoundsCache::new(s),
                )
                .draw
                .into_iter()
                .map(|d| d.id)
                .map(|id| {
                    let mesh = tile_mesh(&s, id);
                    DrawnPlanetTile {
                        id,
                        model: LocalFrame::AUTHORED.model_at(mesh.origin),
                        vertex_base: 0,
                        radial: mesh.radial,
                        morph: Default::default(),
                        morph_extent: 0.0,
                    }
                })
                .collect();
            (tiles, cam)
        }

        #[test]
        fn the_frame_grows_every_layer_over_the_tiles_in_reach() {
            let mut f = field();
            resize(&mut f, [1600, 900]);
            assert!(f.capacity().total() > 0 && f.shadow_capacity() > 1);
            let (tiles, cam) = standing();
            let mut history = GrassHistory::default();
            let frame = f.frame(
                &GrassFrameInputs {
                    shadow: Some(light()),
                    planet_tiles: &tiles,
                    ..inputs(cam)
                },
                &mut history,
            );
            let view = &frame.view;
            let cast = frame.shadow.as_ref().unwrap();
            assert_eq!(view.params.ground, GRASS_GROUND_PLANET);
            assert_eq!(view.params.layer_count, 2);
            assert_eq!(view.dispatch[2], 1);
            let slots: hashbrown::HashSet<u32> = run_tiles(&frame, view)
                .iter()
                .map(|t| t.level_layer >> 8)
                .collect();
            assert_eq!(slots.len(), 2);
            assert!(
                run_tiles(&frame, view)
                    .iter()
                    .all(|t| t.planet_tile < view.params.planet_tile_count)
            );
            assert_eq!(view.params.planet_tile_base, 0);
            assert_eq!(cast.params.planet_tile_base, view.params.planet_tile_count);
            assert_eq!(
                frame.planet_tiles.len() as u32,
                view.params.planet_tile_count + cast.params.planet_tile_count
            );
            assert!(cast.dispatch[1] < view.dispatch[1]);
            assert_eq!(
                view.params.planet_density[..2],
                [Grass::default().density, 30.0]
            );
            let cells = f.planet.as_ref().unwrap().layers[0].cells;
            assert_eq!(view.dispatch[0], (cells * cells).div_ceil(GRASS_GROUP_SIZE));
            let c = view.params.planet_center;
            assert!((c[1] + 50_000.0).abs() < 0.05, "{c:?}");
        }

        // Before any tile is drawn, nothing grows, but the frame still runs.
        #[test]
        fn a_planet_with_no_drawn_tiles_grows_nothing() {
            let mut history = GrassHistory::default();
            let frame = field().frame(&inputs([0.0, 42.0, 0.0]), &mut history);
            assert_eq!(frame.view.params.layer_count, 0);
            assert_eq!(frame.view.dispatch, [1, 1, 1]);
            assert!(frame.planet_tiles.is_empty() && frame.tiles.is_empty());
        }

        // A bender standing on the planet presses fully; one well above the
        // blades does not.
        #[test]
        fn benders_press_from_the_planets_own_ground() {
            let f = field();
            let (tiles, cam) = standing();
            let s = shape();
            let under = crate::math::dvec3::normalize_or([0.5, 50_000.0, 0.5], [0.0; 3]);
            let ground = s.height(under) as f32;
            let benders = [
                GrassBender {
                    base: [0.5, ground, 0.5],
                    radius: 0.3,
                    height: 1.7,
                },
                GrassBender {
                    base: [1.5, ground + 5.0, 0.5],
                    radius: 0.3,
                    height: 1.7,
                },
            ];
            let mut history = GrassHistory::default();
            let frame = f.frame(
                &GrassFrameInputs {
                    benders: &benders,
                    planet_tiles: &tiles,
                    ..inputs(cam)
                },
                &mut history,
            );
            let b = frame.bend.params;
            assert_eq!(b.stamp_count, 1);
            assert!((b.stamps[0][3] - 1.0).abs() < 1e-3, "{:?}", b.stamps[0]);
        }
    }
}
