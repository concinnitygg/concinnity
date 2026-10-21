//! A terrain: the `Terrain` a world declares, and its cooked surface, the one
//! square height grid its render mesh, its physics collider and its grass
//! placement are all read from, so the three agree triangle for triangle.
//!
//! The grid's heights come from a generated noise field ([`noise_heights`]) or
//! a decoded heightmap image ([`heightmap_heights`]); the cook decodes images,
//! this crate only samples pixels. A grass layer's density mask rides beside
//! the grid as a [`DensityMask`]. [`payload`] is the codec the cook writes and
//! the runtime reads.

mod grid;
mod mask;
mod mesh;
pub mod payload;
mod rule;
mod source;

use crate::components::Grass;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{MaterialHandle, PayloadLocator, Ref, TextureHandle};
use alloc::vec::Vec;
pub use grid::TerrainGrid;
pub use mask::{DensityMask, MAX_MASK_SIZE};
pub use mesh::{TERRAIN_CHUNK_CELLS, TerrainChunk, terrain_chunks, terrain_chunks_colored};
pub use rule::{GroundRule, MIN_ALTITUDE_EDGE, MIN_SLOPE_EDGE};
pub use source::{heightmap_heights, noise_heights};

/// Fewest grid cells along a terrain's side.
pub const MIN_TERRAIN_RESOLUTION: u32 = 4;

/// Most grid cells along a terrain's side.
pub const MAX_TERRAIN_RESOLUTION: u32 = 1024;

/// Smallest half-extent a terrain keeps, in meters.
pub const MIN_TERRAIN_EXTENT: f32 = 0.5;

/// The grid resolution an authored `resolution` builds at.
pub fn clamp_resolution(resolution: u32) -> u32 {
    resolution.clamp(MIN_TERRAIN_RESOLUTION, MAX_TERRAIN_RESOLUTION)
}

/// One ground cover a [Terrain](#terrain) grows: a [Grass](#grass) look, and
/// where on the terrain it grows.
///
/// Without a `density_mask` the grass covers the whole terrain. With one, the
/// mask's red channel scales the grass's density across the terrain: white
/// grows the full density, black leaves the ground bare. The mask is stretched
/// over the terrain's extent, its first row along the terrain's `-Z` edge and
/// its first column along `-X`.
///
/// On a [Planet](#planet) a layer grows over the planet's own meadow instead,
/// and `density_mask` is not supported. There the layer grows only where its
/// `rule` holds, thinning out over the rule's soft edges, so a meadow can end
/// above a beach. A terrain does not read `rule`.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct TerrainLayer {
    /// The [Grass](#grass) whose blades this layer grows.
    #[asset(default = Ref::new(AssetId::default()))]
    pub grass: Ref<Grass>,
    /// A [Texture](#texture) whose red channel scales the density across the
    /// terrain. Unset grows the grass everywhere.
    pub density_mask: Option<TextureHandle>,
    /// Where on a planet the grass grows. The default grows it everywhere.
    pub rule: GroundRule,
}

/// A rectangle of ground shaped by a height grid: rendered as a mesh with
/// `material`, collided by physics, and covered by grass `layers`.
///
/// The terrain spans `extent` (half-width and half-depth) around `center`, whose
/// height is the terrain's base. `resolution` cells along each side make up the
/// grid, so every corner is one height sample. The heights come from one of two
/// sources:
///
/// - Generated: rolling hills of up to `amplitude` meters above the base,
///   shaped by `seed`. An `amplitude` of 0 gives a flat field.
/// - A `heightmap` [Texture](#texture): its red channel maps black to
///   `elevation_min` and white to `elevation_max` above the base, stretched
///   over the extent the same way as a layer's density mask.
///
/// The rendered surface, the surface bodies collide with, and the ground every
/// grass blade roots in are the same triangles. Grass grows only on terrain:
/// each entry in `layers` grows one [Grass](#grass) look over the terrain,
/// optionally masked. Several terrains, and several layers on one terrain, may
/// overlap. Where the blades thin out in the distance, the surface blends from
/// its material toward the color its grass shows from afar, in proportion to
/// how densely the grass grows there, so a covered field reaches the horizon.
///
/// ```rust
/// # use concinnity_engine::components::Terrain;
/// Terrain {
///     extent: [40.0, 40.0],
///     resolution: 96,
///     amplitude: 1.5,
///     seed: 7,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Terrain {
    /// World-space center; its height is the terrain's base.
    pub center: [f32; 3],
    /// Half-width and half-depth `[x, z]`, in meters.
    #[asset(default = [50.0, 50.0])]
    pub extent: [f32; 2],
    /// Grid cells along each side. Clamped to [4, 1024].
    #[asset(default = 128)]
    pub resolution: u32,
    /// The [Material](#material) the surface renders with.
    pub material: Option<MaterialHandle>,
    /// Tallest generated hill above the base, in meters. Ignored with a
    /// `heightmap`.
    #[asset(default = 2.0)]
    pub amplitude: f32,
    /// Shapes the generated hills: each seed gives a different landscape.
    /// Ignored with a `heightmap`.
    pub seed: u32,
    /// A [Texture](#texture) whose red channel sets the heights. Unset
    /// generates them from `amplitude` and `seed`.
    pub heightmap: Option<TextureHandle>,
    /// Height of a black `heightmap` texel above the base, in meters.
    pub elevation_min: f32,
    /// Height of a white `heightmap` texel above the base, in meters.
    #[asset(default = 10.0)]
    pub elevation_max: f32,
    /// The grass that grows on this terrain, one entry per look.
    pub layers: Vec<TerrainLayer>,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

/// Blob indices of every `Terrain`'s payload. Graphics init spares these
/// blobs: the physics system inits after it and builds each terrain's collider
/// from the same payload.
pub fn terrain_blob_indices(
    ctx: &crate::ecs::PipelineContext<'_>,
) -> alloc::collections::BTreeSet<u32> {
    ctx.query::<Terrain>()
        .filter_map(|t| t.locator.as_ref().map(|l| l.blob_index))
        .collect()
}
