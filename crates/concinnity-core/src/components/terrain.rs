//! The `Terrain` asset, and the blob-residency helper its colliders rely on.

use crate::components::Grass;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{MaterialHandle, PayloadLocator, Ref, TextureHandle};
use alloc::vec::Vec;

/// One ground cover a [Terrain](#terrain) grows: a [Grass](#grass) look, and
/// where on the terrain it grows.
///
/// Without a `density_mask` the grass covers the whole terrain. With one, the
/// mask's red channel scales the grass's density across the terrain: white
/// grows the full density, black leaves the ground bare. The mask is stretched
/// over the terrain's extent, its first row along the terrain's `-Z` edge and
/// its first column along `-X`.
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
/// overlap.
///
/// ```rust
/// # use concinnity_core::components::Terrain;
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
    ctx: &crate::ecs::PipelineContext,
) -> alloc::collections::BTreeSet<u32> {
    ctx.query::<Terrain>()
        .filter_map(|t| t.locator.as_ref().map(|l| l.blob_index))
        .collect()
}
