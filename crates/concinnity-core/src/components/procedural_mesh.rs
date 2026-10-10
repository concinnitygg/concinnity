//! `ProceduralMesh`'s `Component` impl is generated centrally (see
//! `cn_impl_components!`).

use crate::ecs::PayloadLocator;
use alloc::string::String;
use alloc::vec::Vec;

/// Geometry built by a named generator at compile time. Use for standard shapes.
///
/// For custom / hand-authored geometry use [Mesh](#mesh) instead.
///
/// **Built-in generators:**
///
/// ```rust
/// # use concinnity_core::components::ProceduralMesh;
/// ProceduralMesh {
///     generator: "room".into(),
///     half_width: 16.0,
///     half_depth: 20.0,
///     ceiling_height: 3.5,
///     ..Default::default()
/// };
/// ```
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
pub struct ProceduralMesh {
    /// Built-in generator name (required), e.g. `room`, `box`, `cylinder`,
    /// `sphere`, `plane`, `water_grid`, or `extrude`.
    pub generator: String,

    // Room / box / plane dimensions
    /// Half-width along X (room / plane / water grid), in world units.
    #[asset(default = 8.0)]
    pub half_width: f32,
    /// Half-depth along Z (room / plane / water grid), in world units.
    #[asset(default = 10.0)]
    pub half_depth: f32,
    /// Ceiling height for the `room` generator, in world units.
    #[asset(default = 3.5)]
    pub ceiling_height: f32,

    // Box
    /// Half-extents `[x, y, z]` for the `box` generator.
    pub half_extents: Option<[f32; 3]>,

    // Cylinder / sphere
    /// Radius for the `cylinder` and `sphere` generators.
    pub radius: Option<f32>,
    /// Height for the `cylinder` and `extrude` generators.
    pub height: Option<f32>,
    /// Number of radial segments around the `cylinder` and `sphere` generators.
    pub segments: Option<u32>,

    // Sphere
    /// Number of horizontal rings on the `sphere` generator.
    pub rings: Option<u32>,

    // Water grid
    /// Grid subdivisions for the `water_grid` generator. Higher is more
    /// detailed.
    pub subdivisions: Option<u32>,

    // Extrude
    /// 2D outline `[[x, z], ...]` extruded by the `extrude` generator.
    pub profile: Option<Vec<[f32; 2]>>,
    /// Corner-rounding radius for the `extrude` generator. 0 keeps sharp corners.
    pub corner_radius: Option<f32>,
    /// Number of segments used to round each corner in the `extrude` generator.
    pub corner_segments: Option<u32>,

    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates none; values are clamped to `[1, 8]`.
    #[asset(default = 1)]
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version; length
    /// should be `lod_levels - 1`. Empty lets the build choose defaults.
    pub lod_distances: Vec<f32>,

    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}
