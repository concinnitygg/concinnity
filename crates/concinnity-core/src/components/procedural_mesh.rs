//! `ProceduralMesh`'s `Component` impl is generated centrally (see
//! `cn_impl_components!`); this module keeps the blob-residency helper
//! `PhysicsSystem` relies on.

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
    /// `sphere`, `terrain`, `heightfield`, `skybox`, or `extrude`.
    pub generator: String,

    // Room / box / plane dimensions
    /// Half-width along X (room / box / plane / terrain), in world units.
    #[asset(default = 8.0)]
    pub half_width: f32,
    /// Half-depth along Z (room / box / plane / terrain), in world units.
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

    // Terrain
    /// Grid subdivisions for the `terrain` and `heightfield` generators. Higher
    /// is more detailed.
    pub subdivisions: Option<u32>,
    /// Maximum height variation for the `terrain` generator, in world units.
    pub amplitude: Option<f32>,

    // Heightfield (grayscale image → height grid)
    /// Path to a grayscale heightmap image for the `heightfield` generator.
    pub source: Option<String>,
    /// Height mapped to black pixels in the `heightfield` source, in world units.
    pub elevation_min: Option<f32>,
    /// Height mapped to white pixels in the `heightfield` source, in world units.
    pub elevation_max: Option<f32>,

    // Skybox
    /// Half-extent on all axes for the `skybox` generator, in world units.
    /// Keep it below the camera's `far` plane so the sky is not clipped.
    pub size: Option<f32>,

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

/// Blob indices of heightfield-generator ProceduralMeshes. GraphicsSystem's
/// init release sweep must spare these blobs: PhysicsSystem inits afterwards and
/// reads the baked heightfield collider grid from the payload, mirroring the
/// AudioClip / SdfVolume precedent of holding a blob resident for a later system.
pub fn heightfield_blob_indices(
    ctx: &crate::ecs::PipelineContext,
) -> alloc::collections::BTreeSet<u32> {
    ctx.query::<ProceduralMesh>()
        .filter(|m| m.generator == "heightfield")
        .filter_map(|m| m.locator.as_ref().map(|l| l.blob_index))
        .collect()
}
