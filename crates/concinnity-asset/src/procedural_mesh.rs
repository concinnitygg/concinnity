// Procedural-mesh generator schema.

use crate::{AssetId, PayloadLocator};
use alloc::string::String;
use alloc::vec::Vec;

/// Geometry built by a named generator at compile time. Use for standard shapes.
///
/// For custom / hand-authored geometry use [Mesh](#mesh) instead.
///
/// **Built-in generators:**
///
/// ```rust
/// # use concinnity_asset::ProceduralMesh;
/// ProceduralMesh {
///     generator: "room".into(),
///     half_width: 16.0,
///     half_depth: 20.0,
///     ceiling_height: 3.5,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProceduralMesh {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Built-in generator name (required), e.g. `room`, `box`, `cylinder`,
    /// `sphere`, `terrain`, `heightfield`, `skybox`, or `extrude`.
    pub generator: String,

    // Room / box / plane dimensions
    /// Half-width along X (room / box / plane / terrain), in world units.
    pub half_width: f32,
    /// Half-depth along Z (room / box / plane / terrain), in world units.
    pub half_depth: f32,
    /// Ceiling height for the `room` generator, in world units.
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
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version; length
    /// should be `lod_levels - 1`. Empty lets the build choose defaults.
    pub lod_distances: Vec<f32>,

    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for ProceduralMesh {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            generator: String::new(),
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            half_extents: None,
            radius: None,
            height: None,
            segments: None,
            rings: None,
            subdivisions: None,
            amplitude: None,
            source: None,
            elevation_min: None,
            elevation_max: None,
            size: None,
            profile: None,
            corner_radius: None,
            corner_segments: None,
            lod_levels: 1,
            lod_distances: Vec::new(),
            locator: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_generator_specific_field_starts_unset() {
        // The generator decides which fields it reads, so an unset field has to
        // mean "this generator's own default", not a shared number.
        let m = ProceduralMesh::default();
        assert!(m.generator.is_empty());
        assert_eq!(m.half_extents, None);
        assert_eq!(m.radius, None);
        assert_eq!(m.height, None);
        assert_eq!(m.segments, None);
        assert_eq!(m.rings, None);
        assert_eq!(m.subdivisions, None);
        assert_eq!(m.amplitude, None);
        assert_eq!(m.source, None);
        assert_eq!(m.elevation_min, None);
        assert_eq!(m.elevation_max, None);
        assert_eq!(m.size, None);
        assert_eq!(m.profile, None);
        assert_eq!(m.corner_radius, None);
        assert_eq!(m.corner_segments, None);
        // The room dimensions are shared, so they carry real defaults.
        assert_eq!(m.half_width, 8.0);
        assert_eq!(m.half_depth, 10.0);
        assert_eq!(m.ceiling_height, 3.5);
        assert_eq!(m.lod_levels, 1);
        assert!(m.lod_distances.is_empty());
        assert!(m.locator.is_none());
    }

    #[test]
    fn a_heightfield_reads_its_own_fields_and_leaves_the_rest_unset() {
        let m: ProceduralMesh = serde_json::from_str(
            r#"{"generator":"heightfield","source":"terrain.png","subdivisions":128,
                "elevation_min":-4,"elevation_max":40,"lod_levels":3,"lod_distances":[20,80]}"#,
        )
        .unwrap();
        assert_eq!(m.generator, "heightfield");
        assert_eq!(m.source.as_deref(), Some("terrain.png"));
        assert_eq!(m.subdivisions, Some(128));
        assert_eq!(m.elevation_min, Some(-4.0));
        assert_eq!(m.elevation_max, Some(40.0));
        assert_eq!(m.radius, None);
        assert_eq!(m.rings, None);
    }

    #[test]
    fn an_extruded_profile_round_trips_through_postcard() {
        let m: ProceduralMesh = serde_json::from_str(
            r#"{"generator":"extrude","profile":[[0,0],[1,0],[1,1]],"height":2.5,
                "corner_radius":0.1,"corner_segments":4,"half_extents":[1,2,3],
                "radius":0.5,"segments":32,"rings":16,"amplitude":3,"size":100}"#,
        )
        .unwrap();
        let bytes = postcard::to_allocvec(&m).unwrap();
        let back: ProceduralMesh = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, m);
        assert_eq!(
            back.profile.as_deref(),
            Some(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]][..])
        );
        assert_eq!(back.corner_segments, Some(4));
        assert_eq!(back.half_extents, Some([1.0, 2.0, 3.0]));
        assert_eq!(back.size, Some(100.0));
        assert_eq!(back.asset_id, AssetId::default());
    }
}
