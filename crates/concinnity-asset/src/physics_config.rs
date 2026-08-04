// World-level physics configuration schema.

use crate::{AssetId, de_opt_asset_ref};

/// Configures the world's physics floor / terrain.
///
/// Optional: a world with physics bodies but no `PhysicsConfig` simulates over a
/// flat floor at Y = 0. Physics runs whenever the world declares a
/// `PhysicsConfig`, a [RigidBody](#rigidbody), or a [PropBody](#propbody).
/// Declare a `PhysicsConfig` to put bodies on terrain or a non-zero floor.
///
/// For terrain-based outdoor scenes the terrain parameters must match the
/// terrain mesh exactly.
///
/// ```jsonl
/// // Indoor (flat floor): no PhysicsConfig needed, just declare bodies.
///
/// // Outdoor (heightfield terrain):
/// {"name":"physics","type":"PhysicsConfig","args":{"terrain_mesh":"ground_heightfield_mesh","terrain_offset_y":-0.5}}
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PhysicsConfig {
    /// Y coordinate of the floor. When left at 0.0 it is auto-detected from the
    /// camera; set it explicitly to override.
    pub floor_y: f32,
    /// Half-width of the terrain mesh along X. Must match the terrain mesh.
    /// Leave at 0.0 (with `terrain_subdivisions` = 0) for flat-floor scenes.
    pub terrain_half_width: f32,
    /// Half-depth of the terrain mesh along Z. Must match the terrain mesh.
    pub terrain_half_depth: f32,
    /// Subdivision count of the terrain mesh. When 0, a flat floor at Y = 0 is
    /// used instead of a heightfield.
    pub terrain_subdivisions: u32,
    /// Height variation of the terrain mesh. Must match the terrain mesh.
    pub terrain_amplitude: f32,
    /// World-space Y offset of the terrain: the height of the prop that renders
    /// the terrain mesh. Leave at 0.0 when the terrain sits at the origin.
    pub terrain_offset_y: f32,
    /// Name of a [ProceduralMesh](#proceduralmesh) with `generator:
    /// "heightfield"`. When set, the physics surface is built from that mesh's
    /// source image so props rest on the visible terrain. Takes precedence over
    /// the `terrain_*` values above.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub terrain_mesh: Option<AssetId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_config_is_a_flat_floor_at_the_origin() {
        let p = PhysicsConfig::default();
        assert_eq!(p.floor_y, 0.0);
        assert_eq!(p.terrain_amplitude, 0.0);
        assert_eq!(p.terrain_subdivisions, 0);
        assert_eq!(p.terrain_offset_y, 0.0);
        // No mesh named means the generated terrain values are what is used.
        assert!(p.terrain_mesh.is_none());
    }

    #[test]
    fn a_named_terrain_mesh_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let p: PhysicsConfig = serde_json::from_str(
            r#"{"floor_y":-1.5,"terrain_half_width":128,"terrain_half_depth":128,
                "terrain_subdivisions":64,"terrain_amplitude":12,"terrain_offset_y":2,
                "terrain_mesh":"ground"}"#,
        )
        .unwrap();
        assert_eq!(p.terrain_mesh, Some(AssetId(6)));

        let bytes = postcard::to_allocvec(&p).unwrap();
        let back: PhysicsConfig = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.floor_y, -1.5);
        assert_eq!(back.terrain_half_width, 128.0);
        assert_eq!(back.terrain_half_depth, 128.0);
        assert_eq!(back.terrain_subdivisions, 64);
        assert_eq!(back.terrain_amplitude, 12.0);
        assert_eq!(back.terrain_offset_y, 2.0);
        assert_eq!(back.terrain_mesh, Some(AssetId(6)));
    }
}
