// World-level physics configuration schema.

use crate::{AssetId, de_opt_asset_ref};
use alloc::string::String;
use alloc::vec::Vec;

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
/// ```rust
/// # use concinnity_asset::PhysicsConfig;
/// PhysicsConfig {
///     terrain_offset_y: -0.5,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Extra collision layer names beyond the built-ins (`world`, `prop`,
    /// `character`, `trigger`). At most 28; referenced by collider `layer`
    /// fields and `no_collide` pairs.
    pub layers: Vec<String>,
    /// Unordered layer-name pairs that do not collide. Everything collides by
    /// default; each pair here disables collision (and contact solving) between
    /// its two layers symmetrically. Pairs naming `character` also filter the
    /// character controller's movement.
    pub no_collide: Vec<[String; 2]>,
    /// Minimum contact impulse (mass times velocity change) for a collision to
    /// publish a contact event. Resting contact stays below it; raise to hear
    /// only hard impacts.
    pub contact_min_impulse: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            floor_y: 0.0,
            terrain_half_width: 0.0,
            terrain_half_depth: 0.0,
            terrain_subdivisions: 0,
            terrain_amplitude: 0.0,
            terrain_offset_y: 0.0,
            terrain_mesh: None,
            layers: Vec::new(),
            no_collide: Vec::new(),
            contact_min_impulse: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn a_blank_config_is_a_flat_floor_at_the_origin() {
        let p = PhysicsConfig::default();
        assert_eq!(p.floor_y, 0.0);
        assert_eq!(p.terrain_amplitude, 0.0);
        assert_eq!(p.terrain_subdivisions, 0);
        assert_eq!(p.terrain_offset_y, 0.0);
        // No mesh named means the generated terrain values are what is used.
        assert!(p.terrain_mesh.is_none());
        // Everything collides by default; light impacts stay silent.
        assert!(p.layers.is_empty());
        assert!(p.no_collide.is_empty());
        assert_eq!(p.contact_min_impulse, 1.0);
    }

    #[test]
    fn layers_and_no_collide_parse_and_round_trip_through_postcard() {
        let p: PhysicsConfig = serde_json::from_str(
            r#"{"layers":["debris"],"no_collide":[["debris","character"]],
                "contact_min_impulse":2.5}"#,
        )
        .unwrap();
        assert_eq!(p.layers, vec!["debris".to_string()]);
        assert_eq!(
            p.no_collide,
            vec![["debris".to_string(), "character".to_string()]]
        );

        let bytes = postcard::to_allocvec(&p).unwrap();
        let back: PhysicsConfig = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.layers, p.layers);
        assert_eq!(back.no_collide, p.no_collide);
        assert_eq!(back.contact_min_impulse, 2.5);
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
