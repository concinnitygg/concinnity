//! Character-model schema: a body conforming to a CharacterSchema.

use concinnity_core::components::CharacterCapsule;
use concinnity_core::ecs::MaterialHandle;
use concinnity_core::ecs::de_opt_material_handle;

/// A character body that conforms to a [CharacterSchema](#characterschema).
///
/// The build validates the source against the schema (joint names and
/// parentage, complete `+` / `-` pairs for bipolar keys), imports it,
/// generates the schema's synthesized targets, and emits one
/// [SkinnedMesh](#skinnedmesh) under this asset's name. A
/// [CharacterShape](#charactershape) or [Animation](#animation) targets the
/// model by this name exactly as it would a `SkinnedMesh`. Lower levels of
/// detail come from `lod_levels`, as on a `SkinnedMesh`.
///
/// The source's extra shape keys (ones the schema does not list) are
/// imported and appear under the editor panel's "Other" section.
///
/// ```rust
/// # use concinnity_world::registry::build_only::CharacterModel;
/// CharacterModel {
///     schema: "builtin:humanoid".into(),
///     source: "./base_humanoid.glb".into(),
///     lod_levels: 3,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CharacterModel {
    /// The [CharacterSchema](#characterschema) the source conforms to, by
    /// asset name or the reserved `builtin:humanoid`.
    pub schema: String,
    /// Path to the `.glb` / `.gltf` body.
    pub source: String,
    /// Which skinned mesh of `source` to import, in file order.
    pub skin_index: u32,
    /// [Material](#material) of the emitted mesh.
    #[serde(deserialize_with = "de_opt_material_handle")]
    pub material: Option<MaterialHandle>,
    /// World-space position.
    pub position: [f32; 3],
    /// World rotation, Euler degrees [pitch, yaw, roll], YXZ order.
    pub rotation_deg: [f32; 3],
    /// World scale.
    pub scale: [f32; 3],
    /// Number of level-of-detail versions to generate, including the
    /// original. `1` (the default) generates none; values are clamped to
    /// `[1, 8]`.
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version.
    /// When non-empty, must have exactly `lod_levels - 1` entries; empty
    /// lets the build choose defaults.
    pub lod_distances: Vec<f32>,
    /// Runtime copies the mesh may spawn beyond the authored one.
    pub max_instances: u32,
    /// Character capsule of the emitted mesh.
    pub capsule: Option<CharacterCapsule>,
}

impl Default for CharacterModel {
    fn default() -> Self {
        Self {
            schema: String::from("builtin:humanoid"),
            source: String::new(),
            skin_index: 0,
            material: None,
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            lod_levels: 1,
            lod_distances: Vec::new(),
            max_instances: 0,
            capsule: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_model_names_the_builtin_schema_and_no_source() {
        let m = CharacterModel::default();
        assert_eq!(m.schema, "builtin:humanoid");
        assert!(m.source.is_empty());
        assert_eq!(m.lod_levels, 1);
        assert_eq!(m.scale, [1.0, 1.0, 1.0]);
        assert!(m.capsule.is_none());
    }

    // The resolver seam is process-global and install-once, so the stand-in goes
    // in behind a `Once`: a name resolves to its own byte length, which is what
    // lets a named reference deserialize to a predictable handle.
    fn install_len_resolvers() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            concinnity_core::ecs::resolver::set_material_handle_resolver(|n| Some(n.len() as u32));
        });
    }

    #[test]
    fn a_model_round_trips_through_postcard() {
        install_len_resolvers();
        let m: CharacterModel = serde_json::from_str(
            r#"{"schema":"humanoid","material":"skin","source":"hero.glb","skin_index":1,
                "position":[0,0.1,0],"lod_levels":3,"lod_distances":[6,12],"max_instances":2,
                "capsule":{"half_height":0.9,"radius":0.3}}"#,
        )
        .unwrap();
        assert_eq!(m.material, Some(MaterialHandle(4)));
        let bytes = postcard::to_allocvec(&m).unwrap();
        let back: CharacterModel = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.schema, "humanoid");
        assert_eq!(back.source, "hero.glb");
        assert_eq!(back.skin_index, 1);
        assert_eq!(back.lod_levels, 3);
        assert_eq!(back.lod_distances, [6.0, 12.0]);
        assert_eq!(back.max_instances, 2);
        assert_eq!(back.capsule.unwrap().half_height, 0.9);
    }
}
