// Prefab schema: a reusable template of props / lights / nested prefabs.

use crate::PropCollider;
use alloc::string::String;
use alloc::vec::Vec;

/// A reusable template of [Prop](#prop)s, [PointLight](#pointlight)s, and nested
/// prefabs.
///
/// Placed as a unit at a world-space transform. Add a `prefab` field to a
/// [Prop](#prop) to instantiate it; each instance expands into concrete assets
/// positioned relative to the instance's transform.
///
/// **Expanded asset names:** `<instance_name>_<entry_name>` (nested:
/// `<instance>_<outer>_<inner>`).
///
/// **Instantiation:** add a `prefab` field to a [Prop](#prop). The prop's other
/// fields (`position`, `rotation_deg`, `scale`) act as the instance's world
/// transform.
///
/// ```jsonl
/// // Define the template:
/// {"type":"Prefab","name":"table_set","args":{"props":[
///   {"name":"table","kind":"prop","model":"model_table","position":[0,0,0]},
///   {"name":"chair_n","kind":"prop","model":"model_chair","position":[0,0,0.7],"rotation_deg":[0,180,0]},
///   {"name":"chair_s","kind":"prop","model":"model_chair","position":[0,0,-0.7]},
///   {"name":"lamp","kind":"point_light","position":[0,2.2,0],"light_color":[1.0,0.9,0.7],"light_intensity":6.0,"light_range":5.0}
/// ]}}
///
/// // Place two instances:
/// {"type":"Prop","name":"dining_a","args":{"prefab":"table_set","position":[3,0,-5]}}
/// {"type":"Prop","name":"dining_b","args":{"prefab":"table_set","position":[-3,0,-5],"rotation_deg":[0,45,0]}}
/// ```
///
/// **Library presets** (JSON files in `assets/prefabs/`):
///
/// ```jsonl
/// // From library preset:
/// {"type":"Prop","name":"table_a","args":{"prefab":"prefab_table_4chair","position":[0,0,-6]}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Prefab {
    /// Ordered list of entries. Each is a prop, a point light, or a nested
    /// prefab (selected by `kind`), placed relative to the instance transform.
    pub props: Vec<PrefabEntry>,
}

/// Which kind of asset a [PrefabEntry] expands into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum PrefabKind {
    /// A [Prop](#prop) built from the entry's `model` / `mesh` / `material` /
    /// `texture` and transform fields.
    #[default]
    Prop,
    /// A [PointLight](#pointlight) built from the entry's `light_*` fields at the
    /// entry's `position`.
    PointLight,
    /// A nested prefab named by the entry's `prefab` field, expanded relative to
    /// this entry's transform.
    Prefab,
}

/// One entry in a [Prefab]'s `props` list. The fields consulted depend on
/// `kind`: a `prop` uses the render / collision / transform fields, a
/// `point_light` uses the `light_*` fields, and a `prefab` uses `prefab`. Names
/// in `model` / `mesh` / `material` / `texture` / `parent` / `prefab` are
/// unresolved references to other assets, resolved when the entry expands.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PrefabEntry {
    /// Entry name; the expanded asset is named `<instance>_<name>`.
    pub name: String,
    /// Which asset this entry expands into.
    pub kind: PrefabKind,
    /// Local position relative to the instance transform.
    pub position: [f32; 3],
    /// Local rotation, Euler degrees [pitch, yaw, roll], YXZ order.
    pub rotation_deg: [f32; 3],
    /// Local scale.
    pub scale: [f32; 3],
    /// `prop`: [Model](#model) name.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub model: String,
    /// `prop`: [Mesh](#mesh) / [ProceduralMesh](#proceduralmesh) name.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub mesh: String,
    /// `prop`: [Material](#material) name.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub material: String,
    /// `prop`: [Texture](#texture) name (older path; `material` takes priority).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub texture: String,
    /// `prop`: parent asset name for the expanded prop.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub parent: String,
    /// `prop`: optional collision shape for the expanded prop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collider: Option<PropCollider>,
    /// `prop`: whether the expanded prop is interactable.
    pub interactable: bool,
    /// `prop`: whether the expanded prop is a pickup.
    pub pickup: bool,
    /// `point_light`: linear-space RGB colour.
    pub light_color: [f32; 3],
    /// `point_light`: intensity multiplier.
    pub light_intensity: f32,
    /// `point_light`: maximum reach in world units.
    pub light_range: f32,
    /// `prefab`: name of another [Prefab] to expand at this entry's transform.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub prefab: String,
}

impl Default for PrefabEntry {
    fn default() -> Self {
        Self {
            name: String::new(),
            kind: PrefabKind::Prop,
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            model: String::new(),
            mesh: String::new(),
            material: String::new(),
            texture: String::new(),
            parent: String::new(),
            collider: None,
            interactable: false,
            pickup: false,
            light_color: [1.0, 1.0, 1.0],
            light_intensity: 8.0,
            light_range: 6.0,
            prefab: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_kind_deserialises_with_its_fields() {
        let prop: PrefabEntry = serde_json::from_str(
            r#"{"name":"table","kind":"prop","model":"model_table","position":[1.0,0.0,2.0]}"#,
        )
        .unwrap();
        assert_eq!(prop.kind, PrefabKind::Prop);
        assert_eq!(prop.name, "table");
        assert_eq!(prop.model, "model_table");
        assert_eq!(prop.position, [1.0, 0.0, 2.0]);
        // Omitted scale falls back to unit.
        assert_eq!(prop.scale, [1.0, 1.0, 1.0]);

        let light: PrefabEntry =
            serde_json::from_str(r#"{"name":"lamp","kind":"point_light","light_intensity":5.0}"#)
                .unwrap();
        assert_eq!(light.kind, PrefabKind::PointLight);
        assert_eq!(light.light_intensity, 5.0);
        // Omitted light fields fall back to the point-light defaults.
        assert_eq!(light.light_range, 6.0);
        assert_eq!(light.light_color, [1.0, 1.0, 1.0]);

        let nested: PrefabEntry =
            serde_json::from_str(r#"{"name":"inner","kind":"prefab","prefab":"other"}"#).unwrap();
        assert_eq!(nested.kind, PrefabKind::Prefab);
        assert_eq!(nested.prefab, "other");
    }

    #[test]
    fn kind_defaults_to_prop_when_omitted() {
        let e: PrefabEntry = serde_json::from_str(r#"{"name":"x","mesh":"box"}"#).unwrap();
        assert_eq!(e.kind, PrefabKind::Prop);
        assert_eq!(e.mesh, "box");
    }
}
