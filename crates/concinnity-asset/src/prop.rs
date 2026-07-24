// Scene-object prop schema.

use crate::{
    AssetId, MaterialHandle, MeshHandle, TextureHandle, de_opt_asset_ref, de_opt_material_handle,
    de_opt_mesh_handle, de_opt_texture_handle,
};
use alloc::string::{String, ToString};

/// Collision volume attached to a [Prop](#prop).
///
/// The shape dimensions are in the prop's local space and are scaled by the
/// prop's `scale`. `ball` and `capsule` use the X scale component (they assume
/// uniform scaling).
///
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PropCollider {
    /// Collision shape: "aabb" (alias "cuboid"), "ball", or "capsule".
    pub shape: String,
    /// Box half-extents in local space [x, y, z]. Used by cuboid shapes.
    pub half_extents: [f32; 3],
    /// Radius in local space. Used by ball and capsule shapes.
    pub radius: f32,
    /// Half the cylinder height in local space. Used by capsule shapes.
    pub half_height: f32,
}

impl Default for PropCollider {
    fn default() -> Self {
        Self {
            shape: "cuboid".to_string(),
            half_extents: [0.5, 0.5, 0.5],
            radius: 0.5,
            half_height: 0.5,
        }
    }
}

/// A scene object: places geometry at a world-space transform.
///
/// Reference either a [Model](#model) (multi-mesh) or a single
/// [Mesh](#mesh)/[ProceduralMesh](#proceduralmesh). `model` takes precedence
/// when both are set.
///
/// ```jsonl
/// // single mesh
/// {"name":"crate_a","type":"Prop","args":{"mesh":"box_mesh","material":"mat_brick","position":[4.0,0.4,-8.0],"collider":{"shape":"aabb","half_extents":[0.4,0.4,0.4]}}}
/// {"name":"column_ne","type":"Prop","args":{"mesh":"column_mesh","material":"mat_stone","position":[8.0,1.7,-10.0],"collider":{"shape":"aabb","half_extents":[0.18,1.7,0.18]}}}
/// {"name":"room_floor","type":"Prop","args":{"mesh":"room_mesh","material":"mat_plaster","position":[0.0,0.0,0.0]}}
///
/// // multi-mesh model
/// {"name":"crate_a","type":"Prop","args":{"model":"wooden_crate","position":[2.0,0.3,-4.0],"collider":{"shape":"aabb","half_extents":[0.3,0.3,0.3]}}}
///
/// // parent-child hierarchy: door panel inherits the frame's world transform
/// {"name":"door_frame","type":"Prop","args":{"model":"wooden_frame","position":[3,0,-2]}}
/// {"name":"door_panel","type":"Prop","args":{"model":"door","parent":"door_frame","position":[0,0,0.05]}}
/// ```
///
/// Rotation notes:
/// - `rotation_deg[0]` = pitch (tilt forward/back)
/// - `rotation_deg[1]` = yaw (spin on vertical axis), most common
/// - `rotation_deg[2]` = roll (tilt side-to-side)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Prop {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// A [Model](#model) asset. When set, the prop renders all sub-meshes of
    /// that model (each with its own material) sharing this prop's transform.
    /// Takes precedence over `mesh` and `material`.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub model: Option<AssetId>,
    /// A [Mesh](#mesh) or [ProceduralMesh](#proceduralmesh) asset this prop
    /// renders. Used when `model` is unset.
    #[serde(deserialize_with = "de_opt_mesh_handle")]
    pub mesh: Option<MeshHandle>,
    /// A [Material](#material) to use for this prop. When set it takes
    /// precedence over `texture` and provides the albedo texture plus the
    /// lighting parameters (roughness, metallic, tint, emissive). Used when
    /// `model` is unset.
    #[serde(deserialize_with = "de_opt_material_handle")]
    pub material: Option<MaterialHandle>,
    /// A [Texture](#texture) to use for this prop. Older field: ignored when
    /// `material` is set. Unset uses the first declared texture (or a white
    /// fallback).
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub texture: Option<TextureHandle>,
    /// World-space position [x, y, z].
    pub position: [f32; 3],
    /// Euler rotation in degrees [pitch, yaw, roll], applied in YXZ order
    /// (yaw first so that rotating around the vertical axis is intuitive).
    pub rotation_deg: [f32; 3],
    /// Non-uniform scale [x, y, z]. Defaults to [1, 1, 1].
    pub scale: [f32; 3],
    /// Optional collision volume. When present, the prop blocks the player; when
    /// absent the prop is non-solid.
    pub collider: Option<PropCollider>,
    /// When true, the player can interact with this prop: pressing the interact
    /// key (E) while close and facing it triggers its rotation behaviour.
    pub interactable: bool,
    /// When true, the player can pick up and carry this prop with the interact
    /// key (E). A companion [PropBody](#propbody) must also be declared so the
    /// prop falls correctly after being dropped.
    pub pickup: bool,
    /// Another [Prop](#prop) whose world transform this prop inherits. When set,
    /// `position`, `rotation_deg`, and `scale` are relative to the parent's
    /// world transform. The parent must be declared in the same world; circular
    /// chains are treated as an error.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub parent: Option<AssetId>,
    /// [Scene](#scene) this prop belongs to. Resolved automatically from the
    /// naming convention (a prop named `<scene>_*` belongs to scene `<scene>`);
    /// you don't set this directly. `None` means the prop is visible in every
    /// scene. Used by scene switches for per-scene visibility.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub scene: Option<AssetId>,
    /// Name of a [Prefab](#prefab) to instantiate at this prop's transform. When
    /// set, it expands into concrete child props and lights, replacing this
    /// prop. Cannot be combined with `model` or `mesh`.
    pub prefab: String,
    /// Optional view-distance cutoff in world units. When > 0 the prop is hidden
    /// once the camera is further than this from it. 0 (default) keeps the prop
    /// visible at any distance.
    pub cull_distance: f32,
    /// Set at runtime while the prop is being carried. Not serialised.
    /// While true, PhysicsSystem drives the prop as a kinematic body that
    /// follows the camera instead of simulating it dynamically.
    #[serde(skip)]
    pub is_held: bool,
}

impl Default for Prop {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            model: None,
            mesh: None,
            material: None,
            texture: None,
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            collider: None,
            interactable: false,
            pickup: false,
            parent: None,
            scene: None,
            prefab: String::new(),
            cull_distance: 0.0,
            is_held: false,
        }
    }
}
