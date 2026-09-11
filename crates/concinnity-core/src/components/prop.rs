// Scene-object prop schema.

use crate::ecs::MaterialHandle;
use crate::ecs::MeshHandle;
use crate::ecs::asset_id::AssetId;
use crate::ecs::asset_id::de_opt_asset_ref;
use crate::ecs::de_opt_material_handle;
use crate::ecs::de_opt_mesh_handle;
use alloc::string::{String, ToString};

/// The collision volume a [PropCollider](#propcollider)'s `shape` names. The
/// single accepted vocabulary: the build rejects an authored name this does not
/// recognize, and the runtime resolves the same name through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropColliderShape {
    /// Box sized by `half_extents`. Authored as `aabb` or `cuboid`.
    Cuboid,
    /// Sphere sized by `radius`. Authored as `ball` or `sphere`.
    Ball,
    /// Capsule sized by `radius` and `half_height`.
    Capsule,
}

impl PropColliderShape {
    /// Every authored name, canonical and alias, this accepts. The build lists
    /// these when it rejects an unknown shape.
    pub const NAMES: [&'static str; 5] = ["aabb", "cuboid", "ball", "sphere", "capsule"];

    /// The shape an authored name selects, case-insensitively and accepting the
    /// aliases. `None` for an unknown name.
    pub fn from_str_norm(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "aabb" | "cuboid" => Some(Self::Cuboid),
            "ball" | "sphere" => Some(Self::Ball),
            "capsule" => Some(Self::Capsule),
            _ => None,
        }
    }

    /// The shape's canonical authored name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cuboid => "cuboid",
            Self::Ball => "ball",
            Self::Capsule => "capsule",
        }
    }
}

/// Collision volume attached to a [Prop](#prop).
///
/// The shape dimensions are in the prop's local space and are scaled by the
/// prop's `scale`. `ball` and `capsule` use the X scale component (they assume
/// uniform scaling).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PropCollider {
    /// Collision shape: `aabb` (alias `cuboid`), `ball` (alias `sphere`), or
    /// `capsule`. See [PropColliderShape].
    pub shape: String,
    /// Box half-extents in local space [x, y, z]. Used by cuboid shapes.
    pub half_extents: [f32; 3],
    /// Radius in local space. Used by ball and capsule shapes.
    pub radius: f32,
    /// Half the cylinder height in local space. Used by capsule shapes.
    pub half_height: f32,
    /// Collision layer name. Built-in layers are `world`, `prop`, `character`,
    /// and `trigger`; extra names come from [PhysicsConfig](#physicsconfig)
    /// `layers`. Empty derives the layer from the body kind: `world` for a
    /// static prop, `prop` when a [PropBody](#propbody) makes it dynamic.
    pub layer: String,
}

impl Default for PropCollider {
    fn default() -> Self {
        Self {
            shape: "cuboid".to_string(),
            half_extents: [0.5, 0.5, 0.5],
            radius: 0.5,
            half_height: 0.5,
            layer: String::new(),
        }
    }
}

/// A scene object: places geometry at a world-space transform.
///
/// Reference either a [Model](#model) (multi-mesh) or a single
/// [Mesh](#mesh)/[ProceduralMesh](#proceduralmesh). `model` takes precedence
/// when both are set.
///
/// Rotation notes:
/// - `rotation_deg[0]` = pitch (tilt forward/back)
/// - `rotation_deg[1]` = yaw (spin on vertical axis), most common
/// - `rotation_deg[2]` = roll (tilt side-to-side)
///
/// ```rust
/// # use concinnity_core::components::Prop;
/// Prop {
///     position: [4.0, 0.4, -8.0],
///     ..Default::default()
/// };
/// ```
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
    /// A [Material](#material) to use for this prop: the albedo texture plus
    /// the lighting parameters (roughness, metallic, tint, emissive). Used when
    /// `model` is unset.
    #[serde(deserialize_with = "de_opt_material_handle")]
    pub material: Option<MaterialHandle>,
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
    /// key (E) while close and facing it triggers its rotation behavior.
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
    /// Set at runtime while the prop is being carried. Not serialized.
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

#[cfg(test)]
mod tests {
    use super::*;

    // The alias list and the parser are one vocabulary: every listed name
    // resolves, and every shape's canonical name is listed. A name added to one
    // and not the other fails here.
    #[test]
    fn every_listed_collider_shape_name_resolves() {
        for name in PropColliderShape::NAMES {
            assert!(
                PropColliderShape::from_str_norm(name).is_some(),
                "{name} is listed but does not resolve"
            );
        }
        for shape in [
            PropColliderShape::Cuboid,
            PropColliderShape::Ball,
            PropColliderShape::Capsule,
        ] {
            assert!(
                PropColliderShape::NAMES.contains(&shape.as_str()),
                "{} is a shape whose own name is unlisted",
                shape.as_str()
            );
            assert_eq!(
                PropColliderShape::from_str_norm(shape.as_str()),
                Some(shape)
            );
        }
        assert_eq!(PropColliderShape::from_str_norm("wedge"), None);
    }

    #[test]
    fn a_blank_collider_names_a_listed_shape() {
        assert!(PropColliderShape::from_str_norm(&PropCollider::default().shape).is_some());
    }

    #[test]
    fn a_blank_collider_is_a_unit_cuboid() {
        let c = PropCollider::default();
        assert_eq!(c.shape, "cuboid");
        assert_eq!(c.half_extents, [0.5, 0.5, 0.5]);
        assert_eq!(c.radius, 0.5);
        assert_eq!(c.half_height, 0.5);
    }

    #[test]
    fn a_blank_prop_is_an_unscaled_non_interactive_placement() {
        let p = Prop::default();
        assert_eq!(p.position, [0.0, 0.0, 0.0]);
        assert_eq!(p.rotation_deg, [0.0, 0.0, 0.0]);
        assert_eq!(p.scale, [1.0, 1.0, 1.0]);
        // No collider means the prop is decoration: physics ignores it.
        assert!(p.collider.is_none());
        assert!(!p.interactable);
        assert!(!p.pickup);
        assert!(!p.is_held);
        assert_eq!(p.cull_distance, 0.0);
        assert!(p.prefab.is_empty());
        assert!(p.model.is_none());
        assert!(p.mesh.is_none());
        assert!(p.material.is_none());
        assert!(p.parent.is_none());
        assert!(p.scene.is_none());
    }

    #[test]
    fn every_reference_resolves_through_its_own_seam() {
        let p: Prop = crate::test_support::from_json(
            r#"{"model":"crate_model","mesh":"crate_mesh","material":"wood","parent":"shelf","scene":"vault"}"#,
        );
        // A Model is still an interned name; the resource kinds are handles.
        assert_eq!(p.model, Some(AssetId(11)));
        assert_eq!(p.mesh, Some(MeshHandle(10)));
        assert_eq!(p.material, Some(MaterialHandle(4)));
        assert_eq!(p.parent, Some(AssetId(5)));
        assert_eq!(p.scene, Some(AssetId(5)));
    }

    #[test]
    fn a_pickup_with_a_ball_collider_round_trips_through_postcard() {
        let p: Prop = crate::test_support::from_json(
            r#"{"position":[1,2,3],"rotation_deg":[0,90,0],"scale":[2,2,2],
                "collider":{"shape":"ball","radius":0.25},
                "interactable":true,"pickup":true,"prefab":"lantern","cull_distance":60}"#,
        );
        let collider = p.collider.as_ref().expect("collider");
        assert_eq!(collider.shape, "ball");
        assert_eq!(collider.radius, 0.25);
        // Unmentioned collider dimensions keep the schema defaults.
        assert_eq!(collider.half_height, 0.5);

        let bytes = postcard::to_allocvec(&p).unwrap();
        let back: Prop = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.position, [1.0, 2.0, 3.0]);
        assert_eq!(back.rotation_deg, [0.0, 90.0, 0.0]);
        assert_eq!(back.scale, [2.0, 2.0, 2.0]);
        assert_eq!(back.collider.expect("collider").shape, "ball");
        assert!(back.interactable);
        assert!(back.pickup);
        assert_eq!(back.prefab, "lantern");
        assert_eq!(back.cull_distance, 60.0);
        // Held state is runtime-only, so it never rides the wire.
        assert!(!back.is_held);
        assert_eq!(back.asset_id, AssetId::default());
    }
}
