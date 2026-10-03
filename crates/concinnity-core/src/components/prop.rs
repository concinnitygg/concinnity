// Scene-object prop schema.

use crate::components::{Model, Scene};
use crate::components::{Vocabulary, vocabulary_synonyms};
use crate::ecs::MaterialHandle;
use crate::ecs::MeshHandle;
use crate::ecs::{NameRef, Ref, RefTarget};
use alloc::string::String;

/// The collision volume a [PropCollider](#propcollider)'s `shape` names. The
/// single accepted vocabulary: the build rejects an authored name this does not
/// recognize, and the runtime resolves the same name through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Vocabulary)]
pub enum PropColliderShape {
    /// Box sized by `half_extents`. Authored as `aabb` or `cuboid`.
    #[default]
    #[vocab("cuboid")]
    Cuboid,
    /// Sphere sized by `radius`. Authored as `ball` or `sphere`.
    #[vocab("ball")]
    Ball,
    /// Capsule sized by `radius` and `half_height`.
    #[vocab("capsule")]
    Capsule,
}

vocabulary_synonyms!(PropColliderShape, "a collider shape index");

impl PropColliderShape {
    /// Every authored name, canonical and alias, this accepts. The build lists
    /// these when it rejects an unknown shape.
    pub const ACCEPTED: &'static [&'static str] = &["aabb", "cuboid", "ball", "sphere", "capsule"];

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
}

/// Collision volume attached to a [Prop](#prop).
///
/// The shape dimensions are in the prop's local space and are scaled by the
/// prop's `scale`. `ball` and `capsule` use the X scale component (they assume
/// uniform scaling).
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct PropCollider {
    /// Collision shape: `aabb` (alias `cuboid`), `ball` (alias `sphere`), or
    /// `capsule`. See [PropColliderShape].
    pub shape: PropColliderShape,
    /// Box half-extents in local space [x, y, z]. Used by cuboid shapes.
    #[asset(default = [0.5, 0.5, 0.5])]
    pub half_extents: [f32; 3],
    /// Radius in local space. Used by ball and capsule shapes.
    #[asset(default = 0.5)]
    pub radius: f32,
    /// Half the cylinder height in local space. Used by capsule shapes.
    #[asset(default = 0.5)]
    pub half_height: f32,
    /// Collision layer name. Built-in layers are `world`, `prop`, `character`,
    /// and `trigger`; extra names come from [PhysicsConfig](#physicsconfig)
    /// `layers`. Empty derives the layer from the body kind: `world` for a
    /// static prop, `prop` when a [PropBody](#propbody) makes it dynamic.
    pub layer: String,
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
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Prop {
    /// A [Model](#model) asset. When set, the prop renders all sub-meshes of
    /// that model (each with its own material) sharing this prop's transform.
    /// Takes precedence over `mesh` and `material`.
    pub model: Option<Ref<Model>>,
    /// A [Mesh](#mesh) or [ProceduralMesh](#proceduralmesh) asset this prop
    /// renders. Used when `model` is unset.
    pub mesh: Option<MeshHandle>,
    /// A [Material](#material) to use for this prop: the albedo texture plus
    /// the lighting parameters (roughness, metallic, tint, emissive). Used when
    /// `model` is unset.
    pub material: Option<MaterialHandle>,
    /// World-space position [x, y, z].
    pub position: [f32; 3],
    /// Euler rotation in degrees [pitch, yaw, roll], applied in YXZ order
    /// (yaw first so that rotating around the vertical axis is intuitive).
    pub rotation_deg: [f32; 3],
    /// Non-uniform scale [x, y, z]. Defaults to [1, 1, 1].
    #[asset(default = [1.0, 1.0, 1.0])]
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
    pub parent: Option<Ref<PropParent>>,
    /// [Scene](#scene) this prop belongs to. `None` means the prop is visible
    /// in every scene. Used by scene switches for per-scene visibility.
    #[serde(default)]
    pub scene: Option<Ref<Scene>>,
    /// A [Prefab](#prefab) to instantiate at this prop's transform. When
    /// set, it expands into concrete child props and lights, replacing this
    /// prop. Cannot be combined with `model` or `mesh`.
    pub prefab: NameRef<PrefabTemplate>,
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

/// What a [Prop](#prop)'s `parent` may name: another prop, or the
/// [SkyRotation](#skyrotation) pivot.
#[derive(Debug, Clone, Copy)]
pub struct PropParent;

impl RefTarget for PropParent {
    const TYPES: &'static [&'static str] = &["Prop", "SkyRotation"];
}

/// What a [Prop](#prop)'s `prefab` may name: a [Prefab](#prefab), the template
/// the build expands the prop into.
#[derive(Debug, Clone, Copy)]
pub struct PrefabTemplate;

impl RefTarget for PrefabTemplate {
    const TYPES: &'static [&'static str] = &["Prefab"];
}

#[cfg(test)]
mod tests {
    use super::*;

    // The accepted list, the parser and the load are one vocabulary: every
    // listed name resolves and deserializes, and every shape's canonical name
    // is listed. A name added to one and not the other fails here.
    #[test]
    fn every_accepted_collider_shape_name_resolves_and_loads() {
        for name in PropColliderShape::ACCEPTED {
            let resolved = PropColliderShape::from_str_norm(name);
            assert!(resolved.is_some(), "{name} is listed but does not resolve");
            assert_eq!(
                serde_json::from_str::<PropColliderShape>(&alloc::format!(r#""{name}""#)).ok(),
                resolved,
                "{name} resolves but does not load"
            );
        }
        for shape in PropColliderShape::ALL {
            assert!(
                PropColliderShape::ACCEPTED.contains(&shape.as_str()),
                "{} is a shape whose own name is unlisted",
                shape.as_str()
            );
            assert_eq!(
                PropColliderShape::from_str_norm(shape.as_str()),
                Some(*shape)
            );
        }
        assert_eq!(PropColliderShape::from_str_norm("wedge"), None);
        // A typo is a load failure now, not a silent box.
        serde_json::from_str::<PropCollider>(r#"{"shape":"wedge"}"#)
            .expect_err("an unknown shape does not deserialize");
    }

    #[test]
    fn held_state_is_runtime_only_and_never_rides_the_wire() {
        let held = Prop {
            is_held: true,
            ..Default::default()
        };
        let bytes = postcard::to_allocvec(&held).unwrap();
        let back: Prop = postcard::from_bytes(&bytes).unwrap();
        assert!(!back.is_held);
    }
}
