// Dynamic physics body schema for a companion Prop.

use crate::components::{Buoyancy, Prop};
use crate::ecs::{AudioClipHandle, Ref};

/// Makes a companion [Prop](#prop) a dynamic physics body.
///
/// Attach a `PropBody` to give a [Prop](#prop) real physics: it falls, collides,
/// stacks, tumbles, and (with `pickup: true` on the prop) can be carried and
/// thrown. A Prop with a `collider` but no `PropBody` is a static, immovable
/// obstacle.
///
/// ```json
/// ["PropBody", { "prop_name": "crate_a", "mass": 4.0, "friction": 0.6 }]
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
pub struct PropBody {
    /// The [Prop](#prop) this body drives. Must match a Prop declared in the
    /// same world.
    pub prop_name: Option<Ref<Prop>>,
    /// Mass in kilograms. 0 lets the simulation derive mass from the collider
    /// shape and a default density.
    pub mass: f32,
    /// Friction coefficient used for contacts with this body.
    #[asset(default = 0.5)]
    pub friction: f32,
    /// Bounciness in [0, 1]. 0 is fully inelastic.
    pub restitution: f32,
    /// Multiplier applied to world gravity for this body. 1.0 is normal.
    #[asset(default = 1.0)]
    pub gravity_scale: f32,
    /// Linear velocity damping, modeling air drag.
    #[asset(default = 0.05)]
    pub linear_damping: f32,
    /// Angular velocity damping, slowing the body's spin.
    #[asset(default = 0.05)]
    pub angular_damping: f32,
    /// Rolling resistance: the share of its load a rolling contact resists
    /// with, acting at the collider's rolling radius (a ball's or a capsule's
    /// radius). It stops a ball rolling on level ground. 0 rolls freely; a
    /// box does not roll.
    pub rolling_resistance: f32,
    /// Optional [`AudioClip`](#audioclip) played at the contact point when this
    /// body collides hard enough to pass the world's `contact_min_impulse`
    /// (see [`PhysicsConfig`](#physicsconfig)). Louder impacts play louder.
    pub impact_clip: Option<AudioClipHandle>,
    /// Linear gain applied to the impact clip at full impulse.
    #[asset(default = 1.0)]
    pub impact_volume: f32,
    /// Start the body asleep: it holds its authored pose, ignoring gravity,
    /// until something strikes it or a [Behavior](#behavior)'s `wake` node
    /// wakes it. A body leaning on an awake one wakes with it.
    pub asleep: bool,
    /// Makes the body float in water, as [`Buoyancy`](#buoyancy) describes.
    /// Unset, it sinks.
    pub buoyancy: Option<Buoyancy>,
}
