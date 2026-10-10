// World-level physics configuration schema.

use alloc::string::String;
use alloc::vec::Vec;

/// Configures the world's physics: the floor, collision layers, and how many
/// bodies to reserve.
///
/// Optional: a world with physics bodies but no `PhysicsConfig` simulates over a
/// flat floor at Y = 0, and receives one carrying these values at start so the
/// settings are a component rather than a fallback. Physics runs whenever the world
/// declares a `PhysicsConfig`, a [RigidBody](#rigidbody), a
/// [PropBody](#propbody), a [TriggerVolume](#triggervolume), or a
/// [SkinnedMesh](#skinnedmesh) with a `capsule`.
///
/// Every [Terrain](#terrain) in the world is solid ground. A world with no
/// terrain stands on a flat floor at Y = 0.
///
/// ```rust
/// # use concinnity_core::components::PhysicsConfig;
/// PhysicsConfig {
///     contact_min_impulse: 2.0,
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
pub struct PhysicsConfig {
    /// Y coordinate of the floor. When left at 0.0 it is auto-detected from the
    /// camera; set it explicitly to override.
    pub floor_y: f32,
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
    #[asset(default = 1.0)]
    pub contact_min_impulse: f32,
    /// Extra physics bodies reserved for props created while the world runs
    /// (by a [Spawner](#spawner), a [Behavior](#behavior) `spawn` node, or the
    /// host). Physics reserves every body it will ever need when the world
    /// loads and never grows: once the declared bodies plus this many are
    /// live, a further spawn gets no physics body and is reported as an error.
    ///
    /// This is a floor beneath what the build reserves on its own, not the
    /// whole reservation. Every [Spawner](#spawner) whose `interval` and
    /// `lifetime` bound how many copies can be alive at once is already
    /// reserved for, and the larger of the two numbers wins. Set a value here
    /// for the sources the build cannot count: a `Spawner` with `lifetime: 0`
    /// (its copies live forever), a `spawn` node in a behavior, and spawns the
    /// host drives itself.
    pub spawn_headroom: u32,
}
