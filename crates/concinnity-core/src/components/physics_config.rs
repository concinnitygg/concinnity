// World-level physics configuration schema.

use crate::components::ProceduralMesh;
use crate::ecs::Ref;
use alloc::string::String;
use alloc::vec::Vec;

/// Configures the world's physics floor / terrain.
///
/// Optional: a world with physics bodies but no `PhysicsConfig` simulates over a
/// flat floor at Y = 0, and receives one carrying these values at start so the
/// settings are a component rather than a fallback. Physics runs whenever the world
/// declares a `PhysicsConfig`, a [RigidBody](#rigidbody), a
/// [PropBody](#propbody), a [TriggerVolume](#triggervolume), or a
/// [SkinnedMesh](#skinnedmesh) with a `capsule`. Declare a `PhysicsConfig` to
/// put bodies on terrain or a non-zero floor.
///
/// For terrain-based outdoor scenes the terrain parameters must match the
/// terrain mesh exactly.
///
/// ```rust
/// # use concinnity_core::components::PhysicsConfig;
/// PhysicsConfig {
///     terrain_offset_y: -0.5,
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
    #[serde(default)]
    pub terrain_mesh: Option<Ref<ProceduralMesh>>,
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
