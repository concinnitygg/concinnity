// src/assets/held.rs

/// Marks an entity currently being carried by the player.
///
/// Runtime-only zero-size tag, added and removed by the physics system on
/// pickup and drop. While present, the entity is driven as a kinematic body
/// that follows the camera instead of simulating dynamically.
#[derive(Debug, Clone, Copy, Default)]
pub struct Held;
