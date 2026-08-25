// src/components/pickup.rs

/// Marks an entity the player can pick up and carry with the interact key.
///
/// Runtime-only zero-size tag. Present on an entity whose `Prop` set `pickup`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pickup;
