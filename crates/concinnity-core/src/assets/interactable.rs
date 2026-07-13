// src/assets/interactable.rs

/// Marks an entity the player can interact with (press the interact key while
/// close and facing it to trigger its behavior).
///
/// Runtime-only zero-size tag. Present on an entity whose `Prop` set
/// `interactable`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Interactable;
