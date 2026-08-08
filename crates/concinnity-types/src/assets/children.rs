// src/assets/children.rs

use concinnity_memory::InlineVec;

use crate::ecs::Entity;

/// The entities that name this entity as their `Parent`.
///
/// Runtime-only, maintained automatically alongside `Parent` so a transform-
/// propagation pass can walk the hierarchy top-down and so despawning a parent
/// can cascade to its children.
///
/// The list is held inline up to one child, which most parents have.
#[derive(Debug, Clone, Default)]
pub struct Children(pub InlineVec<Entity>);
