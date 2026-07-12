// src/assets/parent.rs

use crate::ecs::Entity;

/// The entity whose world transform this entity inherits.
///
/// Runtime-only. When present, this entity's `Transform` is relative to the
/// parent's world transform. Carries the relationship a `Prop` declares with
/// its `parent` field, resolved from a name to a live `Entity`.
#[derive(Debug, Clone, Copy)]
pub struct Parent(pub Entity);

impl Default for Parent {
    fn default() -> Self {
        // Never observed: Parent is inserted at runtime with a real parent, not
        // built from serialized args.
        Parent(Entity::dangling())
    }
}
