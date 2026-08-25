// src/components/contact_event.rs

use crate::ecs::Entity;

/// Runtime-only event published by the physics system when two bodies collide
/// hard enough to pass the world's contact impulse threshold. `a` is always a
/// simulated prop entity; `b` is the other side's entity, or None when the
/// other body has no entity (terrain, the floor slab, a character capsule).
/// `normal` points from `a`'s surface toward `b`; `impulse` is the contact's
/// total impulse magnitude (mass times velocity change). World authors never
/// declare this type directly.
#[derive(Debug, Clone, Copy)]
pub struct ContactEvent {
    /// The simulated prop entity on one side of the contact.
    pub a: Entity,
    /// The other side's entity, or `None` when that body has none.
    pub b: Option<Entity>,
    /// World-space contact point.
    pub point: [f32; 3],
    /// Contact normal, pointing from `a`'s surface toward `b`.
    pub normal: [f32; 3],
    /// Total contact impulse magnitude (mass times velocity change).
    pub impulse: f32,
}
