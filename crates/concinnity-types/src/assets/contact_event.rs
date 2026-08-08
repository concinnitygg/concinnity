// src/assets/contact_event.rs

use crate::ecs::Entity;

// Runtime-only event published by the physics system when two bodies collide
// hard enough to pass the world's contact impulse threshold. `a` is always a
// simulated prop entity; `b` is the other side's entity, or None when the
// other body has no entity (terrain, the floor slab, a character capsule).
// `normal` points from `a`'s surface toward `b`; `impulse` is the contact's
// total impulse magnitude (mass times velocity change). World authors never
// declare this type directly.
#[derive(Debug, Clone, Copy)]
pub struct ContactEvent {
    pub a: Entity,
    pub b: Option<Entity>,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub impulse: f32,
}
