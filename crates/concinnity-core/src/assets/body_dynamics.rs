// src/assets/body_dynamics.rs

use crate::ecs::AudioClipHandle;

/// Dynamic-body parameters attached to an entity with a `Collider`.
///
/// Runtime-only. Carries the physical values a `PropBody` declares, resolved
/// onto the owning prop's entity at load so runtime spawns can copy them; an
/// entity with a `Collider` but no `BodyDynamics` is a static obstacle.
#[derive(Debug, Clone, Copy)]
pub struct BodyDynamics {
    /// Mass in kilograms. 0 derives mass from the collider shape.
    pub mass: f32,
    /// Coulomb friction coefficient.
    pub friction: f32,
    /// Bounciness in [0, 1].
    pub restitution: f32,
    /// Multiplier applied to world gravity for this body.
    pub gravity_scale: f32,
    /// Linear velocity damping, modelling air drag.
    pub linear_damping: f32,
    /// Clip played at the contact point when this body collides hard enough
    /// to publish a contact event.
    pub impact_clip: Option<AudioClipHandle>,
    /// Linear gain applied to the impact clip at full impulse.
    pub impact_volume: f32,
}

impl Default for BodyDynamics {
    fn default() -> Self {
        Self {
            mass: 0.0,
            friction: 0.5,
            restitution: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.05,
            impact_clip: None,
            impact_volume: 1.0,
        }
    }
}
