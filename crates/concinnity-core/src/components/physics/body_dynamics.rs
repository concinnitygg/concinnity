use crate::ecs::AudioClipHandle;

/// Dynamic-body parameters attached to an entity with a `Collider`.
///
/// Runtime-only. Carries the physical values a `PropBody` declares, resolved
/// onto the owning prop's entity at load so runtime spawns can copy them; an
/// entity with a `Collider` but no `BodyDynamics` is a static obstacle.
#[derive(Debug, Clone, Copy, crate::ecs::AssetDefault)]
pub struct BodyDynamics {
    /// Mass in kilograms. 0 derives mass from the collider shape.
    pub mass: f32,
    /// Coulomb friction coefficient.
    #[asset(default = 0.5)]
    pub friction: f32,
    /// Bounciness in [0, 1].
    pub restitution: f32,
    /// Multiplier applied to world gravity for this body.
    #[asset(default = 1.0)]
    pub gravity_scale: f32,
    /// Linear velocity damping, modeling air drag.
    #[asset(default = 0.05)]
    pub linear_damping: f32,
    /// Clip played at the contact point when this body collides hard enough
    /// to publish a contact event.
    pub impact_clip: Option<AudioClipHandle>,
    /// Linear gain applied to the impact clip at full impulse.
    #[asset(default = 1.0)]
    pub impact_volume: f32,
    /// Whether the body starts asleep, holding its pose until woken.
    pub asleep: bool,
}
