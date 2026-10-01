use crate::ecs::Entity;

/// Runtime-only event requesting that an entity's physics body wake, so a
/// body resting or authored asleep is simulated again. PhysicsSystem drains
/// these each step; an entity with no dynamic body is left alone. World
/// authors never declare this type directly.
#[derive(Debug, Clone, Copy)]
pub struct WakeRequest {
    /// The entity whose body wakes.
    pub target: Entity,
}
