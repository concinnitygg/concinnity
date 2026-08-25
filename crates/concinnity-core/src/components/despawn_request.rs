// src/components/despawn_request.rs

use crate::components::EntityTarget;

/// Runtime-only event requesting that an authored placement be removed from the
/// world at runtime. A named target needs no live Entity handle; an
/// entity-addressed one reaches placements that never had a name. GraphicsSystem
/// reads these from its `Events<DespawnRequest>` queue each step, resolves the
/// target to its entity, hides that entity's GPU draw slots, and despawns it and
/// its descendants. World authors never declare this type directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct DespawnRequest {
    /// The placement to remove, by name or entity.
    pub target: EntityTarget,
}
