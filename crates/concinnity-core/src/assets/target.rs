// src/assets/target.rs

use crate::ecs::Entity;
use crate::ecs::asset_id::AssetId;

/// How a runtime request addresses the entity it acts on. Authored placements
/// are named, so a producer holding no live handle addresses them by asset name
/// and the applying system resolves it. Logic that already holds an entity --
/// one it spawned, one a query yielded -- addresses it directly, which is the
/// only way to reach an entity that never had a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// Addressed by the placement's authored name.
    Name(AssetId),
    /// Addressed by a live entity handle.
    Entity(Entity),
}

impl Default for Target {
    fn default() -> Self {
        Target::Name(AssetId::default())
    }
}

impl From<AssetId> for Target {
    fn from(name: AssetId) -> Self {
        Target::Name(name)
    }
}

impl From<Entity> for Target {
    fn from(entity: Entity) -> Self {
        Target::Entity(entity)
    }
}

impl Target {
    /// The asset name this target addresses, if it is name-addressed.
    pub fn name(self) -> Option<AssetId> {
        match self {
            Target::Name(name) => Some(name),
            Target::Entity(_) => None,
        }
    }
}
