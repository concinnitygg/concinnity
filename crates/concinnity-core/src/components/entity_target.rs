use crate::ecs::Entity;
use crate::ecs::asset_id::AssetId;

/// How a runtime request addresses the entity it acts on. Authored placements
/// are named, so a producer holding no live handle addresses them by asset name
/// and the applying system resolves it. Logic that already holds an entity --
/// one it spawned, one a query yielded -- addresses it directly, which is the
/// only way to reach an entity that never had a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityTarget {
    /// Addressed by the placement's authored name.
    Name(AssetId),
    /// Addressed by a live entity handle.
    Entity(Entity),
}

impl Default for EntityTarget {
    fn default() -> Self {
        Self::Name(AssetId::default())
    }
}

impl From<AssetId> for EntityTarget {
    fn from(name: AssetId) -> Self {
        Self::Name(name)
    }
}

impl From<Entity> for EntityTarget {
    fn from(entity: Entity) -> Self {
        Self::Entity(entity)
    }
}

impl EntityTarget {
    /// The asset name this target addresses, if it is name-addressed.
    pub fn name(self) -> Option<AssetId> {
        match self {
            Self::Name(name) => Some(name),
            Self::Entity(_) => None,
        }
    }
}
