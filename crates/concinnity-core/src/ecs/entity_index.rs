// src/ecs/entity_index.rs
//
// The name -> Entity index published by the load-time Prop decomposition pass.
// The pass itself (`decompose::run`) lives in the client crate, but the index it
// produces is renderer-free, so it lives here where the physics / audio subsystem
// crates can resolve a name reference (a Prop parent, a PropBody owner, an audio
// emitter target) to an Entity without depending on the renderer. The client
// `ecs::decompose` module re-exports it under the historical
// `crate::ecs::decompose::EntityByName` path.

use crate::ecs::Entity;
use crate::ecs::asset_id::AssetId;
use alloc::collections::BTreeMap;

/// Maps a placement's asset identity (its declared name) to the live Entity it
/// was loaded into. Built by the decomposition pass so later passes can resolve
/// a name reference (a Prop parent, a PropBody owner, an audio emitter target)
/// to an Entity without scanning.
#[derive(Debug, Default)]
pub struct EntityByName(pub BTreeMap<AssetId, Entity>);

impl EntityByName {
    /// The entity a name was loaded into, if the name is known.
    pub fn get(&self, name: AssetId) -> Option<Entity> {
        self.0.get(&name).copied()
    }
}
