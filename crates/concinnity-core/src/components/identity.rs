use crate::ecs::asset_id::AssetId;

/// Runtime-only: the asset this entity was built from. The loader gives one to
/// every entity it mints from a blob def, and anything that creates an asset at
/// runtime gives one through `PipelineContext::identify`, which also indexes it
/// in [`EntityById`](crate::ecs::EntityById). Never authored directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity(pub(crate) AssetId);

impl Identity {
    /// The asset id this entity carries.
    pub fn id(self) -> AssetId {
        self.0
    }
}
