// src/assets/spawn_request.rs

use crate::assets::Transform;
use crate::ecs::asset_id::AssetId;

/// Runtime-only event requesting that a copy of an existing placement be created
/// in the world at runtime. The symmetric counterpart to DespawnRequest.
///
/// `template` names a placement already present in the world; the new instance
/// reuses that placement's geometry and material at a fresh `transform`. A
/// `name` registers the instance so it can later be addressed (despawned,
/// reparented) like any authored placement; `None` spawns a transient copy,
/// like a Spawner's cadence spawns. An optional `lifetime_secs` attaches a
/// Lifetime so the instance auto-despawns after that many seconds, the churn
/// that lets freed draw slots be recycled. GraphicsSystem reads these from its
/// `Events<SpawnRequest>` queue each step. World authors never declare this type
/// directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpawnRequest {
    /// The placement to copy geometry and material from.
    pub template: AssetId,
    /// Name to register the instance under, or `None` for a transient copy.
    pub name: Option<AssetId>,
    /// Where to place the new instance.
    pub transform: Transform,
    /// Seconds before the instance auto-despawns, when set.
    pub lifetime_secs: Option<f32>,
}
