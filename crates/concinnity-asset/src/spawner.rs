// Spawner authoring schema. The runtime `Spawner` component (with its spawn
// accumulator) lives in core.

use crate::AssetId;

/// Authored fields of a [`Spawner`]; the runtime accumulator is not declared.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SpawnerArgs {
    /// Name of the placement to copy on each spawn.
    pub template: AssetId,
    /// Seconds between spawns.
    pub interval: f32,
    /// Seconds each spawned copy lives before auto-removal; 0 keeps it forever.
    pub lifetime: f32,
}

impl Default for SpawnerArgs {
    fn default() -> Self {
        Self {
            template: AssetId::default(),
            interval: 1.0,
            lifetime: 0.0,
        }
    }
}
