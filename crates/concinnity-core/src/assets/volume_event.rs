// src/assets/volume_event.rs

use crate::ecs::asset_id::AssetId;

/// Runtime-only event published by the physics system when something crossing
/// a TriggerVolume boundary passes the volume's `detects` filter. `entered` is
/// true on the way in and false on the way out. BehaviorSystem reads these
/// from its `Events<VolumeEvent>` queue each step (one tick after the crossing,
/// since physics runs later in the schedule). World authors never declare
/// this type directly.
#[derive(Debug, Clone, Copy)]
pub struct VolumeEvent {
    /// The TriggerVolume that was crossed.
    pub volume: AssetId,
    /// `true` on the way in, `false` on the way out.
    pub entered: bool,
}
