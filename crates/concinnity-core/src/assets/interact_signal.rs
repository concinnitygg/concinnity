// src/assets/interact_signal.rs

use crate::ecs::asset_id::AssetId;

/// Runtime-only event published by the camera controller when the interact
/// key fires on an Interactable entity: `target` is the entity's declared
/// name. BehaviorSystem reads these from its `Events<InteractSignal>` queue
/// each step (one tick after the press, since the controller runs later in
/// the schedule). World authors never declare this type directly.
#[derive(Debug, Clone, Copy)]
pub struct InteractSignal {
    /// The interacted entity's declared name.
    pub target: AssetId,
}
