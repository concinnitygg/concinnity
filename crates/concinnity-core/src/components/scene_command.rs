// src/components/scene_command.rs

use crate::components::SceneTransition;
use crate::ecs::asset_id::AssetId;

/// Runtime-only event sent by UiInputSystem when a scene-jump HitRegion fires.
/// GraphicsSystem reads these from its `Events<SceneCommand>` queue each step and
/// applies the scene jump. World authors never declare this type directly.
#[derive(Debug, Clone, Default)]
pub struct SceneCommand {
    /// The scene to jump to.
    pub scene: AssetId,
    /// Transition to play across the jump.
    pub transition: SceneTransition,
}
