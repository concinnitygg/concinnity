// src/assets/scene_command.rs

use alloc::string::String;
use alloc::string::ToString;

use crate::ecs::asset_id::AssetId;

/// Runtime-only event sent by UiInputSystem when a scene-jump HitRegion fires.
/// GraphicsSystem reads these from its `Events<SceneCommand>` queue each step and
/// applies the scene jump. World authors never declare this type directly.
#[derive(Debug, Clone)]
pub struct SceneCommand {
    /// The scene to jump to.
    pub scene: AssetId,
    /// Named transition to play across the jump.
    pub transition: String,
}

impl Default for SceneCommand {
    fn default() -> Self {
        Self {
            scene: AssetId::default(),
            transition: "FadeBlack".to_string(),
        }
    }
}
