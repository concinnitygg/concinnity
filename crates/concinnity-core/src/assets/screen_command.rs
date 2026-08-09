// src/assets/screen_command.rs

use crate::ecs::asset_id::AssetId;

// Runtime-only event sent by UiInputSystem when a `screen:*` action fires.
// UiInputSystem reads these from its Events<ScreenCommand> queue on the next
// step and applies the stack transition. World authors never declare this type
// directly.
#[derive(Debug, Clone, Default)]
pub enum ScreenCommand {
    // Replace the top of the stack (menu navigation).
    Show(AssetId),
    // Pop the top of the stack, revealing what was beneath.
    #[default]
    Hide,
    // Pop if the screen is topmost, push it otherwise.
    Toggle(AssetId),
    // Push on top of whatever is already showing.
    Push(AssetId),
    // Empty the stack (a scene change dismisses every open screen).
    Clear,
}
