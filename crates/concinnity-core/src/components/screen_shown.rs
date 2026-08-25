// src/components/screen_shown.rs

use crate::ecs::asset_id::AssetId;

/// Runtime-only event sent by UiInputSystem whenever a screen reaches the top
/// of the stack: the initial screen at world start, a `screen:show` /
/// `screen:push` / `screen:toggle` navigation, or a pop revealing the screen
/// beneath. Read by AudioSystem to fire AudioCues. World authors never declare
/// this type directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScreenShown {
    /// The screen that reached the top of the stack.
    pub screen: AssetId,
}
