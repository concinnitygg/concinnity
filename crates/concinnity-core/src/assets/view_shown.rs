// src/assets/view_shown.rs

use crate::ecs::asset_id::AssetId;

// Runtime-only event sent by UiInputSystem whenever a view becomes the active
// view: the initial view at world start, a `view:show` / `view:toggle`
// navigation, or a dismissal restoring the previous view. Read by AudioSystem
// to fire AudioCues. World authors never declare this type directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct ViewShown {
    pub view: AssetId,
}
