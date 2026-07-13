// Overlay-view schema.

use crate::AssetId;

/// A named overlay layer drawn on top of the active [Scene](#scene).
///
/// UI elements ([Sprite](#sprite), [TextLabel](#textlabel),
/// [HitRegion](#hitregion)) belong to a view by name prefix `<view_name>_*`,
/// mirroring the [Scene](#scene) → [Prop](#prop) convention. Views are shown /
/// hidden via [HitRegion](#hitregion) or [KeyBinding](#keybinding) actions:
/// - `view:show:<name>`
/// - `view:hide`
/// - `view:toggle:<name>`
///
/// When a view is active, its UI elements become visible and the underlying
/// scene's [HitRegion](#hitregion)s stop firing. Hiding the view restores the
/// scene exactly as it was. Only one view can be active at a time.
///
/// ```jsonl
/// {"name":"pause_menu","type":"View","args":{}}
/// // UI assets prefixed pause_menu_* belong to this view:
/// {"name":"pause_menu_dim","type":"Sprite","args":{"x":0,"y":0,"width":1280,"height":720,"tint":[0,0,0,0.55]}}
/// {"name":"pause_menu_btn_resume","type":"HitRegion","args":{"action":"view:hide", ...}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct View {
    #[serde(skip)]
    pub asset_id: AssetId,
    /// When true, this view is shown as soon as the world loads.
    pub initial: bool,
    /// Seconds to fade the view in when it's shown. 0 shows it instantly.
    pub fade_in_secs: f32,
}
