// Key-to-action binding schema.

use crate::{AssetId, de_opt_asset_ref};
use alloc::string::String;

/// Maps a keyboard key to an action string.
///
/// When the bound key is pressed, the action fires once per press (like a
/// [HitRegion](#hitregion) click). Bindings only run while the cursor is free:
/// they're inactive in worlds that capture the cursor for camera control.
/// While a [TextInput](#textinput) has keyboard focus, bindings are suspended
/// so typing cannot trigger actions; a [Screen](#screen)'s `toggle_key` stays
/// live.
///
/// The action vocabulary is the same as [HitRegion](#hitregion)'s:
/// - `"scene:<name>"`:         jump to the named [Scene](#scene)
/// - `"quit"`:                 stop the application
/// - `"screen:show:<name>"`:   show the named [Screen](#screen), replacing the top of the stack
/// - `"screen:push:<name>"`:   open the named [Screen](#screen) on top of what is showing
/// - `"screen:hide"`:          close the top [Screen](#screen)
/// - `"screen:toggle:<name>"`: toggle the named [Screen](#screen)
///
/// Key names are case-sensitive canonical names (e.g. `"Escape"`, `"Space"`,
/// `"Enter"`).
///
/// ```jsonl
/// {"name":"esc_binding","type":"KeyBinding","args":{"key":"Escape","action":"screen:toggle:pause_menu"}}
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct KeyBinding {
    /// The key name to bind (e.g. `"Escape"`).
    pub key: String,
    /// The action to fire when the key is pressed.
    pub action: String,
    /// [Screen](#screen) this binding is scoped to: the binding only fires
    /// while that screen is on top of the stack. Unset, the binding is global.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub screen: Option<AssetId>,
}
