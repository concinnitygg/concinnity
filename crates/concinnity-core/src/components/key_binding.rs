// InputKey-to-action binding schema.

use crate::components::Screen;
use crate::components::UiAction;
use crate::ecs::Ref;
use alloc::string::String;

/// Maps a keyboard key to an action.
///
/// When the bound key is pressed, the action fires once per press (like a
/// [HitRegion](#hitregion) click). Bindings only run while the cursor is free:
/// they're inactive in worlds that capture the cursor for camera control.
/// While a [TextInput](#textinput) has keyboard focus, bindings are suspended
/// so typing cannot trigger actions; a [Screen](#screen)'s `toggle_key` stays
/// live.
///
/// InputKey names are case-sensitive canonical names (e.g. `"Escape"`, `"Space"`,
/// `"Enter"`).
///
/// ```rust
/// # use concinnity_core::components::{KeyBinding, ScreenCommand, UiAction};
/// # use concinnity_core::ecs::asset_id::AssetId;
/// KeyBinding {
///     key: "Escape".into(),
///     action: Some(UiAction::Screen(ScreenCommand::Toggle(AssetId(3)))),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct KeyBinding {
    /// The key name to bind (e.g. `"Escape"`).
    pub key: String,
    /// What a press of the key fires. Unset fires nothing.
    pub action: Option<UiAction>,
    /// [Screen](#screen) this binding is scoped to: the binding only fires
    /// while that screen is on top of the stack. Unset, the binding is global.
    pub screen: Option<Ref<Screen>>,
}
