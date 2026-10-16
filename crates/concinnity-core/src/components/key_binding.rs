// InputKey-to-action binding schema.

use crate::components::{KeyMods, Screen, UiAction};
use crate::ecs::Ref;
use alloc::string::String;

/// Maps a keyboard key to an action.
///
/// When the bound key is pressed, the action fires once per press (like a
/// [HitRegion](#hitregion) click), whether or not the camera holds the cursor.
/// While a [TextInput](#textinput) has keyboard focus, bindings are suspended
/// so typing cannot trigger actions; a [Screen](#screen)'s `toggle_key` stays
/// live.
///
/// InputKey names are case-sensitive canonical names (e.g. `"Escape"`, `"Space"`,
/// `"Enter"`).
///
/// A binding with `mods` fires only while exactly those modifiers are held, and
/// wins over a binding of the same key without them; a binding without `mods`
/// fires whatever is held. So `P` and Shift+`P` can do different things.
///
/// ```rust
/// # use concinnity_core::components::{KeyBinding, KeyMods, UiAction};
/// let shot = KeyBinding {
///     key: "P".into(),
///     action: Some(UiAction::Screenshot(1)),
///     ..Default::default()
/// };
/// let burst = KeyBinding {
///     key: "P".into(),
///     mods: Some(KeyMods::SHIFT),
///     action: Some(UiAction::Screenshot(8)),
///     ..Default::default()
/// };
/// ```
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
    /// The modifiers that must be held, exactly. Unset, any are allowed.
    pub mods: Option<KeyMods>,
    /// What a press of the key fires. Unset fires nothing.
    pub action: Option<UiAction>,
    /// [Screen](#screen) this binding is scoped to: the binding only fires
    /// while that screen is on top of the stack. Unset, the binding is global.
    pub screen: Option<Ref<Screen>>,
}
