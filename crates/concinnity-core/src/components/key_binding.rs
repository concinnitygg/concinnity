// InputKey-to-action binding schema.

use crate::components::UiAction;
use crate::ecs::asset_id::AssetId;
use crate::ecs::asset_id::de_opt_asset_ref;
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
/// The action vocabulary is the same as [HitRegion](#hitregion)'s:
/// - `"quit"`:                 stop the application
/// - `"scene:<name>"`:         jump to the named [Scene](#scene)
/// - `"screen:show:<name>"`:   show the named [Screen](#screen), replacing the top of the stack
/// - `"screen:push:<name>"`:   open the named [Screen](#screen) on top of what is showing
/// - `"screen:toggle:<name>"`: toggle the named [Screen](#screen)
/// - `"screen:hide"`:          close the top [Screen](#screen)
/// - `"story:<verb>"`:         drive the story
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct KeyBinding {
    /// The key name to bind (e.g. `"Escape"`).
    pub key: String,
    /// The action to fire when the key is pressed. Empty fires nothing.
    #[serde(with = "crate::components::ui_action::optional")]
    pub action: Option<UiAction>,
    /// [Screen](#screen) this binding is scoped to: the binding only fires
    /// while that screen is on top of the stack. Unset, the binding is global.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub screen: Option<AssetId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::ScreenCommand;

    #[test]
    fn a_binding_with_no_screen_is_global() {
        let b = KeyBinding::default();
        assert!(b.key.is_empty());
        assert!(b.action.is_none());
        assert!(b.screen.is_none());
    }

    #[test]
    fn a_screen_scoped_binding_parses_and_round_trips_through_postcard() {
        let b: KeyBinding = crate::test_support::from_json(
            r#"{"key":"Escape","action":"screen:hide","screen":"menu"}"#,
        );
        assert_eq!(b.key, "Escape");
        assert_eq!(b.action, Some(UiAction::Screen(ScreenCommand::Hide)));
        assert_eq!(b.screen, Some(AssetId(4)));

        let bytes = postcard::to_allocvec(&b).unwrap();
        let back: KeyBinding = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.key, "Escape");
        assert_eq!(back.screen, Some(AssetId(4)));
        assert_eq!(back.action, b.action);
    }
}
