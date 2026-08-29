// Overlay-screen schema.

use crate::ecs::asset_id::AssetId;
use crate::ecs::asset_id::de_opt_asset_ref;
use alloc::string::String;

/// How a [Screen](#screen) treats input while it is active.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScreenInput {
    /// The screen owns input while it is the topmost capturing screen:
    /// gameplay input is suppressed and lower screens' [HitRegion](#hitregion)s
    /// stop firing.
    #[default]
    Capture,
    /// The screen only draws; input passes through to whatever is beneath it.
    Passthrough,
}

/// A named full-screen layer of UI drawn over the world: a pause menu, a
/// settings page, a console, a score overlay.
///
/// UI elements ([Sprite](#sprite), [TextLabel](#textlabel),
/// [TextInput](#textinput), [HitRegion](#hitregion)) belong to a screen by
/// name prefix `<screen_name>_*`, mirroring the [Scene](#scene) →
/// [Prop](#prop) convention. Active screens form a stack; each is shown /
/// hidden via [HitRegion](#hitregion) or [KeyBinding](#keybinding) actions:
/// - `screen:show:<name>` replaces the top of the stack (menu navigation)
/// - `screen:push:<name>` opens on top of what is already showing
/// - `screen:hide` closes the top screen, revealing what was beneath
/// - `screen:toggle:<name>` closes the screen if it is on top, opens it otherwise
///
/// Screens draw in stack order (later on top); `layer` orders a screen
/// against the always-on HUD and other screens independent of stack position.
/// While any active screen has `pauses_world` set, the world freezes exactly
/// as today's pause menu does. A `toggle_key` opens and closes the screen from
/// anywhere. `focus` names a [TextInput](#textinput) that receives keyboard
/// focus whenever the screen reaches the top of the stack. Worlds that need no
/// menus simply declare no screens.
///
/// ```rust
/// # use concinnity_core::components::Screen;
/// Screen {
///     toggle_key: "Escape".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Screen {
    /// Assigned by the loader; not authored.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// When true, this screen is shown as soon as the world loads.
    pub initial: bool,
    /// Seconds to fade the screen in when it's shown. 0 shows it instantly.
    pub fade_in_secs: f32,
    /// InputKey that toggles this screen open / closed from anywhere, by the same
    /// canonical key names a [KeyBinding](#keybinding) uses (e.g. "Escape",
    /// "Backtick"). Empty leaves the screen action-driven only.
    pub toggle_key: String,
    /// Input policy while the screen is active.
    pub input: ScreenInput,
    /// When true (the default), the world pauses beneath this screen while it
    /// is active: gameplay input, physics, and animation freeze.
    pub pauses_world: bool,
    /// [TextInput](#textinput) that receives keyboard focus whenever this
    /// screen reaches the top of the stack.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub focus: Option<AssetId>,
    /// Draw-order bias against the always-on HUD and other screens. Screens
    /// default above the HUD in stack order; a negative layer draws beneath
    /// the HUD, a higher layer stays above later-pushed screens.
    pub layer: i32,
}

impl Default for Screen {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            initial: false,
            fade_in_secs: 0.0,
            toggle_key: String::new(),
            input: ScreenInput::Capture,
            pauses_world: true,
            focus: None,
            layer: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_screen_captures_input_and_pauses_the_world() {
        // An overlay is modal by default: the world underneath neither ticks nor
        // sees the input the screen is consuming.
        let s = Screen::default();
        assert_eq!(s.input, ScreenInput::Capture);
        assert!(s.pauses_world);
        assert!(!s.initial);
        assert_eq!(s.fade_in_secs, 0.0);
        assert_eq!(s.layer, 0);
        assert!(s.toggle_key.is_empty());
        assert!(s.focus.is_none());
        assert_eq!(ScreenInput::default(), ScreenInput::Capture);
    }

    #[test]
    fn a_passthrough_hud_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let s: Screen = serde_json::from_str(
            r#"{"initial":true,"fade_in_secs":0.5,"toggle_key":"Tab","input":"passthrough",
                "pauses_world":false,"focus":"first_button","layer":-1}"#,
        )
        .unwrap();
        assert_eq!(s.input, ScreenInput::Passthrough);
        assert!(!s.pauses_world);
        assert!(s.initial);
        assert_eq!(s.focus, Some(AssetId(12)));
        assert_eq!(
            serde_json::to_string(&ScreenInput::Passthrough).unwrap(),
            r#""passthrough""#
        );

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: Screen = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.fade_in_secs, 0.5);
        assert_eq!(back.toggle_key, "Tab");
        assert_eq!(back.input, ScreenInput::Passthrough);
        // A negative layer sits below the default overlays.
        assert_eq!(back.layer, -1);
        assert_eq!(back.asset_id, AssetId::default());
    }
}
