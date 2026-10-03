// Overlay-screen schema.

use crate::components::TextInput;
use crate::components::Vocabulary;
use crate::ecs::Ref;
use alloc::string::String;

/// How a [Screen](#screen) treats input while it is active.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum ScreenInput {
    /// The screen owns input while it is the topmost capturing screen:
    /// gameplay input is suppressed and lower screens' [HitRegion](#hitregion)s
    /// stop firing.
    #[default]
    #[vocab("capture")]
    Capture,
    /// The screen only draws; input passes through to whatever is beneath it.
    #[vocab("passthrough")]
    Passthrough,
}

/// A named full-screen layer of UI drawn over the world: a pause menu, a
/// settings page, a console, a score overlay.
///
/// A UI element ([Sprite](#sprite), [TextLabel](#textlabel),
/// [TextInput](#textinput), [HitRegion](#hitregion)) belongs to a screen by
/// naming it in the element's `screen`, mirroring the [Scene](#scene) →
/// [Prop](#prop) relationship. Active screens form a stack; each is shown /
/// hidden via [HitRegion](#hitregion) or [KeyBinding](#keybinding) actions:
/// - `{"show": "<name>"}` replaces the top of the stack (menu navigation)
/// - `{"push": "<name>"}` opens on top of what is already showing
/// - `"hide"` closes the top screen, revealing what was beneath
/// - `{"toggle": "<name>"}` closes the screen if it is on top, opens it otherwise
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
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Screen {
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
    #[asset(default = true)]
    pub pauses_world: bool,
    /// [TextInput](#textinput) that receives keyboard focus whenever this
    /// screen reaches the top of the stack.
    pub focus: Option<Ref<TextInput>>,
    /// Draw-order bias against the always-on HUD and other screens. Screens
    /// default above the HUD in stack order; a negative layer draws beneath
    /// the HUD, a higher layer stays above later-pushed screens.
    pub layer: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_names_parse_in_lowercase() {
        let i = |s: &str| serde_json::from_str::<ScreenInput>(s).unwrap();
        assert_eq!(i(r#""capture""#), ScreenInput::Capture);
        assert_eq!(i(r#""passthrough""#), ScreenInput::Passthrough);
        assert_eq!(
            serde_json::to_string(&ScreenInput::Passthrough).unwrap(),
            r#""passthrough""#
        );
    }
}
