// src/assets/gamepad_map.rs

use crate::assets::GamepadButton;

/// A rebindable gamepad action. Movement and look come from the sticks (with
/// the d-pad as a digital movement fallback), so only the button-driven actions
/// are rebindable; pause (Start) carries menu semantics and stays fixed, like
/// Escape on the keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamepadAction {
    /// Hold to move faster.
    Sprint,
    /// Jump.
    Jump,
    /// Interact with the entity under the cursor.
    Interact,
}

impl GamepadAction {
    /// Every rebindable action, in Controls-tab row order.
    pub const ALL: [GamepadAction; 3] = [
        GamepadAction::Sprint,
        GamepadAction::Jump,
        GamepadAction::Interact,
    ];

    /// The settings key string used in `setting:<key>:rebind` actions and the
    /// engine settings registry. The `pad_` prefix distinguishes a button
    /// capture row from a `key_*` keyboard capture row.
    pub fn setting_key(self) -> &'static str {
        match self {
            GamepadAction::Sprint => "pad_sprint",
            GamepadAction::Jump => "pad_jump",
            GamepadAction::Interact => "pad_interact",
        }
    }

    /// The action for a settings key string, or `None` if it is not a gamepad
    /// rebind key.
    pub fn from_setting_key(key: &str) -> Option<GamepadAction> {
        GamepadAction::ALL
            .into_iter()
            .find(|a| a.setting_key() == key)
    }
}

/// The canonical action -> gamepad button map. Persisted in the engine's
/// controls settings and applied by the input sampling; carried live to
/// consumers via [ControlsCommand](#controlscommand) on a rebind. Each field is
/// `#[serde(default)]` so adding an action in a future build never invalidates
/// an existing settings file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GamepadMap {
    /// Held to sprint while moving.
    #[serde(default = "def_sprint")]
    pub sprint: GamepadButton,
    /// One-frame jump press.
    #[serde(default = "def_jump")]
    pub jump: GamepadButton,
    /// One-frame interact press.
    #[serde(default = "def_interact")]
    pub interact: GamepadButton,
}

impl GamepadMap {
    /// The default bindings: sprint on the left-stick click, jump on the bottom
    /// face button, interact on the left face button.
    pub const DEFAULT: GamepadMap = GamepadMap {
        sprint: GamepadButton::LeftStick,
        jump: GamepadButton::South,
        interact: GamepadButton::West,
    };

    /// The button currently bound to an action.
    pub fn get(self, action: GamepadAction) -> GamepadButton {
        match action {
            GamepadAction::Sprint => self.sprint,
            GamepadAction::Jump => self.jump,
            GamepadAction::Interact => self.interact,
        }
    }

    /// Bind an action to a button directly (no conflict handling).
    pub fn set(&mut self, action: GamepadAction, button: GamepadButton) {
        match action {
            GamepadAction::Sprint => self.sprint = button,
            GamepadAction::Jump => self.jump = button,
            GamepadAction::Interact => self.interact = button,
        }
    }

    /// The action a button is bound to, or `None` if unbound. The map keeps
    /// each button bound to at most one action (the invariant [rebind](#method.rebind)
    /// maintains), so this is the unique holder.
    pub fn action_for_button(self, button: GamepadButton) -> Option<GamepadAction> {
        GamepadAction::ALL
            .into_iter()
            .find(|&a| self.get(a) == button)
    }

    /// Bind `action` to `new_button`, swapping with whichever action already
    /// holds `new_button` so every action stays bound. Rebinding an action to
    /// its own button is a no-op.
    pub fn rebind(&mut self, action: GamepadAction, new_button: GamepadButton) {
        let old_button = self.get(action);
        if old_button == new_button {
            return;
        }
        if let Some(other) = self.action_for_button(new_button)
            && other != action
        {
            self.set(other, old_button);
        }
        self.set(action, new_button);
    }
}

impl Default for GamepadMap {
    fn default() -> Self {
        Self::DEFAULT
    }
}

fn def_sprint() -> GamepadButton {
    GamepadMap::DEFAULT.sprint
}
fn def_jump() -> GamepadButton {
    GamepadMap::DEFAULT.jump
}
fn def_interact() -> GamepadButton {
    GamepadMap::DEFAULT.interact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setting_key_round_trips() {
        for a in GamepadAction::ALL {
            assert_eq!(GamepadAction::from_setting_key(a.setting_key()), Some(a));
        }
        assert_eq!(GamepadAction::from_setting_key("key_jump"), None);
        assert_eq!(GamepadAction::from_setting_key("pad_nope"), None);
    }

    #[test]
    fn get_set_cover_every_action_arm() {
        let buttons = [
            GamepadButton::North,
            GamepadButton::East,
            GamepadButton::RightShoulder,
        ];
        let mut m = GamepadMap::default();
        for (a, b) in GamepadAction::ALL.into_iter().zip(buttons) {
            m.set(a, b);
        }
        for (a, b) in GamepadAction::ALL.into_iter().zip(buttons) {
            assert_eq!(m.get(a), b);
        }
    }

    #[test]
    fn rebind_to_free_button_just_sets_it() {
        let mut m = GamepadMap::default();
        m.rebind(GamepadAction::Jump, GamepadButton::North);
        assert_eq!(m.jump, GamepadButton::North);
        assert_eq!(m.interact, GamepadMap::DEFAULT.interact);
    }

    #[test]
    fn rebind_to_own_button_is_a_noop() {
        let mut m = GamepadMap::default();
        m.rebind(GamepadAction::Jump, GamepadMap::DEFAULT.jump);
        assert_eq!(m, GamepadMap::default());
    }

    #[test]
    fn rebind_to_occupied_button_swaps() {
        // Bind Jump to West, which Interact holds: they swap, so Interact
        // inherits Jump's old button and every action stays bound.
        let mut m = GamepadMap::default();
        m.rebind(GamepadAction::Jump, GamepadButton::West);
        assert_eq!(m.jump, GamepadButton::West);
        assert_eq!(m.interact, GamepadMap::DEFAULT.jump);
        for a in GamepadAction::ALL {
            assert_eq!(m.action_for_button(m.get(a)), Some(a));
        }
    }

    #[test]
    fn missing_field_falls_back_to_default() {
        // A settings file predating a field still loads, the missing field
        // falling back through its `serde(default = "def_*")` helper.
        let partial: GamepadMap = serde_json::from_str(r#"{"jump":"East"}"#).unwrap();
        assert_eq!(partial.jump, GamepadButton::East);
        assert_eq!(partial.sprint, GamepadMap::DEFAULT.sprint);
        assert_eq!(partial.interact, GamepadMap::DEFAULT.interact);
    }
}
