use crate::components::InputKey;
use crate::settings::SettingKey;

/// What a settings-row change does to its value: cycle one step (for a stepper
/// row), jump to an absolute option index (for a dropdown pick), set an absolute
/// position in [0, 1] (for a slider row's drag), or bind a key (for a key-rebind
/// row).
///
/// Eq is intentionally not derived: `SetFraction` carries an f32.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SettingOp {
    /// Step the value one option forward.
    #[default]
    Next,
    /// Step the value one option back.
    Prev,
    /// Jump straight to this option index (clamped to the option count). Sent
    /// once when the user picks an entry from an open dropdown list.
    SetIndex(usize),
    /// Set the value to this fraction of its range, 0.0..=1.0. Sent each frame
    /// while a slider is dragged.
    SetFraction(f32),
    /// Bind the named action (the command's setting) to this key. Sent once when
    /// the user presses a key while a rebind row is capturing.
    Rebind(InputKey),
    /// Bind the named gamepad action (the command's `pad_*` setting) to this
    /// button. Sent once when the user presses a button while a gamepad rebind
    /// row is capturing.
    RebindButton(crate::components::GamepadButton),
}

/// Runtime-only event sent by `UiInputSystem` when a settings row changes (a
/// step or set on a `setting:` binding, a dropdown pick, a slider drag, or a
/// rebind capture).
/// The renderer reads these each step: it applies the change to the named
/// setting (cycling it or setting it from a fraction), publishes the value it
/// now shows to [`SettingValues`](crate::settings::SettingValues), and (when
/// persist is set) writes the new value to the settings store.
/// World authors never declare this type directly.
#[derive(Debug, Clone)]
pub struct SettingCommand {
    /// The setting to change.
    pub setting: SettingKey,
    /// How to change the value.
    pub op: SettingOp,
    /// Whether to write the new value to the settings store. A cycle is one
    /// discrete change and always persists; a slider drag persists only on
    /// release (the in-progress frames apply live but skip the disk write).
    pub persist: bool,
}
