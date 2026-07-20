// src/input/mod.rs
//
// Gamepad input for InputSystem. `gamepad` folds backend-agnostic pad events
// into a per-frame snapshot (with the OS-facing gilrs adapter isolated at its
// edge); `stick` is the pure deadzone / response-curve math; `nav` shapes the
// held d-pad + stick state into auto-repeating UI navigation pulses.

pub(crate) mod gamepad;
pub(crate) mod nav;
pub(crate) mod stick;
