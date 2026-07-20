// src/input/mod.rs
//
// Gamepad input for InputSystem. `gamepad` folds backend-agnostic pad events
// into a per-frame snapshot (with the OS-facing gilrs adapter isolated at its
// edge); `stick` is the pure deadzone / response-curve math.

pub(crate) mod gamepad;
pub(crate) mod stick;
