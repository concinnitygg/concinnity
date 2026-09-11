// src/input/mod.rs
//
// Input for the engine: the two systems that consume a frame's raw input, and
// the gamepad vocabulary they share. `gamepad` folds backend-agnostic pad
// events into a per-frame snapshot (with the OS-facing gilrs adapter isolated
// at its edge); `stick` is the pure deadzone / response-curve math; `nav`
// shapes the held d-pad + stick state into auto-repeating UI navigation pulses.

pub(crate) mod gamepad;
pub(crate) mod nav;
pub(crate) mod stick;

// Per-frame input sampling + FrameInput publish. Internal system, constructed
// alongside GraphicsSystem (same gate) and scheduled immediately after it.
pub(crate) mod system;
// Editable TextInput field drive: focus, character entry, caret editing.
pub(crate) mod text_system;
