//! Backend-agnostic input: the per-frame snapshot a backend reports, the
//! rebindable key map it decodes against, the gamepad fold and stick shaping,
//! navigation pulses, and the text-field editor.

pub mod compose;
pub(crate) mod controls_command;
pub(crate) mod frame_input;
pub(crate) mod gamepad_button;
pub(crate) mod gamepad_map;
pub(crate) mod input_key;
pub(crate) mod key_binding;
pub(crate) mod key_event;
pub mod keymap;
pub mod mailbox;
pub mod nav;
pub(crate) mod nav_direction;
pub mod pad;
pub mod snapshot;
pub mod stick;
pub mod system;
pub mod text_input;
