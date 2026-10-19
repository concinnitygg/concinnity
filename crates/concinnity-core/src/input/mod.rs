//! Backend-agnostic input: the per-frame snapshot a backend reports, the
//! rebindable key map it decodes against, the gamepad fold and stick shaping,
//! navigation pulses, and the text-field editor.

pub mod compose;
pub mod keymap;
pub mod mailbox;
pub mod nav;
pub mod pad;
pub mod snapshot;
pub mod stick;
pub mod text_input;
