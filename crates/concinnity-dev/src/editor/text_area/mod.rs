//! The editor's code text areas: an engine `TextEditor` drawn into a rect of
//! the HUD from reserved ids (`layout`), with the gutter markers a panel shows
//! beside it and the clipboard it falls back to without a system one.

pub(crate) mod clipboard;
pub(crate) mod layout;
pub(crate) mod markers;

use concinnity_engine::presentation::text_edit::KeyConvention;

// The key conventions of the platform the editor runs on.
pub(crate) const fn key_convention() -> KeyConvention {
    if cfg!(target_os = "macos") {
        KeyConvention::Mac
    } else {
        KeyConvention::Other
    }
}
