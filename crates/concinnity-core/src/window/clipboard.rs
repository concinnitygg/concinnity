//! The system clipboard's plain-text slot, as a window layer exposes it.

use alloc::string::String;

/// The operating system's clipboard, read and written as plain text.
///
/// A window layer implements it over the platform's own clipboard. Text read
/// back keeps whatever line endings the source application wrote; a caller
/// that needs one convention normalizes it.
pub trait Clipboard {
    /// The clipboard's text, or `None` when it holds no text.
    fn text(&mut self) -> Option<String>;
    /// Replace the clipboard's contents with `text`.
    fn set_text(&mut self, text: &str);
}
