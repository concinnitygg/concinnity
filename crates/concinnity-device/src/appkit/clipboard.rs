// The macOS system clipboard: the general pasteboard's plain-string slot.

use concinnity_core::window::clipboard::Clipboard;
use objc2_app_kit::{NSPasteboard, NSPasteboardTypeString};
use objc2_foundation::NSString;

#[derive(Debug, Default)]
pub(crate) struct SystemClipboard;

impl Clipboard for SystemClipboard {
    fn text(&mut self) -> Option<String> {
        let board = NSPasteboard::generalPasteboard();
        // SAFETY: `NSPasteboardTypeString` is an immutable NSString constant AppKit
        // initializes before any code can run; reading the static only copies the reference.
        let kind = unsafe { NSPasteboardTypeString };
        board.stringForType(kind).map(|s| s.to_string())
    }

    fn set_text(&mut self, text: &str) {
        let board = NSPasteboard::generalPasteboard();
        // SAFETY: as in `text`, the constant is initialized before any code runs.
        let kind = unsafe { NSPasteboardTypeString };
        board.clearContents();
        if !board.setString_forType(&NSString::from_str(text), kind) {
            tracing::warn!("the system clipboard refused the copied text");
        }
    }
}
