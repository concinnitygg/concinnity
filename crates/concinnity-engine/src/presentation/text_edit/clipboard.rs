// A clipboard held in memory, for text editing without a system clipboard.

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use crate::window::clipboard::Clipboard;

/// A clipboard kept in memory: text copied from one text editor pastes into
/// another, without reaching the operating system's clipboard. What a text
/// editor uses when its window offers no system clipboard (a headless run,
/// or a backend without one).
#[derive(Debug, Clone, Default)]
pub struct LocalClipboard {
    text: Option<String>,
}

impl Clipboard for LocalClipboard {
    fn text(&mut self) -> Option<String> {
        self.text.clone()
    }

    fn set_text(&mut self, text: &str) {
        self.text = Some(text.to_string());
    }
}

/// The operating system's clipboard, which a host lends the engine's own text
/// editors as a resource. Without one they share a [`LocalClipboard`], so
/// copying and pasting still works between them.
pub struct HostClipboard(pub Box<dyn Clipboard + Send + Sync>);

impl core::fmt::Debug for HostClipboard {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("HostClipboard")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_the_last_copy() {
        let mut c = LocalClipboard::default();
        assert_eq!(c.text(), None);
        c.set_text("one");
        c.set_text("two");
        assert_eq!(c.text().as_deref(), Some("two"));
    }
}
