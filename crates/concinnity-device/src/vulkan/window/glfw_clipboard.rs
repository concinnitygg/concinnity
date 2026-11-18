// The Linux system clipboard, reached through the GLFW window (GLFW owns the
// X11 selection / Wayland data-device plumbing).

use concinnity_engine::window::clipboard::Clipboard;

use super::glfw::GlfwWindow;

impl Clipboard for GlfwWindow {
    fn text(&mut self) -> Option<String> {
        self.window.get_clipboard_string()
    }

    fn set_text(&mut self, text: &str) {
        self.window.set_clipboard_string(text);
    }
}

impl GlfwWindow {
    pub(crate) fn clipboard(&mut self) -> &mut dyn Clipboard {
        self
    }

    // GLFW reads and writes the clipboard only on the thread that runs its
    // event loop, so no handle to it may leave the window.
    #[expect(
        clippy::unused_self,
        reason = "every platform window answers alike, and only this one has none"
    )]
    pub(crate) fn shared_clipboard(&self) -> Option<Box<dyn Clipboard + Send + Sync>> {
        None
    }
}
