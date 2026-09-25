// The Linux system clipboard, reached through the GLFW window (GLFW owns the
// X11 selection / Wayland data-device plumbing).

use concinnity_core::window::clipboard::Clipboard;

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
    pub(crate) fn clipboard(&mut self) -> Option<&mut dyn Clipboard> {
        Some(self)
    }
}
