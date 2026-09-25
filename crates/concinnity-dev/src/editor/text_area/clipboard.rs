//! The clipboard the text area falls back to when the window reaches no system
//! clipboard (a headless run, or a backend without one): text copied inside the
//! editor still pastes inside the editor.

use concinnity_core::ecs::World;
use concinnity_core::window::clipboard::Clipboard;

#[derive(Debug, Default)]
pub(crate) struct InternalClipboard {
    text: Option<String>,
}

impl Clipboard for InternalClipboard {
    fn text(&mut self) -> Option<String> {
        self.text.clone()
    }

    fn set_text(&mut self, text: &str) {
        self.text = Some(text.to_string());
    }
}

// The window's system clipboard, or `fallback` when it reaches none.
pub(crate) fn system_or<'a>(
    world: &'a mut World,
    fallback: &'a mut InternalClipboard,
) -> &'a mut dyn Clipboard {
    let system = concinnity_engine::ecs::render_handoff(world)
        .backend
        .and_then(|backend| backend.clipboard());
    match system {
        Some(clipboard) => clipboard,
        None => fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A world with no render backend reaches no system clipboard.
    #[test]
    fn a_backendless_world_falls_back_to_the_internal_clipboard() {
        let mut world = World::new();
        let mut fallback = InternalClipboard::default();
        system_or(&mut world, &mut fallback).set_text("kept");
        assert_eq!(fallback.text().as_deref(), Some("kept"));
    }

    #[test]
    fn holds_the_last_copy() {
        let mut c = InternalClipboard::default();
        assert_eq!(c.text(), None);
        c.set_text("one");
        c.set_text("two");
        assert_eq!(c.text().as_deref(), Some("two"));
    }
}
