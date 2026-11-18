//! The clipboard a text area uses: the window's system clipboard, or an
//! in-memory one when the window reaches none (a headless run, or a backend
//! without one), so text copied inside the editor still pastes inside it.

use concinnity_engine::ecs::World;
use concinnity_engine::presentation::text_edit::LocalClipboard;
use concinnity_engine::window::clipboard::Clipboard;

// The window's system clipboard, or `fallback` when it reaches none.
pub(crate) fn system_or<'a>(
    world: &'a mut World,
    fallback: &'a mut LocalClipboard,
) -> &'a mut dyn Clipboard {
    let system = crate::debug::handoff::render_handoff(world)
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
        let mut fallback = LocalClipboard::default();
        system_or(&mut world, &mut fallback).set_text("kept");
        assert_eq!(fallback.text().as_deref(), Some("kept"));
    }
}
