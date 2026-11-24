//! The editor's in-engine mouse cursor: a single `follow_cursor` Sprite the
//! overlay draws as an arrow, or as a resize cursor over a resizable panel's edge
//! (the hook publishes the shape through the `DesiredCursor` resource). While it
//! is visible the overlay hides the OS cursor for it (its `want_ui_cursor` path),
//! so the pointer stays managed in-engine and looks identical on every backend.

use concinnity_engine::components::Sprite;
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use crate::editor::ui::UiBuilder;

// Cursor height in reference pixels; the overlay scales it for the viewport.
const CURSOR_PX: f32 = 20.0;
// A light fill so the arrow reads over the dark editor chrome and the scene; the
// overlay picks a contrasting outline for it.
const FILL: [f32; 4] = [0.95, 0.95, 0.97, 1.0];

// The cursor sprite, hidden until the tick shows it.
pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<AssetId> {
    ui.sprite(Sprite {
        follow_cursor: true,
        width: CURSOR_PX,
        height: CURSOR_PX,
        tint: FILL,
        visible: false,
        ..Default::default()
    })
}

// Show or hide the cursor for this frame: shown while the editor owns the
// pointer (edit mode), hidden while a captured camera owns it so no stray arrow
// lingers over the frozen pointer.
pub(crate) fn set_visible(world: &mut World, cursor: AssetId, visible: bool) {
    if let Some(s) = world.get_mut_by_id::<Sprite>(cursor) {
        s.visible = visible;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cursor_starts_hidden_and_follows_the_tick() {
        let mut world = World::new();
        let id = build(&mut UiBuilder::new(&mut world, AssetId(9), None)).unwrap();
        let shown = |world: &World| world.get_by_id::<Sprite>(id).unwrap().visible;
        assert!(!shown(&world));
        set_visible(&mut world, id, true);
        assert!(shown(&world));
        assert!(world.get_by_id::<Sprite>(id).unwrap().follow_cursor);
    }
}
