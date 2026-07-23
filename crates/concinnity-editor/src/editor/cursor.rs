// src/editor/cursor.rs
//
// The editor's in-engine mouse cursor: a single `follow_cursor` Sprite the
// overlay draws as an arrow, or as a resize cursor over a resizable panel's edge
// (the hook publishes the shape through the `DesiredCursor` resource). While it
// is visible the overlay hides the OS cursor for it (its `want_ui_cursor` path),
// so the pointer stays managed in-engine and looks identical on every backend.

use super::registry::ID_BASE;
use super::widget;
use crate::assets::Sprite;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved id: the free slot between the marquee rect (0xE00) and the
// billboards (0x1000).
pub(crate) const CURSOR: AssetId = AssetId(ID_BASE + 0xF00);

// Cursor height in reference pixels; the overlay scales it for the viewport.
const CURSOR_PX: f32 = 20.0;
// A light fill so the arrow reads over the dark editor chrome and the scene; the
// overlay picks a contrasting outline for it.
const FILL: [f32; 4] = [0.95, 0.95, 0.97, 1.0];

// The injected cursor sprite, hidden until the tick shows it.
pub(crate) fn sprite() -> Sprite {
    Sprite {
        asset_id: CURSOR,
        follow_cursor: true,
        width: CURSOR_PX,
        height: CURSOR_PX,
        tint: FILL,
        visible: false,
        ..Default::default()
    }
}

// Show or hide the cursor sprite for this frame: shown while the editor owns the
// pointer (edit mode), hidden while a captured camera owns it so no stray arrow
// lingers over the frozen pointer.
pub(crate) fn set_visible(world: &mut World, visible: bool) {
    widget::set_sprite_visible(world, CURSOR, visible);
}
