// src/editor/marquee.rs
//
// The marquee (box-select) rectangle: the pure screen-rect math and the one
// injected sprite drawn while a drag is in flight. The hook
// (`hook/marquee_drag.rs`) owns the drag state and the release-time selection.

use super::registry::ID_BASE;
use super::theme;
use crate::assets::Sprite;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved id: the next free block after the gizmo's 0xD00.
pub(crate) const RECT: AssetId = AssetId(ID_BASE + 0xE00);

// A press must travel this far (either axis) before it becomes a marquee; a
// stiller release is an empty-space click.
pub(crate) const DRAG_START_PX: f32 = 4.0;

const BORDER_W: f32 = 1.5;
// Accent fill faint enough to keep the scene readable through the band.
const FILL_TINT: [f32; 4] = [0.26, 0.42, 0.66, 0.12];

pub(crate) fn rect_sprite() -> Sprite {
    Sprite {
        asset_id: RECT,
        tint: FILL_TINT,
        border_width: BORDER_W,
        border_color: theme::ACCENT_TINT,
        visible: false,
        ..Default::default()
    }
}

// The `[x, y, w, h]` rect spanned by two drag corners, in any order.
pub(crate) fn rect_from_corners(a: [f32; 2], b: [f32; 2]) -> [f32; 4] {
    let x0 = a[0].min(b[0]);
    let y0 = a[1].min(b[1]);
    [x0, y0, (a[0] - b[0]).abs(), (a[1] - b[1]).abs()]
}

// Whether two `[x, y, w, h]` rects overlap (touching edges count, so a
// zero-height sliver of a clamped projection is still selectable).
pub(crate) fn rects_intersect(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] <= b[0] + b[2] && b[0] <= a[0] + a[2] && a[1] <= b[1] + b[3] && b[1] <= a[1] + a[3]
}

pub(crate) fn place(world: &mut World, rect: [f32; 4]) {
    if let Some(s) = sprite_mut(world) {
        s.x = rect[0];
        s.y = rect[1];
        s.width = rect[2];
        s.height = rect[3];
        s.visible = true;
    }
}

pub(crate) fn hide(world: &mut World) {
    if let Some(s) = sprite_mut(world) {
        s.visible = false;
    }
}

fn sprite_mut(world: &mut World) -> Option<&mut Sprite> {
    world.query_mut::<Sprite>().find(|s| s.asset_id == RECT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_from_corners_normalizes_any_drag_direction() {
        assert_eq!(
            rect_from_corners([10.0, 20.0], [30.0, 50.0]),
            [10.0, 20.0, 20.0, 30.0]
        );
        assert_eq!(
            rect_from_corners([30.0, 50.0], [10.0, 20.0]),
            [10.0, 20.0, 20.0, 30.0]
        );
        assert_eq!(
            rect_from_corners([30.0, 20.0], [10.0, 50.0]),
            [10.0, 20.0, 20.0, 30.0]
        );
    }

    #[test]
    fn rects_intersect_covers_overlap_touch_and_miss() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert!(rects_intersect(a, [5.0, 5.0, 10.0, 10.0]), "overlap");
        assert!(rects_intersect(a, [2.0, 2.0, 4.0, 4.0]), "containment");
        assert!(rects_intersect(a, [10.0, 0.0, 5.0, 5.0]), "shared edge");
        assert!(!rects_intersect(a, [10.1, 0.0, 5.0, 5.0]), "clear miss");
        assert!(!rects_intersect(a, [0.0, 20.0, 5.0, 5.0]));
    }
}
