//! The marquee (box-select) rectangle: the pure screen-rect math and the one
//! shape drawn while a drag is in flight. The hook (`hook/drag/marquee.rs`)
//! owns the drag state and the release-time selection.

use crate::editor::theme;
use crate::editor::ui::shapes::{Shape, Shapes};

// A press must travel this far (either axis) before it becomes a marquee; a
// stiller release is an empty-space click.
pub(crate) const DRAG_START_PX: f32 = 4.0;

const BORDER_W: f32 = 1.5;
// Accent fill faint enough to keep the scene readable through the band.
const FILL_TINT: [f32; 4] = [0.26, 0.42, 0.66, 0.12];

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

// The band over `rect`.
pub(crate) fn push(shapes: &mut Shapes<'_>, rect: [f32; 4]) {
    shapes.push(Shape {
        rect,
        fill: FILL_TINT,
        border: theme::ACCENT_TINT,
        border_width: BORDER_W,
        ..Shape::default()
    });
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
