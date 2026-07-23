// src/editor/resize.rs
//
// Panel resizing: the pure edge/corner hit-test and its cursor-shape mapping,
// shared by every resizable floating panel. A panel is grabbed for a resize
// when the pointer lands in the inner border band of its footprint; which edges
// the band covers picks the resize cursor and, once a drag is under way, which
// dimensions grow. The hook owns the drag state and the per-panel size override;
// this module is stateless geometry so it stays unit-testable.

use super::registry::PanelKey;
use crate::ecs::CursorShape;

// The thickness of the grab band along each edge, in window pixels. The band
// sits inside the panel footprint so it never overlaps a neighbour.
pub(crate) const BORDER: f32 = 6.0;

// Which of a panel's edges the pointer's grab band covers. A single edge is an
// edge resize; two adjacent edges are a corner.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Edges {
    pub left: bool,
    pub right: bool,
    pub top: bool,
    pub bottom: bool,
}

impl Edges {
    pub(crate) fn any(self) -> bool {
        self.left || self.right || self.top || self.bottom
    }
}

// The edges grabbed at `(mx, my)` for a panel at origin `o` of size `s`, or
// `None` when the point is outside the panel or in its interior (past the band).
// The band is the inner `t` pixels of each edge; panels are far larger than
// `2*t`, so opposite edges never both match.
pub(crate) fn hit_test(o: [f32; 2], s: [f32; 2], t: f32, mx: f32, my: f32) -> Option<Edges> {
    if mx < o[0] || mx >= o[0] + s[0] || my < o[1] || my >= o[1] + s[1] {
        return None;
    }
    let edges = Edges {
        left: mx - o[0] <= t,
        right: o[0] + s[0] - mx <= t,
        top: my - o[1] <= t,
        bottom: o[1] + s[1] - my <= t,
    };
    edges.any().then_some(edges)
}

// The resize cursor for a set of grabbed edges: a straight arrow for a lone
// edge, a diagonal for a corner (NWSE for the top-left / bottom-right pair,
// NESW for the top-right / bottom-left pair).
pub(crate) fn cursor_shape(e: Edges) -> CursorShape {
    let horizontal = e.left || e.right;
    let vertical = e.top || e.bottom;
    match (horizontal, vertical) {
        (true, true) => {
            if (e.left && e.top) || (e.right && e.bottom) {
                CursorShape::ResizeNWSE
            } else {
                CursorShape::ResizeNESW
            }
        }
        (true, false) => CursorShape::ResizeEW,
        (false, true) => CursorShape::ResizeNS,
        (false, false) => CursorShape::Default,
    }
}

// A resize drag in progress: the grabbed panel, which edges it grows, and the
// pointer + panel geometry captured at the press so the drag follows without
// snapping.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Resize {
    pub key: PanelKey,
    pub edges: Edges,
    pub grab_mouse: [f32; 2],
    pub start_origin: [f32; 2],
    pub start_size: [f32; 2],
}

// The new (origin, size) for an in-flight resize given the live `mouse`, the
// panel's minimum size `min` (its content-derived default), its maximum `max`
// (`f32::INFINITY` per axis for unbounded), the viewport, and the top-bar lower
// edge `top`. Each grabbed edge grows the panel from the press anchor: the right
// / bottom edges extend the size (origin fixed); the left / top edges hold the
// far edge fixed and move the origin. Every dimension stays within `[min, max]`,
// and the panel can never leave the screen or slide under the top bar.
pub(crate) fn apply(
    r: &Resize,
    mouse: [f32; 2],
    min: [f32; 2],
    max: [f32; 2],
    vp: [f32; 2],
    top: f32,
) -> ([f32; 2], [f32; 2]) {
    let dx = mouse[0] - r.grab_mouse[0];
    let dy = mouse[1] - r.grab_mouse[1];
    let (mut x, mut y) = (r.start_origin[0], r.start_origin[1]);
    let (mut w, mut h) = (r.start_size[0], r.start_size[1]);

    if r.edges.right {
        // Origin fixed, so the right edge cannot pass the viewport.
        let cap = max[0].min(vp[0] - x).max(min[0]);
        w = (r.start_size[0] + dx).clamp(min[0], cap);
    }
    if r.edges.left {
        // Far edge fixed; the left edge cannot pass the window's left (x >= 0).
        let right = r.start_origin[0] + r.start_size[0];
        let cap = max[0].min(right).max(min[0]);
        w = (r.start_size[0] - dx).clamp(min[0], cap);
        x = right - w;
    }
    if r.edges.bottom {
        let cap = max[1].min(vp[1] - y).max(min[1]);
        h = (r.start_size[1] + dy).clamp(min[1], cap);
    }
    if r.edges.top {
        // Far edge fixed; the top edge cannot slide under the top bar (y >= top).
        let bottom = r.start_origin[1] + r.start_size[1];
        let cap = max[1].min(bottom - top).max(min[1]);
        h = (r.start_size[1] - dy).clamp(min[1], cap);
        y = bottom - h;
    }
    ([x, y], [w, h])
}

#[cfg(test)]
mod tests {
    use super::*;

    const O: [f32; 2] = [100.0, 100.0];
    const S: [f32; 2] = [200.0, 300.0];
    const T: f32 = 6.0;

    #[test]
    fn interior_and_outside_miss() {
        // Dead centre: no band.
        assert_eq!(hit_test(O, S, T, 200.0, 250.0), None);
        // Fully outside on every side.
        assert_eq!(hit_test(O, S, T, 50.0, 250.0), None);
        assert_eq!(hit_test(O, S, T, 350.0, 250.0), None);
        assert_eq!(hit_test(O, S, T, 200.0, 50.0), None);
        assert_eq!(hit_test(O, S, T, 200.0, 500.0), None);
    }

    #[test]
    fn each_edge_and_corner_resolves() {
        // Just inside each edge, mid-span, is that lone edge.
        let left = hit_test(O, S, T, 102.0, 250.0).unwrap();
        assert_eq!(cursor_shape(left), CursorShape::ResizeEW);
        assert!(left.left && !left.right && !left.top && !left.bottom);
        let right = hit_test(O, S, T, 298.0, 250.0).unwrap();
        assert_eq!(cursor_shape(right), CursorShape::ResizeEW);
        let top = hit_test(O, S, T, 200.0, 102.0).unwrap();
        assert_eq!(cursor_shape(top), CursorShape::ResizeNS);
        let bottom = hit_test(O, S, T, 200.0, 398.0).unwrap();
        assert_eq!(cursor_shape(bottom), CursorShape::ResizeNS);
        // The four corners map to the two diagonals.
        assert_eq!(
            cursor_shape(hit_test(O, S, T, 102.0, 102.0).unwrap()),
            CursorShape::ResizeNWSE
        );
        assert_eq!(
            cursor_shape(hit_test(O, S, T, 298.0, 398.0).unwrap()),
            CursorShape::ResizeNWSE
        );
        assert_eq!(
            cursor_shape(hit_test(O, S, T, 298.0, 102.0).unwrap()),
            CursorShape::ResizeNESW
        );
        assert_eq!(
            cursor_shape(hit_test(O, S, T, 102.0, 398.0).unwrap()),
            CursorShape::ResizeNESW
        );
    }

    #[test]
    fn no_edges_is_the_arrow() {
        assert_eq!(cursor_shape(Edges::default()), CursorShape::Default);
    }

    fn resize(edges: Edges) -> Resize {
        Resize {
            key: PanelKey::Assets,
            edges,
            grab_mouse: [200.0, 250.0],
            start_origin: O,
            start_size: S,
        }
    }

    const VP: [f32; 2] = [1280.0, 720.0];
    const TOP: f32 = 30.0;
    // A generous min so the clamps below are exercised without the min biting.
    const MIN: [f32; 2] = [120.0, 80.0];
    const UNBOUNDED: [f32; 2] = [f32::INFINITY, f32::INFINITY];

    #[test]
    fn right_and_bottom_grow_from_the_fixed_origin() {
        let r = resize(Edges {
            right: true,
            bottom: true,
            ..Default::default()
        });
        let (o, s) = apply(&r, [260.0, 300.0], MIN, UNBOUNDED, VP, TOP);
        assert_eq!(o, O, "origin stays put for a right / bottom drag");
        assert_eq!(s, [S[0] + 60.0, S[1] + 50.0]);
    }

    #[test]
    fn left_and_top_move_the_origin_and_hold_the_far_edge() {
        let r = resize(Edges {
            left: true,
            top: true,
            ..Default::default()
        });
        // Drag the top-left corner up-left by (40, 30): the panel grows and its
        // bottom-right corner stays where it was.
        let (o, s) = apply(&r, [160.0, 220.0], MIN, UNBOUNDED, VP, TOP);
        assert_eq!(o, [60.0, 70.0]);
        assert_eq!(s, [S[0] + 40.0, S[1] + 30.0]);
        assert_eq!(
            [o[0] + s[0], o[1] + s[1]],
            [O[0] + S[0], O[1] + S[1]],
            "the far corner is pinned"
        );
    }

    #[test]
    fn never_shrinks_below_min() {
        // Drag the right edge far left and the top edge far down.
        let r = resize(Edges {
            right: true,
            top: true,
            ..Default::default()
        });
        let big_min = [180.0, 200.0];
        let (o, s) = apply(&r, [0.0, 700.0], big_min, UNBOUNDED, VP, TOP);
        assert_eq!(s[0], big_min[0], "width floors at min");
        assert_eq!(s[1], big_min[1], "height floors at min");
        // The top edge held the bottom fixed as it clamped at min.
        assert_eq!(o[1] + s[1], O[1] + S[1]);
    }

    #[test]
    fn never_grows_past_max() {
        let r = resize(Edges {
            right: true,
            bottom: true,
            ..Default::default()
        });
        let max = [260.0, 340.0];
        let (_, s) = apply(&r, [5000.0, 5000.0], MIN, max, VP, TOP);
        assert_eq!(s, max, "each dimension caps at max");
    }

    #[test]
    fn clamps_to_the_screen_and_top_bar() {
        // Right / bottom cannot extend past the viewport.
        let r = resize(Edges {
            right: true,
            bottom: true,
            ..Default::default()
        });
        let (_, s) = apply(&r, [5000.0, 5000.0], MIN, UNBOUNDED, VP, TOP);
        assert_eq!(s[0], VP[0] - O[0]);
        assert_eq!(s[1], VP[1] - O[1]);
        // The top edge cannot slide under the top bar.
        let r = resize(Edges {
            top: true,
            ..Default::default()
        });
        let (o, _) = apply(&r, [200.0, -5000.0], MIN, UNBOUNDED, VP, TOP);
        assert_eq!(o[1], TOP);
    }
}
