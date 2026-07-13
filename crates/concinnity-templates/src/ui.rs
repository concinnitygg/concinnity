// Generic UI geometry primitives shared across menus and the editor HUD. Pure
// `f32` math with no engine dependency, so any UI builder or hit-test can reuse it.

// Whether `(x, y)` lies inside `rect` ([x, y, w, h], top-left origin). The top-left
// edge is inside; the bottom-right edge is outside.
pub fn point_in(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_in_includes_top_left_excludes_bottom_right() {
        let r = [10.0, 20.0, 100.0, 40.0];
        assert!(point_in(10.0, 20.0, r), "top-left corner is inside");
        assert!(point_in(50.0, 40.0, r), "interior is inside");
        assert!(!point_in(110.0, 40.0, r), "right edge is outside");
        assert!(!point_in(50.0, 60.0, r), "bottom edge is outside");
        assert!(!point_in(9.9, 40.0, r), "left of the rect is outside");
    }
}
