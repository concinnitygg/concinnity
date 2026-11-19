//! Rectangle helpers for the hook's pointer and panel geometry.

// Whether `(x, y)` lies inside `rect` (`[x, y, width, height]`).
pub(crate) fn point_in(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}

// The rectangle of a box at `o`, `s` big.
pub(crate) fn rect_of(o: [f32; 2], s: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], s[0], s[1]]
}

// `pos` moved so a box `size` big stays whole inside the viewport, its top no
// higher than `top`.
pub(crate) fn clamp_origin(
    pos: [f32; 2],
    size: [f32; 2],
    viewport: [f32; 2],
    top: f32,
) -> [f32; 2] {
    [
        pos[0].clamp(0.0, (viewport[0] - size[0]).max(0.0)),
        pos[1].clamp(top, (viewport[1] - size[1]).max(top)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_in_is_half_open() {
        let r = [10.0, 20.0, 30.0, 40.0];
        assert!(point_in(10.0, 20.0, r));
        assert!(!point_in(40.0, 30.0, r));
        assert!(!point_in(9.9, 30.0, r));
    }

    #[test]
    fn clamp_origin_keeps_the_box_on_screen_below_the_top() {
        let vp = [800.0, 600.0];
        assert_eq!(
            clamp_origin([-50.0, -50.0], [100.0, 100.0], vp, 40.0),
            [0.0, 40.0]
        );
        assert_eq!(
            clamp_origin([900.0, 900.0], [100.0, 100.0], vp, 40.0),
            [700.0, 500.0]
        );
        // A box bigger than the screen pins to its top-left.
        assert_eq!(
            clamp_origin([5.0, 5.0], [900.0, 900.0], vp, 40.0),
            [0.0, 40.0]
        );
    }
}
