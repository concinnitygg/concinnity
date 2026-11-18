use super::*;
use crate::presentation::text_edit::Scroll;
use alloc::string::String;
use alloc::vec::Vec;

const M: Metrics = Metrics {
    line_h: 20.0,
    advance: 10.0,
    text_px: 14.0,
    label_scale: 0.7,
};
const RECT: [f32; 4] = [100.0, 50.0, 400.0, 206.0];

fn lines(n: usize) -> String {
    (0..n)
        .map(|i| alloc::format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn laid_out(text: &str) -> (TextEditor, Geometry) {
    let mut area = TextEditor::from_text(text);
    let g = geometry(RECT, M, area.line_count());
    area.set_view(g.view());
    (area, g)
}

// The gutter holds two digits at least (plus the marker column and padding),
// and the grid fills what the reserved scrollbar strips leave.
#[test]
fn geometry_sizes_the_gutter_and_grid() {
    let g = geometry(RECT, M, 5);
    assert_eq!(g.gutter_w, MARKER_W + 2.0 * 10.0 + GUTTER_PAD);
    assert_eq!(g.text_x, 100.0 + g.gutter_w + TEXT_PAD);
    assert_eq!(g.view(), ViewSize { rows: 10, cols: 34 });
    let wide = geometry(RECT, M, 12_345);
    assert_eq!(wide.gutter_w, MARKER_W + 5.0 * 10.0 + GUTTER_PAD);
    assert_eq!(
        height_for_rows(10, M.line_h),
        206.0,
        "RECT shows exactly 10 rows"
    );
    let huge = geometry([0.0, 0.0, 400.0, 10_000.0], M, 5);
    assert_eq!(huge.view().rows, ROW_POOL, "capped at the pool");
}

#[test]
fn pos_at_maps_points_to_characters() {
    let (area, g) = laid_out("abcdef\n\tx\nlast");
    let x = |cell: f32| g.text_x + cell * M.advance;
    assert_eq!(g.pos_at(&area, x(2.3), 55.0), Pos::new(0, 2));
    assert_eq!(
        g.pos_at(&area, x(2.6), 55.0),
        Pos::new(0, 3),
        "the nearer boundary"
    );
    assert_eq!(
        g.pos_at(&area, x(3.0), 75.0),
        Pos::new(1, 1),
        "past the tab's middle"
    );
    assert_eq!(
        g.pos_at(&area, x(50.0), 95.0),
        Pos::new(2, 4),
        "past the line's end"
    );
    assert_eq!(
        g.pos_at(&area, 101.0, 55.0),
        Pos::new(0, 0),
        "the gutter is column 0"
    );
    assert_eq!(
        g.pos_at(&area, x(1.0), 500.0),
        Pos::new(2, 1),
        "below the text"
    );
    let (mut long, g) = laid_out(&lines(30));
    long.set_scroll(Scroll { top: 5, left: 0 });
    assert_eq!(
        g.pos_at(&long, x(0.0), 30.0).line,
        4,
        "above the window reaches up"
    );
    assert_eq!(
        g.pos_at(&long, x(0.0), 260.0).line,
        15,
        "below it reaches down"
    );
}

#[test]
fn thumbs_track_the_scroll_position() {
    assert_eq!(thumb(100.0, 10, 10, 0), None, "everything fits");
    assert_eq!(thumb(100.0, 10, 40, 0), Some((0.0, 25.0)));
    assert_eq!(thumb(100.0, 10, 40, 30), Some((75.0, 25.0)));
    assert_eq!(thumb(100.0, 1, 1000, 0).unwrap().1, MIN_THUMB);
    assert_eq!(scroll_on_track(100.0, 12.5, 10, 40), 0);
    assert_eq!(scroll_on_track(100.0, 87.5, 10, 40), 30);
    assert_eq!(scroll_on_track(100.0, 50.0, 10, 40), 15);
    assert_eq!(scroll_on_track(100.0, 50.0, 10, 5), 0, "nothing to scroll");
}

#[test]
fn pressing_a_scrollbar_drags_the_window() {
    let (mut area, g) = laid_out(&lines(100));
    let t = g.v_track();
    assert!(area.press_at(&g, t[0] + 1.0, t[1] + t[3] - 1.0, false, 0.0));
    assert_eq!(area.scroll().top, 90, "the bottom of the track");
    assert_eq!(area.caret(), Pos::new(0, 0), "the caret stays");
    area.pointer(&g, t[0] + 1.0, t[1] + 1.0, true);
    assert_eq!(area.scroll().top, 0, "dragged to the top");
    area.pointer(&g, t[0] + 1.0, t[1] + t[3], false);
    assert!(!area.pointer_busy());
    area.pointer(&g, t[0] + 1.0, t[1] + t[3], true);
    assert_eq!(area.scroll().top, 0, "a released bar no longer drags");
}

#[test]
fn a_text_press_drags_a_selection() {
    let (mut area, g) = laid_out("abcdef\nghij");
    let x = |cell: f32| g.text_x + cell * M.advance;
    assert!(area.press_at(&g, x(1.0), 55.0, false, 0.0));
    area.pointer(&g, x(3.0), 75.0, true);
    assert_eq!(area.selection(), Some((Pos::new(0, 1), Pos::new(1, 3))));
    area.pointer(&g, x(3.0), 75.0, false);
    assert!(!area.pointer_busy());
    assert!(
        !area.press_at(&g, 0.0, 0.0, false, 1.0),
        "a miss is not taken"
    );
}

#[test]
fn a_gutter_press_lands_at_the_line_start() {
    let (mut area, g) = laid_out("abc\ndef");
    area.press_at(&g, RECT[0] + 5.0, RECT[1] + 25.0, false, 0.0);
    assert_eq!(area.caret(), Pos::new(1, 0));
}
