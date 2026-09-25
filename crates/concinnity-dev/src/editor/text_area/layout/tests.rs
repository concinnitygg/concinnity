use concinnity_core::components::{Sprite, TextLabel};

use super::*;
use crate::editor::text_area::Scroll;
use crate::editor::text_area::markers::{GutterMarker, Severity};

const M: Metrics = Metrics {
    line_h: 20.0,
    advance: 10.0,
    text_px: 14.0,
    label_scale: 0.7,
};
const IDS: TextAreaIds = TextAreaIds::new(0x100);
const RECT: [f32; 4] = [100.0, 50.0, 400.0, 206.0];

fn world() -> World {
    crate::test_support::injected_world(&IDS.sprite_ids(), &IDS.code_label_ids(), &[])
}

fn lines(n: usize) -> String {
    (0..n)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn laid_out(text: &str) -> (TextArea, Geometry) {
    let mut area = TextArea::from_text(text);
    let g = geometry(RECT, M, area.line_count());
    area.set_view(g.view());
    (area, g)
}

fn sprite(world: &World, id: AssetId) -> Sprite {
    world.get_by_id::<Sprite>(id).unwrap().clone()
}

fn label(world: &World, id: AssetId) -> TextLabel {
    world.get_by_id::<TextLabel>(id).unwrap().clone()
}

fn show(world: &mut World, area: &TextArea, g: &Geometry, focused: bool, markers: &[GutterMarker]) {
    let view = TextAreaView {
        area,
        focused,
        markers,
        mouse: [0.0, 0.0],
    };
    place(world, IDS, Some(&view), g);
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
fn place_draws_the_visible_window_of_lines() {
    let mut world = world();
    let (mut area, g) = laid_out(&lines(30));
    area.go_to(15, 0);
    show(&mut world, &area, &g, true, &[]);
    assert_eq!(area.scroll().top, 10, "the jump centered the line");
    let first = label(&world, IDS.text(0));
    assert_eq!((first.content.as_str(), first.x), ("line 10", g.text_x));
    assert_eq!(first.scale, M.label_scale);
    assert_eq!(
        label(&world, IDS.number(0)).content,
        "11",
        "numbered from 1"
    );
    assert!(label(&world, IDS.number(9)).visible);
    assert!(!label(&world, IDS.number(10)).visible, "past the grid");
    assert!(
        sprite(&world, IDS.v_thumb()).visible,
        "30 lines overflow 10 rows"
    );
    assert!(
        !sprite(&world, IDS.h_thumb()).visible,
        "nothing is wider than the grid"
    );
}

#[test]
fn a_short_text_leaves_the_pool_blank() {
    let mut world = world();
    let (area, g) = laid_out("only");
    show(&mut world, &area, &g, false, &[]);
    assert!(label(&world, IDS.text(0)).visible);
    assert!(!label(&world, IDS.text(1)).visible);
    assert!(!sprite(&world, IDS.v_track()).visible);
}

#[test]
fn the_caret_and_line_band_show_only_while_focused() {
    let mut world = world();
    let (mut area, g) = laid_out("abc\ndef");
    area.go_to(1, 2);
    show(&mut world, &area, &g, true, &[]);
    let caret = sprite(&world, IDS.caret());
    assert!(caret.visible);
    assert_eq!(caret.x, g.text_x + 2.0 * M.advance - CARET_W * 0.5);
    assert_eq!(caret.y, RECT[1] + M.line_h + 1.0);
    assert!(sprite(&world, IDS.current_line()).visible);
    show(&mut world, &area, &g, false, &[]);
    assert!(!sprite(&world, IDS.caret()).visible);
    assert!(!sprite(&world, IDS.current_line()).visible);
}

#[test]
fn selection_rects_cover_the_selected_cells() {
    let mut world = world();
    let (mut area, g) = laid_out("abcdef\nxy\nlast");
    area.press(Pos::new(0, 2), false, 0.0);
    area.drag_to(Pos::new(2, 1));
    show(&mut world, &area, &g, true, &[]);
    let first = sprite(&world, IDS.selection(0));
    assert_eq!(first.x, g.text_x + 2.0 * M.advance);
    assert_eq!(
        first.width,
        5.0 * M.advance,
        "four cells and the line break"
    );
    let middle = sprite(&world, IDS.selection(1));
    assert_eq!(
        middle.width,
        3.0 * M.advance,
        "the whole line and its break"
    );
    let last = sprite(&world, IDS.selection(2));
    assert_eq!((last.x, last.width), (g.text_x, M.advance));
    assert!(!sprite(&world, IDS.selection(3)).visible);
}

#[test]
fn markers_draw_in_the_gutter_and_read_out_on_hover() {
    let mut world = world();
    let (area, g) = laid_out(&lines(5));
    let markers = [
        GutterMarker {
            line: 1,
            column: 0,
            severity: Severity::Warning,
            message: "unused".into(),
        },
        GutterMarker {
            line: 3,
            column: 0,
            severity: Severity::Error,
            message: "undeclared identifier".into(),
        },
    ];
    show(&mut world, &area, &g, false, &markers);
    assert!(!sprite(&world, IDS.marker(0)).visible);
    assert!(sprite(&world, IDS.marker(1)).visible);
    let error = sprite(&world, IDS.marker(3));
    assert_eq!(error.tint[..3], theme::LOG_ERROR);
    let hover = |mouse| {
        let view = TextAreaView {
            area: &area,
            focused: false,
            markers: &markers,
            mouse,
        };
        hovered_marker(&view, &g).map(|m| m.message.clone())
    };
    let row3 = RECT[1] + 3.5 * M.line_h;
    assert_eq!(
        hover([RECT[0] + 4.0, row3]).as_deref(),
        Some("undeclared identifier")
    );
    assert_eq!(
        hover([RECT[0] + 4.0, RECT[1] + 5.0]),
        None,
        "an unmarked line"
    );
    assert_eq!(
        hover([g.text_x + 5.0, row3]),
        None,
        "only the gutter reads out"
    );
}

#[test]
fn long_lines_draw_their_scrolled_window() {
    let mut world = world();
    let (mut area, g) = laid_out(&"0123456789".repeat(6));
    area.go_to(0, 60);
    show(&mut world, &area, &g, true, &[]);
    let left = area.scroll().left;
    assert!(left > 0);
    let text = label(&world, IDS.text(0)).content;
    assert_eq!(
        text.chars().next(),
        char::from_digit((left % 10) as u32, 10)
    );
    assert!(sprite(&world, IDS.h_thumb()).visible);
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

#[test]
fn hide_blanks_every_element() {
    let mut world = world();
    let (area, g) = laid_out(&lines(30));
    show(&mut world, &area, &g, true, &[]);
    place(&mut world, IDS, None, &g);
    assert!(world.query::<Sprite>().all(|s| !s.visible));
    assert!(world.query::<TextLabel>().all(|l| !l.visible));
}
