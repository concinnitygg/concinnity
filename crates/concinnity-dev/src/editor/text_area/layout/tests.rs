use concinnity_core::math::SaturatingFrom;
use concinnity_engine::components::{Sprite, TextLabel};

use super::*;
use crate::editor::text_area::markers::{GutterMarker, Severity};
use crate::test_support::{label, sprite};
use concinnity_engine::presentation::text_edit::{Pos, geometry};

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

fn laid_out(text: &str) -> (TextEditor, Geometry) {
    let mut area = TextEditor::from_text(text);
    let g = geometry(RECT, M, area.line_count());
    area.set_view(g.view());
    (area, g)
}

fn show(
    world: &mut World,
    area: &TextEditor,
    g: &Geometry,
    focused: bool,
    markers: &[GutterMarker],
) {
    let view = TextAreaView {
        area,
        focused,
        markers,
        mouse: [0.0, 0.0],
    };
    place(world, IDS, Some(&view), g);
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
        char::from_digit(u32::saturating_from(left % 10), 10)
    );
    assert!(sprite(&world, IDS.h_thumb()).visible);
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
