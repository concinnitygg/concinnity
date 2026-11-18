// A TextArea's draw: the well and gutter, the line numbers, the visible window
// of lines with their colors, the selection, the caret line and caret, and the
// scrollbars. Also measures each editor's monospace grid from its font.

use crate::components::{TextAlign, TextArea};
use crate::ecs::PipelineContext;
use crate::presentation::text_edit::{GUTTER_PAD, Metrics, SCROLLBAR, geometry, view};
use crate::render::overlay::text::{self, FontSet};
use crate::render::overlay::widgets::WidgetScratch;

use super::{ControlFrame, Pen};

const GUTTER_TINT: [f32; 4] = [0.09, 0.09, 0.115, 1.0];
const CURRENT_LINE_TINT: [f32; 4] = [0.14, 0.15, 0.20, 1.0];
const SELECTION_TINT: [f32; 4] = [0.20, 0.32, 0.55, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const NUMBER_CURRENT: [f32; 3] = [0.78, 0.80, 0.88];
const CARET_W: f32 = 2.0;
// Space between two lines, beyond the face's own height.
const LINE_GAP: f32 = 4.0;

// The grid a text editor of `area`'s style lays out on, from its font.
fn metrics_for(area: &TextArea, fonts: &FontSet) -> Option<Metrics> {
    let font = fonts.resolve(area.style.font)?;
    let text_px = font.size_px * area.style.scale;
    Some(Metrics {
        line_h: text_px + LINE_GAP,
        advance: text::text_advance_width("M", font, area.style.scale).max(1.0),
        text_px,
        label_scale: area.style.scale,
    })
}

// Measure every text editor's grid, so its input maps the cursor onto the
// text the way this frame draws it, and scan the lines it can show for color.
pub(in crate::render::overlay) fn measure_text_areas(
    ctx: &mut PipelineContext<'_>,
    fonts: &FontSet,
) {
    for area in ctx.query_mut::<TextArea>() {
        let m = metrics_for(area, fonts);
        if area.metrics != m {
            area.metrics = m;
        }
        area.editor.prepare_highlight();
    }
}

pub(super) fn draw(area: &TextArea, frame: &ControlFrame<'_>, out: &mut WidgetScratch) {
    let Some(m) = area.metrics else {
        return;
    };
    let style = &area.style;
    let editor = &area.editor;
    let rect = [area.x, area.y, area.width, area.height];
    let g = geometry(rect, m, editor.line_count());
    let mut pen = Pen::new(out, frame, style, area.screen, area.fit);
    pen.framed(rect, style.background, style.corner_radius, style.border);
    pen.rect(g.gutter_rect(), GUTTER_TINT, style.corner_radius);
    let scroll = editor.scroll();
    let caret = editor.caret();
    let caret_row = caret.line.checked_sub(scroll.top).filter(|&r| r < g.rows);
    if area.focused
        && let Some(row) = caret_row
    {
        let band = [
            g.rect[0] + g.gutter_w,
            g.row_y(row),
            g.rect[2] - g.gutter_w - SCROLLBAR,
            m.line_h,
        ];
        pen.rect(band, CURRENT_LINE_TINT, 0.0);
    }
    for row in 0..g.rows {
        let line = scroll.top + row;
        if line >= editor.line_count() {
            break;
        }
        let y = g.row_y(row);
        if let Some((a, b)) = editor.selected_cells(line) {
            let a = a.max(scroll.left);
            let b = b.min(scroll.left + g.cols);
            if b > a {
                let x = g.text_x + (a - scroll.left) as f32 * m.advance;
                let sel = [x, y, (b - a) as f32 * m.advance, m.line_h];
                pen.rect(sel, SELECTION_TINT, 0.0);
            }
        }
        let text_y = y + (m.line_h - m.text_px) * 0.5;
        let number = if line == caret.line {
            NUMBER_CURRENT
        } else {
            style.dim_text
        };
        let number_x = g.rect[0] + g.gutter_w - GUTTER_PAD * 0.5;
        pen.number(line + 1, [number_x, text_y], number, TextAlign::Right);
        let label = pen.text("", [g.text_x, text_y], style.text, TextAlign::Left, 0.0);
        view::display_slice_into(editor.line(line), scroll.left, g.cols, &mut label.content);
        editor.line_runs(line, scroll.left, g.cols, &mut label.color_runs);
    }
    let left = scroll.left;
    let cell = editor.caret_cell();
    if area.focused
        && frame.caret_visible
        && let Some(row) = caret_row
        && cell >= left
        && cell <= left + g.cols
    {
        let x = g.text_x + (cell - left) as f32 * m.advance - CARET_W * 0.5;
        let [r, gr, b] = style.text;
        pen.rect(
            [x, g.row_y(row) + 1.0, CARET_W, m.line_h - 2.0],
            [r, gr, b, 1.0],
            0.0,
        );
    }
    if let Some(thumb) = g.v_thumb(editor) {
        pen.rect(g.v_track(), TRACK_TINT, SCROLLBAR * 0.5);
        pen.rect(thumb, THUMB_TINT, SCROLLBAR * 0.5);
    }
    if let Some(thumb) = g.h_thumb(editor) {
        pen.rect(g.h_track(), TRACK_TINT, SCROLLBAR * 0.5);
        pen.rect(thumb, THUMB_TINT, SCROLLBAR * 0.5);
    }
}
