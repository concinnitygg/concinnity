// The draw of a Canvas: its background, then each of its shapes that reaches
// it, panned into place, its text on one line inside it, then its scrollbars.

use crate::components::{Canvas, TextAlign, UiShape, UiState};
use crate::render::overlay::widgets::WidgetScratch;

use super::{ControlFrame, Pen};

// Whether `a` and `b` overlap.
fn overlaps(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]
}

pub(super) fn draw(
    canvas: &Canvas,
    state: Option<&UiState>,
    frame: &ControlFrame<'_>,
    out: &mut WidgetScratch,
) {
    let style = &canvas.style;
    let mut pen = Pen::new(out, frame, style, canvas.screen, canvas.fit);
    let area = canvas.rect();
    if style.background[3] > 0.0 {
        pen.framed(area, style.background, style.corner_radius, style.border);
    }
    let shapes = state.map_or(&[][..], |s| s.shapes(&canvas.shapes));
    for shape in shapes {
        let rect = canvas.placed(shape);
        if canvas.clip && !overlaps(rect, area) {
            continue;
        }
        if shape.fill[3] > 0.0 || shape.border_width > 0.0 {
            pen.sprite(
                rect,
                shape.fill,
                shape.radius,
                (shape.border, shape.border_width),
                None,
            );
        }
        if shape.text.is_empty() {
            continue;
        }
        let x = match shape.align {
            TextAlign::Left => rect[0] + UiShape::TEXT_INSET,
            TextAlign::Center => rect[0] + rect[2] * 0.5,
            TextAlign::Right => rect[0] + rect[2] - UiShape::TEXT_INSET,
        };
        let scale = if shape.text_scale > 0.0 {
            shape.text_scale
        } else {
            1.0
        };
        // The pen measures at the style's scale, so the room is in those units.
        let width = (rect[2] - 2.0 * UiShape::TEXT_INSET) / scale;
        if width <= 0.0 {
            continue;
        }
        let y = rect[1] + (rect[3] - pen.line_h() * scale) * 0.5;
        let label = pen.text(&shape.text, [x, y], shape.text_color, shape.align, width);
        label.scale *= scale;
    }
    for bar in canvas.scrollbars(shapes).into_iter().flatten() {
        super::list::draw_bar(&mut pen, bar);
    }
}
