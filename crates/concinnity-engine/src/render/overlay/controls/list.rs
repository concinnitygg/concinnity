// A List's draw: its background, the rows in view with the selection and
// hover highlights, tree expanders and indents, and the scrollbar.

use crate::components::{List, TextAlign, UiState};
use crate::ecs::PipelineContext;
use crate::presentation::ui::bound;
use crate::presentation::ui::widgets::list::{EXPANDER_W, ROW_PAD};
use crate::render::overlay::widgets::WidgetScratch;

use super::{ControlFrame, Pen, inset};

pub(super) fn draw(
    list: &List,
    ctx: &PipelineContext<'_>,
    state: Option<&UiState>,
    frame: &ControlFrame<'_>,
    out: &mut WidgetScratch,
) {
    let style = &list.style;
    let mut pen = Pen::new(out, frame, style, list.screen, list.fit);
    pen.framed(
        list.rect(),
        style.background,
        style.corner_radius,
        style.border,
    );
    let items = state.map_or(&[][..], |s| s.list(&list.items));
    let count = items.len();
    let selected = list
        .selected
        .as_ref()
        .and_then(|b| bound::int_of(ctx, b))
        .and_then(|i| usize::try_from(i).ok());
    for i in list.rows_in_view(count) {
        let item = &items[i];
        let row = list.row_rect(i, count);
        if selected == Some(i) {
            pen.rect(inset(row), style.selected, style.corner_radius);
        } else if list.hovered == Some(i) {
            pen.rect(inset(row), style.hover, style.corner_radius);
        }
        let text_y = pen.text_y(row[1], row[3]);
        let ex = list.expander_x(item.depth);
        if item.expandable {
            let glyph = if item.expanded { "v" } else { ">" };
            pen.text(glyph, [ex, text_y], style.dim_text, TextAlign::Left, 0.0);
        }
        let tx = ex
            + if item.depth > 0 || item.expandable {
                EXPANDER_W
            } else {
                0.0
            };
        let right = row[0] + row[2] - ROW_PAD;
        let color = match item.color {
            Some(color) => color,
            None if item.dimmed => style.dim_text,
            None => style.text,
        };
        pen.text(&item.text, [tx, text_y], color, TextAlign::Left, right - tx);
        if !item.detail.is_empty() {
            pen.text(
                &item.detail,
                [right, text_y],
                style.dim_text,
                TextAlign::Right,
                0.0,
            );
        }
    }
    if let Some((track, thumb)) = list.scrollbar(count) {
        pen.rect(track, [0.12, 0.12, 0.15, 0.9], track[2] * 0.5);
        pen.rect(thumb, [0.40, 0.44, 0.56, 0.95], thumb[2] * 0.5);
    }
}
