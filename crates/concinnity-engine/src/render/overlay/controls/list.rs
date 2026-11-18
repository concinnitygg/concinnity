// The draw of the widgets that show a list's items. A List: its background,
// the rows in view with the selection and hover highlights, tree expanders and
// indents, toggle cells and images, and the scrollbar. A Grid: the cells in
// view, each an image or its placeholder over the item's text.

use crate::components::{Grid, List, TextAlign, UiItem, UiState, WidgetStyle};
use crate::ecs::PipelineContext;
use crate::presentation::ui::bound;
use crate::presentation::ui::widgets::grid::CELL_PAD;
use crate::presentation::ui::widgets::list::{CELL_GAP, ROW_PAD};
use crate::presentation::ui::widgets::scroll_view::ScrollView;
use crate::render::overlay::widgets::WidgetScratch;

use super::{ControlFrame, Pen, faded, inset};

const TRACK: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB: [f32; 4] = [0.40, 0.44, 0.56, 0.95];

// The index a widget's `selected` binding holds, if any.
fn selected_index(
    binding: Option<&crate::components::Binding>,
    ctx: &PipelineContext<'_>,
) -> Option<usize> {
    binding
        .and_then(|b| bound::int_of(ctx, b))
        .and_then(|i| usize::try_from(i).ok())
}

// The fill behind item `i`: selected, hovered, or none.
fn highlight(
    style: &WidgetStyle,
    item: &UiItem,
    i: usize,
    selected: Option<usize>,
    hovered: Option<usize>,
) -> Option<[f32; 4]> {
    if selected == Some(i) || item.selected {
        Some(style.selected)
    } else if hovered == Some(i) {
        Some(style.hover)
    } else {
        None
    }
}

fn text_color(style: &WidgetStyle, item: &UiItem) -> [f32; 3] {
    match item.color {
        Some(color) => color,
        None if item.dimmed => style.dim_text,
        None => style.text,
    }
}

fn scrollbar(pen: &mut Pen<'_>, view: &ScrollView) {
    if let Some((track, thumb)) = view.scrollbar() {
        pen.rect(track, TRACK, track[2] * 0.5);
        pen.rect(thumb, THUMB, thumb[2] * 0.5);
    }
}

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
    let selected = selected_index(list.selected.as_ref(), ctx);
    let side = list.cell_side();
    for i in list.rows_in_view(count) {
        let item = &items[i];
        let row = list.row_rect(i, count);
        if let Some(fill) = highlight(style, item, i, selected, list.hovered) {
            pen.rect(inset(row), fill, style.corner_radius);
        }
        let text_y = pen.text_y(row[1], row[3]);
        if item.expandable {
            let glyph = if item.expanded { "v" } else { ">" };
            let ex = list.expander_x(item.depth);
            pen.text(glyph, [ex, text_y], style.dim_text, TextAlign::Left, 0.0);
        }
        let cell_y = row[1] + (row[3] - side) * 0.5;
        if let Some(flags) = item.toggles {
            for (t, toggle) in list.shown_toggles().iter().enumerate() {
                let cell = [list.toggle_x(item, t), cell_y, side];
                let on = flags & (1_u8 << t) != 0;
                if on {
                    pen.icon(&toggle.on, cell, toggle.on_color);
                } else if !toggle.off.is_empty() {
                    pen.icon(&toggle.off, cell, toggle.off_color);
                } else if list.hovered == Some(i) {
                    pen.icon(&toggle.on, cell, toggle.off_color);
                }
            }
        }
        let tx = list.text_x(item);
        if let Some(image) = &item.image {
            pen.image(image, [tx - side - CELL_GAP, cell_y, side, side]);
        }
        let right = row[0] + row[2] - ROW_PAD;
        let color = text_color(style, item);
        pen.text(&item.text, [tx, text_y], color, TextAlign::Left, right - tx);
        if !item.detail.is_empty() {
            let detail = item.detail_color.unwrap_or(style.dim_text);
            pen.text(&item.detail, [right, text_y], detail, TextAlign::Right, 0.0);
        }
    }
    scrollbar(&mut pen, &list.view(count));
}

pub(super) fn draw_grid(
    grid: &Grid,
    ctx: &PipelineContext<'_>,
    state: Option<&UiState>,
    frame: &ControlFrame<'_>,
    out: &mut WidgetScratch,
) {
    let style = &grid.style;
    let mut pen = Pen::new(out, frame, style, grid.screen, grid.fit);
    pen.framed(
        grid.rect(),
        style.background,
        style.corner_radius,
        style.border,
    );
    let items = state.map_or(&[][..], |s| s.list(&grid.items));
    let count = items.len();
    let selected = selected_index(grid.selected.as_ref(), ctx);
    let line_h = pen.line_h();
    for i in grid.cells_in_view(count) {
        let item = &items[i];
        let cell = grid.cell_rect(i, count);
        let fill =
            highlight(style, item, i, selected, grid.hovered).unwrap_or_else(|| faded(style.hover));
        pen.rect(cell, fill, style.corner_radius);
        let area = Grid::image_area(cell, line_h);
        match &item.image {
            Some(image) => pen.image(image, area),
            None if !item.detail.is_empty() => {
                let at = [area[0] + area[2] * 0.5, pen.text_y(area[1], area[3])];
                let color = item.detail_color.unwrap_or(style.dim_text);
                pen.text(&item.detail, at, color, TextAlign::Center, area[2]);
            }
            None => {}
        }
        let caption = [area[0], area[1] + area[3] + CELL_PAD];
        pen.text(
            &item.text,
            caption,
            text_color(style, item),
            TextAlign::Left,
            area[2],
        );
    }
    scrollbar(&mut pen, &grid.view(count));
}
