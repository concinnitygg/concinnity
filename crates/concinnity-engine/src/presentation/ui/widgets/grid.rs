// A scrolling grid of cells over a list in the app's UI state: each cell an
// item's image over its text.

use alloc::string::String;
use concinnity_core::math::SaturatingFrom;

use crate::components::{Binding, Screen, SpriteFit};
use crate::ecs::Ref;

use super::WidgetStyle;
use super::scroll_view::{SCROLLBAR_W, ScrollView};

/// A scrolling grid of the items of a list in the app's
/// [`UiStateValues`](#uistatevalues), as many columns of cells as its width
/// holds, drawing only the cells in view.
///
/// Each item draws as a cell of its image over its text, the text in the
/// item's color; an item with no image shows its detail in the image's place,
/// in its detail color. The wheel
/// scrolls the rows of cells and a scrollbar shows where the view sits.
///
/// A click selects a cell, setting the whole number `selected` names to its
/// index, -1 for none, and a double click activates it; a right click asks
/// for its context menu, and a Shift, Ctrl, or Cmd click sends `on_extend` in
/// place of `on_select`, so the app can keep a selection of many items it
/// marks `selected`. A press dragged off a cell sends `on_drag`. Each event
/// carries the cell's index. With a `selected` binding and the grid last
/// clicked, the arrow keys move the selection and Enter activates it.
///
/// ```rust
/// # use concinnity_engine::components::{Binding, Grid};
/// Grid {
///     items: "textures".into(),
///     selected: Some(Binding::State("texture".into())),
///     on_drag: "drag_texture".into(),
///     width: 400.0,
///     height: 360.0,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Grid {
    /// The list in the app's UI state whose items the cells show.
    pub items: String,
    /// A whole number set to the selected cell's index, -1 for none: a
    /// `state:` or `var:` binding. Unset, cells select only by the events
    /// they send.
    pub selected: Option<Binding>,
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    #[asset(default = 400.0)]
    pub width: f32,
    /// Height of the view; rows of cells past it scroll.
    #[asset(default = 360.0)]
    pub height: f32,
    /// Width of each cell.
    #[asset(default = 92.0)]
    pub cell_width: f32,
    /// Height of each cell, its text included.
    #[asset(default = 112.0)]
    pub cell_height: f32,
    /// Space between neighboring cells.
    #[asset(default = 6.0)]
    pub gap: f32,
    /// The event a click on a cell sends.
    pub on_select: String,
    /// The event a double click or Enter on a cell sends.
    pub on_activate: String,
    /// The event a right click on a cell sends.
    pub on_context: String,
    /// The event a Shift, Ctrl, or Cmd click on a cell sends in place of
    /// `on_select`. Unset, such a click selects as a plain one does.
    pub on_extend: String,
    /// The event a press on a cell sends once the pointer moves a few pixels
    /// with the button held.
    pub on_drag: String,
    /// How the grid draws.
    pub style: WidgetStyle,
    /// When false the grid is not drawn and takes no input.
    #[asset(default = true)]
    pub visible: bool,
    /// How a screen-owned grid maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) this grid belongs to. Unset, it joins the world's
    /// HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime scroll offset in pixels from the first row. Not authored.
    #[serde(skip)]
    pub scroll: f32,
    /// Runtime index of the cell under the cursor. Not authored.
    #[serde(skip)]
    pub hovered: Option<usize>,
}

// Inset of a cell's image from its edges.
pub(crate) const CELL_PAD: f32 = 4.0;

impl Grid {
    // The grid's `[x, y, width, height]` rectangle.
    pub(crate) fn rect(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    fn cell_w(&self) -> f32 {
        self.cell_width.max(1.0)
    }

    fn row_h(&self) -> f32 {
        self.cell_height.max(1.0) + self.gap.max(0.0)
    }

    fn columns_in(&self, width: f32) -> usize {
        let step = self.cell_w() + self.gap.max(0.0);
        usize::saturating_from(concinnity_core::math::floor(
            (width + self.gap.max(0.0)) / step,
        ))
        .max(1)
    }

    fn rows_of(count: usize, columns: usize) -> usize {
        count.div_ceil(columns)
    }

    fn content_of(&self, count: usize, columns: usize) -> f32 {
        (Self::rows_of(count, columns) as f32 * self.row_h() - self.gap.max(0.0)).max(0.0)
    }

    // How many cells a row holds for `count` items: as many as the width
    // fits, short of the scrollbar when one shows.
    pub(crate) fn columns(&self, count: usize) -> usize {
        let wide = self.columns_in(self.width);
        if self.content_of(count, wide) > self.height {
            self.columns_in(self.width - SCROLLBAR_W)
        } else {
            wide
        }
    }

    // The scroll over the rows of cells of `count` items.
    pub(crate) fn view(&self, count: usize) -> ScrollView {
        ScrollView {
            rect: self.rect(),
            content: self.content_of(count, self.columns(count)),
            scroll: self.scroll,
        }
    }

    /// How far the rows of cells of `count` items scroll at most.
    pub fn max_scroll(&self, count: usize) -> f32 {
        self.view(count).max_scroll()
    }

    // The cells in view: the first index and how many, past the scroll.
    pub(crate) fn cells_in_view(&self, count: usize) -> core::ops::Range<usize> {
        let columns = self.columns(count);
        let first = concinnity_core::math::floor(self.scroll.max(0.0) / self.row_h());
        let first = usize::saturating_from(first);
        let shown = usize::saturating_from(concinnity_core::math::ceil(self.height / self.row_h()));
        (first * columns).min(count)..((first + shown + 1) * columns).min(count)
    }

    // Cell `i`'s rectangle, placed by the scroll.
    pub(crate) fn cell_rect(&self, i: usize, count: usize) -> [f32; 4] {
        let columns = self.columns(count);
        let (col, row) = (i % columns, i / columns);
        [
            self.x + col as f32 * (self.cell_w() + self.gap.max(0.0)),
            self.y + row as f32 * self.row_h() - self.scroll,
            self.cell_w(),
            self.cell_height.max(1.0),
        ]
    }

    // The cell under `(x, y)`, if any: none in a gap or over the scrollbar.
    pub(crate) fn cell_at(&self, x: f32, y: f32, count: usize) -> Option<usize> {
        let view = self.view(count);
        if y < self.y || y >= self.y + self.height || x < self.x || x >= self.x + view.inner_width()
        {
            return None;
        }
        let columns = self.columns(count);
        let col = concinnity_core::math::floor((x - self.x) / (self.cell_w() + self.gap.max(0.0)));
        let row = concinnity_core::math::floor((y - self.y + self.scroll) / self.row_h());
        if col < 0.0 || row < 0.0 {
            return None;
        }
        let i = usize::saturating_from(row) * columns + usize::saturating_from(col);
        (usize::saturating_from(col) < columns && i < count)
            .then_some(i)
            .filter(|&i| super::point_in(x, y, self.cell_rect(i, count)))
    }

    // Where cell `cell`'s image or placeholder draws: above its text, which
    // takes a line of `line_h`.
    pub(crate) fn image_area(cell: [f32; 4], line_h: f32) -> [f32; 4] {
        [
            cell[0] + CELL_PAD,
            cell[1] + CELL_PAD,
            (cell[2] - CELL_PAD * 2.0).max(0.0),
            (cell[3] - CELL_PAD * 3.0 - line_h).max(0.0),
        ]
    }

    // Scroll so cell `i` is fully in view.
    pub(crate) fn reveal(&mut self, i: usize, count: usize) {
        let row = i / self.columns(count);
        self.scroll = self
            .view(count)
            .revealing(row as f32 * self.row_h(), self.cell_height.max(1.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid {
        Grid {
            width: 295.0,
            height: 200.0,
            cell_width: 90.0,
            cell_height: 90.0,
            gap: 10.0,
            ..Default::default()
        }
    }

    #[test]
    fn cells_fill_the_width_and_give_way_to_the_scrollbar() {
        let g = grid();
        assert_eq!(g.columns(3), 3, "295 holds three 90s with two gaps");
        assert!(g.view(6).scrollbar().is_none(), "two rows fit");
        assert_eq!(g.columns(9), 2, "a scrollbar takes a column's room");
        assert_eq!(g.cell_rect(4, 6), [100.0, 100.0, 90.0, 90.0]);
    }

    #[test]
    fn a_point_finds_its_cell_and_never_a_gap() {
        let mut g = grid();
        assert_eq!(g.cell_at(10.0, 10.0, 6), Some(0));
        assert_eq!(g.cell_at(95.0, 10.0, 6), None, "the gap between cells");
        assert_eq!(g.cell_at(250.0, 150.0, 6), Some(5));
        assert_eq!(g.cell_at(250.0, 150.0, 5), None, "past the last item");
        g.scroll = 100.0;
        assert_eq!(g.cell_at(10.0, 10.0, 12), Some(2), "a row scrolled away");
        assert_eq!(g.cells_in_view(12), 2..8);
    }

    #[test]
    fn reveal_scrolls_a_cell_row_into_view() {
        let mut g = grid();
        g.reveal(9, 12);
        assert_eq!(g.scroll, 490.0 - 200.0, "row 4 at the bottom edge");
        g.reveal(0, 12);
        assert_eq!(g.scroll, 0.0);
    }
}
