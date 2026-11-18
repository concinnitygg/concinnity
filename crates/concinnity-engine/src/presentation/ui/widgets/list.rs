// A scrolling list of the rows of a list in the app's UI state, flat or as a
// tree.

use alloc::string::String;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

use crate::components::{Binding, Screen, SpriteFit};
use crate::ecs::Ref;

use super::scroll_view::ScrollView;
use super::{Icon, WidgetStyle};
use crate::components::UiItem;

/// A scrolling list of the items of a list in the app's
/// [`UiStateValues`](#uistatevalues), drawing only the rows in view, so it
/// stays cheap at thousands of items.
///
/// The app's systems fill the list `items` names; each item draws as a row of
/// its text, with its dimmer detail at the row's right end. Items with a
/// `depth` indent, and an expandable one draws an expander, so a tree is a list
/// the app flattens. The wheel scrolls the rows and a scrollbar shows where the
/// view sits.
///
/// A click selects a row, setting the whole number `selected` names to its
/// index, -1 for none. A double click activates it, a click on the expander
/// expands or collapses it, and a right click asks for its context menu: each
/// sends a `UiEvent` named by `on_select`, `on_activate`, `on_expand`, or
/// `on_context`, carrying the row's index. With a `selected` binding and the
/// list last clicked, the Up and Down keys move the selection and Enter
/// activates it.
///
/// An app that selects many items at once marks them `selected` and reads
/// `on_extend` for the modified clicks that add to the selection. A row can
/// lead with an image and with the list's toggle columns, and a press dragged
/// off a row sends `on_drag`.
///
/// ```rust
/// # use concinnity_engine::components::{Binding, List};
/// List {
///     items: "files".into(),
///     selected: Some(Binding::State("file".into())),
///     on_activate: "open_file".into(),
///     width: 320.0,
///     height: 400.0,
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
pub struct List {
    /// The list in the app's UI state whose items the rows show.
    pub items: String,
    /// A whole number set to the selected row's index, -1 for none: a
    /// `state:` or `var:` binding. Unset, rows do not select.
    pub selected: Option<Binding>,
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    #[asset(default = 240.0)]
    pub width: f32,
    /// Height of the view; rows past it scroll.
    #[asset(default = 320.0)]
    pub height: f32,
    /// Height of each row.
    #[asset(default = 26.0)]
    pub row_height: f32,
    /// How far each level of a tree indents its rows.
    #[asset(default = 16.0)]
    pub indent: f32,
    /// The event a click on a row sends.
    pub on_select: String,
    /// The event a double click or Enter on a row sends.
    pub on_activate: String,
    /// The event a click on a row's expander sends.
    pub on_expand: String,
    /// The event a right click on a row sends.
    pub on_context: String,
    /// The event a Shift, Ctrl, or Cmd click on a row sends in place of
    /// `on_select`, so the app can add the row to a selection of many. Unset,
    /// such a click selects as a plain one does.
    pub on_extend: String,
    /// The event a press on a row sends once the pointer moves a few pixels
    /// with the button held, so the app can drag the row's item out.
    pub on_drag: String,
    /// Icon columns at the head of the rows, each flipping a flag of its row:
    /// at most 8, and building the world drops any past it.
    pub toggles: Vec<ListToggle>,
    /// How the list draws.
    pub style: WidgetStyle,
    /// When false the list is not drawn and takes no input.
    #[asset(default = true)]
    pub visible: bool,
    /// How a screen-owned list maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) this list belongs to. Unset, it joins the world's
    /// HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime scroll offset in pixels from the first row. Not authored.
    #[serde(skip)]
    pub scroll: f32,
    /// Runtime index of the row under the cursor. Not authored.
    #[serde(skip)]
    pub hovered: Option<usize>,
}

/// A column of icons at the head of a [List](#list)'s rows, each cell
/// flipping one flag of its row, such as whether an item shows or is locked.
///
/// A row draws the cell when its item has toggles, the `on` icon while its
/// flag is set and the `off` one while it is not. With no `off` icon, an unset
/// cell is blank until the row is hovered, when it shows the `on` icon in the
/// `off` color so it can be found.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct ListToggle {
    /// The icon drawn while the flag is set.
    pub on: Icon,
    /// The icon drawn while it is not.
    pub off: Icon,
    /// RGBA color of the `on` icon.
    #[asset(default = [0.82, 0.84, 0.90, 1.0])]
    pub on_color: [f32; 4],
    /// RGBA color of the `off` icon, or of the hovered hint.
    #[asset(default = [0.48, 0.49, 0.55, 1.0])]
    pub off_color: [f32; 4],
    /// The event a click on the cell sends, carrying the row's index.
    pub on_toggle: String,
}

/// The most toggle columns a [List](#list) shows: one per bit of an item's
/// `toggles` flags.
pub const MAX_LIST_TOGGLES: usize = 8;

// Gap from a row's left edge to its expander or text.
pub(crate) const ROW_PAD: f32 = 8.0;
// Width an expander takes before an item's text.
pub(crate) const EXPANDER_W: f32 = 14.0;
// Gap after each toggle cell and a row's image.
pub(crate) const CELL_GAP: f32 = 5.0;

// Where a point on a row lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowPart {
    Expander,
    Toggle(usize),
    Body,
}

impl List {
    // The list's `[x, y, width, height]` rectangle.
    pub(crate) fn rect(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    fn row_h(&self) -> f32 {
        self.row_height.max(1.0)
    }

    // The scroll over the rows of `count` items.
    pub(crate) fn view(&self, count: usize) -> ScrollView {
        ScrollView {
            rect: self.rect(),
            content: count as f32 * self.row_h(),
            scroll: self.scroll,
        }
    }

    /// How far the rows of `count` items scroll at most.
    pub fn max_scroll(&self, count: usize) -> f32 {
        self.view(count).max_scroll()
    }

    // Width of the rows, short of the scrollbar when one shows.
    pub(crate) fn row_width(&self, count: usize) -> f32 {
        self.view(count).inner_width()
    }

    // The rows in view: the first index and how many, past the scroll.
    pub(crate) fn rows_in_view(&self, count: usize) -> core::ops::Range<usize> {
        let first = concinnity_core::math::floor(self.scroll.max(0.0) / self.row_h());
        let first = usize::saturating_from(first).min(count);
        let shown = usize::saturating_from(concinnity_core::math::ceil(self.height / self.row_h()));
        first..(first + shown + 1).min(count)
    }

    // Row `i`'s rectangle, placed by the scroll.
    pub(crate) fn row_rect(&self, i: usize, count: usize) -> [f32; 4] {
        [
            self.x,
            self.y + i as f32 * self.row_h() - self.scroll,
            self.row_width(count),
            self.row_h(),
        ]
    }

    // The row under `(x, y)`, if any: none over the scrollbar strip.
    pub(crate) fn row_at(&self, x: f32, y: f32, count: usize) -> Option<usize> {
        if y < self.y
            || y >= self.y + self.height
            || x < self.x
            || x >= self.x + self.row_width(count)
        {
            return None;
        }
        let i = concinnity_core::math::floor((y - self.y + self.scroll) / self.row_h());
        (i >= 0.0)
            .then(|| usize::saturating_from(i))
            .filter(|&i| i < count)
    }

    // Left edge of a row's expander at tree depth `depth`.
    pub(crate) fn expander_x(&self, depth: u16) -> f32 {
        self.x + ROW_PAD + f32::from(depth) * self.indent
    }

    // Side of a square cell on a row: a toggle or an image.
    pub(crate) fn cell_side(&self) -> f32 {
        (self.row_h() - 8.0).clamp(1.0, 16.0)
    }

    // Left edge of an item's toggle cells, after its indent and expander.
    pub(crate) fn toggles_x(&self, item: &UiItem) -> f32 {
        let x = self.expander_x(item.depth);
        if item.depth > 0 || item.expandable {
            x + EXPANDER_W
        } else {
            x
        }
    }

    // Left edge of toggle cell `t` of `item`.
    pub(crate) fn toggle_x(&self, item: &UiItem, t: usize) -> f32 {
        self.toggles_x(item) + t as f32 * (self.cell_side() + CELL_GAP)
    }

    // The toggle columns a row shows: no more than its flags have bits for.
    pub(crate) fn shown_toggles(&self) -> &[ListToggle] {
        &self.toggles[..self.toggles.len().min(MAX_LIST_TOGGLES)]
    }

    // Left edge of an item's text: after its toggles and image.
    pub(crate) fn text_x(&self, item: &UiItem) -> f32 {
        let cell = self.cell_side() + CELL_GAP;
        let mut x = self.toggles_x(item);
        if item.toggles.is_some() {
            x += self.shown_toggles().len() as f32 * cell;
        }
        if item.image.is_some() {
            x += cell;
        }
        x
    }

    // The part of `item`'s row a point at `x` lands on.
    pub(crate) fn part_at(&self, x: f32, item: &UiItem) -> RowPart {
        let ex = self.expander_x(item.depth);
        if item.expandable && x >= ex && x < ex + EXPANDER_W {
            return RowPart::Expander;
        }
        if item.toggles.is_some() {
            for t in 0..self.shown_toggles().len() {
                let tx = self.toggle_x(item, t);
                if x >= tx && x < tx + self.cell_side() + CELL_GAP {
                    return RowPart::Toggle(t);
                }
            }
        }
        RowPart::Body
    }

    /// Scroll so row `i` of `count` is fully in view, moving no further than
    /// it must.
    pub fn reveal(&mut self, i: usize, count: usize) {
        self.scroll = self
            .view(count)
            .revealing(i as f32 * self.row_h(), self.row_h());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn list() -> List {
        List {
            width: 200.0,
            height: 100.0,
            row_height: 20.0,
            ..Default::default()
        }
    }

    #[test]
    fn rows_in_view_follow_the_scroll() {
        let mut l = list();
        assert_eq!(l.rows_in_view(3), 0..3);
        assert_eq!(l.rows_in_view(100), 0..6, "a partial row below");
        l.scroll = 45.0;
        assert_eq!(l.rows_in_view(100), 2..8);
        assert_eq!(l.row_rect(2, 100)[1], -5.0, "the top row cut by the edge");
        assert_eq!(l.row_at(20.0, 10.0, 100), Some(2));
        assert_eq!(l.row_at(20.0, 200.0, 100), None, "below the view");
        assert_eq!(l.row_at(198.0, 10.0, 100), None, "on the scrollbar");
        assert_eq!(l.row_at(198.0, 10.0, 3), Some(2), "no scrollbar to be on");
    }

    #[test]
    fn a_row_finds_its_toggle_cells_after_its_indent() {
        let mut l = list();
        l.toggles = vec![ListToggle::default(), ListToggle::default()];
        let item = UiItem {
            depth: 1,
            toggles: Some(0b01),
            ..UiItem::new("a")
        };
        let side = l.cell_side();
        let first = l.toggle_x(&item, 0);
        assert_eq!(first, ROW_PAD + l.indent + EXPANDER_W);
        assert_eq!(l.part_at(first + 1.0, &item), RowPart::Toggle(0));
        assert_eq!(
            l.part_at(first + side + CELL_GAP + 1.0, &item),
            RowPart::Toggle(1)
        );
        assert_eq!(l.text_x(&item), first + 2.0 * (side + CELL_GAP));
        assert_eq!(l.part_at(l.text_x(&item) + 1.0, &item), RowPart::Body);
        let plain = UiItem::new("b");
        assert_eq!(l.text_x(&plain), ROW_PAD, "no toggles, no room for them");
    }

    #[test]
    fn toggles_past_the_flag_bits_take_no_cells() {
        let mut l = list();
        l.toggles = vec![ListToggle::default(); MAX_LIST_TOGGLES + 2];
        let item = UiItem {
            toggles: Some(u8::MAX),
            ..UiItem::new("a")
        };
        let cell = l.cell_side() + CELL_GAP;
        let past = l.toggles_x(&item) + MAX_LIST_TOGGLES as f32 * cell + 1.0;
        assert_eq!(l.text_x(&item), past - 1.0);
        assert_eq!(l.part_at(past, &item), RowPart::Body);
    }

    #[test]
    fn reveal_scrolls_a_row_just_into_view() {
        let mut l = list();
        l.reveal(10, 20);
        assert_eq!(l.scroll, 120.0, "row 10 at the bottom edge");
        l.reveal(2, 20);
        assert_eq!(l.scroll, 40.0, "row 2 at the top edge");
    }
}
