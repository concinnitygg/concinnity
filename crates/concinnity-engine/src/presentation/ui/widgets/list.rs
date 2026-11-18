// A scrolling list of the rows of a list in the app's UI state, flat or as a
// tree.

use alloc::string::String;
use concinnity_core::math::SaturatingFrom;

use crate::components::{Binding, Screen, SpriteFit};
use crate::ecs::Ref;

use super::WidgetStyle;

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
/// `on_context`, carrying the row's index. With the list last clicked, the Up
/// and Down keys move the selection and Enter activates it.
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

// Width of the scrollbar strip along a scrolling list's right edge.
pub(crate) const SCROLLBAR_W: f32 = 6.0;
// Gap from a row's left edge to its expander or text.
pub(crate) const ROW_PAD: f32 = 8.0;
// Width an expander takes before an item's text.
pub(crate) const EXPANDER_W: f32 = 14.0;
const MIN_THUMB: f32 = 18.0;

impl List {
    // The list's `[x, y, width, height]` rectangle.
    pub(crate) fn rect(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    fn row_h(&self) -> f32 {
        self.row_height.max(1.0)
    }

    // How far the rows of `count` items scroll at most.
    pub(crate) fn max_scroll(&self, count: usize) -> f32 {
        (count as f32 * self.row_h() - self.height).max(0.0)
    }

    // Whether `count` items overflow the view, so a scrollbar shows.
    pub(crate) fn scrolls(&self, count: usize) -> bool {
        self.max_scroll(count) > 0.0
    }

    // Width of the rows, short of the scrollbar when one shows.
    pub(crate) fn row_width(&self, count: usize) -> f32 {
        if self.scrolls(count) {
            (self.width - SCROLLBAR_W).max(0.0)
        } else {
            self.width
        }
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

    // The scrollbar's track and thumb for `count` items, or `None` when every
    // row shows.
    pub(crate) fn scrollbar(&self, count: usize) -> Option<([f32; 4], [f32; 4])> {
        if !self.scrolls(count) {
            return None;
        }
        let track = [
            self.x + self.width - SCROLLBAR_W,
            self.y,
            SCROLLBAR_W,
            self.height,
        ];
        let total = count as f32 * self.row_h();
        let len =
            (self.height * self.height / total).clamp(MIN_THUMB.min(self.height), self.height);
        let at = (self.height - len) * (self.scroll / self.max_scroll(count)).clamp(0.0, 1.0);
        Some((track, [track[0], self.y + at, SCROLLBAR_W, len]))
    }

    // The scroll that puts the thumb's top `thumb_top` down its track.
    pub(crate) fn scroll_for_thumb(&self, thumb_top: f32, count: usize) -> f32 {
        let Some((track, thumb)) = self.scrollbar(count) else {
            return 0.0;
        };
        let travel = (track[3] - thumb[3]).max(1.0);
        ((thumb_top - track[1]) / travel).clamp(0.0, 1.0) * self.max_scroll(count)
    }

    // Scroll so row `i` is fully in view.
    pub(crate) fn reveal(&mut self, i: usize, count: usize) {
        let top = i as f32 * self.row_h();
        if top < self.scroll {
            self.scroll = top;
        } else if top + self.row_h() > self.scroll + self.height {
            self.scroll = top + self.row_h() - self.height;
        }
        self.scroll = self.scroll.clamp(0.0, self.max_scroll(count));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn a_scrollbar_shows_only_while_rows_overflow() {
        let mut l = list();
        assert!(l.scrollbar(5).is_none());
        assert_eq!(l.row_width(5), 200.0);
        let (_, thumb) = l.scrollbar(20).unwrap();
        assert_eq!(thumb[3], 25.0, "a quarter of the rows show");
        assert_eq!(l.row_width(20), 200.0 - SCROLLBAR_W);
        l.scroll = l.max_scroll(20);
        let (track, thumb) = l.scrollbar(20).unwrap();
        assert_eq!(thumb[1] + thumb[3], track[1] + track[3], "at the bottom");
        assert_eq!(l.scroll_for_thumb(track[1], 20), 0.0);
        assert_eq!(l.scroll_for_thumb(track[1] + 75.0, 20), l.max_scroll(20));
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
