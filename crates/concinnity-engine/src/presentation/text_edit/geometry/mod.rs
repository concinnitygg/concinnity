// Where each part of a text editor falls inside the rect it is drawn in: the
// line-number gutter, the monospace grid of visible text, and the scrollbars.

use concinnity_core::math::SaturatingFrom;

use super::pointer::Bar;
use super::view::{self, ViewSize};
use super::{Pos, TextEditor};

/// The most rows a text editor lays out; a taller rect leaves the rest blank.
pub const ROW_POOL: usize = 64;

/// Room left of the line numbers for a gutter marker.
pub const MARKER_W: f32 = 14.0;
/// Gap between the line numbers and the gutter's right edge.
pub const GUTTER_PAD: f32 = 10.0;
/// Gap between the gutter and the first text cell.
pub const TEXT_PAD: f32 = 6.0;
/// Thickness of the scrollbar strips along the right and bottom edges.
pub const SCROLLBAR: f32 = 6.0;
const MIN_THUMB: f32 = 18.0;

/// The monospace grid a text editor lays out on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Height of one line.
    pub line_h: f32,
    /// Width of one cell.
    pub advance: f32,
    /// The face's pixel size.
    pub text_px: f32,
    /// The scale the face's labels draw at.
    pub label_scale: f32,
}

/// Where each part of a text editor falls inside its rect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    /// The whole rect.
    pub rect: [f32; 4],
    /// The grid it lays out on.
    pub metrics: Metrics,
    /// Width of the line-number gutter.
    pub gutter_w: f32,
    /// Left edge of the first text cell.
    pub text_x: f32,
    /// Lines shown.
    pub rows: usize,
    /// Cells shown across.
    pub cols: usize,
}

// Decimal digits in `n`.
fn digits(n: usize) -> usize {
    n.max(1).ilog10() as usize + 1
}

/// Whether `(x, y)` lies inside the `[x, y, width, height]` rectangle.
pub fn point_in(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}

/// The rect height that shows exactly `rows` rows of `line_h`.
pub fn height_for_rows(rows: usize, line_h: f32) -> f32 {
    rows as f32 * line_h + SCROLLBAR
}

/// Lay a text of `line_count` lines out in `rect`. The scrollbar strips are
/// always reserved, so the grid does not jump when one appears.
pub fn geometry(rect: [f32; 4], metrics: Metrics, line_count: usize) -> Geometry {
    let adv = metrics.advance.max(1.0);
    let gutter_w = MARKER_W + digits(line_count).max(2) as f32 * adv + GUTTER_PAD;
    let text_x = rect[0] + gutter_w + TEXT_PAD;
    let text_w = rect[0] + rect[2] - SCROLLBAR - text_x;
    let rows = concinnity_core::math::floor((rect[3] - SCROLLBAR) / metrics.line_h.max(1.0));
    Geometry {
        rect,
        metrics,
        gutter_w,
        text_x,
        rows: usize::saturating_from(rows.max(1.0)).min(ROW_POOL),
        cols: usize::saturating_from(concinnity_core::math::floor(text_w / adv).max(1.0)),
    }
}

impl Geometry {
    /// How much of the text the rect shows.
    pub fn view(&self) -> ViewSize {
        ViewSize {
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Whether `(x, y)` is inside the rect.
    pub fn contains(&self, x: f32, y: f32) -> bool {
        point_in(x, y, self.rect)
    }

    /// Top edge of visible row `row`.
    pub fn row_y(&self, row: usize) -> f32 {
        self.rect[1] + row as f32 * self.metrics.line_h
    }

    /// The line-number gutter.
    pub fn gutter_rect(&self) -> [f32; 4] {
        [self.rect[0], self.rect[1], self.gutter_w, self.rect[3]]
    }

    /// The strip the vertical scrollbar rides along the right edge.
    pub fn v_track(&self) -> [f32; 4] {
        let r = self.rect;
        [r[0] + r[2] - SCROLLBAR, r[1], SCROLLBAR, r[3] - SCROLLBAR]
    }

    /// The strip the horizontal scrollbar rides along the bottom edge.
    pub fn h_track(&self) -> [f32; 4] {
        let r = self.rect;
        let x = r[0] + self.gutter_w;
        [
            x,
            r[1] + r[3] - SCROLLBAR,
            r[0] + r[2] - SCROLLBAR - x,
            SCROLLBAR,
        ]
    }

    /// The text position under a point, for a press or a drag. A point above
    /// or below the window reaches one line past it, so a drag scrolls.
    pub fn pos_at(&self, editor: &TextEditor, x: f32, y: f32) -> Pos {
        let row = isize::saturating_from(concinnity_core::math::floor(
            (y - self.rect[1]) / self.metrics.line_h,
        ));
        let last = isize::saturating_from(editor.line_count()) - 1;
        let line = usize::saturating_from(
            (isize::saturating_from(editor.scroll().top) + row).clamp(0, last),
        );
        let cell = (x - self.text_x) / self.metrics.advance + editor.scroll().left as f32;
        Pos::new(line, view::col_at_visual(editor.line(line), cell.max(0.0)))
    }

    /// The line a gutter row at `y` shows, if any.
    pub fn gutter_line(&self, editor: &TextEditor, y: f32) -> Option<usize> {
        let row = concinnity_core::math::floor((y - self.rect[1]) / self.metrics.line_h);
        if row < 0.0 || usize::saturating_from(row) >= self.rows {
            return None;
        }
        let line = editor.scroll().top + usize::saturating_from(row);
        (line < editor.line_count()).then_some(line)
    }

    pub(crate) fn bar_at(&self, editor: &TextEditor, x: f32, y: f32) -> Option<Bar> {
        if point_in(x, y, self.v_track()) && editor.line_count() > self.rows {
            Some(Bar::Vertical)
        } else if point_in(x, y, self.h_track()) && editor.widest() + 1 > self.cols {
            Some(Bar::Horizontal)
        } else {
            None
        }
    }

    /// Whether `(x, y)` is in the line-number gutter.
    pub fn in_gutter(&self, x: f32, y: f32) -> bool {
        point_in(x, y, self.gutter_rect())
    }

    /// The vertical scrollbar's thumb, or `None` when every line shows.
    pub fn v_thumb(&self, editor: &TextEditor) -> Option<[f32; 4]> {
        let t = self.v_track();
        thumb(t[3], self.rows, editor.line_count(), editor.scroll().top)
            .map(|(at, len)| [t[0], t[1] + at, t[2], len])
    }

    /// The horizontal scrollbar's thumb, or `None` when every cell shows.
    pub fn h_thumb(&self, editor: &TextEditor) -> Option<[f32; 4]> {
        let t = self.h_track();
        thumb(t[2], self.cols, editor.widest() + 1, editor.scroll().left)
            .map(|(at, len)| [t[0] + at, t[1], len, t[3]])
    }
}

// One scrollbar's thumb along a track of length `len`: its offset and length,
// or `None` when everything fits.
fn thumb(len: f32, shown: usize, total: usize, first: usize) -> Option<(f32, f32)> {
    if total <= shown {
        return None;
    }
    let size = (len * shown as f32 / total as f32).max(MIN_THUMB).min(len);
    let max_first = (total - shown) as f32;
    Some(((len - size) * (first as f32 / max_first).min(1.0), size))
}

/// The scroll offset that centers a scrollbar's thumb on a point along its
/// track, for a thumb drag or a track click.
pub(crate) fn scroll_on_track(len: f32, at: f32, shown: usize, total: usize) -> usize {
    let Some((_, size)) = thumb(len, shown, total, 0) else {
        return 0;
    };
    let travel = (len - size).max(1.0);
    let frac = ((at - size * 0.5) / travel).clamp(0.0, 1.0);
    usize::saturating_from(concinnity_core::math::round(frac * (total - shown) as f32))
}

#[cfg(test)]
mod tests;
