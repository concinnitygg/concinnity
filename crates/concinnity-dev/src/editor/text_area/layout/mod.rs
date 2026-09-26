//! The text area drawn into a rect of the HUD: a line-number gutter with its
//! markers, the visible window of lines as monospace labels, the selection, the
//! caret, and the scrollbars. Only the visible rows have elements, from a fixed
//! pool of reserved ids the owning panel injects.

use concinnity_core::components::TextAlign;
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use super::markers::{GutterMarker, Severity, marker_on};
use super::pointer::Bar;
use super::view::{self, ViewSize};
use super::{Pos, TextArea};
use crate::editor::code_font;
use crate::editor::theme;
use crate::editor::widget::{self, place_rounded, point_in};

// The most rows a text area draws; a taller rect leaves the rest blank.
pub(crate) const ROW_POOL: usize = 64;

const GUTTER_PAD: f32 = 10.0;
const MARKER_SIZE: f32 = 8.0;
const MARKER_W: f32 = 14.0;
const TEXT_PAD: f32 = 6.0;
const SCROLLBAR: f32 = 6.0;
const MIN_THUMB: f32 = 18.0;
const CARET_W: f32 = 2.0;

const WELL_TINT: [f32; 4] = [0.07, 0.07, 0.09, 1.0];
const GUTTER_TINT: [f32; 4] = [0.09, 0.09, 0.115, 1.0];
const CURRENT_LINE_TINT: [f32; 4] = [0.14, 0.15, 0.20, 1.0];
const SELECTION_TINT: [f32; 4] = [0.20, 0.32, 0.55, 1.0];
const CARET_TINT: [f32; 4] = [0.92, 0.93, 0.97, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const NUMBER_COLOR: [f32; 3] = [0.42, 0.44, 0.52];
const NUMBER_CURRENT: [f32; 3] = [0.78, 0.80, 0.88];

fn marker_tint(s: Severity) -> [f32; 4] {
    let [r, g, b] = match s {
        Severity::Error => theme::LOG_ERROR,
        Severity::Warning => theme::LOG_WARN,
    };
    [r, g, b, 1.0]
}

// A text area's reserved element ids: one family starting at `base`, taking
// `0xD0 + ROW_POOL` ids of its owner's block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextAreaIds {
    base: u32,
}

impl TextAreaIds {
    pub(crate) const fn new(base: u32) -> Self {
        Self { base }
    }

    const fn at(self, offset: u32) -> AssetId {
        AssetId(self.base + offset)
    }
    fn well(self) -> AssetId {
        self.at(0)
    }
    fn gutter(self) -> AssetId {
        self.at(1)
    }
    fn current_line(self) -> AssetId {
        self.at(2)
    }
    fn caret(self) -> AssetId {
        self.at(3)
    }
    fn v_track(self) -> AssetId {
        self.at(4)
    }
    fn v_thumb(self) -> AssetId {
        self.at(5)
    }
    fn h_track(self) -> AssetId {
        self.at(6)
    }
    fn h_thumb(self) -> AssetId {
        self.at(7)
    }
    fn selection(self, row: usize) -> AssetId {
        self.at(0x10 + row as u32)
    }
    fn marker(self, row: usize) -> AssetId {
        self.at(0x50 + row as u32)
    }
    fn number(self, row: usize) -> AssetId {
        self.at(0x90 + row as u32)
    }
    fn text(self, row: usize) -> AssetId {
        self.at(0xD0 + row as u32)
    }

    // Every sprite, in draw order: the well and gutter, then the line
    // highlights and markers, the caret over them, the scrollbars on top.
    pub(crate) fn sprite_ids(self) -> Vec<AssetId> {
        let mut ids = vec![self.well(), self.gutter(), self.current_line()];
        ids.extend((0..ROW_POOL).map(|r| self.selection(r)));
        ids.extend((0..ROW_POOL).map(|r| self.marker(r)));
        ids.extend([
            self.caret(),
            self.v_track(),
            self.v_thumb(),
            self.h_track(),
            self.h_thumb(),
        ]);
        ids
    }

    // Every label: the line numbers and the lines, all in the code face.
    pub(crate) fn code_label_ids(self) -> Vec<AssetId> {
        (0..ROW_POOL)
            .map(|r| self.number(r))
            .chain((0..ROW_POOL).map(|r| self.text(r)))
            .collect()
    }
}

// The monospace grid the area lays out on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Metrics {
    pub(crate) line_h: f32,
    pub(crate) advance: f32,
    pub(crate) text_px: f32,
    pub(crate) label_scale: f32,
}

impl Metrics {
    // The editor's code face.
    pub(crate) fn code() -> Self {
        Self {
            line_h: code_font::LINE_H,
            advance: code_font::advance(),
            text_px: code_font::TEXT_PX,
            label_scale: code_font::SCALE,
        }
    }
}

// Where each part of the area falls inside its rect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Geometry {
    pub(crate) rect: [f32; 4],
    pub(crate) metrics: Metrics,
    gutter_w: f32,
    text_x: f32,
    rows: usize,
    cols: usize,
}

// Decimal digits in `n`.
fn digits(n: usize) -> usize {
    n.max(1).ilog10() as usize + 1
}

// The rect height that shows exactly `rows` rows of `line_h`.
pub(crate) fn height_for_rows(rows: usize, line_h: f32) -> f32 {
    rows as f32 * line_h + SCROLLBAR
}

// Lay the area out in `rect` for a text of `line_count` lines. The scrollbar
// strips are always reserved, so the grid does not jump when one appears.
pub(crate) fn geometry(rect: [f32; 4], metrics: Metrics, line_count: usize) -> Geometry {
    let adv = metrics.advance.max(1.0);
    let gutter_w = MARKER_W + digits(line_count).max(2) as f32 * adv + GUTTER_PAD;
    let text_x = rect[0] + gutter_w + TEXT_PAD;
    let text_w = rect[0] + rect[2] - SCROLLBAR - text_x;
    let rows = ((rect[3] - SCROLLBAR) / metrics.line_h.max(1.0)).floor();
    Geometry {
        rect,
        metrics,
        gutter_w,
        text_x,
        rows: (rows.max(1.0) as usize).min(ROW_POOL),
        cols: (text_w / adv).floor().max(1.0) as usize,
    }
}

impl Geometry {
    pub(crate) fn view(&self) -> ViewSize {
        ViewSize {
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub(crate) fn contains(&self, x: f32, y: f32) -> bool {
        point_in(x, y, self.rect)
    }

    fn row_y(&self, row: usize) -> f32 {
        self.rect[1] + row as f32 * self.metrics.line_h
    }

    fn gutter_rect(&self) -> [f32; 4] {
        [self.rect[0], self.rect[1], self.gutter_w, self.rect[3]]
    }

    // The strip the scrollbars ride: right edge, and the bottom edge.
    pub(crate) fn v_track(&self) -> [f32; 4] {
        let r = self.rect;
        [r[0] + r[2] - SCROLLBAR, r[1], SCROLLBAR, r[3] - SCROLLBAR]
    }

    pub(crate) fn h_track(&self) -> [f32; 4] {
        let r = self.rect;
        let x = r[0] + self.gutter_w;
        [
            x,
            r[1] + r[3] - SCROLLBAR,
            r[0] + r[2] - SCROLLBAR - x,
            SCROLLBAR,
        ]
    }

    // The text position under a point, for a press or a drag. A point above or
    // below the window reaches one line past it, so a drag scrolls.
    pub(crate) fn pos_at(&self, area: &TextArea, x: f32, y: f32) -> Pos {
        let row = ((y - self.rect[1]) / self.metrics.line_h).floor() as isize;
        let last = area.line_count() as isize - 1;
        let line = (area.scroll().top as isize + row).clamp(0, last) as usize;
        let cell = (x - self.text_x) / self.metrics.advance + area.scroll().left as f32;
        Pos::new(line, view::col_at_visual(area.line(line), cell.max(0.0)))
    }

    // The line a gutter row shows, if any.
    fn gutter_line(&self, area: &TextArea, y: f32) -> Option<usize> {
        let row = ((y - self.rect[1]) / self.metrics.line_h).floor();
        if row < 0.0 || row as usize >= self.rows {
            return None;
        }
        let line = area.scroll().top + row as usize;
        (line < area.line_count()).then_some(line)
    }

    // The scrollbar under a point, when that bar has anything to scroll.
    pub(crate) fn bar_at(&self, area: &TextArea, x: f32, y: f32) -> Option<Bar> {
        if point_in(x, y, self.v_track()) && area.line_count() > self.rows {
            Some(Bar::Vertical)
        } else if point_in(x, y, self.h_track()) && area.widest() + 1 > self.cols {
            Some(Bar::Horizontal)
        } else {
            None
        }
    }

    fn in_gutter(&self, x: f32, y: f32) -> bool {
        point_in(x, y, self.gutter_rect())
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

// The scroll offset that centers a scrollbar's thumb on a point along its
// track, for a thumb drag or a track click.
pub(crate) fn scroll_on_track(len: f32, at: f32, shown: usize, total: usize) -> usize {
    let Some((_, size)) = thumb(len, shown, total, 0) else {
        return 0;
    };
    let travel = (len - size).max(1.0);
    let frac = ((at - size * 0.5) / travel).clamp(0.0, 1.0);
    (frac * (total - shown) as f32).round() as usize
}

// The per-frame view the owning panel assembles.
pub(crate) struct TextAreaView<'a> {
    pub(crate) area: &'a TextArea,
    // Whether the area holds the keyboard (draws the caret and line band).
    pub(crate) focused: bool,
    pub(crate) markers: &'a [GutterMarker],
    pub(crate) mouse: [f32; 2],
}

// The marker under the cursor in the gutter, for a status-line readout.
pub(crate) fn hovered_marker<'a>(
    view: &TextAreaView<'a>,
    g: &Geometry,
) -> Option<&'a GutterMarker> {
    let [x, y] = view.mouse;
    if !g.in_gutter(x, y) {
        return None;
    }
    marker_on(view.markers, g.gutter_line(view.area, y)?)
}

// Position and show the area (`Some(view)`), or blank every element (`None`).
pub(crate) fn place(
    world: &mut World,
    ids: TextAreaIds,
    view: Option<&TextAreaView>,
    g: &Geometry,
) {
    let Some(view) = view else {
        return hide(world, ids);
    };
    let area = view.area;
    let m = g.metrics;
    let scroll = area.scroll();
    widget::place_sprite(world, ids.well(), g.rect, WELL_TINT, true);
    widget::place_sprite(world, ids.gutter(), g.gutter_rect(), GUTTER_TINT, true);

    let caret = area.caret();
    let caret_row = caret.line.checked_sub(scroll.top).filter(|&r| r < g.rows);
    let band = view.focused && caret_row.is_some();
    let band_rect = [
        g.rect[0] + g.gutter_w,
        g.row_y(caret_row.unwrap_or(0)),
        g.rect[2] - g.gutter_w - SCROLLBAR,
        m.line_h,
    ];
    widget::place_sprite(
        world,
        ids.current_line(),
        band_rect,
        CURRENT_LINE_TINT,
        band,
    );

    for row in 0..ROW_POOL {
        let line = scroll.top + row;
        if row >= g.rows || line >= area.line_count() {
            widget::set_sprite_visible(world, ids.selection(row), false);
            widget::set_sprite_visible(world, ids.marker(row), false);
            widget::set_label_visible(world, ids.number(row), false);
            widget::set_label_visible(world, ids.text(row), false);
            continue;
        }
        place_row(world, ids, view, g, row, line);
    }
    place_caret(world, ids, view, g, caret_row);
    place_scrollbars(world, ids, area, g);
}

fn place_row(
    world: &mut World,
    ids: TextAreaIds,
    view: &TextAreaView,
    g: &Geometry,
    row: usize,
    line: usize,
) {
    let area = view.area;
    let m = g.metrics;
    let scroll = area.scroll();
    let y = g.row_y(row);
    let text_y = y + (m.line_h - m.text_px) * 0.5;
    let current = line == area.caret().line;
    if let Some(l) = widget::label_mut(world, ids.number(row)) {
        l.x = g.rect[0] + g.gutter_w - GUTTER_PAD * 0.5;
        l.y = text_y;
        l.align = TextAlign::Right;
        l.scale = m.label_scale;
        l.color = if current {
            NUMBER_CURRENT
        } else {
            NUMBER_COLOR
        };
        l.visible = true;
        l.content = (line + 1).to_string();
    }
    if let Some(l) = widget::label_mut(world, ids.text(row)) {
        l.x = g.text_x;
        l.y = text_y;
        l.align = TextAlign::Left;
        l.scale = m.label_scale;
        l.color = theme::LABEL;
        l.content = view::display_slice(area.line(line), scroll.left, g.cols);
        l.color_runs.clear();
        area.line_runs(line, scroll.left, g.cols, &mut l.color_runs);
        l.visible = !l.content.is_empty();
    }
    let marker = marker_on(view.markers, line);
    let marker_rect = [
        g.rect[0] + (MARKER_W - MARKER_SIZE) * 0.5 + 1.0,
        y + (m.line_h - MARKER_SIZE) * 0.5,
        MARKER_SIZE,
        MARKER_SIZE,
    ];
    place_rounded(
        world,
        ids.marker(row),
        marker_rect,
        marker.map_or([0.0; 4], |mk| marker_tint(mk.severity)),
        MARKER_SIZE * 0.5,
        marker.is_some(),
    );
    let span = selected_cells(area, line).and_then(|(a, b)| {
        let a = a.max(scroll.left);
        let b = b.min(scroll.left + g.cols);
        (b > a).then(|| (a - scroll.left, b - scroll.left))
    });
    let sel_rect = span.map_or([0.0; 4], |(a, b)| {
        [
            g.text_x + a as f32 * m.advance,
            y,
            (b - a) as f32 * m.advance,
            m.line_h,
        ]
    });
    widget::place_sprite(
        world,
        ids.selection(row),
        sel_rect,
        SELECTION_TINT,
        span.is_some(),
    );
}

// The cells of `line` the selection covers, a selected line break counting as
// one cell past the line's end.
fn selected_cells(area: &TextArea, line: usize) -> Option<(usize, usize)> {
    let (start, end) = area.selection()?;
    if line < start.line || line > end.line {
        return None;
    }
    let text = area.line(line);
    let a = if line == start.line {
        view::visual_col(text, start.col)
    } else {
        0
    };
    let b = if line == end.line {
        view::visual_col(text, end.col)
    } else {
        view::line_width(text) + 1
    };
    Some((a, b))
}

fn place_caret(
    world: &mut World,
    ids: TextAreaIds,
    view: &TextAreaView,
    g: &Geometry,
    caret_row: Option<usize>,
) {
    let area = view.area;
    let left = area.scroll().left;
    let cell = area.caret_cell();
    let shown = view.focused && caret_row.is_some() && cell >= left && cell <= left + g.cols;
    let rect = [
        g.text_x + cell.saturating_sub(left) as f32 * g.metrics.advance - CARET_W * 0.5,
        g.row_y(caret_row.unwrap_or(0)) + 1.0,
        CARET_W,
        g.metrics.line_h - 2.0,
    ];
    widget::place_sprite(world, ids.caret(), rect, CARET_TINT, shown);
}

fn place_scrollbars(world: &mut World, ids: TextAreaIds, area: &TextArea, g: &Geometry) {
    let scroll = area.scroll();
    let vt = g.v_track();
    let v = thumb(vt[3], g.rows, area.line_count(), scroll.top);
    place_rounded(
        world,
        ids.v_track(),
        vt,
        TRACK_TINT,
        SCROLLBAR * 0.5,
        v.is_some(),
    );
    let (off, len) = v.unwrap_or((0.0, 0.0));
    place_rounded(
        world,
        ids.v_thumb(),
        [vt[0], vt[1] + off, vt[2], len],
        THUMB_TINT,
        SCROLLBAR * 0.5,
        v.is_some(),
    );
    let ht = g.h_track();
    let h = thumb(ht[2], g.cols, area.widest() + 1, scroll.left);
    place_rounded(
        world,
        ids.h_track(),
        ht,
        TRACK_TINT,
        SCROLLBAR * 0.5,
        h.is_some(),
    );
    let (off, len) = h.unwrap_or((0.0, 0.0));
    place_rounded(
        world,
        ids.h_thumb(),
        [ht[0] + off, ht[1], len, ht[3]],
        THUMB_TINT,
        SCROLLBAR * 0.5,
        h.is_some(),
    );
}

pub(crate) fn hide(world: &mut World, ids: TextAreaIds) {
    widget::hide_all(world, &ids.sprite_ids(), &ids.code_label_ids(), &[]);
}

#[cfg(test)]
mod tests;
