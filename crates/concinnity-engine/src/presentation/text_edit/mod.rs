//! Multi-line text editing: a pure editing model ([`TextEditor`], world-free and
//! tested edit by edit) and the geometry it lays out on in a rect. Its owner
//! feeds it the frame's key events and pointer presses, then draws the window
//! of lines it shows. The [`TextArea`](crate::components::TextArea) widget is
//! one owner.

use alloc::string::String;
use alloc::vec::Vec;

mod buffer;
mod clipboard;
mod editing;
mod geometry;
pub mod highlight;
mod history;
mod keys;
mod motion;
mod navigate;
mod pointer;
pub mod view;

use core::sync::atomic::{AtomicUsize, Ordering};

use crate::components::ColorRun;

pub use buffer::Pos;
use buffer::{Buffer, ordered};
pub use clipboard::{HostClipboard, LocalClipboard};
pub use editing::Response;
pub use geometry::{
    GUTTER_PAD, Geometry, MARKER_W, Metrics, ROW_POOL, SCROLLBAR, TEXT_PAD, geometry,
    height_for_rows, point_in,
};
use highlight::Highlighter;
use highlight::states::LineStates;
use history::{Cursor, History};
pub use keys::{Command, KeyConvention, Motion};
pub use view::{Scroll, ViewSize};

// Presses closer together than this, on one spot, count as a double (word) or
// triple (line) click.
const MULTI_CLICK_S: f64 = 0.4;

// The last run of presses on one spot.
#[derive(Debug, Clone, Copy)]
struct ClickRun {
    at: f64,
    pos: Pos,
    count: u8,
}

// The language a text is highlighted as, its lines' start states so far, and
// the spans of the lines highlighted since their last edit.
#[derive(Debug, Clone)]
struct Highlight {
    language: &'static dyn Highlighter,
    states: LineStates,
    // Each line's spans, kept until the line or one above it is edited.
    spans: Vec<Option<Vec<highlight::Span>>>,
}

// The widest line in cells, computed on demand and dropped by every edit.
#[derive(Debug)]
struct WidestCache(AtomicUsize);

impl WidestCache {
    const UNKNOWN: usize = usize::MAX;

    fn get(&self) -> Option<usize> {
        let w = self.0.load(Ordering::Relaxed);
        (w != Self::UNKNOWN).then_some(w)
    }

    fn set(&self, w: usize) {
        self.0.store(w.min(Self::UNKNOWN - 1), Ordering::Relaxed);
    }

    fn clear(&self) {
        self.0.store(Self::UNKNOWN, Ordering::Relaxed);
    }
}

impl Default for WidestCache {
    fn default() -> Self {
        Self(AtomicUsize::new(Self::UNKNOWN))
    }
}

impl Clone for WidestCache {
    fn clone(&self) -> Self {
        Self(AtomicUsize::new(self.0.load(Ordering::Relaxed)))
    }
}

/// An editable multi-line text: the lines, the caret and selection, the undo
/// history, and the window of the text a rect shows.
#[derive(Debug, Clone, Default)]
pub struct TextEditor {
    buffer: Buffer,
    caret: Pos,
    // The other end of the selection; equal to `caret` when nothing is selected.
    anchor: Pos,
    // The cell a run of vertical moves aims for, so passing through a short
    // line does not pull the caret left for good.
    goal: Option<usize>,
    history: History,
    // The history state last saved, for `is_dirty`.
    saved: u64,
    scroll: Scroll,
    view: ViewSize,
    // Whether a mouse press is still extending the selection.
    dragging: bool,
    // The scrollbar a press grabbed, while the button stays down.
    bar: Option<pointer::Bar>,
    // Wheel travel short of a whole line, carried to the next movement.
    wheel_carry: f32,
    clicks: Option<ClickRun>,
    widest: WidestCache,
    highlight: Option<Highlight>,
}

impl TextEditor {
    /// A clean area holding `text`, caret at the start.
    pub fn from_text(text: &str) -> Self {
        Self {
            buffer: Buffer::from_text(text),
            ..Self::default()
        }
    }

    /// Color the text as `language` draws it.
    #[must_use]
    pub fn highlighted(mut self, language: &'static dyn Highlighter) -> Self {
        self.highlight = Some(Highlight {
            language,
            states: LineStates::default(),
            spans: Vec::new(),
        });
        self
    }

    /// Scan the lines a window of the text can show from its top, so
    /// [`line_runs`](Self::line_runs) colors them. A line stays scanned until
    /// it, or a line above it, is edited; call this before each draw.
    pub fn prepare_highlight(&mut self) {
        let Some(hl) = &mut self.highlight else {
            return;
        };
        let top = self.scroll.top;
        let end = (top + self.view.rows.max(ROW_POOL)).min(self.buffer.line_count());
        if hl.spans.len() < end {
            hl.spans.resize(end, None);
        }
        for line in top..end {
            if hl.spans[line].is_some() {
                continue;
            }
            let start = hl
                .states
                .start_of(line, hl.language, |i| self.buffer.line(i));
            let mut found = Vec::new();
            let after = hl.language.line(self.buffer.line(line), start, &mut found);
            hl.states.learn(line, after);
            hl.spans[line] = Some(found);
        }
    }

    /// Append the color runs for the cells `[left, left + width)` of `line` to
    /// `out`, counted from `left`. Nothing without a highlighter, or for a line
    /// [`prepare_highlight`](Self::prepare_highlight) has not scanned since it
    /// changed.
    pub fn line_runs(&self, line: usize, left: usize, width: usize, out: &mut Vec<ColorRun>) {
        let Some(found) = self
            .highlight
            .as_ref()
            .and_then(|hl| hl.spans.get(line))
            .and_then(Option::as_ref)
        else {
            return;
        };
        highlight::window::window_runs(self.buffer.line(line), found, left, width, out);
    }

    // Line `line` changed, or lines were added or removed below it.
    fn edited(&mut self, line: usize) {
        if let Some(hl) = &mut self.highlight {
            hl.states.edited(line);
            hl.spans.truncate(line);
        }
    }

    /// The whole text, in the line ending it was loaded with.
    pub fn text(&self) -> String {
        self.buffer.text()
    }

    /// The current text is what is on disk now; the next edit starts a new
    /// undo step, so undoing back to here reads as clean again.
    pub fn mark_saved(&mut self) {
        self.history.seal();
        self.saved = self.history.state();
    }

    /// Whether the text reads `text`, whatever its line endings.
    pub fn matches(&self, text: &str) -> bool {
        let mut lines = text.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l));
        (0..self.line_count()).all(|i| lines.next() == Some(self.line(i))) && lines.next().is_none()
    }

    /// A count that changes with every edit, undone and redone ones too.
    pub fn revision(&self) -> u64 {
        self.history.state()
    }

    /// Whether the text differs from the last `mark_saved` (or the load).
    pub fn is_dirty(&self) -> bool {
        self.history.state() != self.saved
    }

    /// How many lines the text has.
    pub fn line_count(&self) -> usize {
        self.buffer.line_count()
    }

    /// Line `i`, without its line ending; empty past the end.
    pub fn line(&self, i: usize) -> &str {
        self.buffer.line(i)
    }

    /// Where the caret is.
    pub fn caret(&self) -> Pos {
        self.caret
    }

    /// The selected range, earlier end first, or `None` when nothing is selected.
    pub fn selection(&self) -> Option<(Pos, Pos)> {
        (self.anchor != self.caret).then(|| ordered(self.anchor, self.caret))
    }

    /// The window over the text.
    pub fn scroll(&self) -> Scroll {
        self.scroll
    }

    /// The widest line, in cells.
    pub fn widest(&self) -> usize {
        if let Some(w) = self.widest.get() {
            return w;
        }
        let w = (0..self.buffer.line_count())
            .map(|i| view::line_width(self.buffer.line(i)))
            .max()
            .unwrap_or(0);
        self.widest.set(w);
        w
    }

    /// The cells of `line` the selection covers, first and past-the-end; a
    /// selected line break counts as one cell past the line's end.
    pub fn selected_cells(&self, line: usize) -> Option<(usize, usize)> {
        let (start, end) = self.selection()?;
        if line < start.line || line > end.line {
            return None;
        }
        let text = self.line(line);
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

    /// The cell the caret sits in.
    pub fn caret_cell(&self) -> usize {
        view::visual_col(self.buffer.line(self.caret.line), self.caret.col)
    }

    /// How much of the text the laid-out rect shows. The host sets it before
    /// feeding input, so scrolling keeps the caret inside what is drawn.
    pub fn set_view(&mut self, view: ViewSize) {
        self.view = view;
        self.clamp_scroll();
    }

    /// Put the caret at (line, column), clamped onto the text, and bring it into
    /// view: centered when it was off screen, as a jump to a diagnostic wants.
    pub fn go_to(&mut self, line: usize, col: usize) {
        self.history.seal();
        self.caret = self.buffer.clamp(Pos::new(line, col));
        self.anchor = self.caret;
        self.goal = None;
        let rows = self.view.rows.max(1);
        let top = self.scroll.top;
        if self.caret.line < top || self.caret.line >= top + rows {
            self.scroll.top = self.caret.line.saturating_sub(rows / 2);
        }
        self.reveal_caret();
    }

    /// Move the window by whole lines and cells (a wheel), within the text.
    pub fn scroll_by(&mut self, lines: isize, cols: isize) {
        self.scroll.top = self.scroll.top.saturating_add_signed(lines);
        self.scroll.left = self.scroll.left.saturating_add_signed(cols);
        self.clamp_scroll();
    }

    /// Place the window directly (a scrollbar drag), within the text.
    pub fn set_scroll(&mut self, scroll: Scroll) {
        self.scroll = scroll;
        self.clamp_scroll();
    }

    /// A mouse press at `pos` (already hit-tested) at time `now` in seconds.
    /// One press places the caret (Shift keeps the anchor), a second on the same
    /// spot selects the word, a third the line.
    pub fn press(&mut self, pos: Pos, extend: bool, now: f64) {
        self.history.seal();
        let pos = self.buffer.clamp(pos);
        let count = match self.clicks {
            Some(run) if run.pos == pos && now - run.at <= MULTI_CLICK_S => run.count % 3 + 1,
            _ => 1,
        };
        self.clicks = Some(ClickRun {
            at: now,
            pos,
            count,
        });
        self.goal = None;
        match count {
            1 => {
                self.caret = pos;
                if !extend {
                    self.anchor = pos;
                }
                self.dragging = true;
            }
            2 => {
                let (start, end) = motion::word_at(&self.buffer, pos);
                self.anchor = start;
                self.caret = end;
                self.dragging = false;
            }
            _ => {
                self.anchor = Pos::new(pos.line, 0);
                self.caret = if pos.line + 1 < self.buffer.line_count() {
                    Pos::new(pos.line + 1, 0)
                } else {
                    motion::line_end(&self.buffer, pos)
                };
                self.dragging = false;
            }
        }
        self.reveal_caret();
    }

    /// Extend a press's selection to `pos` while the button is held.
    pub fn drag_to(&mut self, pos: Pos) {
        if self.dragging {
            self.caret = self.buffer.clamp(pos);
            self.reveal_caret();
        }
    }

    /// The button that pressed was let go.
    pub fn release(&mut self) {
        self.dragging = false;
    }

    /// Whether a press is still extending the selection.
    pub fn dragging(&self) -> bool {
        self.dragging
    }

    fn cursor(&self) -> Cursor {
        Cursor {
            anchor: self.anchor,
            caret: self.caret,
        }
    }

    fn set_cursor(&mut self, c: Cursor) {
        let clamp = |p| self.buffer.clamp(p);
        self.anchor = clamp(c.anchor);
        self.caret = clamp(c.caret);
    }

    fn reveal_caret(&mut self) {
        let cell = self.caret_cell();
        self.scroll.reveal(self.caret.line, cell, self.view);
        self.clamp_scroll();
    }

    fn clamp_scroll(&mut self) {
        let (lines, widest) = (self.buffer.line_count(), self.widest());
        self.scroll
            .clamp(lines, widest.max(self.caret_cell()), self.view);
    }
}

#[cfg(test)]
mod tests;
