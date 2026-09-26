//! A multi-line code text surface: a pure editing model (`TextArea`, world-free
//! and unit-tested edit by edit) and its layout into a rect of the editor HUD
//! (`layout`). A panel owns one `TextArea`, feeds it the frame's key events and
//! mouse presses, and lays it out with whatever gutter markers it wants shown.

mod buffer;
pub(crate) mod clipboard;
mod editing;
pub(crate) mod highlight;
mod history;
pub(crate) mod keys;
pub(crate) mod layout;
pub(crate) mod markers;
mod motion;
mod navigate;
mod pointer;
mod view;

use std::cell::{Cell, RefCell};

use concinnity_core::components::ColorRun;

pub(crate) use buffer::Pos;
use buffer::{Buffer, ordered};
use highlight::Highlighter;
use highlight::states::LineStates;
use history::{Cursor, History};
pub(crate) use view::{Scroll, ViewSize};

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

// The language a text is highlighted as, and its lines' start states so far.
#[derive(Debug)]
struct Highlight {
    language: &'static dyn Highlighter,
    states: RefCell<LineStates>,
}

#[derive(Debug, Default)]
pub(crate) struct TextArea {
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
    // The widest line in cells, computed on demand and dropped by every edit.
    widest: Cell<Option<usize>>,
    highlight: Option<Highlight>,
}

impl TextArea {
    // A clean area holding `text`, caret at the start.
    pub(crate) fn from_text(text: &str) -> Self {
        Self {
            buffer: Buffer::from_text(text),
            ..Self::default()
        }
    }

    // Color the text as `language` draws it.
    pub(crate) fn highlighted(mut self, language: &'static dyn Highlighter) -> Self {
        self.highlight = Some(Highlight {
            language,
            states: RefCell::default(),
        });
        self
    }

    // Append the color runs for the cells `[left, left + width)` of `line` to
    // `out`, counted from `left`. Nothing without a highlighter.
    pub(crate) fn line_runs(
        &self,
        line: usize,
        left: usize,
        width: usize,
        out: &mut Vec<ColorRun>,
    ) {
        let Some(hl) = &self.highlight else {
            return;
        };
        let mut states = hl.states.borrow_mut();
        let start = states.start_of(line, hl.language, |i| self.buffer.line(i));
        let mut spans = Vec::new();
        let text = self.buffer.line(line);
        let after = hl.language.line(text, start, &mut spans);
        states.learn(line, after);
        highlight::window::window_runs(text, &spans, left, width, out);
    }

    // Line `line` changed, or lines were added or removed below it.
    fn edited(&self, line: usize) {
        if let Some(hl) = &self.highlight {
            hl.states.borrow_mut().edited(line);
        }
    }

    // The whole text, in the line ending it was loaded with.
    pub(crate) fn text(&self) -> String {
        self.buffer.text()
    }

    // The current text is what is on disk now; the next edit starts a new
    // undo step, so undoing back to here reads as clean again.
    pub(crate) fn mark_saved(&mut self) {
        self.history.seal();
        self.saved = self.history.state();
    }

    // Whether the text differs from the last `mark_saved` (or the load).
    pub(crate) fn is_dirty(&self) -> bool {
        self.history.state() != self.saved
    }

    pub(crate) fn line_count(&self) -> usize {
        self.buffer.line_count()
    }

    pub(crate) fn line(&self, i: usize) -> &str {
        self.buffer.line(i)
    }

    pub(crate) fn caret(&self) -> Pos {
        self.caret
    }

    // The selected range, earlier end first, or `None` when nothing is selected.
    pub(crate) fn selection(&self) -> Option<(Pos, Pos)> {
        (self.anchor != self.caret).then(|| ordered(self.anchor, self.caret))
    }

    pub(crate) fn scroll(&self) -> Scroll {
        self.scroll
    }

    // The widest line, in cells.
    pub(crate) fn widest(&self) -> usize {
        if let Some(w) = self.widest.get() {
            return w;
        }
        let w = (0..self.buffer.line_count())
            .map(|i| view::line_width(self.buffer.line(i)))
            .max()
            .unwrap_or(0);
        self.widest.set(Some(w));
        w
    }

    // The cell the caret sits in.
    pub(crate) fn caret_cell(&self) -> usize {
        view::visual_col(self.buffer.line(self.caret.line), self.caret.col)
    }

    // How much of the text the laid-out rect shows. The host sets it before
    // feeding input, so scrolling keeps the caret inside what is drawn.
    pub(crate) fn set_view(&mut self, view: ViewSize) {
        self.view = view;
        self.clamp_scroll();
    }

    // Put the caret at (line, column), clamped onto the text, and bring it into
    // view: centered when it was off screen, as a jump to a diagnostic wants.
    pub(crate) fn go_to(&mut self, line: usize, col: usize) {
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

    // Move the window by whole lines and cells (a wheel), within the text.
    pub(crate) fn scroll_by(&mut self, lines: isize, cols: isize) {
        self.scroll.top = self.scroll.top.saturating_add_signed(lines);
        self.scroll.left = self.scroll.left.saturating_add_signed(cols);
        self.clamp_scroll();
    }

    // Place the window directly (a scrollbar drag), within the text.
    pub(crate) fn set_scroll(&mut self, scroll: Scroll) {
        self.scroll = scroll;
        self.clamp_scroll();
    }

    // A mouse press at `pos` (already hit-tested) at time `now` in seconds.
    // One press places the caret (Shift keeps the anchor), a second on the same
    // spot selects the word, a third the line.
    pub(crate) fn press(&mut self, pos: Pos, extend: bool, now: f64) {
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

    // Extend a press's selection to `pos` while the button is held.
    pub(crate) fn drag_to(&mut self, pos: Pos) {
        if self.dragging {
            self.caret = self.buffer.clamp(pos);
            self.reveal_caret();
        }
    }

    pub(crate) fn release(&mut self) {
        self.dragging = false;
    }

    pub(crate) fn dragging(&self) -> bool {
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
