//! Caret motion that needs the view: vertical moves aim for a remembered cell,
//! and a page is the window's height.

use super::buffer::{Pos, ordered};
use super::keys::Motion;
use super::{TextArea, motion, view};

impl TextArea {
    pub(super) fn move_caret(&mut self, m: Motion, extend: bool) {
        self.history.seal();
        // An arrow with a selection and no Shift lands on that side of it.
        if !extend && let Some((start, end)) = self.selection() {
            match m {
                Motion::Left => return self.collapse_to(start),
                Motion::Right => return self.collapse_to(end),
                _ => {}
            }
        }
        let b = &self.buffer;
        let p = self.caret;
        let target = match m {
            Motion::Left => motion::left(b, p),
            Motion::Right => motion::right(b, p),
            Motion::WordLeft => motion::word_left(b, p),
            Motion::WordRight => motion::word_right(b, p),
            Motion::LineStart => motion::smart_home(b, p),
            Motion::LineEnd => motion::line_end(b, p),
            Motion::DocStart => Pos::new(0, 0),
            Motion::DocEnd => b.end(),
            Motion::Up => self.vertical(-1),
            Motion::Down => self.vertical(1),
            Motion::PageUp | Motion::PageDown => {
                let page = self.view.rows.saturating_sub(1).max(1) as isize;
                let lines = if m == Motion::PageUp { -page } else { page };
                self.scroll_by(lines, 0);
                self.vertical(lines)
            }
        };
        if !matches!(
            m,
            Motion::Up | Motion::Down | Motion::PageUp | Motion::PageDown
        ) {
            self.goal = None;
        }
        self.caret = target;
        if !extend {
            self.anchor = target;
        }
        self.reveal_caret();
    }

    fn collapse_to(&mut self, p: Pos) {
        self.caret = p;
        self.anchor = p;
        self.goal = None;
        self.reveal_caret();
    }

    // The position `lines` lines away at the goal cell (set from the caret on
    // the first vertical move). Moving past the first line lands at its start,
    // past the last at its end.
    fn vertical(&mut self, lines: isize) -> Pos {
        let b = &self.buffer;
        let goal = *self
            .goal
            .get_or_insert_with(|| view::visual_col(b.line(self.caret.line), self.caret.col));
        let last = b.line_count() - 1;
        let target = self.caret.line as isize + lines;
        if target < 0 {
            return Pos::new(0, 0);
        }
        if target as usize > last {
            return b.end();
        }
        let line = target as usize;
        Pos::new(line, view::col_at_visual(b.line(line), goal as f32))
    }

    // Select everything, caret at the end.
    pub(super) fn select_all(&mut self) {
        self.history.seal();
        self.anchor = Pos::new(0, 0);
        self.caret = self.buffer.end();
        self.goal = None;
        self.reveal_caret();
    }

    // The whole lines a selection covers, for indenting: a selection ending at
    // a line's column 0 does not take that line.
    pub(super) fn selected_lines(&self) -> std::ops::RangeInclusive<usize> {
        let (start, end) = ordered(self.anchor, self.caret);
        let last = if end.col == 0 && end.line > start.line {
            end.line - 1
        } else {
            end.line
        };
        start.line..=last
    }
}
