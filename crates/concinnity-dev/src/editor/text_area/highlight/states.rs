//! Each line's start state, known for a prefix of the text. A line's state
//! depends only on the lines above it, so an edit keeps every state down to
//! the edited line's own and drops the rest; asking for a later line scans
//! forward from the last one still known.

use super::{Highlighter, LineState};

#[derive(Debug, Default)]
pub(crate) struct LineStates {
    // `starts[i]` is the state line `i` starts in.
    starts: Vec<LineState>,
    // Scratch for the spans a forward scan throws away.
    spans: Vec<super::Span>,
}

impl LineStates {
    // Line `line` changed, or lines were added or removed below it.
    pub(crate) fn edited(&mut self, line: usize) {
        self.starts.truncate(line + 1);
    }

    // How many lines' start states are known.
    #[cfg(test)]
    pub(crate) fn known(&self) -> usize {
        self.starts.len()
    }

    // The state `target` starts in, scanning `line(i)` forward from the last
    // known line when it is not known yet.
    pub(crate) fn start_of<'a>(
        &mut self,
        target: usize,
        hl: &dyn Highlighter,
        line: impl Fn(usize) -> &'a str,
    ) -> LineState {
        if self.starts.is_empty() {
            self.starts.push(LineState::default());
        }
        while self.starts.len() <= target {
            let i = self.starts.len() - 1;
            self.spans.clear();
            let next = hl.line(line(i), self.starts[i], &mut self.spans);
            self.starts.push(next);
        }
        self.starts[target]
    }

    // Record the state after `line` when it is the next one unknown, so
    // highlighting lines in order never scans a line twice.
    pub(crate) fn learn(&mut self, line: usize, after: LineState) {
        if self.starts.len() == line + 1 {
            self.starts.push(after);
        }
    }
}
