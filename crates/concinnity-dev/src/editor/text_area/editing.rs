//! Text edits: typing, deletion, line breaks that keep the indentation,
//! indent / outdent, the clipboard, and undo / redo. Every edit goes through
//! the history, so each one can be taken back.

use concinnity_core::components::KeyEvent;
use concinnity_core::window::clipboard::Clipboard;

use super::buffer::{Pos, normalize_newlines, ordered};
use super::history::{Change, Cursor, EditKind};
use super::keys::{Command, Platform, command};
use super::{TextArea, motion};

// Tab inserts this; Shift+Tab takes up to this much indentation back.
const INDENT: &str = "    ";

// What a frame's keys asked of the host beyond the edits themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Response {
    // The save shortcut was pressed.
    pub(crate) save: bool,
}

impl TextArea {
    // Apply a frame's key events in order: typed characters insert, and key
    // presses run the command they map to on `platform`.
    pub(crate) fn handle_events(
        &mut self,
        events: &[KeyEvent],
        platform: Platform,
        clipboard: &mut dyn Clipboard,
    ) -> Response {
        let mut response = Response::default();
        for event in events {
            match *event {
                KeyEvent::Text(c) => self.type_char(c),
                KeyEvent::Press(press) => {
                    if let Some(cmd) = command(press, platform) {
                        response.save |= self.run(cmd, clipboard);
                    }
                }
            }
        }
        response
    }

    // Run one command; `true` when it asks the host to save.
    pub(crate) fn run(&mut self, cmd: Command, clipboard: &mut dyn Clipboard) -> bool {
        match cmd {
            Command::Move { motion, extend } => self.move_caret(motion, extend),
            Command::Backspace { word } => self.backspace(word),
            Command::Delete { word } => self.delete_forward(word),
            Command::Newline => self.newline(),
            Command::Indent => self.indent(),
            Command::Outdent => self.outdent(),
            Command::SelectAll => self.select_all(),
            Command::Cut => {
                self.copy(clipboard);
                self.replace_selection("", EditKind::Other);
            }
            Command::Copy => self.copy(clipboard),
            Command::Paste => self.paste(clipboard),
            Command::Undo => self.undo(),
            Command::Redo => self.redo(),
            Command::Save => return true,
        }
        false
    }

    pub(crate) fn type_char(&mut self, c: char) {
        self.replace_selection(&c.to_string(), EditKind::Typing);
    }

    // Replace the selection (or insert at the caret) with LF-separated `text`,
    // leaving the caret after it.
    pub(crate) fn replace_selection(&mut self, text: &str, kind: EditKind) {
        let (start, end) = ordered(self.anchor, self.caret);
        if start == end && text.is_empty() {
            return;
        }
        let before = self.cursor();
        let change = self.replace(start, end, text);
        let caret = change.inserted_end();
        self.caret = caret;
        self.anchor = caret;
        self.goal = None;
        self.history.record(change, kind, before, self.cursor());
        self.reveal_caret();
    }

    // Replace `[start, end)` with `text` in the buffer, returning the change.
    fn replace(&mut self, start: Pos, end: Pos, text: &str) -> Change {
        self.widest.set(None);
        self.edited(start.line);
        let removed = self.buffer.remove(start, end);
        self.buffer.insert(start, text);
        Change {
            at: start,
            removed,
            inserted: text.to_string(),
        }
    }

    fn backspace(&mut self, word: bool) {
        if self.selection().is_some() {
            return self.replace_selection("", EditKind::Other);
        }
        let from = if word {
            motion::word_left(&self.buffer, self.caret)
        } else {
            motion::left(&self.buffer, self.caret)
        };
        self.delete_span(from, self.caret, word);
    }

    fn delete_forward(&mut self, word: bool) {
        if self.selection().is_some() {
            return self.replace_selection("", EditKind::Other);
        }
        let to = if word {
            motion::word_right(&self.buffer, self.caret)
        } else {
            motion::right(&self.buffer, self.caret)
        };
        self.delete_span(self.caret, to, word);
    }

    // Delete `[from, to)` as a Backspace / Delete: single characters coalesce
    // into word-sized undo steps, a word deletion is a step of its own.
    fn delete_span(&mut self, from: Pos, to: Pos, word: bool) {
        if from == to {
            return;
        }
        let kind = if word {
            EditKind::Other
        } else {
            EditKind::Deleting
        };
        let before = self.cursor();
        let change = self.replace(from, to, "");
        self.caret = from;
        self.anchor = from;
        self.goal = None;
        let after = self.cursor();
        self.history.record(change, kind, before, after);
        self.reveal_caret();
    }

    // Enter: a line break carrying the current line's indentation (as far as
    // the caret reaches into it).
    fn newline(&mut self) {
        let (start, _) = ordered(self.anchor, self.caret);
        let line = self.buffer.line(start.line);
        let indent: String = line
            .chars()
            .take(motion::indent_end(line).min(start.col))
            .collect();
        self.replace_selection(&format!("\n{indent}"), EditKind::Other);
    }

    // Tab: indent every line a multi-line selection covers, or insert
    // `INDENT` in place of the selection.
    fn indent(&mut self) {
        let (start, end) = ordered(self.anchor, self.caret);
        if start.line == end.line {
            return self.replace_selection(INDENT, EditKind::Other);
        }
        let before = self.cursor();
        let width = INDENT.chars().count();
        let changes: Vec<Change> = self
            .selected_lines()
            .map(|line| self.replace(Pos::new(line, 0), Pos::new(line, 0), INDENT))
            .collect();
        let lines = self.selected_lines();
        let shift = |p: Pos| {
            if lines.contains(&p.line) && p.col > 0 {
                Pos::new(p.line, p.col + width)
            } else {
                p
            }
        };
        self.anchor = shift(self.anchor);
        self.caret = shift(self.caret);
        self.goal = None;
        self.history.record_step(changes, before, self.cursor());
        self.reveal_caret();
    }

    // Shift+Tab: take up to one indent (four spaces, or a tab) off the front of
    // every line the selection covers, or of the caret's line.
    fn outdent(&mut self) {
        let before = self.cursor();
        let mut changes = Vec::new();
        for line in self.selected_lines() {
            let text = self.buffer.line(line);
            let n = if text.starts_with('\t') {
                1
            } else {
                text.chars()
                    .take(INDENT.len())
                    .take_while(|&c| c == ' ')
                    .count()
            };
            if n == 0 {
                continue;
            }
            changes.push(self.replace(Pos::new(line, 0), Pos::new(line, n), ""));
            let pull = |p: Pos| {
                if p.line == line {
                    Pos::new(line, p.col.saturating_sub(n))
                } else {
                    p
                }
            };
            self.anchor = pull(self.anchor);
            self.caret = pull(self.caret);
        }
        self.goal = None;
        self.history.record_step(changes, before, self.cursor());
        self.reveal_caret();
    }

    fn copy(&mut self, clipboard: &mut dyn Clipboard) {
        if let Some((start, end)) = self.selection() {
            clipboard.set_text(&self.buffer.slice(start, end));
        }
    }

    // Paste in place of the selection, with the pasted line endings made LF.
    fn paste(&mut self, clipboard: &mut dyn Clipboard) {
        if let Some(text) = clipboard.text() {
            self.insert(&text);
        }
    }

    // Replace the selection (or insert at the caret) with `text`, as one undo
    // step of its own.
    pub(crate) fn insert(&mut self, text: &str) {
        let text = normalize_newlines(text);
        if !text.is_empty() {
            self.history.seal();
            self.replace_selection(&text, EditKind::Other);
        }
    }

    pub(crate) fn undo(&mut self) {
        let Some(step) = self.history.undo() else {
            return;
        };
        let mut edited = usize::MAX;
        for c in step.changes.iter().rev() {
            self.buffer.remove(c.at, c.inserted_end());
            self.buffer.insert(c.at, &c.removed);
            edited = edited.min(c.at.line);
        }
        let before = step.before;
        self.edited(edited);
        self.after_history_jump(before);
    }

    pub(crate) fn redo(&mut self) {
        let Some(step) = self.history.redo() else {
            return;
        };
        let mut edited = usize::MAX;
        for c in &step.changes {
            self.buffer.remove(c.at, c.removed_end());
            self.buffer.insert(c.at, &c.inserted);
            edited = edited.min(c.at.line);
        }
        let after = step.after;
        self.edited(edited);
        self.after_history_jump(after);
    }

    fn after_history_jump(&mut self, cursor: Cursor) {
        self.widest.set(None);
        self.set_cursor(cursor);
        self.goal = None;
        self.reveal_caret();
    }
}
