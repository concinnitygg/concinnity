//! Undo / redo over text changes. Consecutive typing (or deleting) coalesces
//! into one step per word, so an undo takes back a word rather than a letter;
//! every other edit is a step of its own.

use super::buffer::{Pos, end_of_insert};

// The most steps kept; the oldest is dropped past this.
const MAX_STEPS: usize = 1000;

// One replacement: `removed` taken out at `at`, `inserted` put in its place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Change {
    pub(crate) at: Pos,
    pub(crate) removed: String,
    pub(crate) inserted: String,
}

impl Change {
    // Where the inserted text ends.
    pub(crate) fn inserted_end(&self) -> Pos {
        end_of_insert(self.at, &self.inserted)
    }

    // Where the removed text ended before it was removed.
    pub(crate) fn removed_end(&self) -> Pos {
        end_of_insert(self.at, &self.removed)
    }
}

// The caret and selection anchor around a step, restored by undo / redo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Cursor {
    pub(crate) anchor: Pos,
    pub(crate) caret: Pos,
}

// How a change was made, which decides whether it may join the step before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EditKind {
    Typing,
    Deleting,
    Other,
}

// One undo step: its changes in the order they were applied.
#[derive(Debug, Clone)]
pub(crate) struct Step {
    pub(crate) changes: Vec<Change>,
    pub(crate) before: Cursor,
    pub(crate) after: Cursor,
    kind: EditKind,
    // Names the text state this step leaves; see `History::state`.
    id: u64,
}

#[derive(Debug, Default)]
pub(crate) struct History {
    undo: Vec<Step>,
    redo: Vec<Step>,
    // Whether the next change may join the top step.
    open: bool,
    next_id: u64,
    // The state below the oldest kept step: the loaded text until a step is
    // dropped, then the text that step left.
    base: u64,
}

// A single character typed at the end of the previous one.
fn continues_typing(prev: &Change, next: &Change) -> bool {
    next.removed.is_empty()
        && next.inserted.chars().count() == 1
        && next.inserted != "\n"
        && next.at == prev.inserted_end()
}

// A single character deleted beside the previous deletion: just before it
// (Backspace) or at the same place (Delete). A line break never joins.
fn continues_deleting(prev: &Change, next: &Change) -> bool {
    next.inserted.is_empty()
        && next.removed.chars().count() == 1
        && next.removed != "\n"
        && prev.removed != "\n"
        && (next.removed_end() == prev.at || next.at == prev.at)
}

// The character a change typed or deleted.
fn edited_char(c: &Change) -> Option<char> {
    c.inserted
        .chars()
        .next()
        .or_else(|| c.removed.chars().next())
}

// Whitespace after a non-whitespace character ends a word, and so a step.
fn word_break(prev: &Change, next: &Change) -> bool {
    match (edited_char(prev), edited_char(next)) {
        (Some(p), Some(n)) => n.is_whitespace() && !p.is_whitespace(),
        _ => false,
    }
}

impl History {
    // Record one change. Typing and deleting join the open top step while they
    // continue it and no word ends; anything else starts a new step.
    pub(crate) fn record(&mut self, change: Change, kind: EditKind, before: Cursor, after: Cursor) {
        self.redo.clear();
        let id = self.fresh_id();
        if self.open
            && let Some(top) = self.undo.last_mut()
            && top.kind == kind
            && let Some(prev) = top.changes.last()
        {
            let continues = match kind {
                EditKind::Typing => continues_typing(prev, &change),
                EditKind::Deleting => continues_deleting(prev, &change),
                EditKind::Other => false,
            };
            if continues && !word_break(prev, &change) {
                top.changes.push(change);
                top.after = after;
                top.id = id;
                return;
            }
        }
        self.push(Step {
            changes: vec![change],
            before,
            after,
            kind,
            id,
        });
        self.open = kind != EditKind::Other;
    }

    // Record several changes as one step (an indent over many lines).
    pub(crate) fn record_step(&mut self, changes: Vec<Change>, before: Cursor, after: Cursor) {
        if changes.is_empty() {
            return;
        }
        self.redo.clear();
        let id = self.fresh_id();
        self.push(Step {
            changes,
            before,
            after,
            kind: EditKind::Other,
            id,
        });
        self.open = false;
    }

    fn push(&mut self, step: Step) {
        self.undo.push(step);
        if self.undo.len() > MAX_STEPS {
            self.base = self.undo.remove(0).id;
        }
    }

    // End the open step: the next change starts a new one. Called on anything
    // that is not a continuing edit (a caret move, a click, a save).
    pub(crate) fn seal(&mut self) {
        self.open = false;
    }

    // The step to take back, moved onto the redo stack.
    pub(crate) fn undo(&mut self) -> Option<&Step> {
        self.open = false;
        let step = self.undo.pop()?;
        self.redo.push(step);
        self.redo.last()
    }

    // The step to apply again, moved back onto the undo stack.
    pub(crate) fn redo(&mut self) -> Option<&Step> {
        self.open = false;
        let step = self.redo.pop()?;
        self.undo.push(step);
        self.undo.last()
    }

    // A name for the current text state: equal for two states exactly when no
    // change separates them, so a saved state can be recognized after undo.
    pub(crate) fn state(&self) -> u64 {
        self.undo.last().map_or(self.base, |s| s.id)
    }

    fn fresh_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(col: usize, c: char) -> Change {
        Change {
            at: Pos::new(0, col),
            removed: String::new(),
            inserted: c.to_string(),
        }
    }

    fn deleted(col: usize, c: char) -> Change {
        Change {
            at: Pos::new(0, col),
            removed: c.to_string(),
            inserted: String::new(),
        }
    }

    fn cur(col: usize) -> Cursor {
        Cursor {
            anchor: Pos::new(0, col),
            caret: Pos::new(0, col),
        }
    }

    fn type_str(h: &mut History, from: usize, s: &str) {
        for (i, c) in s.chars().enumerate() {
            h.record(
                typed(from + i, c),
                EditKind::Typing,
                cur(from + i),
                cur(from + i + 1),
            );
        }
    }

    #[test]
    fn typing_coalesces_into_word_sized_steps() {
        let mut h = History::default();
        type_str(&mut h, 0, "hello world");
        let second = h.undo().unwrap();
        let text: String = second.changes.iter().map(|c| c.inserted.as_str()).collect();
        assert_eq!(text, " world", "the last word and the space before it");
        assert_eq!(second.before, cur(5));
        let first = h.undo().unwrap();
        assert_eq!(first.changes.len(), 5, "\"hello\"");
        assert!(h.undo().is_none());
    }

    #[test]
    fn a_caret_move_seals_the_step() {
        let mut h = History::default();
        type_str(&mut h, 0, "ab");
        h.seal();
        type_str(&mut h, 2, "cd");
        assert_eq!(h.undo().unwrap().changes.len(), 2);
        assert_eq!(h.undo().unwrap().changes.len(), 2);
    }

    #[test]
    fn typing_elsewhere_starts_a_new_step() {
        let mut h = History::default();
        type_str(&mut h, 0, "ab");
        type_str(&mut h, 10, "cd");
        assert_eq!(h.undo().unwrap().changes.len(), 2);
        assert!(h.undo().is_some());
    }

    #[test]
    fn backspaces_coalesce_until_a_word_ends() {
        let mut h = History::default();
        // "ab cd" deleted from the end: d, c, then the space ends the word.
        for (col, c) in [(4, 'd'), (3, 'c'), (2, ' '), (1, 'b')] {
            h.record(deleted(col, c), EditKind::Deleting, cur(col + 1), cur(col));
        }
        assert_eq!(h.undo().unwrap().changes.len(), 2, "' ' then 'b'");
        assert_eq!(h.undo().unwrap().changes.len(), 2, "'d' then 'c'");
    }

    #[test]
    fn other_edits_never_coalesce() {
        let mut h = History::default();
        h.record(typed(0, 'x'), EditKind::Other, cur(0), cur(1));
        h.record(typed(1, 'y'), EditKind::Other, cur(1), cur(2));
        assert!(h.undo().is_some());
        assert!(h.undo().is_some());
    }

    #[test]
    fn redo_replays_and_a_new_edit_clears_it() {
        let mut h = History::default();
        type_str(&mut h, 0, "ab");
        assert!(h.undo().is_some());
        assert_eq!(h.redo().unwrap().changes.len(), 2);
        assert!(h.redo().is_none());
        assert!(h.undo().is_some());
        type_str(&mut h, 0, "z");
        assert!(h.redo().is_none(), "a new edit drops the redo stack");
    }

    #[test]
    fn state_names_the_text_across_undo_and_redo() {
        let mut h = History::default();
        let empty = h.state();
        type_str(&mut h, 0, "ab");
        let typed = h.state();
        assert_ne!(typed, empty);
        h.undo();
        assert_eq!(h.state(), empty);
        h.redo();
        assert_eq!(h.state(), typed);
        h.seal();
        type_str(&mut h, 2, "c");
        assert_ne!(h.state(), typed);
    }

    #[test]
    fn a_coalesced_change_renames_the_state() {
        let mut h = History::default();
        type_str(&mut h, 0, "a");
        let one = h.state();
        type_str(&mut h, 1, "b");
        assert_ne!(h.state(), one, "the joined step leaves a different text");
    }

    // Past the bound the oldest step goes, and undoing everything that is left
    // does not read as the loaded text.
    #[test]
    fn the_oldest_step_drops_past_the_bound() {
        let mut h = History::default();
        let loaded = h.state();
        for i in 0..=MAX_STEPS {
            h.record(typed(i, 'x'), EditKind::Other, cur(i), cur(i + 1));
        }
        let mut undone = 0;
        while h.undo().is_some() {
            undone += 1;
        }
        assert_eq!(undone, MAX_STEPS);
        assert_ne!(h.state(), loaded, "the first step is gone for good");
    }

    #[test]
    fn record_step_is_one_undo() {
        let mut h = History::default();
        h.record_step(vec![typed(0, 'x'), typed(5, 'y')], cur(0), cur(0));
        assert_eq!(h.undo().unwrap().changes.len(), 2);
        h.record_step(Vec::new(), cur(0), cur(0));
        assert!(h.undo().is_none(), "an empty step records nothing");
    }
}
