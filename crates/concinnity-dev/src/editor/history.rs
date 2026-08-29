// src/editor/history.rs
//
// Bounded undo/redo stacks over the editor's authored entry list. Pure data:
// the hook records the pre-edit list on every committed entry mutation
// (`mark_changed`) and swaps whole lists back in on undo/redo. Worlds are a
// few dozen JSON lines, so whole-list snapshots stay cheap.

type Snapshot = Vec<serde_json::Value>;

// Oldest snapshots are dropped past this depth, bounding a long session.
const MAX_DEPTH: usize = 64;

#[derive(Default)]
pub(crate) struct History {
    past: Vec<Snapshot>,
    future: Vec<Snapshot>,
}

impl History {
    // Record the pre-edit state of a committed edit. A fresh edit invalidates
    // any redo branch.
    pub(crate) fn record(&mut self, before: Snapshot) {
        self.future.clear();
        if self.past.len() == MAX_DEPTH {
            self.past.remove(0);
        }
        self.past.push(before);
    }

    // Step back: returns the list to restore, moving `current` to the redo
    // stack. `None` when there is nothing to undo.
    pub(crate) fn undo(&mut self, current: Snapshot) -> Option<Snapshot> {
        let snap = self.past.pop()?;
        self.future.push(current);
        Some(snap)
    }

    // Step forward again: returns the list to restore, moving `current` back
    // to the undo stack. `None` when there is nothing to redo.
    pub(crate) fn redo(&mut self, current: Snapshot) -> Option<Snapshot> {
        let snap = self.future.pop()?;
        self.past.push(current);
        Some(snap)
    }

    pub(crate) fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    pub(crate) fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(tag: &str) -> Snapshot {
        vec![serde_json::json!({ "name": tag })]
    }

    #[test]
    fn empty_history_has_nothing_to_step() {
        let mut h = History::default();
        assert!(!h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.undo(snap("now")), None);
        assert_eq!(h.redo(snap("now")), None);
    }

    #[test]
    fn undo_returns_the_recorded_state_and_arms_redo() {
        let mut h = History::default();
        h.record(snap("v1"));
        assert!(h.can_undo());

        let restored = h.undo(snap("v2")).expect("one step to undo");
        assert_eq!(restored, snap("v1"));
        assert!(!h.can_undo());
        assert!(h.can_redo());

        let forward = h.redo(snap("v1")).expect("one step to redo");
        assert_eq!(forward, snap("v2"));
        assert!(h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn a_fresh_edit_clears_the_redo_branch() {
        let mut h = History::default();
        h.record(snap("v1"));
        h.undo(snap("v2")).unwrap();
        assert!(h.can_redo());

        // Editing from the restored v1 forks the timeline: v2 is unreachable.
        h.record(snap("v1"));
        assert!(!h.can_redo());
        assert_eq!(h.undo(snap("v3")), Some(snap("v1")));
    }

    #[test]
    fn multiple_steps_unwind_in_order() {
        let mut h = History::default();
        h.record(snap("v1"));
        h.record(snap("v2"));
        h.record(snap("v3"));

        assert_eq!(h.undo(snap("v4")), Some(snap("v3")));
        assert_eq!(h.undo(snap("v3")), Some(snap("v2")));
        assert_eq!(h.redo(snap("v2")), Some(snap("v3")));
        assert_eq!(h.undo(snap("v3")), Some(snap("v2")));
        assert_eq!(h.undo(snap("v2")), Some(snap("v1")));
        assert_eq!(h.undo(snap("v1")), None);
    }

    #[test]
    fn depth_is_bounded_by_dropping_the_oldest() {
        let mut h = History::default();
        for i in 0..(MAX_DEPTH + 10) {
            h.record(snap(&format!("v{i}")));
        }
        // Unwind everything: the earliest reachable state is v10, not v0.
        let mut last = None;
        let mut current = snap("current");
        while let Some(s) = h.undo(current.clone()) {
            current = s.clone();
            last = Some(s);
        }
        assert_eq!(last, Some(snap("v10")));
    }
}
