//! EditorHook: the Story panel's session state, beside its actions in `story.rs`.

// Shown state, the loaded source's lines / edit line / window scroll, whether
// the edit line holds keyboard focus, the source path shown in the header, and
// the last parse / IO error. `blur` suppresses the edit line's focus for one
// frame after a Backspace line join, so the text system does not also apply
// that Backspace to the freshly joined content. `touched` is the unapplied-edit
// marker behind the heading's "*".
#[derive(Debug)]
pub(in crate::editor::hook) struct StoryState {
    pub(in crate::editor::hook) open: bool,
    pub(in crate::editor::hook) lines: Vec<String>,
    pub(in crate::editor::hook) line: usize,
    pub(in crate::editor::hook) scroll: usize,
    pub(in crate::editor::hook) focus: bool,
    pub(in crate::editor::hook) path: String,
    pub(in crate::editor::hook) status: Option<String>,
    pub(in crate::editor::hook) blur: bool,
    pub(in crate::editor::hook) touched: bool,
}

impl Default for StoryState {
    fn default() -> Self {
        Self {
            open: false,
            lines: vec![String::new()],
            line: 0,
            scroll: 0,
            focus: false,
            path: String::new(),
            status: None,
            blur: false,
            touched: false,
        }
    }
}

impl StoryState {
    // Drop the source read out of the world being left. Shown state and the
    // one-frame blur are not the world's.
    pub(in crate::editor::hook) fn reset_for_world(&mut self) {
        self.lines = vec![String::new()];
        self.line = 0;
        self.scroll = 0;
        self.focus = false;
        self.path = String::new();
        self.status = None;
        self.touched = false;
    }

    // Scroll the least distance that shows the edit line in a window of
    // `rows_shown` rows.
    pub(in crate::editor::hook) fn ensure_line_visible(&mut self, rows_shown: usize) {
        if self.line < self.scroll {
            self.scroll = self.line;
        } else if self.line >= self.scroll + rows_shown {
            self.scroll = self.line + 1 - rows_shown;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_for_world_keeps_shown_state_and_the_blur() {
        let mut s = StoryState {
            open: true,
            lines: vec!["a".into(), "b".into()],
            line: 1,
            scroll: 1,
            focus: true,
            path: "story.md".into(),
            status: Some("bad".into()),
            blur: true,
            touched: true,
        };
        s.reset_for_world();
        assert_eq!(s.lines, vec![String::new()]);
        assert_eq!((s.line, s.scroll, s.focus), (0, 0, false));
        assert_eq!((s.path.as_str(), s.status.as_deref()), ("", None));
        assert!(!s.touched);
        assert!(s.open && s.blur);
    }
}
