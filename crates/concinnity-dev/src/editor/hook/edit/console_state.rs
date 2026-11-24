//! `EditorHook`: the Console panel's view state, beside its actions in
//! `console.rs`. The log sink and the build guard are the session's, not the
//! panel's, so they stay on the hook.

// Whether the command line holds keyboard focus (suppressed for one frame
// after a backtick open so the text system does not type the backtick into
// it), and whether the log follows its tail (it auto-scrolls on new lines
// until the user scrolls up).
#[derive(Debug)]
pub(in crate::editor::hook) struct ConsoleState {
    pub(in crate::editor::hook) focus: bool,
    pub(in crate::editor::hook) blur: bool,
    pub(in crate::editor::hook) pinned: bool,
}

impl Default for ConsoleState {
    fn default() -> Self {
        Self {
            focus: false,
            blur: false,
            pinned: true,
        }
    }
}
