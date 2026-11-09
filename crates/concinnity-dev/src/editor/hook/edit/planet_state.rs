//! `EditorHook`: the Planet panel's session state: which row's text input holds
//! keyboard focus, the first row its window shows, the message from the last
//! rejected edit, and whether typed edits are waiting on Apply.

#[derive(Debug, Default)]
pub(in crate::editor::hook) struct PlanetState {
    pub(in crate::editor::hook) focus: Option<usize>,
    pub(in crate::editor::hook) scroll: usize,
    pub(in crate::editor::hook) status: Option<String>,
    pub(in crate::editor::hook) touched: bool,
}

impl PlanetState {
    // Forget everything the last world left behind.
    pub(in crate::editor::hook) fn reset_for_world(&mut self) {
        *self = Self::default();
    }
}
