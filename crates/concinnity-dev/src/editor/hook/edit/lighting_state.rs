//! EditorHook: the Lighting panel's session state: which text binding holds
//! keyboard focus, the message from the last rejected Apply, and whether typed
//! or clicked control state is waiting on Apply (the heading's "*").

#[derive(Debug, Default)]
pub(in crate::editor::hook) struct LightingState {
    pub(in crate::editor::hook) focus: Option<usize>,
    pub(in crate::editor::hook) status: Option<String>,
    pub(in crate::editor::hook) touched: bool,
}
