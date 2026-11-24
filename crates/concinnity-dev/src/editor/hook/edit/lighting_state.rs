//! `EditorHook`: the Lighting panel's session state: what each binding's text
//! field holds, which one holds keyboard focus, the message from the last
//! rejected Apply, whether typed or clicked control state is waiting on Apply
//! (the title's "*"), and what the panel shows of the entries.

use crate::editor::panels::form::FormField;
use crate::editor::panels::lighting_panel::LightingIds;

#[derive(Debug, Default)]
pub(in crate::editor::hook) struct LightingState {
    pub(in crate::editor::hook) texts: Vec<String>,
    pub(in crate::editor::hook) focus: Option<usize>,
    pub(in crate::editor::hook) status: Option<String>,
    pub(in crate::editor::hook) touched: bool,
    // Each section's presence and each binding's field, as of the entries
    // revision they were read at.
    pub(in crate::editor::hook) view: Option<(u64, Vec<bool>, Vec<Option<FormField>>)>,
    // The rows built into the current world.
    pub(in crate::editor::hook) shown: Option<LightingIds>,
}
