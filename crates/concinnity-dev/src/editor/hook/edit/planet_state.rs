//! `EditorHook`: the Planet panel's session state: each row's typed text and
//! which row's holds keyboard focus, the message from the last rejected edit,
//! whether typed edits are waiting on Apply, the rows the panel shows, the
//! paint tool's pick it names, and the rows built into the world.

use crate::editor::panels::planet::Row;
use crate::editor::panels::planet::panel::PlanetIds;
use crate::editor::planet_tools::Tool;

#[derive(Debug, Default)]
pub(in crate::editor::hook) struct PlanetState {
    pub(in crate::editor::hook) texts: Vec<String>,
    pub(in crate::editor::hook) focus: Option<usize>,
    pub(in crate::editor::hook) status: Option<String>,
    pub(in crate::editor::hook) touched: bool,
    // The rows, read at an entries revision.
    pub(in crate::editor::hook) view: Option<(u64, Vec<Row>)>,
    // The pick's name, read at an entries revision for a tool and its pick.
    pub(in crate::editor::hook) pick: Option<(u64, Option<Tool>, usize, String)>,
    pub(in crate::editor::hook) shown: Option<PlanetIds>,
}

impl PlanetState {
    // Forget everything the last world left behind.
    pub(in crate::editor::hook) fn reset_for_world(&mut self) {
        *self = Self::default();
    }
}
