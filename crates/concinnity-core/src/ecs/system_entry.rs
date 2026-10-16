//! The system table a host hands [`World::start`](crate::ecs::World::start):
//! one entry per system in run order, plus the load-time passes that bracket
//! them.
//!
//! The table is a document. Table order is run order, and the `after` / `before`
//! edges are checked against it rather than resolved into one, so a reader can
//! take the file top to bottom as the tick. What builds the systems is a gate
//! per entry: it inspects the world's content and returns the constructed
//! system, or `None` to leave it out.
//!
//! Each entry also names the [`Phase`] it runs in, and a table's entries are in
//! phase order. That is what a system registered on a world from outside the
//! table anchors to: it names a phase, never an entry, so the entries stay
//! internal and their order stays the table's own business.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::ecs::{Access, EventStore, Phase, PipelineContext, System, World};
use crate::error::WorldError;

/// One row of the system table. Table order is run order.
pub struct SystemEntry {
    /// The entry name; the system's stable display name.
    pub name: &'static str,
    /// Human-readable gate condition, for docs and CLI reporting.
    pub present_when: &'static str,
    /// Where in the tick this entry runs. Entries appear in the table in phase
    /// order, and a system registered on a world runs after every entry sharing
    /// its phase.
    pub phase: Phase,
    /// Constructs the system from world content when its gate holds. Runs from
    /// `World::start` and from `World::system_manifest`, which discards the
    /// value, so a system's constructor must stay cheap and side-effect-free.
    pub gate: fn(&World) -> Option<Box<dyn System>>,
    /// Systems (by entry name) that must run earlier in the tick than this one.
    /// Validated against table order at schedule build: the table stays the one
    /// execution order, and an edge that contradicts it is a startup panic, not
    /// a silent reorder.
    pub after: &'static [&'static str],
    /// Systems this one must run before.
    pub before: &'static [&'static str],
}

/// A host's completion pass: it runs over the world before the gates read it,
/// and fails the start when the world cannot be completed.
pub type CompleteWorld = fn(&mut PipelineContext) -> Result<(), WorldError>;

/// A host's system table and the load-time passes only the host can supply.
///
/// The entries name the host's own system types, so the table is written where
/// those types live; everything that runs it is here. A world can also carry
/// systems no table lists ([`World::add_system`](crate::ecs::World::add_system));
/// those anchor to a [`Phase`] rather than to an entry, so a table stays a
/// closed document whatever a world registers beside it.
pub struct SystemTable {
    /// One entry per system, in run order.
    pub entries: &'static [SystemEntry],
    /// Entries (by name) whose `init` runs ahead of the rest, in this order;
    /// the rest init in run order. For a system that runs late in the tick but
    /// whose `init` builds what other inits read.
    pub init_first: &'static [&'static str],
    /// Runs over the world before the gates read it, so a component the pass
    /// injects can still bring its system into the schedule. Absent leaves the
    /// world exactly as it was loaded.
    pub complete_world: Option<CompleteWorld>,
    /// Runs over the world once its systems are built and before their `init`.
    /// Absent leaves the loaded content exactly as it was added.
    pub before_init: Option<fn(&mut PipelineContext)>,
    /// Pre-creates the event queues a scheduled system's declared access can
    /// touch, so its `events_mut` never grows the store's map mid-tick. Absent
    /// leaves every queue to be created on first use.
    pub prepare_events: Option<fn(&mut EventStore, Access)>,
}

impl SystemTable {
    /// A table with no systems and no load-time passes: what a world runs when
    /// its host contributes none.
    pub const EMPTY: Self = Self {
        entries: &[],
        init_first: &[],
        complete_world: None,
        before_init: None,
        prepare_events: None,
    };

    /// The order the built systems named `built` (in run order) init in: the
    /// table's `init_first` entries that are present, in that order, then the
    /// rest in run order. Indices into `built`.
    pub fn init_order(&self, built: &[&str]) -> Vec<usize> {
        let first = self
            .init_first
            .iter()
            .filter_map(|name| built.iter().position(|b| b == name));
        let rest = (0..built.len()).filter(|&i| !self.init_first.contains(&built[i]));
        first.chain(rest).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(init_first: &'static [&'static str]) -> SystemTable {
        SystemTable {
            init_first,
            ..SystemTable::EMPTY
        }
    }

    #[test]
    fn without_init_first_systems_init_in_run_order() {
        assert_eq!(table(&[]).init_order(&["A", "B", "C"]), [0, 1, 2]);
    }

    #[test]
    fn init_first_entries_lead_in_their_own_order() {
        let order = table(&["C", "A"]).init_order(&["A", "B", "C", "D"]);
        assert_eq!(order, [2, 0, 1, 3]);
    }

    // A gate that left an `init_first` entry out leaves no hole in the order.
    #[test]
    fn an_absent_init_first_entry_is_skipped() {
        let order = table(&["Gone", "C"]).init_order(&["A", "C"]);
        assert_eq!(order, [1, 0]);
    }
}
