//! The system table a host hands [`World::start`](crate::ecs::World::start):
//! one entry per system in run order, plus the load-time passes that bracket
//! them.
//!
//! The table is a document. Table order is run order, and the `after` / `before`
//! edges are checked against it rather than resolved into one, so a reader can
//! take the file top to bottom as the tick. What builds the systems is a gate
//! per entry: it inspects the world's content and returns the constructed
//! system, or `None` to leave it out.

use alloc::boxed::Box;

use crate::ecs::{Access, EventStore, PipelineContext, System, World};

/// One row of the system table. Table order is run order.
pub struct SystemEntry {
    /// The entry name; the system's stable display name.
    pub name: &'static str,
    /// Human-readable gate condition, for docs and CLI reporting.
    pub present_when: &'static str,
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

/// A host's system table and the load-time passes only the host can supply.
///
/// The entries name the host's own system types, so the table is written where
/// those types live; everything that runs it is here.
pub struct SystemTable {
    /// One entry per system, in run order.
    pub entries: &'static [SystemEntry],
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
        before_init: None,
        prepare_events: None,
    };
}
