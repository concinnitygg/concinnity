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

/// `system` as a gate returns it.
pub fn boxed<S: System + 'static>(system: Option<S>) -> Option<Box<dyn System>> {
    system.map(|s| Box::new(s) as Box<dyn System>)
}

/// One row of the system table. Table order is run order.
#[derive(Clone, Copy)]
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
    /// Checked against table order (by [`SystemTable::check`] in tests, and at
    /// schedule build in debug builds): the table stays the one execution
    /// order, and an edge never reorders it.
    pub after: &'static [&'static str],
    /// Systems this one must run before.
    pub before: &'static [&'static str],
}

/// A host's completion pass: it runs over the world before the gates read it,
/// and fails the start when the world cannot be completed.
pub type CompleteWorld = fn(&mut PipelineContext<'_>) -> Result<(), WorldError>;

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
    pub before_init: Option<fn(&mut PipelineContext<'_>)>,
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

    /// What is wrong with this table, if anything: entries out of phase order,
    /// an edge that contradicts table order or names a system neither this
    /// table nor `order` holds, an entry with no gate description, an
    /// `init_first` name that is no entry, or entries of `order` held out of
    /// their relative order here. A table may hold a subset of `order`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn check(&self, order: &[SystemEntry]) -> Result<(), alloc::string::String> {
        use alloc::format;
        let position = |name: &str| self.entries.iter().position(|e| e.name == name);
        let known = |name: &str| position(name).is_some() || order.iter().any(|e| e.name == name);
        if let Some(name) = self.init_first.iter().find(|n| position(n).is_none()) {
            return Err(format!("init_first names {name}, which is no entry"));
        }
        for pair in self.entries.windows(2) {
            if pair[0].phase > pair[1].phase {
                return Err(format!(
                    "{} runs in {} and {} after it in {}",
                    pair[0].name,
                    pair[0].phase.as_str(),
                    pair[1].name,
                    pair[1].phase.as_str(),
                ));
            }
        }
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.present_when.is_empty() {
                return Err(format!("{} has no present_when", entry.name));
            }
            for name in entry.after.iter().chain(entry.before) {
                if !known(name) {
                    return Err(format!("{} names unknown system {name}", entry.name));
                }
            }
            if let Some(after) = entry
                .after
                .iter()
                .find(|a| position(a).is_some_and(|at| at > i))
            {
                return Err(format!(
                    "{} runs after {after}, which the table runs later",
                    entry.name
                ));
            }
            if let Some(before) = entry
                .before
                .iter()
                .find(|b| position(b).is_some_and(|at| at < i))
            {
                return Err(format!(
                    "{} runs before {before}, which the table runs earlier",
                    entry.name
                ));
            }
        }
        let mut held = order
            .iter()
            .filter_map(|e| position(e.name).map(|at| (e.name, at)));
        if let Some(mut previous) = held.next() {
            for next in held {
                if next.1 < previous.1 {
                    return Err(format!(
                        "{} runs before {}, out of order",
                        next.0, previous.0
                    ));
                }
                previous = next;
            }
        }
        Ok(())
    }

    /// [`check`](Self::check), and every entry of `order` is held: what a
    /// host table owes the full core list.
    #[cfg(any(test, feature = "test-support"))]
    pub fn check_complete(&self, order: &[SystemEntry]) -> Result<(), alloc::string::String> {
        self.check(order)?;
        match order
            .iter()
            .find(|e| !self.entries.iter().any(|mine| mine.name == e.name))
        {
            Some(missing) => Err(alloc::format!("the table drops {}", missing.name)),
            None => Ok(()),
        }
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

    fn entry(name: &'static str, phase: Phase, after: &'static [&'static str]) -> SystemEntry {
        SystemEntry {
            name,
            present_when: "always",
            phase,
            gate: |_| None,
            after,
            before: &[],
        }
    }

    // A table's entries are `'static`; a test's synthetic ones are leaked.
    fn leak(entries: &[SystemEntry]) -> &'static [SystemEntry] {
        Vec::leak(entries.to_vec())
    }

    #[test]
    fn a_well_formed_table_checks_clean() {
        let entries = [
            entry("A", Phase::Early, &[]),
            entry("B", Phase::Late, &["A", "C"]),
        ];
        let order = [entries[0], entry("C", Phase::Late, &[]), entries[1]];
        let table = SystemTable {
            entries: leak(&entries),
            init_first: &["B"],
            ..SystemTable::EMPTY
        };
        assert_eq!(table.check(&order), Ok(()));
        assert!(table.check_complete(&order).is_err(), "C is dropped");
        let whole = SystemTable {
            entries: leak(&order),
            ..SystemTable::EMPTY
        };
        assert_eq!(whole.check_complete(&order), Ok(()));
    }

    #[test]
    fn a_check_reports_each_kind_of_fault() {
        let phases = [entry("A", Phase::Late, &[]), entry("B", Phase::Early, &[])];
        let edge = [
            entry("A", Phase::Early, &["B"]),
            entry("B", Phase::Early, &[]),
        ];
        let unknown = [entry("A", Phase::Early, &["Z"])];
        for entries in [&phases[..], &edge[..], &unknown[..]] {
            let table = SystemTable {
                entries: leak(entries),
                ..SystemTable::EMPTY
            };
            assert!(table.check(&[]).is_err(), "{}", entries[0].name);
        }
        let entries = [entry("A", Phase::Early, &[]), entry("B", Phase::Early, &[])];
        let table = SystemTable {
            entries: leak(&entries),
            ..SystemTable::EMPTY
        };
        let reversed = [entries[1], entries[0]];
        assert!(table.check(&reversed).is_err());
        let stray = SystemTable {
            entries: leak(&entries),
            init_first: &["Z"],
            ..SystemTable::EMPTY
        };
        assert!(
            stray.check(&[]).is_err(),
            "an init_first name must be an entry"
        );
    }

    #[test]
    fn boxing_keeps_the_built_system() {
        #[derive(Debug)]
        struct Inert;

        impl System for Inert {
            fn step(&mut self, _ctx: &mut PipelineContext<'_>) -> crate::ecs::StepResult {
                crate::ecs::StepResult::Continue
            }
        }

        assert!(boxed(Some(Inert)).is_some());
        assert!(boxed(None::<Inert>).is_none());
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
