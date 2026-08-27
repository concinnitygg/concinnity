//! Two tables, one schedule.
//!
//! concinnity-core writes its own system table for a world that runs with no
//! host (`HEADLESS_SYSTEMS`), listing the simulation systems it owns. This
//! crate's `SYSTEMS` lists those same systems among its own, and table order is
//! run order in both. Keeping each table one readable document is the reason
//! there are two; this is the price of that, and the only thing that stops them
//! drifting apart.
//!
//! What must agree: every core entry is present here, in the same relative
//! order, under the same name, with the same `present_when`, and with the same
//! ordering edges among the systems core also knows about. Edges to systems
//! core has no idea exist (the render band, audio, story) are this crate's
//! business and are not compared.

use concinnity_core::ecs::{HEADLESS_SYSTEMS, SystemEntry};

use crate::ecs::SYSTEMS;

// The entry of the same name, or a failure naming what went missing.
fn matching<'a>(entries: &'a [SystemEntry], name: &str) -> &'a SystemEntry {
    entries.iter().find(|e| e.name == name).unwrap_or_else(|| {
        let present: Vec<&str> = entries.iter().map(|e| e.name).collect();
        panic!(
            "core's headless table lists '{name}', which this crate's table does not: {present:?}"
        )
    })
}

// Edges restricted to the systems both tables know about, in declaration order.
fn shared_edges<'a>(edges: &'a [&'a str], known: &[&str]) -> Vec<&'a str> {
    edges
        .iter()
        .copied()
        .filter(|e| known.contains(e))
        .collect()
}

#[test]
fn every_headless_entry_appears_here_in_the_same_order() {
    let core = HEADLESS_SYSTEMS.entries;
    let mine = SYSTEMS.entries;

    let mut previous: Option<(&str, usize)> = None;
    for entry in core {
        let here = matching(mine, entry.name);
        let at = mine
            .iter()
            .position(|e| e.name == entry.name)
            .expect("the entry was just found by name");
        if let Some((earlier, earlier_at)) = previous {
            assert!(
                earlier_at < at,
                "core's table runs {earlier} before {}, but this crate's runs it after",
                here.name,
            );
        }
        previous = Some((here.name, at));
    }
}

#[test]
fn every_headless_entry_keeps_its_gate_description_and_shared_edges() {
    let core = HEADLESS_SYSTEMS.entries;
    let mine = SYSTEMS.entries;
    let known: Vec<&str> = core.iter().map(|e| e.name).collect();

    for entry in core {
        let here = matching(mine, entry.name);
        assert_eq!(
            here.present_when, entry.present_when,
            "{}'s gate reads '{}' here and '{}' in core's headless table",
            entry.name, here.present_when, entry.present_when,
        );
        assert_eq!(
            shared_edges(here.after, &known),
            shared_edges(entry.after, &known),
            "{}'s `after` edges among {known:?} differ: {:?} here, {:?} in core's headless table",
            entry.name,
            here.after,
            entry.after,
        );
        assert_eq!(
            shared_edges(here.before, &known),
            shared_edges(entry.before, &known),
            "{}'s `before` edges among {known:?} differ: {:?} here, {:?} in core's headless table",
            entry.name,
            here.before,
            entry.before,
        );
    }
}

// The gates are written separately (core's builds a bare system, this crate's
// attaches its job pool and its state store), so agreeing on `present_when` is
// only half the claim. The other half is that they fire on the same content.
#[test]
fn the_two_tables_gate_the_same_systems_in_for_the_same_world() {
    use crate::components::{Behavior, PhysicsConfig, RigidBody, TriggerVolume};
    use crate::ecs::World;

    // One world per gating component, so a gate that fires on the wrong half
    // of an `or` is caught rather than hidden by a neighbour.
    let worlds: [fn(&mut World); 4] = [
        |w| w.add_component(Behavior::default()),
        |w| w.add_component(PhysicsConfig::default()),
        |w| w.add_component(RigidBody::default()),
        |w| w.add_component(TriggerVolume::default()),
    ];
    for declare in worlds {
        let mut world = World::new();
        declare(&mut world);

        let mine = world.system_manifest(SYSTEMS);
        let headless = world.system_manifest(HEADLESS_SYSTEMS);
        assert!(
            !headless.is_empty(),
            "the content gates nothing in core's headless table",
        );
        for name in headless {
            assert!(
                mine.contains(&name),
                "core's headless table gates {name} in for this world and this crate's does not: \
                 {mine:?}",
            );
        }
    }
}
