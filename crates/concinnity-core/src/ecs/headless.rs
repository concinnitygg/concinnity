// The system table for a world that runs without a host: the simulation
// systems this crate owns, in run order.
//
// A second table, not a slice a bigger one splices in. Table order is run
// order, so a table read top to bottom is the tick, and that property only
// survives if each table is one document. A host with a renderer writes its
// own, listing these same systems in the same relative order among its own;
// a test in that host asserts the two agree.
//
// Every gate here inspects the world's content and returns the constructed
// system, or `None` to leave it out. Nothing is loaded from a blob: a system is
// internal, has no declarable asset, and carries no discriminant.

use alloc::boxed::Box;

use crate::behavior::BehaviorSystem;
use crate::components::{Behavior, PhysicsConfig, PropBody, RigidBody, SkyRotation, TriggerVolume};
use crate::ecs::{System, SystemEntry, SystemTable, World};
use crate::physics::PhysicsSystem;
use crate::resource::SkinnedMeshTable;
use crate::sky::SkyRotationSystem;

// SkyRotationSystem: present whenever the world declares a `SkyRotation`. Runs
// ahead of everything that reads the sky's orientation.
fn sky_rotation(world: &World) -> Option<Box<dyn System>> {
    let rotation = world.query::<SkyRotation>().next()?;
    Some(Box::new(SkyRotationSystem::new(rotation)))
}

// BehaviorSystem: present whenever the world declares any `Behavior`. Runs
// serially and persists nothing; a host lends it a thread pool and a state
// store when it has them.
fn behavior(world: &World) -> Option<Box<dyn System>> {
    world.query::<Behavior>().next()?;
    Some(Box::new(BehaviorSystem::new()))
}

// PhysicsSystem: present whenever the world has physics content. Steps on the
// calling thread; a host lends it a job pool when it has one.
fn physics(world: &World) -> Option<Box<dyn System>> {
    let needs = world.query::<PhysicsConfig>().next().is_some()
        || world.query::<RigidBody>().next().is_some()
        || world.query::<PropBody>().next().is_some()
        || world.query::<TriggerVolume>().next().is_some()
        || world
            .resource::<SkinnedMeshTable>()
            .is_some_and(|t| t.has_capsule());
    if !needs {
        return None;
    }
    // A shipped world carries the config cook injected; one built directly
    // (a test, an in-memory tool) gets the flat-floor default.
    let config = world
        .query::<PhysicsConfig>()
        .next()
        .cloned()
        .unwrap_or_default();
    Some(Box::new(PhysicsSystem::new(config)))
}

/// The simulation systems a world runs with no host beyond this crate, in run
/// order. What [`App`](crate::App) starts a headless world against.
pub const HEADLESS_SYSTEMS: &SystemTable = &SystemTable {
    entries: &[
        SystemEntry {
            name: "SkyRotationSystem",
            present_when: "the world declares a SkyRotation",
            gate: sky_rotation,
            after: &[],
            before: &[],
        },
        SystemEntry {
            name: "BehaviorSystem",
            present_when: "the world declares any Behavior",
            gate: behavior,
            after: &[],
            before: &[],
        },
        SystemEntry {
            name: "PhysicsSystem",
            present_when: "the world declares a PhysicsConfig, RigidBody, PropBody, or TriggerVolume, or a skinned mesh bakes a character capsule",
            gate: physics,
            after: &[],
            before: &[],
        },
    ],
    // The headless tier renders nothing, so the completion pass leaves it with
    // the one default that is not a rendering concern: the PhysicsConfig its
    // simulation already runs on.
    complete_world: Some(crate::defaults::run),
    before_init: None,
    prepare_events: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // Every declared edge agrees with table order, and names a real entry: an
    // edge that contradicts the table is a schedule-build panic at world start,
    // and a typo silently drops the constraint.
    #[test]
    fn declared_edges_respect_table_order_and_name_real_entries() {
        let entries = HEADLESS_SYSTEMS.entries;
        let position = |name: &str| entries.iter().position(|e| e.name == name);
        for (i, entry) in entries.iter().enumerate() {
            for after in entry.after {
                let at = position(after).unwrap_or_else(|| panic!("{} names {after}", entry.name));
                assert!(
                    at < i,
                    "{} runs after {after}, but the table runs {after} later",
                    entry.name,
                );
            }
            for before in entry.before {
                let at =
                    position(before).unwrap_or_else(|| panic!("{} names {before}", entry.name));
                assert!(
                    i < at,
                    "{} runs before {before}, but the table runs {before} earlier",
                    entry.name,
                );
            }
            assert!(
                !entry.present_when.is_empty(),
                "{} has no present_when",
                entry.name
            );
        }
    }

    // A world with no behaviors gates nothing in, so a headless run over it has
    // nothing to do.
    #[test]
    fn an_empty_world_gates_no_systems() {
        let world = World::new();
        assert!(world.system_manifest(HEADLESS_SYSTEMS).is_empty());
    }

    // A declared SkyRotation gates the sky in, ahead of every system that
    // reads the orientation it publishes.
    #[test]
    fn a_sky_rotation_gates_the_sky_system_first() {
        let mut world = World::new();
        world.add_component(SkyRotation::default());
        world.add_component(Behavior::default());
        assert_eq!(
            world.system_manifest(HEADLESS_SYSTEMS),
            ["SkyRotationSystem", "BehaviorSystem"]
        );
    }

    // A declared Behavior is what puts the system in the table's manifest.
    #[test]
    fn a_behavior_gates_the_behavior_system() {
        let mut world = World::new();
        world.add_component(Behavior::default());
        let manifest: Vec<&str> = world.system_manifest(HEADLESS_SYSTEMS);
        assert_eq!(manifest, ["BehaviorSystem"]);
    }

    // Physics content gates the simulation driver in, on its own and beside a
    // behavior, in table order.
    #[test]
    fn physics_content_gates_the_physics_system() {
        let mut world = World::new();
        world.add_component(PhysicsConfig::default());
        assert_eq!(world.system_manifest(HEADLESS_SYSTEMS), ["PhysicsSystem"]);

        let mut world = World::new();
        world.add_component(RigidBody::default());
        assert_eq!(world.system_manifest(HEADLESS_SYSTEMS), ["PhysicsSystem"]);

        let mut world = World::new();
        world.add_component(Behavior::default());
        world.add_component(PhysicsConfig::default());
        assert_eq!(
            world.system_manifest(HEADLESS_SYSTEMS),
            ["BehaviorSystem", "PhysicsSystem"]
        );
    }
}
