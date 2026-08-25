// src/physics/budget.rs
//
// What the driver reserves for a world's physics, and how it learns the size.
//
// Cook ships a budget in the blob; a world built in memory (a unit test, an
// uncooked world) has none, so the same counts are scanned off the loaded
// components instead. Both go through `PhysicsBudget::derive`, which is what
// lets the runtime debug-assert that the shipped record and the live world
// agree.
//
// The capacities and their byte cost are pure functions of the budget, kept
// apart from the code that applies them so both can be tested without a
// simulation.

use std::collections::HashSet;

use concinnity_core::assets::{
    BodyDynamics, Camera3D, CharacterRig, Collider, PhysicsJoint, Transform, TriggerVolume,
};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{Entity, EntityByName, PhysicsBudgetRecord, PipelineContext};
use concinnity_memory::{Ledger, MemTag, Realm};
use concinnity_physics::{
    BodyHandle, ContactHit, PhysicsBudget, PhysicsCounts, SensorCrossing, Simulation,
};

use super::props::{PropCollSnap, PropPhysics};
use super::rig::RigPhysics;

// Contacts one dynamic body can report in a single tick. A body wedged against
// several neighbours reports one per pair; four keeps the drain scratch from
// growing mid-frame without reserving for a pile-up.
const CONTACTS_PER_DYNAMIC_BODY: usize = 4;
// A sensor reports at most an entry and an exit per body crossing it.
const CROSSINGS_PER_SENSOR: usize = 2;

/// The budget a shipped record describes.
pub(crate) fn budget_of(record: &PhysicsBudgetRecord) -> PhysicsBudget {
    PhysicsBudget {
        fixed: record.fixed,
        dynamic: record.dynamic,
        kinematic: record.kinematic,
        sensors: record.sensors,
        joints: record.joints,
        anchors: record.anchors,
        spawn_headroom: record.spawn_headroom,
    }
}

/// The record a budget ships as. Cook writes the shipped copy; this is the
/// inverse the tests compare against.
#[cfg(test)]
pub(crate) fn record_of(budget: &PhysicsBudget) -> PhysicsBudgetRecord {
    PhysicsBudgetRecord {
        fixed: budget.fixed,
        dynamic: budget.dynamic,
        kinematic: budget.kinematic,
        sensors: budget.sensors,
        joints: budget.joints,
        anchors: budget.anchors,
        spawn_headroom: budget.spawn_headroom,
    }
}

/// Count the world's physics content off its loaded components, mirroring what
/// [`super::PhysicsSystem`]'s init goes on to build.
///
/// Must run before init drains the `PhysicsJoint` column.
pub(crate) fn scan_counts(ctx: &PipelineContext) -> PhysicsCounts {
    // A body is built per collider-bearing entity that also has a transform to
    // place it at, and it simulates freely when the entity carries dynamics.
    let bodies: HashSet<Entity> = ctx
        .join2::<Collider, Transform>()
        .map(|(e, ..)| e)
        .collect();
    let dynamics: HashSet<Entity> = ctx
        .query_with_entity::<BodyDynamics>()
        .map(|(e, _)| e)
        .collect();
    let dynamic_colliders = bodies.iter().filter(|e| dynamics.contains(e)).count() as u32;

    let named = ctx.resource::<EntityByName>();
    let has_body = |id: AssetId| {
        named
            .and_then(|index| index.0.get(&id))
            .is_some_and(|entity| bodies.contains(entity))
    };

    let mut counts = PhysicsCounts {
        static_colliders: bodies.len() as u32 - dynamic_colliders,
        dynamic_colliders,
        trigger_volumes: ctx.query::<TriggerVolume>().count() as u32,
        rig_capsules: ctx.query::<CharacterRig>().count() as u32,
        ..PhysicsCounts::default()
    };

    for joint in ctx.query::<PhysicsJoint>() {
        if !joint.body_a.is_some_and(has_body) {
            continue;
        }
        match joint.body_b {
            Some(body_b) if has_body(body_b) => counts.joints += 1,
            Some(_) => continue,
            None => {
                counts.joints += 1;
                counts.world_anchored_joints += 1;
            }
        }
    }

    // The first declared camera owns the player capsule, unless it orbits a
    // followed character instead.
    let first_person = ctx.query::<Camera3D>().next().is_some_and(|camera| {
        camera
            .controller
            .as_ref()
            .is_none_or(|ctrl| ctrl.follow.is_none())
    });
    if first_person {
        counts.player_capsules = 1;
    }

    counts
}

/// How many entries each of the driver's containers reserves for a budget.
///
/// Everything the driver holds per body is sized from here once at init and
/// never grown, so a container that outgrows its entry is a budget that was
/// counted wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DriverCapacities {
    /// Prop bodies: the authored colliders plus the runtime-spawn headroom.
    pub(crate) props: usize,
    /// Entities refused a body, at most the headroom's worth of spawns.
    pub(crate) refused: usize,
    /// Dynamic props, whose blended poses are written back each frame.
    pub(crate) sampled: usize,
    /// The per-tick scan for freshly spawned collider-bearing entities.
    pub(crate) new_props: usize,
    /// Character rig capsules.
    pub(crate) rigs: usize,
    /// Root-motion displacements drained per frame.
    pub(crate) root_motions: usize,
    /// Contact hits drained per tick.
    pub(crate) contacts: usize,
    /// Body pairs the frame's contact batching and refractory gate track.
    pub(crate) contact_pairs: usize,
    /// Sensor crossings drained per frame.
    pub(crate) sensor_crossings: usize,
    /// Sensor tag -> trigger volume entries.
    pub(crate) sensor_filters: usize,
}

impl DriverCapacities {
    /// Derive the reservations a budget implies.
    pub(crate) fn derive(budget: &PhysicsBudget) -> Self {
        // The floor is a body the driver builds directly, not a prop.
        let statics = budget.fixed.saturating_sub(1) as usize;
        let headroom = budget.spawn_headroom as usize;
        let dynamic = budget.dynamic as usize + headroom;
        Self {
            props: statics + dynamic,
            refused: headroom,
            sampled: dynamic,
            new_props: headroom,
            rigs: budget.kinematic as usize,
            root_motions: budget.kinematic as usize,
            contacts: dynamic * CONTACTS_PER_DYNAMIC_BODY,
            contact_pairs: dynamic * CONTACTS_PER_DYNAMIC_BODY,
            sensor_crossings: budget.sensors as usize * CROSSINGS_PER_SENSOR,
            sensor_filters: budget.sensors as usize,
        }
    }
}

// Bytes a `Vec<T>` of `capacity` holds.
fn vec_bytes<T>(capacity: usize) -> u64 {
    (capacity as u64).saturating_mul(size_of::<T>() as u64)
}

// Bytes a hash table reserved for `capacity` entries holds: the bucket count is
// rounded up past the 7/8 load factor to a power of two, and each bucket keeps
// a control byte beside it.
fn table_bytes<K, V>(capacity: usize) -> u64 {
    if capacity == 0 {
        return 0;
    }
    let buckets = capacity.saturating_mul(8).div_ceil(7).next_power_of_two() as u64;
    buckets.saturating_mul(size_of::<(K, V)>() as u64 + 1)
}

/// Host bytes the driver's own containers hold once a budget is reserved.
///
/// The simulation reserves against the same budget and reports its own bytes;
/// [`publish_reservation`] adds the two.
pub(crate) fn reserved_bytes(budget: &PhysicsBudget) -> u64 {
    let caps = DriverCapacities::derive(budget);
    vec_bytes::<PropPhysics>(caps.props)
        + table_bytes::<BodyHandle, Entity>(caps.props)
        + table_bytes::<Entity, ()>(caps.props)
        + table_bytes::<Entity, ()>(caps.refused)
        + vec_bytes::<(Entity, [f32; 3], [f32; 3])>(caps.sampled)
        + vec_bytes::<(Entity, PropCollSnap)>(caps.new_props)
        + vec_bytes::<RigPhysics>(caps.rigs)
        + vec_bytes::<concinnity_core::assets::RootMotionEvent>(caps.root_motions)
        + vec_bytes::<ContactHit>(caps.contacts)
        + vec_bytes::<SensorCrossing>(caps.sensor_crossings)
        + table_bytes::<(BodyHandle, BodyHandle), ContactHit>(caps.contact_pairs)
        + table_bytes::<(BodyHandle, BodyHandle), u64>(caps.contact_pairs)
        + table_bytes::<u64, (AssetId, concinnity_core::assets::TriggerFilter)>(caps.sensor_filters)
}

/// Report the reservation under the physics tag: the driver's containers plus
/// the simulation's own storage, which is only knowable once `world` is built.
/// Republished rather than accumulated, so the row is what physics holds now.
pub(crate) fn publish_reservation(ledger: &Ledger, budget: &PhysicsBudget, world: &Simulation) {
    let driver = reserved_bytes(budget);
    let simulation = world.reserved_bytes();
    let bytes = driver.saturating_add(simulation);
    tracing::debug!(
        "PhysicsSystem: {} bodies reserved (+{} spawn headroom), {} host bytes ({} driver, {} simulation)",
        budget.body_total(),
        budget.spawn_headroom,
        bytes,
        driver,
        simulation
    );
    ledger.set(MemTag::Physics, Realm::Host, bytes);
    // Both sides are reserved from the budget at init, so the reservation is
    // both what is held and the ceiling it is held against.
    ledger.set_budget(MemTag::Physics, Realm::Host, Some(bytes));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(counts: PhysicsCounts, headroom: u32) -> PhysicsBudget {
        PhysicsBudget::derive(&counts, headroom)
    }

    #[test]
    fn a_record_round_trips_through_the_budget_it_describes() {
        let original = budget(
            PhysicsCounts {
                static_colliders: 7,
                dynamic_colliders: 3,
                trigger_volumes: 2,
                joints: 1,
                world_anchored_joints: 1,
                player_capsules: 1,
                rig_capsules: 4,
            },
            12,
        );
        assert_eq!(budget_of(&record_of(&original)), original);
    }

    #[test]
    fn capacities_hold_the_authored_content_plus_the_headroom() {
        let caps = DriverCapacities::derive(&budget(
            PhysicsCounts {
                static_colliders: 5,
                dynamic_colliders: 2,
                trigger_volumes: 3,
                player_capsules: 1,
                rig_capsules: 2,
                ..PhysicsCounts::default()
            },
            10,
        ));
        assert_eq!(caps.props, 5 + 2 + 10, "statics, dynamics, and spawns");
        assert_eq!(caps.sampled, 2 + 10, "only dynamic props are written back");
        assert_eq!(caps.refused, 10);
        assert_eq!(caps.new_props, 10);
        assert_eq!(caps.rigs, 3, "the player capsule plus two rigs");
        assert_eq!(caps.sensor_filters, 3);
        assert!(caps.contacts > 0 && caps.sensor_crossings > 0);
    }

    // The floor is built directly, so it must not inflate the prop containers.
    #[test]
    fn an_empty_world_reserves_no_prop_capacity() {
        let caps = DriverCapacities::derive(&budget(PhysicsCounts::default(), 0));
        assert_eq!(caps.props, 0);
        assert_eq!(caps.sampled, 0);
        assert_eq!(caps.contacts, 0);
    }

    #[test]
    fn reserved_bytes_grow_with_the_budget_they_describe() {
        let empty = reserved_bytes(&budget(PhysicsCounts::default(), 0));
        assert_eq!(empty, 0, "a floor-only world holds nothing per body");

        let small = reserved_bytes(&budget(
            PhysicsCounts {
                static_colliders: 4,
                ..PhysicsCounts::default()
            },
            0,
        ));
        assert!(small > 0);

        let with_headroom = reserved_bytes(&budget(
            PhysicsCounts {
                static_colliders: 4,
                ..PhysicsCounts::default()
            },
            64,
        ));
        assert!(
            with_headroom > small,
            "headroom is reserved, not borrowed later"
        );
    }

    // The ledger row is the whole reservation -- the driver's containers and
    // the simulation's storage -- republished rather than accumulated.
    #[test]
    fn the_published_row_counts_the_simulation_as_well_as_the_driver() {
        let ledger = Ledger::new();
        let budget = budget(
            PhysicsCounts {
                static_colliders: 8,
                dynamic_colliders: 4,
                ..PhysicsCounts::default()
            },
            16,
        );
        let world = Simulation::with_capacity(budget.body_cap() as usize);
        publish_reservation(&ledger, &budget, &world);

        let driver = reserved_bytes(&budget);
        let usage = ledger.usage(MemTag::Physics, Realm::Host);
        assert!(driver > 0, "a world with bodies holds something");
        assert_eq!(usage.bytes, driver + world.reserved_bytes());
        assert!(
            usage.bytes > driver,
            "the simulation reserves against the same budget"
        );
        assert_eq!(
            usage.budget,
            Some(usage.bytes),
            "everything is reserved at init, so usage is the ceiling"
        );
    }

    #[test]
    fn a_hash_table_reserves_past_its_load_factor() {
        assert_eq!(table_bytes::<u64, u64>(0), 0);
        // 8 entries need more than 8 buckets at a 7/8 load factor, so the
        // table rounds up to 16 rather than filling exactly.
        assert_eq!(table_bytes::<u64, u64>(8), 16 * (16 + 1));
        assert_eq!(vec_bytes::<u32>(10), 40);
    }
}
