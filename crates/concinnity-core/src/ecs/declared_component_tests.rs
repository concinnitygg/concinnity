// src/ecs/declared_component_tests.rs
//
// A component type declared outside the engine's registry, through the whole
// component API. The point of these is that nothing in them is special: the
// same operations a registered component gets, over a type the registry has
// never heard of, including where the two are mixed on one entity.

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::components::TextLabel;
use crate::declare_components;
use crate::ecs::{ComponentSlot, ComponentTag, EXTENSION_COMPONENT_BASE, World};

#[derive(Debug, PartialEq)]
struct Health(u32);

#[derive(Debug, PartialEq)]
struct Faction(&'static str);

#[derive(Debug, PartialEq)]
struct Untouched(u8);

declare_components!(Health, Faction, Untouched);

fn label(content: &str) -> TextLabel {
    TextLabel {
        content: content.to_string(),
        ..Default::default()
    }
}

// The list numbers its types by position, from the first discriminant no
// registered component uses. Nothing is hand-numbered and nothing collides
// with the registry.
#[test]
fn discriminants_follow_the_registry() {
    assert_eq!(EXTENSION_COMPONENT_BASE, ComponentTag::COUNT);
    assert_eq!(Health::DISCRIMINANT, EXTENSION_COMPONENT_BASE);
    assert_eq!(Faction::DISCRIMINANT, EXTENSION_COMPONENT_BASE + 1);
    assert_eq!(Untouched::DISCRIMINANT, EXTENSION_COMPONENT_BASE + 2);
}

// A type nothing has pushed has no column, and every read of it answers as an
// empty one rather than failing. That is what lets a declared type be used
// without registering it anywhere first.
#[test]
fn a_type_with_no_rows_reads_as_empty() {
    let world = World::new();
    assert_eq!(world.query::<Untouched>().count(), 0);
    assert!(world.component_census().is_empty());
}

// Push mints an entity and the row is queryable, exactly as a registered
// component's would be.
#[test]
fn push_and_query() {
    let mut world = World::new();
    world.push(Health(100));
    world.push(Health(50));

    let seen: Vec<u32> = world.query::<Health>().map(|h| h.0).collect();
    assert_eq!(seen, [100, 50]);
    assert_eq!(world.component_count(), 2);
}

// A bare entity takes both kinds of component, and each is reachable by its
// own type off the same entity.
#[test]
fn one_entity_holds_a_declared_and_a_registered_component() {
    let mut world = World::new();
    let player = world.spawn();
    world.insert(player, Health(10));
    world.insert(player, label("player"));

    assert_eq!(world.get::<Health>(player), Some(&Health(10)));
    assert_eq!(
        world.get::<TextLabel>(player).map(|l| l.content.as_str()),
        Some("player")
    );
}

// A targeted write reaches the row and only that row.
#[test]
fn get_mut_writes_one_row() {
    let mut world = World::new();
    let a = world.push(Health(10));
    let b = world.push(Health(20));

    world.get_mut::<Health>(a).expect("a's row").0 = 11;
    assert_eq!(world.get::<Health>(a), Some(&Health(11)));
    assert_eq!(world.get::<Health>(b), Some(&Health(20)));
}

// A join across the two halves works: the lead column is declared, the joined
// one registered.
#[test]
fn a_declared_component_joins_a_registered_one() {
    let mut world = World::new();
    let both = world.spawn();
    world.insert(both, Health(7));
    world.insert(both, label("named"));
    let health_only = world.push(Health(9));

    let joined: Vec<(u32, &str)> = world
        .join2::<Health, TextLabel>()
        .map(|(_, h, l)| (h.0, l.content.as_str()))
        .collect();
    assert_eq!(joined, [(7, "named")]);
    assert!(world.is_alive(health_only));
}

// A join whose other side has no column at all yields nothing rather than
// failing: the answer to "which entities have an Untouched" is none.
#[test]
fn a_join_against_a_type_with_no_column_is_empty() {
    let mut world = World::new();
    world.push(Health(1));
    assert_eq!(world.join2::<Health, Untouched>().count(), 0);
    assert_eq!(world.join2::<Untouched, Health>().count(), 0);
}

// Despawn sweeps a declared column the same way it sweeps a registered one,
// and patches the row the swap moved so the survivor still reads its own data.
#[test]
fn despawn_sweeps_declared_and_registered_columns() {
    let mut world = World::new();
    let first = world.spawn();
    world.insert(first, Health(1));
    world.insert(first, label("first"));
    let second = world.spawn();
    world.insert(second, Health(2));
    world.insert(second, label("second"));

    world.despawn(first);

    assert!(!world.is_alive(first));
    assert_eq!(world.query::<Health>().count(), 1);
    assert_eq!(world.get::<Health>(second), Some(&Health(2)));
    assert_eq!(
        world.get::<TextLabel>(second).map(|l| l.content.as_str()),
        Some("second"),
        "the registered column's tail row moved with the declared one's"
    );
}

// Remove takes one component off an entity and leaves the rest, and the
// entity stays alive.
#[test]
fn remove_takes_one_component_off_an_entity() {
    let mut world = World::new();
    let entity = world.spawn();
    world.insert(entity, Health(3));
    world.insert(entity, Faction("blue"));

    let taken = world.context().remove::<Health>(entity);
    assert_eq!(taken, Some(Health(3)));
    assert_eq!(world.get::<Health>(entity), None);
    assert_eq!(world.get::<Faction>(entity), Some(&Faction("blue")));
    assert!(world.is_alive(entity));
}

// Drain empties the column and hands back every row; a type with no column
// drains to nothing rather than failing.
#[test]
fn drain_empties_a_declared_column() {
    let mut world = World::new();
    world.push(Health(1));
    world.push(Health(2));

    let drained = world.context().drain::<Health>();
    assert_eq!(drained, [Health(1), Health(2)]);
    assert_eq!(world.query::<Health>().count(), 0);
    assert!(world.context().drain::<Untouched>().is_empty());
}

// A targeted write is reported by row, so a declared component gets the same
// dirty-set treatment a registered one does.
#[test]
fn changed_rows_reports_a_targeted_write() {
    let mut world = World::new();
    let a = world.push(Health(1));
    let _b = world.push(Health(2));
    let since = world.context().changed_tick::<Health>();

    world.get_mut::<Health>(a).expect("a's row").0 = 5;

    let ctx = world.context();
    let changed: Vec<u32> = ctx
        .changed_rows::<Health>(since)
        .map(|(_, h)| h.0)
        .collect();
    assert_eq!(changed, [5]);
}

// The census reports declared columns by discriminant, after every registered
// one, so a debug snapshot sees them.
#[test]
fn the_census_reports_declared_columns_last() {
    let mut world = World::new();
    world.push(label("registered"));
    world.push(Health(1));
    world.push(Faction("blue"));

    let census = world.component_census();
    let declared: Vec<(u8, u32)> = census
        .iter()
        .copied()
        .filter(|(tag, _)| *tag >= EXTENSION_COMPONENT_BASE)
        .collect();
    assert_eq!(
        declared,
        [(Health::DISCRIMINANT, 1), (Faction::DISCRIMINANT, 1)]
    );
    assert!(
        census
            .iter()
            .any(|(tag, _)| *tag == ComponentTag::TextLabel as u8),
        "the registered column is still censused: {census:?}"
    );
    let tags: Vec<u8> = census.iter().map(|(tag, _)| *tag).collect();
    let mut sorted = tags.clone();
    sorted.sort_unstable();
    assert_eq!(tags, sorted, "the whole census stays in tag order");
}

// The headroom a declaration list has: the mask holds 128 ids and the registry
// takes the first `COUNT` of them. Not a requirement, a readout -- it fails
// when the registry grows past what the mask can hold beside it, which is the
// moment `ComponentMask` has to widen.
#[test]
fn the_registry_leaves_room_to_declare_more() {
    let headroom = crate::ecs::ComponentId::MAX - EXTENSION_COMPONENT_BASE + 1;
    assert!(
        headroom >= 8,
        "{} registered components leave only {headroom} ids before the mask ceiling",
        ComponentTag::COUNT,
    );
}
