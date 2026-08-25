//! Guards the registry's `consumed` flag against the load-time passes it
//! describes.
//!
//! A column marked `consumed` is drained during `World::start`, so a behavior
//! naming it in `scope` or `queries` would match nothing forever; the world
//! checker rejects those names on the strength of the flag alone. The flag is
//! declared in concinnity-core while the drains live here, so nothing but this
//! test stops the two from drifting apart: it starts a world holding one of
//! each component it can build without a graphics device and asserts the
//! census agrees with what the registry claims.
//!
//! The graphics drains (`Window`, `GraphicsConfig`, `Model`, `Decal`, and the
//! rest of the `GraphicsSystem` init sweep) need a device to reach, so they
//! stay covered by the flag alone.

use crate::components::{
    AudioEmitter, Behavior, HitRegion, KeyBinding, PhysicsJoint, Prop, PropBody, Screen,
    ScrollPanel, TriggerVolume,
};
use crate::ecs::World;
use concinnity_core::ecs::ComponentTag;

// One of each component under test, added before the world starts. The returned
// tags are what the census is then checked against.
fn seed(world: &mut World) -> Vec<ComponentTag> {
    world.add_component(Prop {
        scale: [1.0; 3],
        ..Default::default()
    });
    world.add_component(PropBody::default());
    world.add_component(Screen::default());
    world.add_component(KeyBinding::default());
    world.add_component(HitRegion::default());
    world.add_component(ScrollPanel::default());
    world.add_component(PhysicsJoint::default());
    world.add_component(Behavior::default());
    world.add_component(AudioEmitter::default());
    world.add_component(TriggerVolume::default());
    vec![
        ComponentTag::Prop,
        ComponentTag::PropBody,
        ComponentTag::Screen,
        ComponentTag::KeyBinding,
        ComponentTag::HitRegion,
        ComponentTag::ScrollPanel,
        ComponentTag::PhysicsJoint,
        ComponentTag::Behavior,
        ComponentTag::AudioEmitter,
        ComponentTag::TriggerVolume,
    ]
}

#[test]
fn the_registry_agrees_with_what_start_drains() {
    let mut world = World::new();
    let covered = seed(&mut world);
    for tag in &covered {
        assert!(
            world
                .component_census()
                .iter()
                .any(|(t, n)| *t == *tag as u8 && *n > 0),
            "{} was not seeded, so the check below would pass vacuously",
            tag.as_str(),
        );
    }

    world.start().unwrap();

    let census = world.component_census();
    for tag in covered {
        let held = census
            .iter()
            .find(|(t, _)| *t == tag as u8)
            .map(|(_, n)| *n)
            .unwrap_or(0);
        let survives = tag.surviving_tag() == Some(tag);
        assert_eq!(
            held > 0,
            survives,
            "{} is registered as {}, but start() left {held} of them",
            tag.as_str(),
            if survives { "surviving" } else { "consumed" },
        );
    }
}

// The half of the flag that names a replacement: a drained Prop must leave the
// marker its `consumed: PropInstance` registration promises, or a behavior
// scoped to "Prop" resolves to an empty column.
#[test]
fn a_drained_prop_leaves_the_tag_it_promises() {
    let mut world = World::new();
    world.add_component(Prop {
        scale: [1.0; 3],
        ..Default::default()
    });
    world.start().unwrap();

    let surviving = ComponentTag::Prop
        .surviving_tag()
        .expect("Prop registers a surviving tag");
    assert_ne!(surviving, ComponentTag::Prop);
    let held = world
        .component_census()
        .iter()
        .find(|(t, _)| *t == surviving as u8)
        .map(|(_, n)| *n)
        .unwrap_or(0);
    assert_eq!(held, 1, "{} holds the drained prop", surviving.as_str());
}
