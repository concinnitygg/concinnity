//! The engine's access-id registry and the debug-build access validator.
//!
//! Resource and event types this crate defines get their access ids here,
//! after core's (`concinnity_core::ecs::access_id`); a type absent from both
//! lists cannot be declared, so a system touching it stays exclusive.
//!
//! The validator half installs the core `access_check` hooks: `World::step`
//! announces the stepping system through one, and every context accessor's touch
//! is asserted against that system's declared access through the other.
//! Exclusive systems pass everything; structural change and blob access require
//! exclusivity. Debug builds only.

use concinnity_core::ecs::Access;
use concinnity_core::ecs::EventStore;
use concinnity_core::ecs::access_id::{self, ACCESS_ID_COUNT, ACCESS_IDS, AccessEntry};

concinnity_core::define_access_ids! {
    static ENGINE_ACCESS_IDS;
    const ENGINE_ACCESS_END;
    base: ACCESS_ID_COUNT,
    resources: [
        crate::ecs::InputMailbox,
    ],
    events: [],
}

const TABLES: [&[AccessEntry]; 2] = [ACCESS_IDS, ENGINE_ACCESS_IDS];

// Ensure every event queue a declared access can touch exists before the
// first tick.
pub(crate) fn ensure_event_queues(store: &mut EventStore, access: Access) {
    access_id::ensure_event_queues(&TABLES, store, access);
}

#[cfg(debug_assertions)]
mod validate {
    use super::{TABLES, access_id};
    use concinnity_core::ecs::Access;
    use concinnity_core::ecs::ComponentId;
    use concinnity_core::ecs::access_check::{self, Touch};

    // The schedule id of a registered resource or event type, from its `TypeId`.
    fn resolve(type_id: std::any::TypeId) -> Option<ComponentId> {
        access_id::resolve(&TABLES, type_id)
    }

    std::thread_local! {
        static ACTIVE: std::cell::Cell<Option<(Access, &'static str)>> =
            const { std::cell::Cell::new(None) };
    }

    // Mark the given system's declared access active on this thread for the
    // duration of its step. `None` clears it (init, decompose, and editor
    // drives run unvalidated).
    fn set_active(active: Option<(Access, &'static str)>) {
        ACTIVE.with(|a| a.set(active));
    }

    pub(crate) fn install_hook() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            access_check::install(check);
            access_check::install_active(set_active);
        });
    }

    fn check(touch: &Touch) {
        let Some((access, system)) = ACTIVE.with(|a| a.get()) else {
            return;
        };
        if access.is_exclusive() {
            return;
        }
        match touch {
            Touch::ComponentRead { id, type_name } => assert!(
                access.may_read_component(ComponentId::new(*id)),
                "{system} reads {type_name} without declaring it",
            ),
            Touch::ComponentWrite { id, type_name } => assert!(
                access.may_write_component(ComponentId::new(*id)),
                "{system} writes {type_name} without declaring it",
            ),
            Touch::Structural { op } => {
                panic!("{system} performs structural change ({op}) without exclusive access")
            }
            Touch::Blob { op } => {
                panic!("{system} touches the blob store ({op}) without exclusive access")
            }
            Touch::Resource {
                type_id,
                type_name,
                write,
            } => match resolve(*type_id) {
                Some(id) if *write => assert!(
                    access.may_write_resource(id),
                    "{system} writes resource {type_name} without declaring it",
                ),
                Some(id) => assert!(
                    access.may_read_resource(id),
                    "{system} reads resource {type_name} without declaring it",
                ),
                None => panic!(
                    "{system} touches resource {type_name}, which is not in the \
                     access-id registry; register it or keep the system exclusive \
                     (registry holds {} entries)",
                    access_id::registered(&TABLES),
                ),
            },
        }
    }
}

#[cfg(debug_assertions)]
pub(crate) use validate::install_hook;

#[cfg(test)]
mod tests {
    use super::*;

    use concinnity_core::components::{ScreenCommand, TextLabel};
    use concinnity_core::ecs::access_id::AccessId;
    use concinnity_core::ecs::{FlyCam, MenuActive, ScreenStack};
    use concinnity_core::{component_mask, resource_mask};
    use std::any::TypeId;

    use crate::ecs::InputMailbox;

    #[test]
    fn engine_types_resolve_after_core_ones() {
        let mailbox = access_id::resolve(&TABLES, TypeId::of::<InputMailbox>()).unwrap();
        assert_eq!(mailbox, InputMailbox::ID);
        assert!(mailbox.get() >= ACCESS_ID_COUNT);
        assert_eq!(
            access_id::resolve(&TABLES, TypeId::of::<ScreenCommand>()),
            Some(ScreenCommand::ID)
        );
        struct NotRegistered;
        assert!(access_id::resolve(&TABLES, TypeId::of::<NotRegistered>()).is_none());
    }

    #[cfg(debug_assertions)]
    mod hook {
        use super::super::*;
        use concinnity_core::components::Sprite;
        use concinnity_core::components::TextLabel;
        use concinnity_core::ecs::access_check::set_active;
        use concinnity_core::ecs::{Access, MenuActive, World};
        use concinnity_core::{component_mask, resource_mask};

        #[test]
        #[should_panic(expected = "Sprite without declaring it")]
        fn undeclared_component_read_panics() {
            install_hook();
            set_active(Some((
                Access::new().writes_components(component_mask![TextLabel]),
                "TestSystem",
            )));
            let mut world = World::new();
            let _ = world.context().query::<Sprite>().count();
        }

        #[test]
        #[should_panic(expected = "structural change")]
        fn undeclared_structural_change_panics() {
            install_hook();
            set_active(Some((Access::new(), "TestSystem")));
            let mut world = World::new();
            world.context().push(TextLabel::default());
        }

        #[test]
        #[should_panic(expected = "not in the access-id registry")]
        fn unregistered_resource_touch_panics() {
            install_hook();
            set_active(Some((Access::new(), "TestSystem")));
            struct Unregistered;
            let mut world = World::new();
            let _ = world.context().resource::<Unregistered>();
        }

        #[test]
        fn declared_touches_pass_and_exclusive_passes_everything() {
            install_hook();
            let mut world = World::new();
            set_active(Some((
                Access::new()
                    .writes_components(component_mask![TextLabel])
                    .reads_resources(resource_mask![MenuActive]),
                "TestSystem",
            )));
            {
                let mut c = world.context();
                let _ = c.query::<TextLabel>().count();
                let _ = c.query_mut::<TextLabel>().count();
                let _ = c.resource::<MenuActive>();
            }
            set_active(Some((Access::new().exclusive(), "TestSystem")));
            {
                let mut c = world.context();
                c.push(TextLabel::default());
                let _ = c.query::<Sprite>().count();
            }
            set_active(None);
        }
    }

    #[test]
    fn masks_build_from_registered_types() {
        let m = resource_mask![MenuActive, ScreenStack, InputMailbox];
        assert!(m.contains(InputMailbox::ID));
        assert!(!m.contains(FlyCam::ID));

        let c = component_mask![TextLabel];
        assert!(!c.is_empty());
    }
}
