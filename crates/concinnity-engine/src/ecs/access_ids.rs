// src/ecs/access_ids.rs
//
// The schedule's resource-id registry and the debug-build access validator.
//
// Component access ids are the component registry's discriminants; resources
// and event types have no such number, so this module assigns them one (by
// list position) in the `Access` resource-id space. A type absent from the
// list cannot be declared: a system touching it must stay exclusive, or the
// type gets registered here. Event send is a resource write of the event
// type's id, event read a resource read, which is what lets the schedule
// serialize same-queue systems in table order and keep event order
// bit-identical to serial.
//
// The validator half installs the core `access_check` hook: while a stepping
// system's declared access is active (set by `World::step`), every context
// accessor's touch is asserted against it. Exclusive systems pass everything;
// structural change and blob access require exclusivity. Debug builds only.

use std::any::TypeId;
use std::sync::OnceLock;

use crate::ecs::{Access, ComponentId, EventStore};

macro_rules! define_access_ids {
    (
        resources: [ $( $res:path, )* ],
        events: [ $( $ev:path, )* ] $(,)?
    ) => {
        fn table() -> &'static [(TypeId, &'static str, u8)] {
            static TABLE: OnceLock<Vec<(TypeId, &'static str, u8)>> = OnceLock::new();
            TABLE.get_or_init(|| {
                let types = [
                    $( (TypeId::of::<$res>(), stringify!($res)), )*
                    $( (TypeId::of::<$ev>(), stringify!($ev)), )*
                ];
                assert!(types.len() <= 128, "access-id registry exceeds the 128-bit mask");
                types
                    .into_iter()
                    .enumerate()
                    .map(|(id, (type_id, name))| (type_id, name, id as u8))
                    .collect()
            })
        }

        // Ensure every event queue a declared access can touch exists, so a
        // validated system's `events_mut` never grows the store's map
        // mid-tick. Exclusive systems keep today's lazy creation.
        pub fn ensure_event_queues(store: &mut EventStore, access: Access) {
            if access.is_exclusive() {
                return;
            }
            $(
                if let Some(id) = id_of::<$ev>() {
                    if access.may_write_resource(id) || access.may_read_resource(id) {
                        store.get_mut_or_create::<$ev>();
                    }
                }
            )*
        }
    };
}

define_access_ids! {
    resources: [
        crate::assets::FrameInput,
        crate::ecs::MenuActive,
        crate::ecs::SimTiming,
        crate::ecs::MenuOverride,
        crate::ecs::DesiredCursor,
        crate::ecs::HudLayers,
        crate::ecs::ScreenStack,
        crate::ecs::FlyCam,
        crate::ecs::CursorState,
        crate::ecs::HudPrefs,
        crate::ecs::OpenDropdown,
        crate::ecs::DisabledSettingRows,
        crate::ecs::DisplayModes,
        crate::ecs::InputMailbox,
        crate::ecs::ScheduleMode,
        crate::ecs::ActiveSceneFlow,
        crate::ecs::SceneResidencyStatus,
        crate::ecs::decompose::EntityByName,
        crate::app::budget::MemoryBudget,
        crate::app::budget::ThreadBudget,
        crate::gfx::overlay::OverlayFrame,
        crate::gfx::overlay::OverlayAssets,
        crate::gfx::overlay::OverlayRecycle,
    ],
    events: [
        crate::assets::ControlsCommand,
        crate::assets::InteractSignal,
        crate::assets::RootMotion,
        crate::assets::ScreenCommand,
        crate::assets::ScreenShown,
        crate::assets::SettingCommand,
        crate::assets::SceneCommand,
        crate::assets::StoryCommand,
        crate::assets::StoryReload,
        crate::assets::PlayCue,
        crate::assets::AudioCommand,
    ],
}

// The schedule id of a registered resource or event type.
pub fn id_of<T: 'static>() -> Option<ComponentId> {
    resolve(TypeId::of::<T>())
}

fn resolve(type_id: TypeId) -> Option<ComponentId> {
    table()
        .iter()
        .find(|(t, _, _)| *t == type_id)
        .map(|&(_, _, id)| ComponentId::new(id))
}

// Build a component mask from registered component types.
#[macro_export]
macro_rules! component_mask {
    ( $( $ty:ty ),* $(,)? ) => {{
        #[allow(unused_mut)]
        let mut m = $crate::ecs::ComponentMask::EMPTY;
        $( m.insert($crate::ecs::ComponentId::new(
            <$ty as $crate::ecs::ComponentSlot>::DISCRIMINANT,
        )); )*
        m
    }};
}

// Build a resource mask from types in the access-id registry. Panics on an
// unregistered type: called once at schedule build, so a typo fails loudly at
// world start, not silently at runtime.
#[macro_export]
macro_rules! resource_mask {
    ( $( $ty:ty ),* $(,)? ) => {{
        #[allow(unused_mut)]
        let mut m = $crate::ecs::ComponentMask::EMPTY;
        $( m.insert(
            $crate::ecs::access_ids::id_of::<$ty>()
                .unwrap_or_else(|| panic!(
                    "not in the access-id registry: {}",
                    stringify!($ty),
                )),
        ); )*
        m
    }};
}

#[cfg(debug_assertions)]
mod validate {
    use super::{resolve, table};
    use crate::ecs::Access;
    use concinnity_core::ecs::access_check::{self, Touch};

    std::thread_local! {
        static ACTIVE: std::cell::Cell<Option<(Access, &'static str)>> =
            const { std::cell::Cell::new(None) };
    }

    // Mark the given system's declared access active on this thread for the
    // duration of its step. `None` clears it (init, decompose, and editor
    // drives run unvalidated).
    pub fn set_active(active: Option<(Access, &'static str)>) {
        ACTIVE.with(|a| a.set(active));
    }

    pub fn install_hook() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| access_check::install(check));
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
                access.may_read_component(crate::ecs::ComponentId::new(*id)),
                "{system} reads {type_name} without declaring it",
            ),
            Touch::ComponentWrite { id, type_name } => assert!(
                access.may_write_component(crate::ecs::ComponentId::new(*id)),
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
                    table().len(),
                ),
            },
        }
    }
}

#[cfg(debug_assertions)]
pub use validate::{install_hook, set_active};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::Access;

    #[test]
    fn registered_types_resolve_to_distinct_ids() {
        let a = id_of::<crate::ecs::MenuActive>().unwrap();
        let b = id_of::<crate::assets::ScreenCommand>().unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn unregistered_types_do_not_resolve() {
        struct NotRegistered;
        assert!(id_of::<NotRegistered>().is_none());
    }

    #[test]
    fn ensure_event_queues_creates_declared_queues_only() {
        let mut store = EventStore::new();
        let access = Access::new().writes_resources(resource_mask![crate::assets::ScreenCommand]);
        ensure_event_queues(&mut store, access);
        assert!(store.get::<crate::assets::ScreenCommand>().is_some());
        assert!(store.get::<crate::assets::PlayCue>().is_none());

        // Exclusive systems keep lazy creation.
        let mut lazy = EventStore::new();
        ensure_event_queues(&mut lazy, Access::new().exclusive());
        assert!(lazy.get::<crate::assets::ScreenCommand>().is_none());
    }

    #[cfg(debug_assertions)]
    mod hook {
        use super::super::*;
        use crate::blob::BlobData;
        use crate::ecs::{Access, ComponentStorage, FrameContext, PipelineContext, Resources};
        use crate::gfx::profile::FrameProfile;
        use concinnity_core::ecs::Arena;

        struct Parts {
            components: ComponentStorage,
            blob: BlobData,
            profile: FrameProfile,
            resources: Resources,
            scratch: Arena,
        }

        fn parts() -> Parts {
            Parts {
                components: ComponentStorage::default(),
                blob: BlobData::empty(),
                profile: FrameProfile::default(),
                resources: Resources::default(),
                scratch: Arena::with_capacity(0),
            }
        }

        fn ctx(p: &mut Parts) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut p.components,
                blob: &mut p.blob,
                profile: &mut p.profile,
                resources: &mut p.resources,
                frame: FrameContext::new(&p.scratch),
            }
        }

        #[test]
        #[should_panic(expected = "Sprite without declaring it")]
        fn undeclared_component_read_panics() {
            install_hook();
            set_active(Some((
                Access::new().writes_components(component_mask![crate::assets::TextLabel]),
                "TestSystem",
            )));
            let mut p = parts();
            let _ = ctx(&mut p).query::<crate::assets::Sprite>().count();
        }

        #[test]
        #[should_panic(expected = "structural change")]
        fn undeclared_structural_change_panics() {
            install_hook();
            set_active(Some((Access::new(), "TestSystem")));
            let mut p = parts();
            ctx(&mut p).push(crate::assets::TextLabel::default());
        }

        #[test]
        #[should_panic(expected = "not in the access-id registry")]
        fn unregistered_resource_touch_panics() {
            install_hook();
            set_active(Some((Access::new(), "TestSystem")));
            struct Unregistered;
            let mut p = parts();
            let _ = ctx(&mut p).resource::<Unregistered>();
        }

        #[test]
        fn declared_touches_pass_and_exclusive_passes_everything() {
            install_hook();
            let mut p = parts();
            set_active(Some((
                Access::new()
                    .writes_components(component_mask![crate::assets::TextLabel])
                    .reads_resources(resource_mask![crate::ecs::MenuActive]),
                "TestSystem",
            )));
            {
                let mut c = ctx(&mut p);
                let _ = c.query::<crate::assets::TextLabel>().count();
                let _ = c.query_mut::<crate::assets::TextLabel>().count();
                let _ = c.resource::<crate::ecs::MenuActive>();
            }
            set_active(Some((Access::new().exclusive(), "TestSystem")));
            {
                let mut c = ctx(&mut p);
                c.push(crate::assets::TextLabel::default());
                let _ = c.query::<crate::assets::Sprite>().count();
            }
            set_active(None);
        }
    }

    #[test]
    fn masks_build_from_registered_types() {
        let m = resource_mask![crate::ecs::MenuActive, crate::ecs::ScreenStack];
        assert!(m.contains(id_of::<crate::ecs::MenuActive>().unwrap()));
        assert!(!m.contains(id_of::<crate::ecs::FlyCam>().unwrap()));

        let c = component_mask![crate::assets::TextLabel];
        assert!(!c.is_empty());
    }
}
