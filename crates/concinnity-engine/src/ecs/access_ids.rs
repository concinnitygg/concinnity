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
// The validator half installs the core `access_check` hooks: `World::step`
// announces the stepping system through one, and every context accessor's touch
// is asserted against that system's declared access through the other.
// Exclusive systems pass everything; structural change and blob access require
// exclusivity. Debug builds only.

use concinnity_core::components::AudioCommand;
use concinnity_core::components::ControlsCommand;
use concinnity_core::components::FrameInput;
use concinnity_core::components::InteractEvent;
use concinnity_core::components::PlayCue;
use concinnity_core::components::RootMotionEvent;
use concinnity_core::components::SceneCommand;
use concinnity_core::components::ScreenCommand;
use concinnity_core::components::ScreenShown;
use concinnity_core::components::SettingCommand;
use concinnity_core::components::StoryCommand;
use concinnity_core::components::StoryReload;
use concinnity_core::ecs::{
    Access, ComponentId, CursorState, DesiredCursor, EventStore, FlyCam, HudLayers, HudPrefs,
    MenuActive, MenuOverride, OpenDropdown, ScheduleMode, ScreenStack, SimTiming,
};
use std::any::TypeId;
use std::sync::OnceLock;

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
        pub(crate) fn ensure_event_queues(store: &mut EventStore, access: Access) {
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
        FrameInput,
        MenuActive,
        SimTiming,
        MenuOverride,
        DesiredCursor,
        HudLayers,
        ScreenStack,
        FlyCam,
        CursorState,
        HudPrefs,
        OpenDropdown,
        crate::ecs::DisabledSettingRows,
        crate::ecs::DisplayModes,
        crate::ecs::InputMailbox,
        ScheduleMode,
        crate::ecs::ActiveSceneFlow,
        crate::ecs::SceneResidencyStatus,
        concinnity_core::ecs::EntityByName,
        crate::app::budget::MemoryBudget,
        crate::app::budget::ThreadBudget,
        crate::gfx::overlay::OverlayFrame,
        crate::gfx::overlay::OverlayAssets,
        crate::gfx::overlay::OverlayRecycle,
    ],
    events: [
        ControlsCommand,
        InteractEvent,
        RootMotionEvent,
        ScreenCommand,
        ScreenShown,
        SettingCommand,
        SceneCommand,
        StoryCommand,
        StoryReload,
        PlayCue,
        AudioCommand,
    ],
}

// The schedule id of a registered resource or event type.
pub(crate) fn id_of<T: 'static>() -> Option<ComponentId> {
    resolve(TypeId::of::<T>())
}

fn resolve(type_id: TypeId) -> Option<ComponentId> {
    table()
        .iter()
        .find(|(t, _, _)| *t == type_id)
        .map(|&(_, _, id)| ComponentId::new(id))
}

// Build a resource mask from types in the access-id registry. Panics on an
// unregistered type: called once at schedule build, so a typo fails loudly at
// world start, not silently at runtime.
macro_rules! resource_mask {
    ( $( $ty:ty ),* $(,)? ) => {{
        let mut m = ::concinnity_core::ecs::ComponentMask::EMPTY;
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

pub(crate) use concinnity_core::component_mask;
pub(crate) use resource_mask;

#[cfg(debug_assertions)]
mod validate {
    use super::{resolve, table};
    use concinnity_core::ecs::Access;
    use concinnity_core::ecs::ComponentId;
    use concinnity_core::ecs::access_check::{self, Touch};

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
                    table().len(),
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

    use concinnity_core::components::TextLabel;
    use concinnity_core::ecs::Access;

    #[test]
    fn registered_types_resolve_to_distinct_ids() {
        let a = id_of::<MenuActive>().unwrap();
        let b = id_of::<ScreenCommand>().unwrap();
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
        let access = Access::new().writes_resources(resource_mask![ScreenCommand]);
        ensure_event_queues(&mut store, access);
        assert!(store.get::<ScreenCommand>().is_some());
        assert!(store.get::<PlayCue>().is_none());

        // Exclusive systems keep lazy creation.
        let mut lazy = EventStore::new();
        ensure_event_queues(&mut lazy, Access::new().exclusive());
        assert!(lazy.get::<ScreenCommand>().is_none());
    }

    #[cfg(debug_assertions)]
    mod hook {
        use super::super::*;
        use concinnity_core::components::Sprite;
        use concinnity_core::components::TextLabel;
        use concinnity_core::ecs::Arena;
        use concinnity_core::ecs::access_check::set_active;
        use concinnity_core::ecs::{
            Access, ComponentStorage, FrameContext, PipelineContext, Resources,
        };
        use concinnity_core::gfx::profile::FrameProfile;
        use concinnity_host::store::blob::BlobData;

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
                Access::new().writes_components(component_mask![TextLabel]),
                "TestSystem",
            )));
            let mut p = parts();
            let _ = ctx(&mut p).query::<Sprite>().count();
        }

        #[test]
        #[should_panic(expected = "structural change")]
        fn undeclared_structural_change_panics() {
            install_hook();
            set_active(Some((Access::new(), "TestSystem")));
            let mut p = parts();
            ctx(&mut p).push(TextLabel::default());
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
                    .writes_components(component_mask![TextLabel])
                    .reads_resources(resource_mask![MenuActive]),
                "TestSystem",
            )));
            {
                let mut c = ctx(&mut p);
                let _ = c.query::<TextLabel>().count();
                let _ = c.query_mut::<TextLabel>().count();
                let _ = c.resource::<MenuActive>();
            }
            set_active(Some((Access::new().exclusive(), "TestSystem")));
            {
                let mut c = ctx(&mut p);
                c.push(TextLabel::default());
                let _ = c.query::<Sprite>().count();
            }
            set_active(None);
        }
    }

    #[test]
    fn masks_build_from_registered_types() {
        let m = resource_mask![MenuActive, ScreenStack];
        assert!(m.contains(id_of::<MenuActive>().unwrap()));
        assert!(!m.contains(id_of::<FlyCam>().unwrap()));

        let c = component_mask![TextLabel];
        assert!(!c.is_empty());
    }
}
