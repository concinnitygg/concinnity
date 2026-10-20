//! Access ids for resources and event types, the ids a system's declared
//! [`Access`] names them by.
//!
//! Component access ids are the component registry's discriminants; resources
//! and events have no such number, so a registry assigns each one an id in the
//! `Access` resource-id space, fixed at compile time through [`AccessId`]. A
//! type with no id cannot be declared: a system touching it stays exclusive, or
//! the type gets registered. Event send is a resource write of the event type's
//! id and event read a resource read, which lets the schedule serialize
//! same-queue systems in table order and keep event order bit-identical to
//! serial.
//!
//! This crate registers its own types in [`ACCESS_IDS`], from id 0. A crate
//! registering more types starts at [`ACCESS_ID_COUNT`] with
//! [`define_access_ids!`](crate::define_access_ids) and resolves a runtime
//! `TypeId` across both tables with [`resolve`].

use core::any::TypeId;

use crate::ecs::{Access, ComponentId, EventStore};

/// A resource or event type with an access id, the bit a declared
/// [`Access`] names it by in [`resource_mask!`](crate::resource_mask).
pub trait AccessId: 'static {
    /// The type's id.
    const ID: ComponentId;
}

/// One registered type, for resolving a `TypeId` at runtime.
pub struct AccessEntry {
    /// The type's `TypeId`.
    pub type_id: fn() -> TypeId,
    /// The type's name as registered.
    pub name: &'static str,
    /// The type's access id.
    pub id: ComponentId,
    /// Creates the type's event queue, for an event type.
    pub ensure_queue: Option<fn(&mut EventStore)>,
}

#[doc(hidden)]
pub fn ensure_queue<E: Send + 'static>(store: &mut EventStore) {
    store.get_mut_or_create::<E>();
}

/// The access id registered for `type_id` in any of `tables`.
pub fn resolve(tables: &[&[AccessEntry]], type_id: TypeId) -> Option<ComponentId> {
    tables
        .iter()
        .flat_map(|t| t.iter())
        .find(|e| (e.type_id)() == type_id)
        .map(|e| e.id)
}

/// How many types `tables` register.
pub fn registered(tables: &[&[AccessEntry]]) -> usize {
    tables.iter().map(|t| t.len()).sum()
}

/// Create every event queue `access` may touch, so a validated system's
/// `events_mut` never grows the store's map mid-tick. Exclusive systems keep
/// lazy creation.
pub fn ensure_event_queues(tables: &[&[AccessEntry]], store: &mut EventStore, access: Access) {
    if access.is_exclusive() {
        return;
    }
    for entry in tables.iter().flat_map(|t| t.iter()) {
        if let Some(ensure) = entry.ensure_queue
            && (access.may_write_resource(entry.id) || access.may_read_resource(entry.id))
        {
            ensure(store);
        }
    }
}

/// Create every event queue `access` may touch among this crate's
/// [`ACCESS_IDS`]: the table's `prepare_events` for a host with no ids of its
/// own.
pub fn prepare_events(store: &mut EventStore, access: Access) {
    ensure_event_queues(&[ACCESS_IDS], store, access);
}

/// Register resource and event types for access declaration: each gets an
/// [`AccessId`] by list position from `base`, and the named table lists them
/// for runtime resolution, and the named const is the first id past them. More ids than the 128-bit mask holds is a compile
/// error.
///
/// ```
/// use concinnity_core::define_access_ids;
/// use concinnity_core::ecs::access_id::{self, ACCESS_ID_COUNT, ACCESS_IDS, AccessId};
/// use concinnity_core::resource_mask;
///
/// pub struct Budget;
/// pub struct Ping;
///
/// define_access_ids! {
///     static MY_ACCESS_IDS;
///     const MY_ACCESS_END;
///     base: ACCESS_ID_COUNT,
///     resources: [Budget],
///     events: [Ping],
/// }
///
/// assert_eq!(Budget::ID.get(), ACCESS_ID_COUNT);
/// assert_eq!(MY_ACCESS_END, ACCESS_ID_COUNT + 2);
/// let tables = [ACCESS_IDS, MY_ACCESS_IDS];
/// assert_eq!(access_id::resolve(&tables, core::any::TypeId::of::<Ping>()), Some(Ping::ID));
/// assert!(resource_mask![Budget, Ping].contains(Ping::ID));
/// ```
#[macro_export]
macro_rules! define_access_ids {
    (
        $( #[$tmeta:meta] )* $vis:vis static $table:ident;
        $( #[$cmeta:meta] )* $cvis:vis const $end:ident;
        base: $base:expr,
        resources: [ $( $res:ty ),* $(,)? ],
        events: [ $( $ev:ty ),* $(,)? ] $(,)?
    ) => {
        $crate::define_access_ids!(@ids ($base); $( $res, )* $( $ev, )*);

        $( #[$tmeta] )*
        $vis static $table: &[$crate::ecs::access_id::AccessEntry] = &[
            $( $crate::ecs::access_id::AccessEntry {
                type_id: ::core::any::TypeId::of::<$res>,
                name: stringify!($res),
                id: <$res as $crate::ecs::access_id::AccessId>::ID,
                ensure_queue: None,
            }, )*
            $( $crate::ecs::access_id::AccessEntry {
                type_id: ::core::any::TypeId::of::<$ev>,
                name: stringify!($ev),
                id: <$ev as $crate::ecs::access_id::AccessId>::ID,
                ensure_queue: Some($crate::ecs::access_id::ensure_queue::<$ev>),
            }, )*
        ];

        $( #[$cmeta] )*
        $cvis const $end: u8 = $crate::define_access_ids!(@count ($base); $( $res, )* $( $ev, )*);

        const _: () = assert!(
            ($end as usize) <= $crate::ecs::ComponentId::MAX as usize + 1,
            "the access-id registry exceeds the 128-bit mask",
        );
    };
    (@ids ($n:expr); ) => {};
    (@ids ($n:expr); $ty:ty, $( $rest:ty, )*) => {
        impl $crate::ecs::access_id::AccessId for $ty {
            const ID: $crate::ecs::ComponentId = $crate::ecs::ComponentId::new($n);
        }
        $crate::define_access_ids!(@ids ($n + 1); $( $rest, )*);
    };
    (@count ($n:expr); ) => { $n };
    (@count ($n:expr); $ty:ty, $( $rest:ty, )*) => {
        $crate::define_access_ids!(@count ($n + 1); $( $rest, )*)
    };
}

/// The [`ComponentMask`](crate::ecs::ComponentMask) holding the given
/// resource or event types' [`AccessId`]s, for the [`Access`] a system
/// declares. A type with no access id is a compile error.
///
/// ```
/// use concinnity_core::components::ScreenCommand;
/// use concinnity_core::ecs::{Access, MenuActive};
/// use concinnity_core::resource_mask;
///
/// let access = Access::new()
///     .reads_resources(resource_mask![MenuActive])
///     .writes_resources(resource_mask![ScreenCommand]);
/// ```
#[macro_export]
macro_rules! resource_mask {
    ( $( $ty:ty ),* $(,)? ) => {{
        let mut m = $crate::ecs::ComponentMask::EMPTY;
        $( m.insert(<$ty as $crate::ecs::access_id::AccessId>::ID); )*
        m
    }};
}

crate::define_access_ids! {
    /// The resource and event types this crate defines.
    pub static ACCESS_IDS;
    /// The first access id past [`ACCESS_IDS`], where another registry starts.
    pub const ACCESS_ID_COUNT;
    base: 0,
    resources: [
        crate::components::FrameInput,
        crate::ecs::MenuActive,
        crate::ecs::SimTiming,
        crate::ecs::FrameTime,
        crate::ecs::MenuOverride,
        crate::ecs::DesiredCursor,
        crate::ecs::HudLayers,
        crate::ecs::ScreenStack,
        crate::ecs::FlyCam,
        crate::ecs::CursorState,
        crate::ecs::WindowMinimized,
        crate::ecs::HudPrefs,
        crate::ecs::OpenDropdown,
        crate::ecs::ScheduleMode,
        crate::ecs::EntityById,
        crate::planet::FrameRebases,
        crate::camera::track::CameraTrackStatus,
        crate::ecs::DisabledSettingRows,
        crate::ecs::DisplayModes,
        crate::ecs::InputSampledAt,
        crate::ecs::ScreenshotRequest,
        crate::ecs::FrameCaptureRequest,
        crate::render::scene::flow::ActiveSceneFlow,
        crate::render::streaming::scene_residency::SceneResidencyStatus,
        crate::budget::MemoryBudget,
        crate::budget::ThreadBudget,
        crate::budget::ResidentMemory,
        crate::platform::CompiledPlatforms,
        crate::platform::GraphicsApiStatus,
        crate::settings::StoredSettings,
        crate::settings::SettingsRevision,
        crate::render::overlay::OverlayFrame,
        crate::render::overlay::OverlayAssets,
        crate::render::overlay::OverlayRecycle,
        crate::input::mailbox::InputMailbox,
    ],
    events: [
        crate::components::ControlsCommand,
        crate::components::InteractEvent,
        crate::components::RootMotionEvent,
        crate::components::ScreenCommand,
        crate::components::ScreenShown,
        crate::components::SettingCommand,
        crate::components::SceneCommand,
        crate::components::StoryCommand,
        crate::components::StoryReload,
        crate::components::PlayCue,
        crate::components::AudioCommand,
    ],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{PlayCue, ScreenCommand};
    use crate::ecs::{FlyCam, MenuActive, ScreenStack};

    struct Extra;
    struct ExtraEvent;

    crate::define_access_ids! {
        static EXTRA_IDS;
        const EXTRA_END;
        base: ACCESS_ID_COUNT,
        resources: [Extra],
        events: [ExtraEvent],
    }

    const TABLES: [&[AccessEntry]; 2] = [ACCESS_IDS, EXTRA_IDS];

    #[test]
    fn ids_follow_list_order_and_chain_across_registries() {
        let ids: alloc::vec::Vec<u8> = ACCESS_IDS.iter().map(|e| e.id.get()).collect();
        let expected: alloc::vec::Vec<u8> = (0..ACCESS_ID_COUNT).collect();
        assert_eq!(ids, expected);
        assert_eq!(Extra::ID.get(), ACCESS_ID_COUNT);
        assert_eq!(ExtraEvent::ID.get(), ACCESS_ID_COUNT + 1);
        assert_eq!(EXTRA_END, ACCESS_ID_COUNT + 2);
        assert_eq!(registered(&TABLES), EXTRA_END as usize);
    }

    #[test]
    fn resolve_finds_registered_types_only() {
        assert_eq!(
            resolve(&TABLES, TypeId::of::<MenuActive>()),
            Some(MenuActive::ID)
        );
        assert_eq!(resolve(&TABLES, TypeId::of::<Extra>()), Some(Extra::ID));
        struct NotRegistered;
        assert_eq!(resolve(&TABLES, TypeId::of::<NotRegistered>()), None);
    }

    #[test]
    fn masks_hold_the_named_ids() {
        let m = crate::resource_mask![MenuActive, ScreenStack];
        assert!(m.contains(MenuActive::ID));
        assert!(m.contains(ScreenStack::ID));
        assert!(!m.contains(FlyCam::ID));
        assert_ne!(MenuActive::ID, ScreenCommand::ID);
    }

    #[test]
    fn ensure_event_queues_creates_declared_queues_only() {
        let mut store = EventStore::new();
        let access =
            Access::new().writes_resources(crate::resource_mask![ScreenCommand, ExtraEvent]);
        ensure_event_queues(&TABLES, &mut store, access);
        assert!(store.get::<ScreenCommand>().is_some());
        assert!(store.get::<ExtraEvent>().is_some());
        assert!(store.get::<PlayCue>().is_none());

        let mut lazy = EventStore::new();
        ensure_event_queues(&TABLES, &mut lazy, Access::new().exclusive());
        assert!(lazy.get::<ScreenCommand>().is_none());
    }
}
