//! Debug-build access validation hook. `PipelineContext` accessors report each
//! touch here; the client installs a hook that asserts the touch against the
//! stepping system's declared [`Access`](crate::ecs::Access) (tracked
//! client-side, since this crate is no_std and holds no per-thread state).
//! Compiled out of release builds entirely: release parallel-safety never rests
//! on these checks, only on the executor handing conflicting systems to
//! different waves.
#![cfg(debug_assertions)]

use core::any::TypeId;
use core::sync::atomic::{AtomicPtr, Ordering};

/// What a context accessor touched. Component ids are the registry
/// discriminants; resources and events report their `TypeId` for the client's
/// id registry to resolve.
pub enum Touch {
    /// A component column was read.
    ComponentRead {
        /// The component's registry discriminant.
        id: u8,
        /// The component type's name, for the assertion message.
        type_name: &'static str,
    },
    /// A component column was written.
    ComponentWrite {
        /// The component's registry discriminant.
        id: u8,
        /// The component type's name, for the assertion message.
        type_name: &'static str,
    },
    /// Structural change (push/insert/remove/despawn/drain): reorders columns
    /// and the join index, so it requires exclusive access.
    Structural {
        /// The operation that changed structure.
        op: &'static str,
    },
    /// A resource or event queue was touched.
    Resource {
        /// The resource type's id, resolved by the client's id registry.
        type_id: TypeId,
        /// The resource type's name, for the assertion message.
        type_name: &'static str,
        /// `true` for a write, `false` for a read.
        write: bool,
    },
    /// Compiled-payload access mutates the blob store's residency.
    Blob {
        /// The operation that touched the blob store.
        op: &'static str,
    },
}

type Hook = fn(&Touch);

static HOOK: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// Install the process-wide validation hook. Idempotent by usage (the client
/// installs one hook once); the last install wins.
pub fn install(hook: Hook) {
    HOOK.store(hook as *mut (), Ordering::Release);
}

#[inline]
pub(crate) fn touch(t: Touch) {
    let raw = HOOK.load(Ordering::Acquire);
    if raw.is_null() {
        return;
    }
    // SAFETY: the pointer only ever holds a `Hook` stored by `install`.
    let hook: Hook = unsafe { core::mem::transmute::<*mut (), Hook>(raw) };
    hook(&t);
}
