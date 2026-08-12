// concinnity-core/src/ecs/access_check.rs
//
// Debug-build access validation hook. `PipelineContext` accessors report each
// touch here; the client installs a hook that asserts the touch against the
// stepping system's declared `Access` (tracked client-side, since this crate
// is no_std and holds no per-thread state). Compiled out of release builds
// entirely: release parallel-safety never rests on these checks, only on the
// executor handing conflicting systems to different waves.
#![cfg(debug_assertions)]

use core::any::TypeId;
use core::sync::atomic::{AtomicPtr, Ordering};

// What a context accessor touched. Component ids are the registry
// discriminants; resources and events report their `TypeId` for the client's
// id registry to resolve.
pub enum Touch {
    ComponentRead {
        id: u8,
        type_name: &'static str,
    },
    ComponentWrite {
        id: u8,
        type_name: &'static str,
    },
    // Structural change (push/insert/remove/despawn/drain): reorders columns
    // and the join index, so it requires exclusive access.
    Structural {
        op: &'static str,
    },
    Resource {
        type_id: TypeId,
        type_name: &'static str,
        write: bool,
    },
    // Compiled-payload access mutates the blob store's residency.
    Blob {
        op: &'static str,
    },
}

type Hook = fn(&Touch);

static HOOK: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

// Install the process-wide validation hook. Idempotent by usage (the client
// installs one hook once); the last install wins.
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
