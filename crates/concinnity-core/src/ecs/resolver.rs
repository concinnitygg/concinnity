//! Name -> id resolution seam.
//!
//! A reference deserializes either from an already-resolved integer id (the
//! compiled-args / runtime form) or from a name string (the authoring form).
//! Turning a name into a dense id is engine policy -- the build assigns ids in
//! world declaration order -- so this data crate does not own it.
//! concinnity-host installs a resolver here, backed by its build-time interner,
//! before it deserializes named references. A name seen with no resolver
//! installed is a configuration error, surfaced as a deserialization failure
//! (the resolver is always installed during a build; only an out-of-engine tool
//! reading authoring JSON would hit the unset case).
//!
//! Each resolver is a plain function pointer held in an atomic, so this stays
//! `no_std` and thread-safe: the pointer is written once (install) and only read
//! afterward, and the installed function keeps its own (per-thread) state in
//! concinnity-host. The two slot types below centralize the single unavoidable
//! piece of unsafe -- `core` has no atomic function-pointer type, so reading a
//! `fn` back out of a `usize` requires a `transmute` -- into one audited place
//! per function-pointer shape.

use core::sync::atomic::{AtomicUsize, Ordering};

use crate::ecs::handle::HandleKind;

/// A name -> dense id resolver.
pub(crate) type ResolveFn = fn(&str) -> u32;

/// A name -> resource-handle resolver, given the handle space to resolve in.
/// Returns the resource's dense handle, or `None` when the name is not a known
/// resource of that space in the current build (or no build map is installed).
/// Unlike the name interner a handle is not assignable on demand: it is a
/// position in the build's declaration-ordered resource table, so a name with
/// no matching resource has no handle.
pub(crate) type HandleResolveFn = fn(HandleKind, &str) -> Option<u32>;

// An atomically-installable `ResolveFn` slot. Holds the function pointer as a
// `usize` (0 = unset): written once at install, only read afterward.
struct NameResolverSlot(AtomicUsize);

impl NameResolverSlot {
    const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    fn set(&self, f: ResolveFn) {
        self.0.store(f as usize, Ordering::Release);
    }

    fn resolve(&self, name: &str) -> Option<u32> {
        let v = self.0.load(Ordering::Acquire);
        if v == 0 {
            return None;
        }
        // SAFETY: `v` is non-zero here, so it is a `ResolveFn` address stored by
        // `set`; the transmute reverses that exact `fn as usize`.
        let f: ResolveFn = unsafe { core::mem::transmute::<usize, ResolveFn>(v) };
        Some(f(name))
    }
}

// An atomically-installable `HandleResolveFn` slot. Same install-once /
// read-many discipline as `NameResolverSlot`; one instance backs each handle
// space.
struct HandleResolverSlot(AtomicUsize);

impl HandleResolverSlot {
    const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    fn set(&self, f: HandleResolveFn) {
        self.0.store(f as usize, Ordering::Release);
    }

    fn resolve(&self, kind: HandleKind, name: &str) -> Option<u32> {
        let v = self.0.load(Ordering::Acquire);
        if v == 0 {
            return None;
        }
        // SAFETY: `v` is non-zero here, so it is a `HandleResolveFn` address
        // stored by `set`; the transmute reverses that exact `fn as usize`.
        let f: HandleResolveFn = unsafe { core::mem::transmute::<usize, HandleResolveFn>(v) };
        f(kind, name)
    }
}

#[cfg(not(test))]
static RESOLVER: NameResolverSlot = NameResolverSlot::new();

/// Install the name -> id resolver. Called once by concinnity-host, backed by
/// its build-time interner. Idempotent; the last writer wins.
#[cfg(not(test))]
pub fn set_name_resolver(f: ResolveFn) {
    RESOLVER.set(f);
}

/// Resolve a name to a dense id via the installed resolver, or `None` if none is
/// installed (only expected outside a build).
#[cfg(not(test))]
pub(crate) fn resolve_name(name: &str) -> Option<u32> {
    RESOLVER.resolve(name)
}

// Under test the slot is per-thread rather than process-wide. The harness runs
// tests in parallel and the crate carries two stand-in policies -- a
// declaration-order interner and a name-length map -- so a shared pointer lets
// whichever installed last answer the other's tests.
#[cfg(test)]
std::thread_local! {
    static RESOLVER: core::cell::Cell<Option<ResolveFn>> =
        const { core::cell::Cell::new(None) };
}

/// Install the name -> id resolver.
#[cfg(test)]
pub fn set_name_resolver(f: ResolveFn) {
    RESOLVER.with(|slot| slot.set(Some(f)));
}

/// Resolve a name to a dense id via the installed resolver.
#[cfg(test)]
pub(crate) fn resolve_name(name: &str) -> Option<u32> {
    RESOLVER.with(|slot| slot.get()).map(|f| f(name))
}

// One handle resolver slot per handle space, indexed by `HandleKind`.
#[cfg(not(test))]
static HANDLE_RESOLVERS: [HandleResolverSlot; HandleKind::COUNT] =
    [const { HandleResolverSlot::new() }; HandleKind::COUNT];

// Per-thread under test, for the reason given on the name slot above: some
// tests install a stand-in and others pin what happens with none installed,
// which a process-wide slot cannot serve at the same time.
#[cfg(test)]
std::thread_local! {
    static HANDLE_RESOLVERS: core::cell::Cell<[Option<HandleResolveFn>; HandleKind::COUNT]> =
        const { core::cell::Cell::new([None; HandleKind::COUNT]) };
}

/// Install the name -> handle resolver for one handle space. Called by
/// concinnity-cook, backed by the current build's declaration-ordered handle
/// map. Idempotent; the last writer wins.
#[cfg(not(test))]
pub fn set_handle_resolver(kind: HandleKind, f: HandleResolveFn) {
    if let Some(slot) = HANDLE_RESOLVERS.get(kind.index()) {
        slot.set(f);
    }
}

/// Install the name -> handle resolver for one handle space.
#[cfg(test)]
pub fn set_handle_resolver(kind: HandleKind, f: HandleResolveFn) {
    HANDLE_RESOLVERS.with(|slots| {
        let mut table = slots.get();
        if let Some(slot) = table.get_mut(kind.index()) {
            *slot = Some(f);
        }
        slots.set(table);
    });
}

/// Resolve a reference name to its dense handle in `kind`'s space via the
/// installed resolver. `None` means either no resolver is installed or the
/// name is not a declared resource of that space; the caller decides whether
/// to fall back (a validation context) or to fail (a real build).
#[cfg(not(test))]
pub(crate) fn resolve_handle(kind: HandleKind, name: &str) -> Option<u32> {
    HANDLE_RESOLVERS
        .get(kind.index())
        .and_then(|slot| slot.resolve(kind, name))
}

#[cfg(test)]
pub(crate) fn resolve_handle(kind: HandleKind, name: &str) -> Option<u32> {
    HANDLE_RESOLVERS
        .with(|slots| slots.get().get(kind.index()).copied().flatten())
        .and_then(|f| f(kind, name))
}

#[cfg(test)]
mod tests {
    // These tests own the process-global resolver: each installs the same
    // deterministic stand-in first, so they stay correct regardless of the order
    // the test harness runs them in (installs are idempotent, last-writer-wins).
    use super::*;
    use crate::ecs::asset_id::AssetId;
    use crate::ecs::{Ref, RefTarget};
    use crate::test_support::{install_resolvers, len_handle_resolver, len_name_resolver};

    struct Clip;

    impl RefTarget for Clip {
        const TYPES: &'static [&'static str] = &["AudioClip"];
    }

    #[test]
    fn a_slot_reads_back_the_function_pointer_it_was_given() {
        // The slots hold their function pointer as a `usize` and transmute it
        // back, the one piece of unsafe here. Exercising a fresh slot rather
        // than the process-global statics is the only way to see the unset
        // state, which a test cannot restore once something has installed.
        let name_slot = NameResolverSlot::new();
        assert_eq!(name_slot.resolve("floor"), None);
        name_slot.set(len_name_resolver);
        assert_eq!(name_slot.resolve("floor"), Some(5));

        let handle_slot = HandleResolverSlot::new();
        assert_eq!(handle_slot.resolve(HandleKind::Texture, "floor"), None);
        handle_slot.set(len_handle_resolver);
        assert_eq!(handle_slot.resolve(HandleKind::Texture, "floor"), Some(5));
        // A handle resolver may also answer "no such resource of this kind",
        // which the name interner slot has no way to express.
        assert_eq!(handle_slot.resolve(HandleKind::Texture, "unknown_x"), None);
    }

    #[test]
    fn installed_resolver_is_used() {
        set_name_resolver(len_name_resolver);
        assert_eq!(resolve_name("abcd"), Some(4));
    }

    #[test]
    fn asset_id_resolves_a_name_through_the_seam() {
        set_name_resolver(len_name_resolver);
        let id: AssetId = serde_json::from_str("\"floor\"").unwrap();
        assert_eq!(id, AssetId(5));
    }

    #[test]
    fn a_ref_resolves_a_name_through_the_seam() {
        set_name_resolver(len_name_resolver);
        let r: Ref<Clip> = serde_json::from_str("\"wall\"").unwrap();
        assert_eq!(r.id(), AssetId(4));
    }

    #[test]
    fn every_handle_space_resolves_through_its_own_slot() {
        // One slot per space: a name is a position in that space's declaration-
        // ordered table, so the spaces never share an answer by accident.
        install_resolvers();
        fn textures_only(kind: HandleKind, name: &str) -> Option<u32> {
            (kind == HandleKind::Texture).then_some(name.len() as u32)
        }
        set_handle_resolver(HandleKind::Texture, textures_only);
        set_handle_resolver(HandleKind::Shader, textures_only);
        assert_eq!(resolve_handle(HandleKind::Texture, "floor"), Some(5));
        assert_eq!(resolve_handle(HandleKind::Shader, "floor"), None);

        // A handle is not assignable on demand: a name the build declares no
        // resource of that space for has none, even with a resolver installed.
        for kind in HandleKind::ALL {
            set_handle_resolver(*kind, len_handle_resolver);
            assert_eq!(resolve_handle(*kind, "floor"), Some(5));
            assert_eq!(resolve_handle(*kind, "unknown_x"), None);
        }
    }
}
