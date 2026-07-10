// Name -> id resolution seam.
//
// A reference deserializes either from an already-resolved integer id (the
// compiled-args / runtime form) or from a name string (the authoring form).
// Turning a name into a dense id is engine policy -- the build assigns ids in
// world declaration order -- so this data crate does not own it. concinnity-core
// installs a resolver here, backed by its build-time interner, before it
// deserializes named references. A name seen with no resolver installed is a
// configuration error, surfaced as a deserialization failure (the resolver is
// always installed during a build; only an out-of-engine tool reading authoring
// JSON would hit the unset case).
//
// The resolver is a plain function pointer held in an atomic, so this stays
// `no_std` and thread-safe: the pointer is written once (install) and only read
// afterward, and the installed function keeps its own (per-thread) state in
// concinnity-core.

use core::sync::atomic::{AtomicUsize, Ordering};

/// A name -> dense id resolver.
pub type ResolveFn = fn(&str) -> u32;

// 0 means "no resolver installed". Any other value is a `ResolveFn` address.
static RESOLVER: AtomicUsize = AtomicUsize::new(0);

/// Install the name -> id resolver. Called once by concinnity-core, backed by
/// its build-time interner. Idempotent; the last writer wins.
pub fn set_name_resolver(f: ResolveFn) {
    RESOLVER.store(f as usize, Ordering::Release);
}

/// Resolve a name to a dense id via the installed resolver, or `None` if none is
/// installed (only expected outside a build).
pub(crate) fn resolve_name(name: &str) -> Option<u32> {
    let v = RESOLVER.load(Ordering::Acquire);
    if v == 0 {
        None
    } else {
        // SAFETY: `v` is non-zero here, so it is a `ResolveFn` address stored by
        // `set_name_resolver`; the transmute reverses that exact `fn as usize`.
        let f: ResolveFn = unsafe { core::mem::transmute::<usize, ResolveFn>(v) };
        Some(f(name))
    }
}

#[cfg(test)]
mod tests {
    // These tests own the process-global resolver: each installs the same
    // deterministic stand-in first, so they stay correct regardless of the order
    // the test harness runs them in (installs are idempotent, last-writer-wins).
    use super::*;
    use crate::{AssetId, AssetRef, de_opt_asset_ref, de_opt_asset_ref_typed};

    // A name resolves to its byte length: a simple, order-independent mapping.
    fn len_resolver(name: &str) -> u32 {
        name.len() as u32
    }

    struct Clip;

    #[test]
    fn installed_resolver_is_used() {
        set_name_resolver(len_resolver);
        assert_eq!(resolve_name("abcd"), Some(4));
    }

    #[test]
    fn asset_id_resolves_a_name_through_the_seam() {
        set_name_resolver(len_resolver);
        let id: AssetId = serde_json::from_str("\"floor\"").unwrap();
        assert_eq!(id, AssetId(5));
    }

    #[test]
    fn asset_ref_resolves_a_name_through_the_seam() {
        set_name_resolver(len_resolver);
        let r: AssetRef<Clip> = serde_json::from_str("\"wall\"").unwrap();
        assert_eq!(r.id(), Some(AssetId(4)));
        assert!(r.is_resolved());
    }

    #[test]
    fn opt_helpers_resolve_a_name_and_pass_through_an_id() {
        set_name_resolver(len_resolver);

        #[derive(serde::Deserialize)]
        struct Bare {
            #[serde(default, deserialize_with = "de_opt_asset_ref")]
            r: Option<AssetId>,
        }
        #[derive(serde::Deserialize)]
        struct Typed {
            #[serde(default, deserialize_with = "de_opt_asset_ref_typed")]
            r: Option<AssetRef<Clip>>,
        }

        assert_eq!(
            serde_json::from_str::<Bare>("{\"r\":\"mesh_a\"}")
                .unwrap()
                .r,
            Some(AssetId(6))
        );
        assert_eq!(
            serde_json::from_str::<Typed>("{\"r\":\"abc\"}")
                .unwrap()
                .r
                .unwrap()
                .id(),
            Some(AssetId(3))
        );
    }
}
