//! Name -> id resolution seam.
//!
//! A reference deserializes either from an already-resolved integer id (the
//! compiled-args / runtime form) or from a name string (the authoring form).
//! Turning a name into a dense id is engine policy -- the build assigns ids in
//! world declaration order -- but the tables that hold it live with the build,
//! above this crate. The layer that owns them installs one
//! [`ReferenceResolver`] here, which answers against whichever build's tables
//! the calling thread resolves through, so concurrent builds stay apart. A name
//! seen with none installed is a configuration error, surfaced as a
//! deserialization failure.

use spin::Once;

use crate::ecs::handle::HandleKind;

/// Turns an authored reference name into the id or handle the calling thread's
/// build assigns it.
pub trait ReferenceResolver: Sync {
    /// The id `name` is interned under, assigning the next one to a name not
    /// seen before.
    fn name(&self, name: &str) -> u32;

    /// The handle `name` has in `kind`'s space, or `None` when the build
    /// declares no such resource.
    ///
    /// Unlike a name's id, a handle is not assignable on demand: it is a
    /// position in the build's declaration-ordered resource table.
    fn handle(&self, kind: HandleKind, name: &str) -> Option<u32>;
}

static RESOLVER: Once<&'static dyn ReferenceResolver> = Once::new();

/// Install the process's reference resolver. The first install stands; returns
/// whether this call made it.
pub fn install_reference_resolver(resolver: &'static dyn ReferenceResolver) -> bool {
    install(&RESOLVER, resolver)
}

fn install<T: ?Sized>(slot: &Once<&'static T>, resolver: &'static T) -> bool {
    let mut installed = false;
    slot.call_once(|| {
        installed = true;
        resolver
    });
    installed
}

/// Resolve a name to a dense id, or `None` with no resolver installed.
pub(crate) fn resolve_name(name: &str) -> Option<u32> {
    RESOLVER.get().map(|r| r.name(name))
}

/// Resolve a reference name to its dense handle in `kind`'s space. `None`
/// means either no resolver is installed or the name is not a declared
/// resource of that space; the caller decides whether to fall back (a
/// validation context) or to fail (a real build).
pub(crate) fn resolve_handle(kind: HandleKind, name: &str) -> Option<u32> {
    RESOLVER.get().and_then(|r| r.handle(kind, name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::AssetId;
    use crate::ecs::{Ref, RefTarget};
    use crate::test_support::install_resolvers;

    struct Clip;

    impl RefTarget for Clip {
        const TYPES: &'static [&'static str] = &["AudioClip"];
    }

    struct Fixed(u32);

    impl ReferenceResolver for Fixed {
        fn name(&self, _: &str) -> u32 {
            self.0
        }

        fn handle(&self, _: HandleKind, _: &str) -> Option<u32> {
            Some(self.0)
        }
    }

    // A fresh slot is the only way to see the first install stand: the
    // process's own slots are installed by whichever test gets there first.
    #[test]
    fn the_first_install_stands() {
        static FIRST: Fixed = Fixed(1);
        static SECOND: Fixed = Fixed(2);
        let slot: Once<&'static dyn ReferenceResolver> = Once::new();
        assert!(slot.get().is_none());
        assert!(install(&slot, &FIRST));
        assert!(!install(&slot, &SECOND));
        assert_eq!(slot.get().map(|r| r.name("floor")), Some(1));
    }

    #[test]
    fn asset_id_resolves_a_name_through_the_seam() {
        install_resolvers();
        let id: AssetId = serde_json::from_str("\"floor\"").unwrap();
        assert_eq!(id, AssetId(5));
    }

    #[test]
    fn a_ref_resolves_a_name_through_the_seam() {
        install_resolvers();
        let r: Ref<Clip> = serde_json::from_str("\"wall\"").unwrap();
        assert_eq!(r.id(), AssetId(4));
    }

    // A handle is not assignable on demand: a name the build declares no
    // resource of that space for has none.
    #[test]
    fn every_handle_space_resolves_through_the_installed_resolver() {
        install_resolvers();
        for kind in HandleKind::ALL {
            assert_eq!(resolve_handle(*kind, "floor"), Some(5));
            assert_eq!(resolve_handle(*kind, "unknown_x"), None);
        }
        assert_eq!(resolve_name("abcd"), Some(4));
    }
}
