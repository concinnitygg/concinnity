//! Name -> id resolution seam.
//!
//! A reference deserializes either from an already-resolved integer id (the
//! compiled-args / runtime form) or from a name string (the authoring form).
//! Turning a name into a dense id is engine policy -- the build assigns ids in
//! world declaration order -- but the tables that hold it live with the build,
//! above this crate. The build installs a [`NameResolver`] and a
//! [`HandleResolver`] here before it deserializes named references. A name seen
//! with neither installed is a configuration error, surfaced as a
//! deserialization failure (only an out-of-engine tool reading authoring JSON
//! would hit it).
//!
//! Each resolver is installed once per process. The state it resolves against
//! (one build's tables) is the installer's to keep, so concurrent builds stay
//! apart.

use spin::Once;

use crate::ecs::handle::HandleKind;

/// Turns an authored name into the dense id the build assigns it.
pub trait NameResolver: Sync {
    /// The id `name` is interned under, assigning the next one to a name not
    /// seen before.
    fn resolve(&self, name: &str) -> u32;
}

/// Turns an authored resource name into its dense handle in one handle space.
///
/// Unlike a name's id, a handle is not assignable on demand: it is a position
/// in the build's declaration-ordered resource table, so a name with no
/// matching resource of that space has none.
pub trait HandleResolver: Sync {
    /// The handle `name` has in `kind`'s space, or `None` when the build
    /// declares no such resource.
    fn resolve(&self, kind: HandleKind, name: &str) -> Option<u32>;
}

static NAMES: Once<&'static dyn NameResolver> = Once::new();
static HANDLES: Once<&'static dyn HandleResolver> = Once::new();

/// Install the process's name resolver. The first install stands; returns
/// whether this call made it.
pub fn install_name_resolver(resolver: &'static dyn NameResolver) -> bool {
    install(&NAMES, resolver)
}

/// Install the process's resource-handle resolver. The first install stands;
/// returns whether this call made it.
pub fn install_handle_resolver(resolver: &'static dyn HandleResolver) -> bool {
    install(&HANDLES, resolver)
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
    NAMES.get().map(|r| r.resolve(name))
}

/// Resolve a reference name to its dense handle in `kind`'s space. `None`
/// means either no resolver is installed or the name is not a declared
/// resource of that space; the caller decides whether to fall back (a
/// validation context) or to fail (a real build).
pub(crate) fn resolve_handle(kind: HandleKind, name: &str) -> Option<u32> {
    HANDLES.get().and_then(|r| r.resolve(kind, name))
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

    impl NameResolver for Fixed {
        fn resolve(&self, _: &str) -> u32 {
            self.0
        }
    }

    // A fresh slot is the only way to see the first install stand: the
    // process's own slots are installed by whichever test gets there first.
    #[test]
    fn the_first_install_stands() {
        static FIRST: Fixed = Fixed(1);
        static SECOND: Fixed = Fixed(2);
        let slot: Once<&'static dyn NameResolver> = Once::new();
        assert!(slot.get().is_none());
        assert!(install(&slot, &FIRST));
        assert!(!install(&slot, &SECOND));
        assert_eq!(slot.get().map(|r| r.resolve("floor")), Some(1));
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
