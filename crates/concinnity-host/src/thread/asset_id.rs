//! Build-time name -> dense id interner. Asset names declared in world.jsonl are
//! interned to an `AssetId` in declaration order; the blob and the runtime carry
//! only the integer, so every cross-reference lookup is an integer compare.
//!
//! This module owns the interner and installs it into the schema crate's
//! resolver seam (`concinnity_asset::set_name_resolver`) so a name-string
//! reference deserializes to a dense id during a build. At runtime references
//! are already integers, so the seam is never consulted. The identity types the
//! seam produces are re-exported from concinnity-core under the same path.

use std::cell::RefCell;
use std::sync::Once;

use super::name_interner::NameInterner;

// The asset identity + typed reference primitives, defined in the schema crate.
pub use concinnity_core::ecs::asset_id::{
    AssetId, AssetRef, de_opt_asset_ref, de_opt_asset_ref_typed,
};

thread_local! {
    static INTERNER: RefCell<NameInterner> = RefCell::new(NameInterner::default());
}

/// Install the schema crate's resolver seam so a name-string reference
/// deserializes through this interner. The closure is non-capturing (the
/// interner is a thread-local static), so it coerces to the plain `fn` pointer
/// the seam holds; per-thread interner state stays isolated. Idempotent and cheap
/// after the first call.
pub fn ensure_name_resolver() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        concinnity_asset::set_name_resolver(|name| INTERNER.with(|i| i.borrow_mut().intern(name)));
    });
}

/// Intern a name into the current thread's interner, returning its id. If the
/// name was already interned the existing id is returned (idempotent).
pub fn intern(name: &str) -> AssetId {
    ensure_name_resolver();
    AssetId(INTERNER.with(|i| i.borrow_mut().intern(name)))
}

/// Clear the thread-local interner. Call once at the start of a build so ids
/// are dense and declaration-ordered for that build.
pub fn reset_interner() {
    ensure_name_resolver();
    INTERNER.with(|i| *i.borrow_mut() = NameInterner::default());
}

/// Resolve an already-interned name to its id without inserting: `None` for a
/// name this thread's interner has never seen, so per-frame lookups of unknown
/// names never grow the table. O(1) and allocation-free, unlike snapshotting
/// `name_table`.
pub fn lookup(name: &str) -> Option<AssetId> {
    ensure_name_resolver();
    INTERNER.with(|i| i.borrow().lookup(name).map(AssetId))
}

/// The name `id` was interned under, or `None` when none was recorded for it --
/// an id past the table, or a slot a sparse `prime_name_table` left blank. O(1)
/// and one allocation, for a caller that wants one label rather than the whole
/// table.
pub fn name_of(id: AssetId) -> Option<String> {
    ensure_name_resolver();
    INTERNER.with(|i| match i.borrow().name(id.0) {
        "" => None,
        name => Some(name.to_string()),
    })
}

/// Snapshot every interned name on the current thread, indexed by `AssetId`.
/// Because ids are assigned in world.jsonl declaration order, `table[id]` is
/// the declared name for that id. Used by the editor's debug module to remap
/// runtime `AssetId`s back to their declared names.
pub fn name_table() -> Vec<String> {
    INTERNER.with(|i| i.borrow().names().map(str::to_string).collect())
}

/// Install a recorded (id, name) table into an EMPTY interner, so a process
/// that loads prebuilt blobs (the editor booting without an in-process cook)
/// resolves names to the ids the build baked. Ids may be sparse; unrecorded
/// slots hold an empty name, which no authored asset can carry. Returns false
/// (and changes nothing) when the interner already holds names -- an in-process
/// cook has primed it authoritatively.
pub fn prime_name_table(pairs: &[(u32, String)]) -> bool {
    ensure_name_resolver();
    INTERNER.with(|i| i.borrow_mut().prime(pairs))
}

/// Pre-intern a batch of names in order so identity ids are dense and follow
/// world.jsonl declaration order.
pub fn intern_all(names: &[&str]) {
    ensure_name_resolver();
    INTERNER.with(|i| {
        let mut interner = i.borrow_mut();
        for n in names {
            interner.intern(n);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_is_idempotent_and_dense() {
        reset_interner();
        intern_all(&["a", "b", "c"]);
        assert_eq!(intern("a"), AssetId(0));
        assert_eq!(intern("b"), AssetId(1));
        assert_eq!(intern("c"), AssetId(2));
        // a fresh reference past the pre-interned set gets the next id
        assert_eq!(intern("d"), AssetId(3));
        assert_eq!(intern("a"), AssetId(0));
    }

    #[test]
    fn name_table_snapshots_in_id_order() {
        reset_interner();
        intern_all(&["a", "b", "c"]);
        assert_eq!(name_table(), vec!["a", "b", "c"]);
    }

    // The single-id counterpart to `name_table`. A blank slot reads as absent
    // rather than as an empty name, so a caller labelling by id falls back to
    // its own placeholder instead of printing nothing.
    #[test]
    fn name_of_resolves_one_id_and_reports_unnamed_slots_as_absent() {
        reset_interner();
        intern_all(&["floor", "wall"]);
        assert_eq!(name_of(AssetId(0)).as_deref(), Some("floor"));
        assert_eq!(name_of(AssetId(1)).as_deref(), Some("wall"));
        assert_eq!(name_of(AssetId(7)), None, "an id past the table is absent");

        reset_interner();
        prime_name_table(&[(0, "floor".to_string()), (2, "lamp".to_string())]);
        assert_eq!(name_of(AssetId(1)), None, "a blank slot is absent");
        assert_eq!(name_of(AssetId(2)).as_deref(), Some("lamp"));
    }

    // Callers used to resolve a name by scanning a `name_table()` snapshot for
    // its position. `lookup` replaces that everywhere, so the two must agree on
    // every name a caller can hold -- and where they diverge, `lookup` must be
    // the correct one.
    #[test]
    fn lookup_agrees_with_scanning_a_name_table_snapshot() {
        let scan = |name: &str| {
            name_table()
                .iter()
                .position(|n| n == name)
                .map(|i| AssetId(i as u32))
        };

        reset_interner();
        intern_all(&["floor", "wall", "lamp"]);
        for name in ["floor", "wall", "lamp", "missing"] {
            assert_eq!(lookup(name), scan(name), "disagreed on '{name}'");
        }

        // Sparse table: agreement must hold across the gap too.
        reset_interner();
        prime_name_table(&[(0, "floor".to_string()), (2, "lamp".to_string())]);
        for name in ["floor", "lamp", "missing"] {
            assert_eq!(lookup(name), scan(name), "disagreed on '{name}'");
        }

        // The one divergence, and the reason the scan was worth replacing: a
        // blank slot makes the scan resolve the empty name to whichever asset
        // sits in the first gap. `lookup` refuses it.
        assert_eq!(scan(""), Some(AssetId(1)));
        assert_eq!(lookup(""), None);
    }

    #[test]
    fn lookup_resolves_interned_names_only() {
        reset_interner();
        intern_all(&["a", "b"]);
        assert_eq!(lookup("a"), Some(AssetId(0)));
        assert_eq!(lookup("b"), Some(AssetId(1)));
        // An unknown name resolves to None without interning it, so a
        // per-frame lookup never grows the table.
        assert_eq!(lookup("c"), None);
        assert_eq!(name_table().len(), 2);
        // A sparse prime leaves blank names in the id-indexed table; those
        // slots are not lookupable, so an empty name never resolves to the
        // entity sitting in the first gap.
        reset_interner();
        prime_name_table(&[(0, "floor".to_string()), (2, "lamp".to_string())]);
        assert_eq!(name_table()[1], "");
        assert_eq!(lookup(""), None);
        assert_eq!(lookup("lamp"), Some(AssetId(2)));
    }

    #[test]
    fn prime_installs_a_sparse_table_and_yields_to_a_cook() {
        reset_interner();
        // Sparse ids (1 is unrecorded) land at their exact slots.
        assert!(prime_name_table(&[
            (0, "floor".to_string()),
            (2, "lamp".to_string())
        ]));
        assert_eq!(intern("floor"), AssetId(0));
        assert_eq!(intern("lamp"), AssetId(2));
        assert_eq!(name_table()[1], "", "unrecorded slot stays blank");
        // A fresh name appends past the recorded table, never into a gap.
        assert_eq!(intern("new"), AssetId(3));
        // A populated interner is authoritative; priming over it is refused.
        assert!(!prime_name_table(&[(0, "other".to_string())]));
        assert_eq!(intern("floor"), AssetId(0));
    }

    // Integration: a name string deserializes to a dense id through the resolver
    // seam this module installs, backed by the real interner.
    #[test]
    fn asset_id_deserializes_from_name_string_via_the_seam() {
        reset_interner();
        intern_all(&["floor", "wall"]);
        let id: AssetId = serde_json::from_str("\"wall\"").unwrap();
        assert_eq!(id, AssetId(1));
    }

    #[test]
    fn opt_ref_resolves_a_name_and_treats_empty_as_none() {
        reset_interner();
        intern_all(&["mesh_a"]);

        #[derive(serde::Deserialize)]
        struct Holder {
            #[serde(default, deserialize_with = "de_opt_asset_ref")]
            r: Option<AssetId>,
        }

        let named: Holder = serde_json::from_str("{\"r\":\"mesh_a\"}").unwrap();
        assert_eq!(named.r, Some(AssetId(0)));
        let empty: Holder = serde_json::from_str("{\"r\":\"\"}").unwrap();
        assert_eq!(empty.r, None);
        let by_id: Holder = serde_json::from_str("{\"r\":5}").unwrap();
        assert_eq!(by_id.r, Some(AssetId(5)));
    }
}
