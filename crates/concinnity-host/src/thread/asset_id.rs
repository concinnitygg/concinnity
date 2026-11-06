//! Build-time name -> dense id interner. Asset names declared in world.jsonl are
//! interned to an `AssetId` in declaration order; the blob and the runtime carry
//! only the integer, so every cross-reference lookup is an integer compare.
//!
//! This module owns the interner and installs it into the engine's resolver
//! seam (`concinnity_engine::ecs::install_name_resolver`) so a name-string
//! reference deserializes to a dense id during a build. At runtime references
//! are already integers, so the seam is never consulted.
//!
//! Each thread resolves through one table, which a build starts fresh and the
//! threads it hands work to share, so every thread of a build agrees on every
//! id and concurrent builds stay apart. A reference resolved through the seams
//! ([`resolve_reference`]) on a thread no build gave a table is reported, since
//! its ids belong to no build.

use std::cell::{Cell, RefCell};
use std::sync::{Arc, Mutex, PoisonError};

use concinnity_engine::ecs::NameResolver;
use concinnity_engine::ecs::asset_id::AssetId;

use super::name_interner::NameInterner;

/// A build's interner, shared by every thread the build hands work to.
#[derive(Clone, Default)]
pub struct SharedInterner(Arc<Mutex<NameInterner>>);

thread_local! {
    static INTERNER: RefCell<SharedInterner> = RefCell::new(SharedInterner::default());
    // Whether this thread was given a table to resolve through.
    static ASSIGNED: Cell<bool> = const { Cell::new(false) };
}

// Run `f` on the calling thread's table.
fn with_table<R>(f: impl FnOnce(&mut NameInterner) -> R) -> R {
    INTERNER.with(|t| {
        let table = t.borrow();
        let mut guard = table.0.lock().unwrap_or_else(PoisonError::into_inner);
        f(&mut guard)
    })
}

// Resolve through this thread's table from now on.
fn assign(table: SharedInterner) {
    INTERNER.with(|t| *t.borrow_mut() = table);
    ASSIGNED.with(|a| a.set(true));
}

// The seam's view of the calling thread's interner.
struct ThreadInterner;

impl NameResolver for ThreadInterner {
    fn resolve(&self, name: &str) -> u32 {
        resolve_reference(name).0
    }
}

/// Intern a name a deserialized reference names, as the resolver seams do,
/// warning once on a thread no build gave a table.
pub fn resolve_reference(name: &str) -> AssetId {
    if !ASSIGNED.with(|a| a.replace(true)) {
        tracing::warn!(
            "asset name '{name}' resolved on a thread no build gave its tables; \
             its ids match no build"
        );
    }
    intern(name)
}

/// Install the resolver seam so a name-string reference deserializes through
/// this interner. Idempotent and cheap after the first call.
pub fn ensure_name_resolver() {
    static RESOLVER: ThreadInterner = ThreadInterner;
    concinnity_engine::ecs::install_name_resolver(&RESOLVER);
}

/// Intern a name into the current thread's table, returning its id. If the
/// name was already interned the existing id is returned (idempotent).
pub fn intern(name: &str) -> AssetId {
    ensure_name_resolver();
    AssetId(with_table(|i| i.intern(name)))
}

/// Start the calling thread on a fresh, empty table. Call once at the start of
/// a build so ids are dense and declaration-ordered for that build. A table
/// other threads share is left as it is.
pub fn reset_interner() {
    ensure_name_resolver();
    assign(SharedInterner::default());
}

/// Resolve an already-interned name to its id without inserting: `None` for a
/// name this thread's table has never seen, so per-frame lookups of unknown
/// names never grow the table. O(1) and allocation-free, unlike snapshotting
/// `name_table`.
pub fn lookup(name: &str) -> Option<AssetId> {
    ensure_name_resolver();
    with_table(|i| i.lookup(name).map(AssetId))
}

/// The name `id` was interned under, or `None` when none was recorded for it --
/// an id past the table, or a slot a sparse `prime_name_table` left blank. O(1)
/// and one allocation, for a caller that wants one label rather than the whole
/// table.
pub fn name_of(id: AssetId) -> Option<String> {
    ensure_name_resolver();
    with_table(|i| match i.name(id.0) {
        "" => None,
        name => Some(name.to_string()),
    })
}

/// Snapshot every interned name on the current thread, indexed by `AssetId`.
/// Because ids are assigned in world.jsonl declaration order, `table[id]` is
/// the declared name for that id. Used by the editor's debug module to remap
/// runtime `AssetId`s back to their declared names.
pub fn name_table() -> Vec<String> {
    with_table(|i| i.names().map(str::to_string).collect())
}

/// Install a recorded (id, name) table into an EMPTY interner, so a process
/// that loads prebuilt blobs (the editor booting without an in-process cook)
/// resolves names to the ids the build baked. Ids may be sparse; unrecorded
/// slots hold an empty name, which no authored asset can carry. Returns false
/// (and changes nothing) when the interner already holds names -- an in-process
/// cook has primed it authoritatively.
pub fn prime_name_table(pairs: &[(u32, String)]) -> bool {
    ensure_name_resolver();
    ASSIGNED.with(|a| a.set(true));
    with_table(|i| i.prime(pairs))
}

/// The calling thread's table, for handing to the threads a build gives work.
pub fn shared_interner() -> SharedInterner {
    INTERNER.with(|t| t.borrow().clone())
}

/// Resolve through `table` on the calling thread, so it agrees on every id
/// with the thread `table` came from, and a name it interns is that thread's
/// too. Names past the ones a build declares land in arrival order; the
/// declared ones are interned by position first, so their ids are fixed.
pub fn adopt_interner(table: &SharedInterner) {
    ensure_name_resolver();
    assign(table.clone());
}

/// Pre-intern a batch of names in order so identity ids are dense and follow
/// world.jsonl declaration order.
pub fn intern_all(names: &[&str]) {
    ensure_name_resolver();
    ASSIGNED.with(|a| a.set(true));
    with_table(|interner| {
        for n in names {
            interner.intern(n);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::math::SaturatingFrom;
    use concinnity_engine::components::Prop;
    use concinnity_engine::ecs::Ref;

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
    // rather than as an empty name, so a caller labeling by id falls back to
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
                .map(|i| AssetId(u32::saturating_from(i)))
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

    // An adopted table resolves names to the source thread's ids, sparse slots
    // included, and a name the adopting thread interns is the source's too.
    #[test]
    fn an_adopted_table_is_the_source_threads_own() {
        reset_interner();
        prime_name_table(&[(0, "floor".to_string()), (3, "lamp".to_string())]);
        intern("wall");
        let table = shared_interner();

        let resolved = std::thread::spawn(move || {
            adopt_interner(&table);
            (
                lookup("floor"),
                lookup("lamp"),
                lookup("wall"),
                name_of(AssetId(1)),
                intern("chair"),
            )
        })
        .join()
        .unwrap();

        assert_eq!(
            resolved,
            (
                Some(AssetId(0)),
                Some(AssetId(3)),
                Some(AssetId(4)),
                None,
                AssetId(5)
            )
        );
        assert_eq!(lookup("chair"), Some(AssetId(5)), "the source sees it");
    }

    // A new build on one thread starts its own table and leaves the one it was
    // sharing with another thread as it was.
    #[test]
    fn a_reset_leaves_a_shared_table_alone() {
        reset_interner();
        intern_all(&["floor", "wall"]);
        let table = shared_interner();
        reset_interner();
        assert_eq!(lookup("floor"), None, "this thread starts fresh");

        let kept = std::thread::spawn(move || {
            adopt_interner(&table);
            lookup("wall")
        })
        .join()
        .unwrap();
        assert_eq!(kept, Some(AssetId(1)));
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
    fn opt_ref_resolves_a_name_and_treats_null_as_none() {
        reset_interner();
        intern_all(&["mesh_a"]);

        #[derive(serde::Deserialize)]
        struct Holder {
            #[serde(default)]
            r: Option<Ref<Prop>>,
        }

        let named: Holder = serde_json::from_str("{\"r\":\"mesh_a\"}").unwrap();
        assert_eq!(named.r, Some(Ref::new(AssetId(0))));
        let null: Holder = serde_json::from_str("{\"r\":null}").unwrap();
        assert_eq!(null.r, None);
        let by_id: Holder = serde_json::from_str("{\"r\":5}").unwrap();
        assert_eq!(by_id.r, Some(Ref::new(AssetId(5))));
    }
}
