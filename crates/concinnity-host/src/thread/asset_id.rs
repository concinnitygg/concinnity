//! Build-time name -> dense id interner and resource-handle map. Asset names
//! declared in world.jsonl are interned to an `AssetId` in declaration order;
//! the blob and the runtime carry only the integer, so every cross-reference
//! lookup is an integer compare.
//!
//! A build's tables are one [`BuildTables`]: its name interner and the resource
//! handles it assigns. Each thread resolves through one, which a build starts
//! fresh ([`begin_build`]) and the threads it hands work to enter for as long
//! as they work for it ([`BuildScope`]), so every thread of a build agrees on
//! every id and concurrent builds stay apart. This module installs the engine's
//! reference resolver seam, which answers a name-string reference against the
//! calling thread's tables. At runtime references are already integers, so the
//! seam is never consulted.
//!
//! A reference resolved through the seam ([`resolve_reference`]) on a thread
//! no build gave tables is reported, since its ids belong to no build.

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{HandleKind, ReferenceResolver};
use concinnity_engine::resource::ResourceHandles;

use super::name_interner::NameInterner;

/// One build's tables: the names it interns and the resource handles it
/// assigns. Shared by every thread the build hands work to.
#[derive(Default)]
pub struct BuildTables {
    names: Mutex<NameInterner>,
    handles: RwLock<Arc<ResourceHandles>>,
}

impl BuildTables {
    /// Resolve resource references through `handles` from now on.
    pub fn set_resource_handles(&self, handles: ResourceHandles) {
        *self.handles.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(handles);
    }

    /// Give the build's assets their ids: `pairs` replaces every name interned
    /// so far, as [`prime_name_table`] would install it into empty tables.
    /// Ids handed out before this were provisional, for the passes that read
    /// references before the build's assets are final.
    pub fn assign_names(&self, pairs: &[(u32, String)]) {
        self.with_names(|names| {
            *names = NameInterner::default();
            names.prime(pairs);
        });
    }

    fn with_names<R>(&self, f: impl FnOnce(&mut NameInterner) -> R) -> R {
        f(&mut self.names.lock().unwrap_or_else(PoisonError::into_inner))
    }

    fn handle(&self, kind: HandleKind, id: AssetId) -> Option<u32> {
        self.handles
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .handle(kind, id)
    }
}

thread_local! {
    static TABLES: RefCell<Arc<BuildTables>> = RefCell::default();
    // Whether this thread was given tables to resolve through.
    static ASSIGNED: Cell<bool> = const { Cell::new(false) };
}

/// Resolves through a build's tables on the calling thread until dropped, then
/// returns the thread to the tables it had before.
#[must_use = "the thread leaves the build's tables when the scope drops"]
pub struct BuildScope {
    previous: Option<(Arc<BuildTables>, bool)>,
    // The scope swaps a thread-local, so it must drop on the thread it entered.
    _thread: PhantomData<*const ()>,
}

impl BuildScope {
    /// Resolve through `tables` on the calling thread for the scope's life, so
    /// it agrees on every id with every other thread in `tables`' build, and a
    /// name it interns is theirs too.
    pub fn enter(tables: Arc<BuildTables>) -> Self {
        let assigned = ASSIGNED.with(Cell::get);
        let previous = Some((assign(tables, true), assigned));
        Self {
            previous,
            _thread: PhantomData,
        }
    }

    /// Resolve through fresh, empty tables for the scope's life, for work whose
    /// ids are throwaway (validating one asset on its own).
    pub fn fresh() -> Self {
        Self::enter(Arc::default())
    }
}

impl Drop for BuildScope {
    fn drop(&mut self) {
        if let Some((tables, assigned)) = self.previous.take() {
            assign(tables, assigned);
        }
    }
}

// Make `tables` the calling thread's, returning the ones it replaces.
fn assign(tables: Arc<BuildTables>, assigned: bool) -> Arc<BuildTables> {
    install_resolver();
    ASSIGNED.with(|a| a.set(assigned));
    TABLES.with(|t| std::mem::replace(&mut *t.borrow_mut(), tables))
}

/// Start the calling thread on fresh, empty tables and keep them, returning
/// them for the build to fill and hand to its workers. Call once at the start
/// of a build so ids are dense and declaration-ordered for that build; the
/// thread's name lookups read them after the build too. Tables other threads
/// share are left as they are.
pub fn begin_build() -> Arc<BuildTables> {
    let tables = Arc::<BuildTables>::default();
    assign(Arc::clone(&tables), true);
    tables
}

/// The calling thread's tables, for handing to the threads a build gives work.
pub fn current_tables() -> Arc<BuildTables> {
    install_resolver();
    TABLES.with(|t| Arc::clone(&t.borrow()))
}

/// Resolve the calling thread's resource references through `handles`.
pub fn set_resource_handles(handles: ResourceHandles) {
    current_tables().set_resource_handles(handles);
}

// Run `f` on the calling thread's name table.
fn with_table<R>(f: impl FnOnce(&mut NameInterner) -> R) -> R {
    install_resolver();
    TABLES.with(|t| t.borrow().with_names(f))
}

// The seam's view of the calling thread's tables: a name interns there, and a
// resource reference resolves to its `AssetId` there and then to that
// resource's handle in the requested space.
struct ThreadTables;

impl ReferenceResolver for ThreadTables {
    fn name(&self, name: &str) -> u32 {
        resolve_reference(name).0
    }

    fn handle(&self, kind: HandleKind, name: &str) -> Option<u32> {
        let id = resolve_reference(name);
        TABLES.with(|t| t.borrow().handle(kind, id))
    }
}

/// Install the engine's reference resolver so a name-string reference
/// deserializes through the calling thread's tables. Every table access here
/// installs it; code that deserializes authored references before touching
/// the tables calls it first. Cheap after the first call.
pub fn install_resolver() {
    static RESOLVER: ThreadTables = ThreadTables;
    concinnity_engine::ecs::install_reference_resolver(&RESOLVER);
}

/// Intern a name a deserialized reference names, as the resolver seam does,
/// warning once on a thread no build gave tables.
pub fn resolve_reference(name: &str) -> AssetId {
    if !ASSIGNED.with(|a| a.replace(true)) {
        tracing::warn!(
            "asset name '{name}' resolved on a thread no build gave its tables; \
             its ids match no build"
        );
    }
    intern(name)
}

/// Intern a name into the current thread's table, returning its id. If the
/// name was already interned the existing id is returned (idempotent).
pub fn intern(name: &str) -> AssetId {
    AssetId(with_table(|i| i.intern(name)))
}

/// Resolve an already-interned name to its id without inserting: `None` for a
/// name this thread's table has never seen, so per-frame lookups of unknown
/// names never grow the table. O(1) and allocation-free, unlike snapshotting
/// `name_table`.
pub fn lookup(name: &str) -> Option<AssetId> {
    with_table(|i| i.lookup(name).map(AssetId))
}

/// The name `id` was interned under, or `None` when none was recorded for it --
/// an id past the table, or a slot a sparse `prime_name_table` left blank. O(1)
/// and one allocation, for a caller that wants one label rather than the whole
/// table.
pub fn name_of(id: AssetId) -> Option<String> {
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
    ASSIGNED.with(|a| a.set(true));
    with_table(|i| i.prime(pairs))
}

/// Pre-intern a batch of names in order so identity ids are dense and follow
/// world.jsonl declaration order.
pub fn intern_all(names: &[&str]) {
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
        begin_build();
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
        begin_build();
        intern_all(&["a", "b", "c"]);
        assert_eq!(name_table(), vec!["a", "b", "c"]);
    }

    // The single-id counterpart to `name_table`. A blank slot reads as absent
    // rather than as an empty name, so a caller labeling by id falls back to
    // its own placeholder instead of printing nothing.
    #[test]
    fn name_of_resolves_one_id_and_reports_unnamed_slots_as_absent() {
        begin_build();
        intern_all(&["floor", "wall"]);
        assert_eq!(name_of(AssetId(0)).as_deref(), Some("floor"));
        assert_eq!(name_of(AssetId(1)).as_deref(), Some("wall"));
        assert_eq!(name_of(AssetId(7)), None, "an id past the table is absent");

        begin_build();
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

        begin_build();
        intern_all(&["floor", "wall", "lamp"]);
        for name in ["floor", "wall", "lamp", "missing"] {
            assert_eq!(lookup(name), scan(name), "disagreed on '{name}'");
        }

        // Sparse table: agreement must hold across the gap too.
        begin_build();
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
        begin_build();
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
        begin_build();
        prime_name_table(&[(0, "floor".to_string()), (2, "lamp".to_string())]);
        assert_eq!(name_table()[1], "");
        assert_eq!(lookup(""), None);
        assert_eq!(lookup("lamp"), Some(AssetId(2)));
    }

    #[test]
    fn prime_installs_a_sparse_table_and_yields_to_a_cook() {
        begin_build();
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

    // Entered tables resolve names to the source thread's ids, sparse slots
    // included, and a name the entering thread interns is the source's too.
    #[test]
    fn entered_tables_are_the_source_threads_own() {
        begin_build();
        prime_name_table(&[(0, "floor".to_string()), (3, "lamp".to_string())]);
        intern("wall");
        let table = current_tables();

        let resolved = std::thread::spawn(move || {
            let _build = BuildScope::enter(table);
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

    // A new build on one thread starts its own tables and leaves the ones it
    // was sharing with another thread as they were.
    #[test]
    fn a_new_build_leaves_shared_tables_alone() {
        begin_build();
        intern_all(&["floor", "wall"]);
        let table = current_tables();
        begin_build();
        assert_eq!(lookup("floor"), None, "this thread starts fresh");

        let kept = std::thread::spawn(move || {
            let _build = BuildScope::enter(table);
            lookup("wall")
        })
        .join()
        .unwrap();
        assert_eq!(kept, Some(AssetId(1)));
    }

    // Assigning the build's names drops the provisional ids handed out before
    // its assets were final, in place, so threads sharing the tables see them.
    #[test]
    fn assigned_names_replace_the_provisional_ones() {
        let tables = begin_build();
        intern("provisional");
        tables.assign_names(&[(0, "floor".to_string()), (1, "wall".to_string())]);
        assert_eq!(lookup("provisional"), None);
        assert_eq!(lookup("wall"), Some(AssetId(1)));
        assert_eq!(intern("lamp"), AssetId(2));
    }

    // A scope's tables last only as long as the scope: the thread returns to
    // the build it was in.
    #[test]
    fn a_scope_returns_the_thread_to_its_tables() {
        begin_build();
        intern_all(&["floor"]);
        {
            let _build = BuildScope::fresh();
            assert_eq!(lookup("floor"), None, "the scope starts fresh");
            intern("throwaway");
        }
        assert_eq!(lookup("floor"), Some(AssetId(0)));
        assert_eq!(lookup("throwaway"), None);
    }

    // Nested scopes unwind in order, each restoring the tables and the
    // assigned flag it found, so a thread no build gave tables still reports
    // its first reference once the scopes end.
    #[test]
    fn nested_scopes_restore_tables_and_assignment() {
        let unassigned = std::thread::spawn(|| {
            {
                let _build = BuildScope::fresh();
                assert!(ASSIGNED.with(Cell::get));
            }
            ASSIGNED.with(Cell::get)
        })
        .join()
        .unwrap();
        assert!(
            !unassigned,
            "a scope leaves an unassigned thread unassigned"
        );

        begin_build();
        intern_all(&["floor"]);
        let outer = current_tables();
        {
            let _entered = BuildScope::enter(Arc::clone(&outer));
            intern("wall");
            {
                let _fresh = BuildScope::fresh();
                assert_eq!(lookup("floor"), None);
                intern("throwaway");
            }
            assert_eq!(lookup("wall"), Some(AssetId(1)));
        }
        assert!(Arc::ptr_eq(&current_tables(), &outer));
        assert_eq!(lookup("throwaway"), None);
        assert!(ASSIGNED.with(Cell::get));
    }

    // A resource reference resolves to its handle in the build's map, and a
    // name the map declares no resource for has none in that space.
    #[test]
    fn a_resource_reference_resolves_through_the_builds_handles() {
        use concinnity_engine::blob::ResourceKind;
        use concinnity_engine::ecs::TextureHandle;

        begin_build();
        intern_all(&["floor", "brick", "moss"]);
        set_resource_handles(ResourceHandles::from_assets([
            (intern("brick"), ResourceKind::Texture),
            (intern("moss"), ResourceKind::Texture),
        ]));
        let moss: TextureHandle = serde_json::from_str("\"moss\"").unwrap();
        assert_eq!(moss, TextureHandle::new(1));
        assert_eq!(ThreadTables.handle(HandleKind::Texture, "floor"), None);
    }

    // Integration: a name string deserializes to a dense id through the resolver
    // seam this module installs, backed by the real interner.
    #[test]
    fn asset_id_deserializes_from_name_string_via_the_seam() {
        begin_build();
        intern_all(&["floor", "wall"]);
        let id: AssetId = serde_json::from_str("\"wall\"").unwrap();
        assert_eq!(id, AssetId(1));
    }

    #[test]
    fn opt_ref_resolves_a_name_and_treats_null_as_none() {
        begin_build();
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
