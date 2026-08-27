// The build-time name -> dense id table, storing each name's bytes once.
//
// The obvious shape -- a `Vec<String>` for id -> name beside a
// `HashMap<String, u32>` for name -> id -- holds every name twice and charges
// two heap blocks for it. On a twenty thousand asset world that is forty
// thousand live blocks, and it was half of the engine's small-block footprint.
//
// Here the names are concatenated into one blob and addressed by span, so
// id -> name is an index and the heap holds a fixed handful of blocks whatever
// the name count. The map cannot key on a borrow into the blob without making
// the struct self-referential, so it keys on the name's hash and resolves the
// rest by comparing against the blob: a bucket is the ids whose names hash
// alike, which is one id except on a true hash collision. `InlineVec` keeps
// that ordinary case off the heap too.
//
// Ids are handed out in insertion order and never derived from a hash, so the
// declaration order a build depends on does not move with the hasher.

use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;

use concinnity_memory::InlineVec;

// Where one name's bytes sit in the blob. A default span is empty, which is
// what an id no name was recorded for reads as.
#[derive(Clone, Copy, Default)]
struct Span {
    start: u32,
    len: u32,
}

// Names interned to dense ids, each name's bytes held once.
//
// Generic over the name hasher so the collision path can be tested; every
// caller outside this module's tests takes the default.
pub(crate) struct NameInterner<S = RandomState> {
    blob: String,
    spans: Vec<Span>,
    buckets: HashMap<u64, InlineVec<u32>>,
    hasher: S,
}

impl<S: Default> Default for NameInterner<S> {
    fn default() -> Self {
        Self {
            blob: String::new(),
            spans: Vec::new(),
            buckets: HashMap::new(),
            hasher: S::default(),
        }
    }
}

impl<S: BuildHasher> NameInterner<S> {
    // Whether any id has been handed out. A primed table counts as populated
    // even where its slots are blank.
    pub(crate) fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    // The name recorded for `id`, or "" for an id past the table or for a slot
    // a sparse prime left blank.
    pub(crate) fn name(&self, id: u32) -> &str {
        self.spans
            .get(id as usize)
            .map_or("", |&span| self.slice(span))
    }

    // Every name in id order, so position `i` is the name for id `i`.
    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.spans.iter().map(|&span| self.slice(span))
    }

    // Resolve a name already interned, without inserting it.
    pub(crate) fn lookup(&self, name: &str) -> Option<u32> {
        self.buckets
            .get(&self.hasher.hash_one(name))?
            .iter()
            .copied()
            .find(|&id| self.name(id) == name)
    }

    // Resolve `name`, interning it at the next id if it is new.
    pub(crate) fn intern(&mut self, name: &str) -> u32 {
        if let Some(id) = self.lookup(name) {
            return id;
        }
        let id = self.spans.len() as u32;
        let span = self.append(name);
        self.spans.push(span);
        self.record(name, id);
        id
    }

    // Install a recorded (id, name) table into an empty interner. Ids may be
    // sparse; the slots between them stay blank and do not resolve. Returns
    // false and changes nothing when the table already holds ids.
    pub(crate) fn prime(&mut self, pairs: &[(u32, String)]) -> bool {
        if !self.is_empty() {
            return false;
        }
        let len = pairs.iter().map(|(id, _)| id + 1).max().unwrap_or(0) as usize;
        self.spans = vec![Span::default(); len];
        for (id, name) in pairs {
            let span = self.append(name);
            self.spans[*id as usize] = span;
            self.record(name, *id);
        }
        true
    }

    fn slice(&self, span: Span) -> &str {
        &self.blob[span.start as usize..span.start as usize + span.len as usize]
    }

    // Add `name`'s bytes to the blob without giving it an id, so both a fresh
    // intern and a sparse prime can place the span where each needs it.
    fn append(&mut self, name: &str) -> Span {
        let start = u32::try_from(self.blob.len()).expect("interned names exceed 4 GiB");
        let len = u32::try_from(name.len()).expect("interned name exceeds 4 GiB");
        self.blob.push_str(name);
        Span { start, len }
    }

    fn record(&mut self, name: &str, id: u32) {
        let hash = self.hasher.hash_one(name);
        self.buckets.entry(hash).or_default().push(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hasher;

    fn interner() -> NameInterner {
        NameInterner::default()
    }

    // A hasher that sends every name to one bucket, so the collision path --
    // which a real hasher reaches too rarely to test -- runs on every name.
    #[derive(Default)]
    struct CollidingHasher;

    impl Hasher for CollidingHasher {
        fn write(&mut self, _bytes: &[u8]) {}
        fn finish(&self) -> u64 {
            0
        }
    }

    #[derive(Default)]
    struct AllCollide;

    impl BuildHasher for AllCollide {
        type Hasher = CollidingHasher;
        fn build_hasher(&self) -> CollidingHasher {
            CollidingHasher
        }
    }

    #[test]
    fn ids_are_dense_in_insertion_order_and_interning_is_idempotent() {
        let mut names = interner();
        assert_eq!(names.intern("floor"), 0);
        assert_eq!(names.intern("wall"), 1);
        assert_eq!(names.intern("floor"), 0, "a repeat returns the first id");
        assert_eq!(names.intern("lamp"), 2);
        assert_eq!(names.names().count(), 3);
    }

    #[test]
    fn ids_round_trip_back_to_their_names() {
        let mut names = interner();
        names.intern("floor");
        names.intern("wall");

        assert_eq!(names.name(0), "floor");
        assert_eq!(names.name(1), "wall");
        assert_eq!(names.names().collect::<Vec<_>>(), ["floor", "wall"]);
        assert_eq!(names.name(9), "", "an id past the table reads blank");
    }

    // The point of the blob: each distinct name contributes its bytes once, so
    // the table cannot drift back into holding two copies.
    #[test]
    fn each_distinct_name_is_stored_exactly_once() {
        let mut names = interner();
        for name in ["aa", "bbb", "aa", "bbb", "c"] {
            names.intern(name);
        }
        assert_eq!(names.blob, "aabbbc");
        assert_eq!(names.names().count(), 3);
    }

    #[test]
    fn lookup_resolves_interned_names_and_never_inserts() {
        let mut names = interner();
        names.intern("floor");

        assert_eq!(names.lookup("floor"), Some(0));
        assert_eq!(names.lookup("missing"), None);
        assert_eq!(names.names().count(), 1, "a miss must not grow the table");
        assert_eq!(names.blob, "floor", "a miss must not grow the blob");
    }

    // A name whose bytes are a prefix or substring of the blob must not
    // resolve just because those bytes are present somewhere in it.
    #[test]
    fn a_substring_of_the_blob_is_not_an_interned_name() {
        let mut names = interner();
        names.intern("floorboard");
        names.intern("board");

        assert_eq!(names.lookup("floor"), None);
        assert_eq!(names.lookup("oorbo"), None);
        assert_eq!(names.lookup("board"), Some(1));
    }

    #[test]
    fn a_sparse_prime_leaves_blank_slots_that_do_not_resolve() {
        let mut names = interner();
        assert!(names.prime(&[(0, "floor".to_string()), (2, "lamp".to_string())]));

        assert_eq!(names.name(0), "floor");
        assert_eq!(names.name(1), "", "the unrecorded slot stays blank");
        assert_eq!(names.name(2), "lamp");
        assert_eq!(names.lookup("lamp"), Some(2));
        assert_eq!(names.lookup(""), None, "a blank slot is not lookupable");
        // A fresh name appends past the recorded table rather than into the gap.
        assert_eq!(names.intern("new"), 3);
    }

    #[test]
    fn priming_a_populated_table_is_refused_and_changes_nothing() {
        let mut names = interner();
        names.intern("floor");

        assert!(!names.prime(&[(0, "other".to_string())]));
        assert_eq!(names.name(0), "floor");
        assert_eq!(names.lookup("other"), None);
    }

    #[test]
    fn priming_nothing_leaves_an_empty_table() {
        let mut names = interner();
        assert!(names.prime(&[]));
        assert!(names.is_empty());
    }

    // Everything above, with every name in one bucket. Identity comes from the
    // blob comparison, so a hasher that distinguishes nothing must not merge
    // two names or lose one.
    #[test]
    fn colliding_names_stay_distinct() {
        let mut names: NameInterner<AllCollide> = NameInterner::default();
        assert_eq!(names.intern("floor"), 0);
        assert_eq!(names.intern("wall"), 1);
        assert_eq!(names.intern("lamp"), 2);
        assert_eq!(names.intern("wall"), 1, "a repeat still finds its own id");

        assert_eq!(names.lookup("floor"), Some(0));
        assert_eq!(names.lookup("wall"), Some(1));
        assert_eq!(names.lookup("lamp"), Some(2));
        assert_eq!(names.lookup("missing"), None);
        assert_eq!(names.names().collect::<Vec<_>>(), ["floor", "wall", "lamp"]);
        assert_eq!(names.buckets.len(), 1, "the test hasher collides them all");
    }

    #[test]
    fn an_empty_name_is_interned_like_any_other_when_asked_for() {
        let mut names = interner();
        assert_eq!(names.intern(""), 0);
        assert_eq!(names.lookup(""), Some(0));
        assert_eq!(names.name(0), "");
    }
}
