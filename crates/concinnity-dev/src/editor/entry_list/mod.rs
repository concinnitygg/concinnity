//! The authored entry list, with a session key beside every entry. The key is
//! what the editor addresses an entry by, because neither of the alternatives
//! carries identity: a position shifts as lines are added and removed (and an
//! undo can shift it back), and a name is the authored content, which the user
//! renames and which an entry need not declare at all.
//!
//! Keys are minted from one session-wide counter, so a key is unique for the
//! whole session and a list that drops an entry never reissues its key. They
//! are session state rather than authored content, so two lists are equal when
//! they would serialize the same, whatever keys their rows hold.
//!
//! The list is the world as the build sees it: an `Include` line is followed by
//! the entries of the file it names, so the `<Type>#<n>` labels over the list
//! are the build's. Beside each entry sits the included file it came from, and
//! a save writes each file whose entries changed: the world file with its
//! `Include` lines, and every included file with its own entries in list order.
//!
//! Three edits stay refused, since the files they would write no longer read
//! back as the list holds them: changing or removing an `Include` line (its
//! inlined entries would go stale), changing an entry of a file the world
//! includes more than once (each copy would claim the one file), and moving an
//! entry out of the run of its file's entries.

mod layout;
pub(crate) mod paths;

use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use concinnity_cook::authoring::world::write_world_jsonl;
use concinnity_cook::build_only::include::{SourcedEntry, is_include};

use layout::Layout;

static NEXT_KEY: AtomicU64 = AtomicU64::new(1);

// A session key for one authored entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct EntryId(u64);

impl EntryId {
    fn mint() -> Self {
        Self(NEXT_KEY.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for EntryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

// Why an edit cannot stand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    // The edit changed or removed the `Include` line at this position of the
    // list before it.
    IncludeLine(usize),
    // The edit changed or removed this entry of the list before it, which
    // comes from a file the world includes more than once.
    Repeated(usize),
    // The edit left the entry at this position outside its file's run.
    Moved(usize),
}

// The working entry list: the parsed world.jsonl values, each with its key
// and the included file it was read from.
//
// Derefs to the values, so every read and every in-place value edit works as
// it does on a `Vec`. The operations that change the length go through
// `push`, `push_to` and `remove`, which keep the other columns in step.
#[derive(Debug, Default, Clone)]
pub(crate) struct EntryList {
    values: Vec<serde_json::Value>,
    keys: Vec<EntryId>,
    files: Vec<Option<Arc<Path>>>,
    // Included files the world includes more than once.
    repeated: Vec<Arc<Path>>,
}

impl EntryList {
    // Take ownership of a parsed list of the world file's own entries, minting
    // a fresh key per entry.
    pub(crate) fn new(values: Vec<serde_json::Value>) -> Self {
        let keys = values.iter().map(|_| EntryId::mint()).collect();
        let files = vec![None; values.len()];
        Self {
            values,
            keys,
            files,
            ..Self::default()
        }
    }

    // Take the world at `world` with its includes resolved, keeping where each
    // entry came from.
    pub(crate) fn with_includes(sourced: Vec<SourcedEntry>, world: &Path) -> Self {
        let mut list = Self::default();
        for SourcedEntry { entry, file } in sourced {
            list.values.push(entry);
            list.keys.push(EntryId::mint());
            list.files.push(file.map(PathBuf::into));
        }
        let layout = list.layout(world);
        for file in list.files.iter().flatten() {
            if !list.repeated.contains(file) && layout.times_opened(file) > 1 {
                list.repeated.push(Arc::clone(file));
            }
        }
        list
    }

    // The include structure, with the world file's `Include` lines resolved
    // against `world`.
    fn layout(&self, world: &Path) -> Layout {
        Layout::of(&self.values, &self.files, world)
    }

    // Append an entry of the world file, returning the key minted for it.
    pub(crate) fn push(&mut self, value: serde_json::Value) -> EntryId {
        self.insert(self.values.len(), value, None)
    }

    // Add an entry to `file` in the list read from the world file `world`,
    // returning the key minted for it. A file the world includes takes it at
    // the end of its entries. Any other file is included first, by an `Include`
    // line appended to the world file naming it relative to the world file's
    // directory; the save that writes the line creates the file. A file the
    // world includes more than once, or the world file itself, takes nothing.
    pub(crate) fn push_to(
        &mut self,
        value: serde_json::Value,
        file: &Path,
        world: &Path,
    ) -> Result<EntryId, String> {
        if paths::same_file(file, world) {
            return Err(format!("{} is the world file", paths::shown(file)));
        }
        let layout = self.layout(world);
        let mut opened = layout
            .openings
            .iter()
            .filter(|o| paths::same_file(&o.target, file));
        match (opened.next(), opened.next()) {
            (Some(o), None) => {
                let tag = Arc::from(o.target.as_path());
                Ok(self.insert(o.end, value, Some(tag)))
            }
            (Some(_), Some(_)) => Err(format!("{} is included more than once", paths::shown(file))),
            (None, _) => {
                let dir = world.parent().unwrap_or_else(|| Path::new(""));
                let path = paths::relative_from(dir, file);
                let tag: Arc<Path> = Arc::from(paths::include_target(world, &path));
                let line = serde_json::json!({"type": "Include", "args": {"path": path}});
                self.insert(self.values.len(), line, None);
                Ok(self.insert(self.values.len(), value, Some(tag)))
            }
        }
    }

    fn insert(
        &mut self,
        index: usize,
        value: serde_json::Value,
        file: Option<Arc<Path>>,
    ) -> EntryId {
        let key = EntryId::mint();
        self.values.insert(index, value);
        self.keys.insert(index, key);
        self.files.insert(index, file);
        key
    }

    // Drop the entry at `index`, returning it. `None` past the end.
    pub(crate) fn remove(&mut self, index: usize) -> Option<serde_json::Value> {
        if index >= self.values.len() {
            return None;
        }
        self.keys.remove(index);
        self.files.remove(index);
        Some(self.values.remove(index))
    }

    // Move the entry at `from` so it sits at `to`, its key and file with it.
    // Out-of-range positions leave the list as it is.
    pub(crate) fn move_entry(&mut self, from: usize, to: usize) {
        if from >= self.values.len() || to >= self.values.len() {
            return;
        }
        let value = self.values.remove(from);
        let key = self.keys.remove(from);
        let file = self.files.remove(from);
        self.values.insert(to, value);
        self.keys.insert(to, key);
        self.files.insert(to, file);
    }

    // The included file the entry at `index` was read from; `None` for an
    // entry of the world file itself.
    pub(crate) fn included_from(&self, index: usize) -> Option<&Path> {
        self.files.get(index)?.as_deref()
    }

    // Whether the entry at `index` is not the editor's to change: an `Include`
    // line, or an entry of a file the world includes more than once.
    pub(crate) fn is_read_only(&self, index: usize) -> bool {
        self.values.get(index).is_some_and(is_include) || self.is_repeated(index)
    }

    fn is_repeated(&self, index: usize) -> bool {
        self.included_from(index)
            .is_some_and(|f| self.repeated.iter().any(|r| **r == *f))
    }

    // Why this list, edited from `before`, cannot stand, or `None` when it can.
    // `world` is the world file both were read from.
    pub(crate) fn refusal(&self, before: &Self, world: &Path) -> Option<Refusal> {
        let changed = (0..before.len()).find(|&i| {
            before.is_read_only(i) && self.by_key(before.keys[i]) != Some(&before.values[i])
        });
        if let Some(i) = changed {
            return Some(if is_include(&before.values[i]) {
                Refusal::IncludeLine(i)
            } else {
                Refusal::Repeated(i)
            });
        }
        self.layout(world).misplaced.map(Refusal::Moved)
    }

    // The text of `file` (`None` for the world file): its own entries in list
    // order, `Include` lines among them.
    pub(crate) fn file_text(&self, file: Option<&Path>) -> std::io::Result<String> {
        let own: Vec<serde_json::Value> = self.entries_of(file).cloned().collect();
        write_world_jsonl(&own)
    }

    fn entries_of<'a>(
        &'a self,
        file: Option<&'a Path>,
    ) -> impl Iterator<Item = &'a serde_json::Value> + 'a {
        self.values
            .iter()
            .zip(&self.files)
            .filter(move |(_, f)| f.as_deref() == file)
            .map(|(value, _)| value)
    }

    // The files whose entries differ from `saved`'s: what a save writes. The
    // included files come first and the world file last, so the world never
    // names a file a failed save left unwritten. A file the world includes more
    // than once is never among them.
    pub(crate) fn unsaved_files(&self, saved: &Self) -> Vec<Option<Arc<Path>>> {
        let mut files: Vec<Option<Arc<Path>>> = Vec::new();
        for file in self.files.iter().chain(&saved.files).flatten() {
            if !files.iter().flatten().any(|f| f == file) {
                files.push(Some(Arc::clone(file)));
            }
        }
        files.push(None);
        files.retain(|file| {
            let file = file.as_deref();
            let repeated = file.is_some_and(|f| self.repeated.iter().any(|r| **r == *f));
            !repeated && !self.entries_of(file).eq(saved.entries_of(file))
        });
        files
    }

    // The key of the entry at `index`.
    pub(crate) fn key_at(&self, index: usize) -> Option<EntryId> {
        self.keys.get(index).copied()
    }

    // Where `key`'s entry currently sits, or `None` once it is gone.
    pub(crate) fn index_of(&self, key: EntryId) -> Option<usize> {
        self.keys.iter().position(|k| *k == key)
    }

    // `key`'s entry, or `None` once it is gone.
    pub(crate) fn by_key(&self, key: EntryId) -> Option<&serde_json::Value> {
        self.values.get(self.index_of(key)?)
    }

    // `key`'s entry for in-place edit.
    pub(crate) fn by_key_mut(&mut self, key: EntryId) -> Option<&mut serde_json::Value> {
        let index = self.index_of(key)?;
        self.values.get_mut(index)
    }
}

// The text a build compiles `entries` from: every entry but the `Include` lines,
// whose files' entries the list already holds in their place.
pub(crate) fn build_text(entries: &[serde_json::Value]) -> std::io::Result<String> {
    let inlined: Vec<serde_json::Value> = entries
        .iter()
        .filter(|entry| !is_include(entry))
        .cloned()
        .collect();
    write_world_jsonl(&inlined)
}

impl From<Vec<serde_json::Value>> for EntryList {
    fn from(values: Vec<serde_json::Value>) -> Self {
        Self::new(values)
    }
}

// Session keys are not authored content: a list is equal to another when the
// two would write the same files.
impl PartialEq for EntryList {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values && self.files == other.files
    }
}

impl Deref for EntryList {
    type Target = [serde_json::Value];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

// Only the values are reachable: a slice cannot change the length, so the key
// column stays in step.
impl DerefMut for EntryList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn list(names: &[&str]) -> EntryList {
        EntryList::new(
            names
                .iter()
                .map(|n| json!({"type": "Prop", "args": {"$id": n}}))
                .collect(),
        )
    }

    #[test]
    fn every_entry_gets_its_own_key_and_no_two_lists_share_one() {
        let a = list(&["x", "y"]);
        let b = list(&["x", "y"]);
        assert_ne!(a.key_at(0), a.key_at(1), "two entries, two keys");
        assert_ne!(
            a.key_at(0),
            b.key_at(0),
            "a second list mints its own keys, so a key means one entry"
        );
    }

    #[test]
    fn a_moved_entry_keeps_its_key() {
        let mut l = list(&["a", "b", "c"]);
        let c = l.key_at(2).unwrap();
        l.move_entry(2, 0);
        assert_eq!(l.index_of(c), Some(0));
        assert_eq!(l[0]["args"]["$id"], "c");
        assert_eq!(l[1]["args"]["$id"], "a");
        l.move_entry(0, 9);
        assert_eq!(l.index_of(c), Some(0), "out of range moves nothing");
    }

    #[test]
    fn a_key_survives_an_insert_and_a_removal_before_it() {
        let mut l = list(&["a", "b"]);
        let b = l.key_at(1).unwrap();
        l.push(json!({"type": "Prop", "args": {"$id": "c"}}));
        assert_eq!(l.index_of(b), Some(1));
        l.remove(0);
        assert_eq!(l.index_of(b), Some(0), "b shifted down but is still b");
        assert_eq!(l.by_key(b).unwrap()["args"]["$id"], "b");
    }

    #[test]
    fn a_key_survives_a_rename_of_its_entry() {
        let mut l = list(&["old"]);
        let key = l.key_at(0).unwrap();
        l.by_key_mut(key).unwrap()["args"]["$id"] = json!("new");
        assert_eq!(l.by_key(key).unwrap()["args"]["$id"], "new");
    }

    #[test]
    fn a_removed_entrys_key_stops_resolving() {
        let mut l = list(&["a", "b"]);
        let a = l.key_at(0).unwrap();
        assert_eq!(l.remove(0).unwrap()["args"]["$id"], "a");
        assert_eq!(l.index_of(a), None);
        assert_eq!(l.by_key(a), None);
        assert_eq!(l.remove(5), None, "past the end removes nothing");
    }

    #[test]
    fn a_reused_position_does_not_reuse_the_key() {
        let mut l = list(&["a"]);
        let first = l.key_at(0).unwrap();
        l.remove(0);
        let second = l.push(json!({"type": "Prop", "args": {"$id": "a"}}));
        assert_ne!(first, second, "the old key must not address the new entry");
        assert_eq!(l.index_of(first), None);
    }

    #[test]
    fn equality_ignores_the_keys() {
        let a = list(&["x"]);
        let b = list(&["x"]);
        assert_ne!(a.key_at(0), b.key_at(0));
        assert_eq!(a, b, "same authored content, different session keys");

        let c = list(&["y"]);
        assert_ne!(a, c);
    }

    #[test]
    fn the_values_read_and_edit_as_a_slice() {
        let mut l = list(&["a", "b"]);
        let key = l.key_at(1);
        assert_eq!(l.len(), 2);
        assert_eq!(l.iter().count(), 2);
        l[1]["type"] = json!("Sphere");
        assert_eq!(l[1]["type"], "Sphere");
        assert_eq!(
            l.key_at(1),
            key,
            "an in-place edit leaves the entry's key alone"
        );
    }

    fn sourced(entry: serde_json::Value, file: Option<&str>) -> SourcedEntry {
        SourcedEntry {
            entry,
            file: file.map(PathBuf::from),
        }
    }

    fn include(path: &str) -> serde_json::Value {
        json!({"type": "Include", "args": {"path": path}})
    }

    const WORLD: &str = "world.jsonl";

    // `Scene`, an `Include` line, the entry it brought in from `lights.jsonl`,
    // then `Prop`.
    fn included() -> EntryList {
        EntryList::with_includes(
            vec![
                sourced(json!({"type": "Scene", "args": {}}), None),
                sourced(include("lights.jsonl"), None),
                sourced(
                    json!({"type": "PointLight", "args": {}}),
                    Some("lights.jsonl"),
                ),
                sourced(json!({"type": "Prop", "args": {}}), None),
            ],
            Path::new(WORLD),
        )
    }

    fn world() -> &'static Path {
        Path::new(WORLD)
    }

    #[test]
    fn only_the_include_line_is_read_only() {
        let l = included();
        let read_only: Vec<bool> = (0..l.len()).map(|i| l.is_read_only(i)).collect();
        assert_eq!(read_only, [false, true, false, false]);
        assert_eq!(l.included_from(2), Some(Path::new("lights.jsonl")));
        assert_eq!(l.included_from(3), None);
    }

    // The file column moves with its entry, and a pushed entry is the world
    // file's own.
    #[test]
    fn push_and_remove_keep_the_file_column_in_step() {
        let mut l = included();
        l.remove(0);
        assert_eq!(l.included_from(1), Some(Path::new("lights.jsonl")));
        l.push(json!({"type": "Prop", "args": {}}));
        assert_eq!(l.included_from(3), None);
        assert!(!l.is_read_only(3));
    }

    #[test]
    fn each_file_text_holds_that_files_own_entries() {
        let l = included();
        assert_eq!(
            l.file_text(None).unwrap(),
            "[\"Scene\",{}]\n[\"Include\",{\"path\":\"lights.jsonl\"}]\n[\"Prop\",{}]\n"
        );
        assert_eq!(
            l.file_text(Some(Path::new("lights.jsonl"))).unwrap(),
            "[\"PointLight\",{}]\n"
        );
    }

    #[test]
    fn the_build_text_keeps_what_was_inlined_and_drops_the_include_line() {
        assert_eq!(
            build_text(&included()).unwrap(),
            "[\"Scene\",{}]\n[\"PointLight\",{}]\n[\"Prop\",{}]\n"
        );
    }

    #[test]
    fn an_included_entry_edits_but_an_include_line_does_not() {
        let before = included();
        let mut after = before.clone();
        after[2]["args"]["intensity"] = json!(2.0);
        after[3]["args"]["mesh"] = json!("m");
        assert_eq!(after.refusal(&before, world()), None);

        let mut after = before.clone();
        after[1]["args"]["path"] = json!("other.jsonl");
        assert_eq!(
            after.refusal(&before, world()),
            Some(Refusal::IncludeLine(1))
        );

        let mut after = before.clone();
        after.remove(1);
        assert_eq!(
            after.refusal(&before, world()),
            Some(Refusal::IncludeLine(1))
        );
    }

    #[test]
    fn an_entry_moved_out_of_its_files_run_is_refused() {
        let before = included();
        let mut after = before.clone();
        after.move_entry(2, 0);
        assert_eq!(after.refusal(&before, world()), Some(Refusal::Moved(0)));

        let mut after = before.clone();
        after.move_entry(3, 0);
        assert_eq!(
            after.refusal(&before, world()),
            None,
            "the world's own entries reorder freely"
        );
    }

    // `a.jsonl` twice: each copy would claim the one file.
    fn repeated() -> EntryList {
        EntryList::with_includes(
            vec![
                sourced(include("a.jsonl"), None),
                sourced(json!({"type": "Prop", "args": {}}), Some("a.jsonl")),
                sourced(include("a.jsonl"), None),
                sourced(json!({"type": "Prop", "args": {}}), Some("a.jsonl")),
            ],
            world(),
        )
    }

    #[test]
    fn a_file_included_twice_is_read_only_and_never_saved() {
        let before = repeated();
        assert!(before.is_read_only(1) && before.is_read_only(3));
        let mut after = before.clone();
        after[3]["args"]["mesh"] = json!("m");
        assert_eq!(after.refusal(&before, world()), Some(Refusal::Repeated(3)));
        assert!(after.unsaved_files(&before).is_empty());
        assert!(
            after
                .push_to(json!({"type": "Prop"}), Path::new("a.jsonl"), world())
                .unwrap_err()
                .contains("more than once")
        );
    }

    #[test]
    fn unsaved_files_are_the_changed_ones_world_last() {
        let saved = included();
        assert!(saved.unsaved_files(&saved).is_empty());

        let mut l = saved.clone();
        l[2]["args"]["intensity"] = json!(2.0);
        assert_eq!(
            l.unsaved_files(&saved),
            [Some(Arc::from(Path::new("lights.jsonl")))]
        );

        l.push(json!({"type": "Prop", "args": {}}));
        assert_eq!(
            l.unsaved_files(&saved),
            [Some(Arc::from(Path::new("lights.jsonl"))), None]
        );
    }

    #[test]
    fn a_file_emptied_of_its_entries_is_still_saved() {
        let saved = included();
        let mut l = saved.clone();
        l.remove(2);
        assert_eq!(
            l.unsaved_files(&saved),
            [Some(Arc::from(Path::new("lights.jsonl")))]
        );
        assert_eq!(l.file_text(Some(Path::new("lights.jsonl"))).unwrap(), "");
    }

    #[test]
    fn an_entry_pushed_to_an_included_file_ends_its_run() {
        let before = included();
        let mut l = before.clone();
        let key = l
            .push_to(
                json!({"type": "SpotLight", "args": {}}),
                Path::new("./lights.jsonl"),
                world(),
            )
            .unwrap();
        assert_eq!(l.index_of(key), Some(3));
        assert_eq!(l.included_from(3), Some(Path::new("lights.jsonl")));
        assert_eq!(l.refusal(&before, world()), None);
    }

    // A nested file's run ends inside its parent's, so the parent's entry goes
    // after both.
    #[test]
    fn an_entry_pushed_to_a_file_with_a_nested_include_ends_after_it() {
        let mut l = EntryList::with_includes(
            vec![
                sourced(include("a.jsonl"), None),
                sourced(include("b.jsonl"), Some("a.jsonl")),
                sourced(json!({"type": "Prop"}), Some("b.jsonl")),
                sourced(json!({"type": "Scene"}), None),
            ],
            world(),
        );
        let key = l
            .push_to(json!({"type": "Sphere"}), Path::new("a.jsonl"), world())
            .unwrap();
        assert_eq!(l.index_of(key), Some(3));
        assert_eq!(
            l.file_text(Some(Path::new("a.jsonl"))).unwrap(),
            "[\"Include\",{\"path\":\"b.jsonl\"}]\n[\"Sphere\",{}]\n"
        );
    }

    #[test]
    fn an_entry_pushed_to_a_new_file_includes_it_from_the_world() {
        let world = Path::new("worlds/w.jsonl");
        let mut l = EntryList::new(vec![json!({"type": "Scene", "args": {}})]);
        let saved = l.clone();
        let file = Path::new("assets/ground/a.jsonl");
        let first = l.push_to(json!({"type": "Prop"}), file, world).unwrap();
        let second = l.push_to(json!({"type": "Sphere"}), file, world).unwrap();
        assert_eq!(l.index_of(first), Some(2));
        assert_eq!(
            l.index_of(second),
            Some(3),
            "the second joins the first's run"
        );
        assert_eq!(l[1], include("../assets/ground/a.jsonl"));
        let tag = Path::new("worlds/../assets/ground/a.jsonl");
        assert_eq!(l.included_from(2), Some(tag), "tagged as a reload tags it");
        assert_eq!(
            l.unsaved_files(&saved),
            [Some(Arc::from(tag)), None],
            "the new file is written ahead of the line naming it"
        );
        assert!(
            l.push_to(json!({"type": "Prop"}), Path::new("worlds/w.jsonl"), world)
                .is_err()
        );
    }

    #[test]
    fn equality_counts_the_file_an_entry_lives_in() {
        let mut a = included();
        let b = a.clone();
        a.files[2] = None;
        assert_ne!(a, b);
    }
}
