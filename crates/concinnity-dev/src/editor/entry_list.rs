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
//! are the build's. Beside each entry sits the included file it came from; a
//! save writes only the entries that have none, which keeps each `Include` line
//! and drops what it inlined.

use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use concinnity_cook::authoring::world::write_world_jsonl;
use concinnity_cook::build_only::include::{SourcedEntry, is_include};

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

// The working entry list: the parsed world.jsonl values, each with its key
// and the included file it was read from.
//
// Derefs to the values, so every read and every in-place value edit works as
// it does on a `Vec`. The two operations that change the length go through
// `push` and `remove`, which keep the other columns in step.
#[derive(Debug, Default, Clone)]
pub(crate) struct EntryList {
    values: Vec<serde_json::Value>,
    keys: Vec<EntryId>,
    files: Vec<Option<Arc<Path>>>,
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
        }
    }

    // Take a world whose includes are resolved, keeping where each entry came
    // from.
    pub(crate) fn with_includes(sourced: Vec<SourcedEntry>) -> Self {
        let mut list = Self::default();
        for SourcedEntry { entry, file } in sourced {
            list.values.push(entry);
            list.keys.push(EntryId::mint());
            list.files.push(file.map(PathBuf::into));
        }
        list
    }

    // Append an entry of the world file, returning the key minted for it.
    pub(crate) fn push(&mut self, value: serde_json::Value) -> EntryId {
        let key = EntryId::mint();
        self.values.push(value);
        self.keys.push(key);
        self.files.push(None);
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

    // The included file the entry at `index` was read from; `None` for an
    // entry of the world file itself.
    pub(crate) fn included_from(&self, index: usize) -> Option<&Path> {
        self.files.get(index)?.as_deref()
    }

    // Whether the entry at `index` is not the editor's to change: it was read
    // from an included file, or it is an `Include` line, whose inlined entries
    // would go stale under an edit.
    pub(crate) fn is_read_only(&self, index: usize) -> bool {
        self.included_from(index).is_some() || self.values.get(index).is_some_and(is_include)
    }

    // The first read-only entry of `before` this list no longer holds as it
    // was: edited, or removed.
    pub(crate) fn changed_read_only(&self, before: &EntryList) -> Option<usize> {
        (0..before.len()).find(|&i| {
            before.is_read_only(i) && self.by_key(before.keys[i]) != Some(&before.values[i])
        })
    }

    // The world file's own text: every entry that was not read from an
    // included file, `Include` lines among them.
    pub(crate) fn file_text(&self) -> std::io::Result<String> {
        let own: Vec<serde_json::Value> = self
            .values
            .iter()
            .zip(&self.files)
            .filter(|(_, file)| file.is_none())
            .map(|(value, _)| value.clone())
            .collect();
        write_world_jsonl(&own)
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
// two would write the same world.jsonl.
impl PartialEq for EntryList {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
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

    // `Scene`, an `Include` line, the entry it brought in from `lights.jsonl`,
    // then `Prop`.
    fn included() -> EntryList {
        let sourced = |entry, file: Option<&str>| SourcedEntry {
            entry,
            file: file.map(PathBuf::from),
        };
        EntryList::with_includes(vec![
            sourced(json!({"type": "Scene", "args": {}}), None),
            sourced(
                json!({"type": "Include", "args": {"path": "lights.jsonl"}}),
                None,
            ),
            sourced(
                json!({"type": "PointLight", "args": {}}),
                Some("lights.jsonl"),
            ),
            sourced(json!({"type": "Prop", "args": {}}), None),
        ])
    }

    #[test]
    fn included_entries_and_include_lines_are_read_only() {
        let l = included();
        let read_only: Vec<bool> = (0..l.len()).map(|i| l.is_read_only(i)).collect();
        assert_eq!(read_only, [false, true, true, false]);
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
    fn the_file_text_keeps_the_include_line_and_drops_what_it_inlined() {
        assert_eq!(
            included().file_text().unwrap(),
            "[\"Scene\",{}]\n[\"Include\",{\"path\":\"lights.jsonl\"}]\n[\"Prop\",{}]\n"
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
    fn a_changed_or_removed_read_only_entry_is_found() {
        let before = included();
        let mut after = before.clone();
        after[3]["args"]["mesh"] = json!("m");
        assert_eq!(
            after.changed_read_only(&before),
            None,
            "Prop is the file's own"
        );

        after[2]["args"]["intensity"] = json!(2.0);
        assert_eq!(after.changed_read_only(&before), Some(2));

        let mut after = before.clone();
        after.remove(1);
        assert_eq!(after.changed_read_only(&before), Some(1));
    }
}
