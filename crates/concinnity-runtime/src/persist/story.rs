// Story saves kept as files under the host's save directory: the auto-save is
// `auto`, manual slots are `save1`.. (1-based on disk, matching the slot-menu
// labels). The save's shape is `concinnity_engine::story::StorySave`; what this
// owns is the files, written off the sim thread. A preview session saves to
// memory instead, so the save UI works without touching the player's saves and
// every session starts fresh.

use concinnity_engine::story::{SaveSlot, StorySave, StoryStore};
use concinnity_host::store::paths::StateTree;
use concinnity_host::store::writer::FileWriter;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

// The store a session saves to: the player's save files, or for a preview
// session (`transient`) a sandbox that lives as long as the store.
pub(crate) fn for_session(tree: &StateTree, transient: bool) -> Box<dyn StoryStore> {
    if transient {
        Box::<SandboxStore>::default()
    } else {
        Box::new(FileStore::at(&tree.saves_dir()))
    }
}

#[derive(Debug)]
struct FileStore {
    dir: PathBuf,
    writer: FileWriter,
}

impl FileStore {
    fn at(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            writer: FileWriter::new("cn-story-save"),
        }
    }

    fn path(&self, slot: SaveSlot) -> PathBuf {
        match slot {
            SaveSlot::Auto => self.dir.join("auto"),
            SaveSlot::Manual(i) => self.dir.join(format!("save{}", i + 1)),
        }
    }
}

impl StoryStore for FileStore {
    fn read(&self, slot: SaveSlot) -> Option<StorySave> {
        self.writer.flush();
        concinnity_host::store::cbor::read(&self.path(slot), "story: save")
    }

    fn write(&self, slot: SaveSlot, save: &StorySave) {
        self.writer.write_cbor(self.path(slot), save, "story:");
    }

    fn remove(&self, slot: SaveSlot) {
        self.writer.remove(self.path(slot));
    }

    fn exists(&self, slot: SaveSlot) -> bool {
        self.writer.flush();
        self.path(slot).exists()
    }
}

// A preview session's saves, held in memory: a rebuilt preview world builds a
// fresh store, so nothing from one session reaches the next or the disk.
#[derive(Debug, Default)]
struct SandboxStore(Mutex<BTreeMap<SaveSlot, StorySave>>);

impl SandboxStore {
    fn slots(&self) -> MutexGuard<'_, BTreeMap<SaveSlot, StorySave>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl StoryStore for SandboxStore {
    fn read(&self, slot: SaveSlot) -> Option<StorySave> {
        self.slots().get(&slot).cloned()
    }

    fn write(&self, slot: SaveSlot, save: &StorySave) {
        self.slots().insert(slot, save.clone());
    }

    fn remove(&self, slot: SaveSlot) {
        self.slots().remove(&slot);
    }

    fn exists(&self, slot: SaveSlot) -> bool {
        self.slots().contains_key(&slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn save(slug: &str) -> StorySave {
        StorySave {
            slug: slug.to_string(),
            page: 2,
            vars: BTreeMap::from([("asked".to_string(), 1), ("gold".to_string(), 7)]),
        }
    }

    // Each slot round-trips through its own file; a missing or unreadable file
    // reads as nothing, and a removal drops it.
    #[test]
    fn saves_round_trip_through_their_files() {
        let tree = concinnity_testing::TempTree::new();
        let store = FileStore::at(&tree.join("saves"));
        assert!(store.read(SaveSlot::Auto).is_none());
        store.write(SaveSlot::Auto, &save("meadow"));
        store.write(SaveSlot::Manual(1), &save("cave"));
        assert_eq!(store.read(SaveSlot::Auto), Some(save("meadow")));
        assert!(store.exists(SaveSlot::Manual(1)));
        assert!(tree.join("saves/save2").exists());
        assert!(!store.exists(SaveSlot::Manual(0)));

        tree.write("saves/auto", b"not cbor");
        assert!(store.read(SaveSlot::Auto).is_none());
        store.remove(SaveSlot::Manual(1));
        assert!(!store.exists(SaveSlot::Manual(1)));
    }

    // A preview session saves to memory: its slots round-trip, and the
    // player's save files are neither read nor written.
    #[test]
    fn a_preview_session_leaves_the_save_files_alone() {
        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());
        for_session(&tree, false).write(SaveSlot::Auto, &save("real"));

        let preview = for_session(&tree, true);
        assert!(preview.read(SaveSlot::Auto).is_none());
        preview.write(SaveSlot::Auto, &save("preview"));
        preview.write(SaveSlot::Manual(0), &save("slot"));
        assert_eq!(preview.read(SaveSlot::Auto), Some(save("preview")));
        assert!(preview.exists(SaveSlot::Manual(0)));
        preview.remove(SaveSlot::Manual(0));
        assert!(!preview.exists(SaveSlot::Manual(0)));
        drop(preview);

        assert_eq!(
            for_session(&tree, false).read(SaveSlot::Auto),
            Some(save("real"))
        );
        assert!(!dir.join("saves/save1").exists());
        assert!(for_session(&tree, true).read(SaveSlot::Auto).is_none());
    }
}
