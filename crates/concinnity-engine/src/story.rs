// Story saves kept as files under the host's save directory: the auto-save is
// `auto`, manual slots are `save1`.. (1-based on disk, matching the slot-menu
// labels). The save's shape is `concinnity_core::story::StorySave`; what this
// owns is the files, written off the sim thread.

use crate::file_writer::FileWriter;
use concinnity_core::story::{SaveSlot, StorySave, StoryStore};
use concinnity_host::store::paths::StateTree;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct FileStore {
    dir: PathBuf,
    writer: FileWriter,
}

impl FileStore {
    // The store under `tree`. A preview session (`transient`) saves to a
    // sandbox emptied here, so the save UI works without touching the
    // player's saves and every session starts fresh.
    pub(crate) fn new(tree: &StateTree, transient: bool) -> FileStore {
        if !transient {
            return FileStore::at(&tree.saves_dir());
        }
        let preview = tree.preview_saves_dir();
        let _ = std::fs::remove_dir_all(&preview);
        FileStore::at(&preview)
    }

    fn at(dir: &Path) -> FileStore {
        FileStore {
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
        crate::cbor_file::read(&self.path(slot), "story: save")
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

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

    // A preview store writes to the emptied preview directory and leaves the
    // player's saves alone.
    #[test]
    fn a_preview_store_starts_empty_beside_the_real_saves() {
        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());
        FileStore::new(&tree, false).write(SaveSlot::Auto, &save("real"));
        let stale = tree.preview_saves_dir().join("save1");
        crate::file_writer::write_file(&stale, b"stale").unwrap();
        let preview = FileStore::new(&tree, true);
        assert!(!preview.exists(SaveSlot::Manual(0)));
        assert!(preview.read(SaveSlot::Auto).is_none());
        preview.write(SaveSlot::Auto, &save("preview"));
        preview.writer.flush();
        assert!(tree.preview_saves_dir().join("auto").exists());
        assert_eq!(
            FileStore::new(&tree, false).read(SaveSlot::Auto),
            Some(save("real"))
        );
    }
}
