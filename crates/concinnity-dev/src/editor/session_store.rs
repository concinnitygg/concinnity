// src/editor/session_store.rs
//
// Persisted editor session state: one small CBOR file per project at
// `<writable state dir>/editor`, keyed by world file stem so several worlds in
// a project keep separate entries. Load tolerates a missing or unreadable
// file (fresh default); save is whole-file, the store is tiny.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::framing::CameraPose;

pub(crate) const BOOKMARK_SLOTS: usize = 9;

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SessionStore {
    #[serde(default)]
    pub worlds: BTreeMap<String, WorldSession>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct WorldSession {
    #[serde(default)]
    pub bookmarks: [Option<CameraPose>; BOOKMARK_SLOTS],
}

// The per-project store file, beside `saves/` and `settings`. `None` when the
// session has no project open, which leaves bookmarks in memory for the run.
pub(crate) fn default_path() -> Option<PathBuf> {
    crate::project::tree().map(|tree| tree.editor_session_path())
}

// The store key for a world file: its stem, so `world.jsonl` and a sibling
// `arena.jsonl` don't share bookmarks.
pub(crate) fn world_key(world_path: &str) -> String {
    Path::new(world_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| world_path.to_string())
}

pub(crate) fn load(path: &Path) -> SessionStore {
    let Ok(file) = std::fs::File::open(path) else {
        return SessionStore::default();
    };
    ciborium::from_reader(std::io::BufReader::new(file)).unwrap_or_default()
}

pub(crate) fn save(path: &Path, store: &SessionStore) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::new();
    ciborium::into_writer(store, &mut bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_the_file() {
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("editor");
        let mut store = SessionStore::default();
        let mut session = WorldSession::default();
        session.bookmarks[3] = Some(CameraPose {
            position: [1.0, 2.0, 3.0],
            yaw: 0.5,
            pitch: -0.25,
        });
        store.worlds.insert("world".into(), session);
        save(&path, &store).unwrap();
        let back = load(&path);
        let pose = back.worlds["world"].bookmarks[3].unwrap();
        assert_eq!(pose.position, [1.0, 2.0, 3.0]);
        assert_eq!((pose.yaw, pose.pitch), (0.5, -0.25));
        assert!(back.worlds["world"].bookmarks[0].is_none());
    }

    #[test]
    fn missing_or_corrupt_files_load_fresh() {
        let tree = concinnity_testing::TempTree::new();
        assert!(load(&tree.join("absent")).worlds.is_empty());
        let bad = tree.write("editor", b"not cbor");
        assert!(load(&bad).worlds.is_empty());
    }

    #[test]
    fn world_key_uses_the_stem() {
        assert_eq!(world_key("some/dir/world.jsonl"), "world");
        assert_eq!(world_key("arena.jsonl"), "arena");
    }
}
