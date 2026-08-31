// Persisted behavior state, kept in a `state` file under the host's save
// directory. The state's shape and its keying are the system's (see
// `concinnity_core::behavior::BehaviorState`); what this owns is the file.

use std::path::{Path, PathBuf};

use concinnity_core::behavior::{BehaviorState, BehaviorStore};

#[derive(Debug)]
pub(crate) struct FileStore {
    dir: PathBuf,
}

impl FileStore {
    // `None` when no host installed a state root: behaviors run, saving does
    // not.
    pub(crate) fn new() -> Option<FileStore> {
        concinnity_host::store::paths::saves_dir().map(|dir| FileStore { dir })
    }

    #[cfg(test)]
    pub(crate) fn at(dir: &Path) -> FileStore {
        FileStore {
            dir: dir.to_path_buf(),
        }
    }
}

impl BehaviorStore for FileStore {
    fn read(&self) -> Option<BehaviorState> {
        crate::cbor_file::read(&state_file(&self.dir), "BehaviorSystem: saved state")
    }

    fn write(&self, state: &BehaviorState) {
        if let Err(e) = crate::cbor_file::write(&state_file(&self.dir), state) {
            tracing::warn!("BehaviorSystem: state save failed: {e}");
        }
    }
}

pub(crate) fn state_file(dir: &Path) -> PathBuf {
    dir.join("state")
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::BehaviorLiteral;

    #[test]
    fn state_round_trips_through_the_file() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.join("state");
        let store = FileStore::at(&dir);
        assert!(store.read().is_none(), "nothing written yet");

        let mut state = BehaviorState::default();
        state.vars.insert("score".into(), BehaviorLiteral::Int(12));
        state.fired.push((3, 0xfeed));
        store.write(&state);
        assert!(state_file(&dir).exists());

        let back = store.read().expect("state readable");
        assert_eq!(back.vars.get("score"), Some(&BehaviorLiteral::Int(12)));
        assert_eq!(back.fired, vec![(3, 0xfeed)]);
    }
}
