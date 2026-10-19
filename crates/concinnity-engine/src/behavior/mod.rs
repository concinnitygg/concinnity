// Persisted behavior state, the one thing this crate lends
// `concinnity_core::behavior::BehaviorSystem`: a `state` file under the host's
// save directory. The state's shape and its keying are the system's (see
// `concinnity_core::behavior::BehaviorState`); what this owns is the file,
// written off the sim thread.

use crate::file_writer::FileWriter;
use concinnity_core::behavior::{BehaviorState, BehaviorStore};
use concinnity_host::store::paths::StateTree;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct FileStore {
    dir: PathBuf,
    writer: FileWriter,
}

impl FileStore {
    // `None` for a world with no state tree: behaviors run, saving does not.
    pub(crate) fn new(tree: Option<&StateTree>) -> Option<FileStore> {
        tree.map(|tree| FileStore::at(&tree.saves_dir()))
    }

    pub(crate) fn at(dir: &Path) -> FileStore {
        FileStore {
            dir: dir.to_path_buf(),
            writer: FileWriter::new("cn-behav-save"),
        }
    }
}

impl BehaviorStore for FileStore {
    fn read(&self) -> Option<BehaviorState> {
        self.writer.flush();
        crate::cbor_file::read(&state_file(&self.dir), "BehaviorSystem: saved state")
    }

    fn write(&self, state: &BehaviorState) {
        self.writer
            .write_cbor(state_file(&self.dir), state, "BehaviorSystem: state");
    }
}

fn state_file(dir: &Path) -> PathBuf {
    dir.join("state")
}

#[cfg(test)]
mod tests;
