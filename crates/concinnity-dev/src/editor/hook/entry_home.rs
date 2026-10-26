//! `EditorHook`: adding an entry to the file it belongs in. Large editor-written
//! data keeps the world file thin by living in its own file under the project's
//! `assets/`, which the world pulls in with an `Include` line.

use std::path::{Path, PathBuf};

use super::{EditorHook, EntryId};
use crate::editor::notify;

// The file an added entry goes into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryHome<'a> {
    // The world file itself.
    World,
    // A file the world includes, or one it will include from now on.
    File(&'a Path),
    // A file under the project's `assets/`, by its path there.
    #[expect(
        dead_code,
        reason = "no editor tool writes its own asset file yet; remove once one adds entries this way"
    )]
    Asset(&'a str),
}

impl EditorHook {
    // Add `value` to the file `home` names and return its key. A file the world
    // does not include yet gets an `Include` line in the world file, and the
    // next SAVE creates it. Nothing is committed, so the caller's commit makes
    // the entry and any new line one undo step. A file that cannot take the
    // entry (the world includes it more than once) says why and takes nothing.
    pub(super) fn add_entry(
        &mut self,
        value: serde_json::Value,
        home: EntryHome<'_>,
    ) -> Option<EntryId> {
        let file = match home {
            EntryHome::World => return Some(self.entries.push(value)),
            EntryHome::File(file) => file.to_path_buf(),
            EntryHome::Asset(name) => asset_file(name),
        };
        match self
            .entries
            .push_to(value, &file, Path::new(&self.world_path))
        {
            Ok(key) => Some(key),
            Err(reason) => {
                self.notifier.push(notify::Level::Error, &reason);
                None
            }
        }
    }
}

// Where an asset file `name` sits: under the open project's `assets/`, or an
// `assets/` beside the working directory when no project is open.
fn asset_file(name: &str) -> PathBuf {
    crate::project::assets_dir()
        .unwrap_or_else(|| PathBuf::from("assets"))
        .join(name)
}
