//! `EditorHook`: adding an entry to the file it belongs in. Large editor-written
//! data keeps the world file thin by living in its own file under the project's
//! `assets/`, which the world pulls in with an `Include` line.

use std::path::{Path, PathBuf};

use super::{EditorHook, EntryId};
use concinnity_cook::build_only::include::rebase_files;

use crate::editor::entry_list::Origin;
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
    // Add `value` to `home`'s file and return its key, adding an `Include` line
    // if needed. The caller commits.
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
        let origin = Origin {
            world: Path::new(self.entries_from.as_deref().unwrap_or(&self.world_path)),
            root: &self.content_root,
        };
        match self.entries.push_to(value, &file, origin) {
            Ok(key) => Some(key),
            Err(reason) => {
                self.notifier.push(notify::Level::Error, &reason);
                None
            }
        }
    }

    // `add_entry` for `value` written in a file whose paths resolve from
    // `from`: its file paths are rebased onto the root of `home`'s file first,
    // so they name the same files there.
    pub(super) fn add_entry_from(
        &mut self,
        mut value: serde_json::Value,
        from: &Path,
        home: EntryHome<'_>,
    ) -> Option<EntryId> {
        let file = match home {
            EntryHome::World => None,
            EntryHome::File(file) => Some(file.to_path_buf()),
            EntryHome::Asset(name) => Some(asset_file(name)),
        };
        let to = self.entries.file_root(file.as_deref(), self.world_origin());
        rebase_files(&mut value, from, &to);
        self.add_entry(value, home)
    }
}

// Where an asset file `name` sits: under the open project's `assets/`, or an
// `assets/` beside the working directory when no project is open.
fn asset_file(name: &str) -> PathBuf {
    crate::project::assets_dir()
        .unwrap_or_else(|| PathBuf::from("assets"))
        .join(name)
}
