//! The project a dev session works on.
//!
//! Every tier below this one takes its paths as arguments: the cook is handed
//! the tree it builds into, the engine's `App` carries the one it runs against,
//! and the two content-addressed caches are told which files they are. What
//! remains is that a dev session works on exactly one project for the length of
//! a process, and that the editor's panels, the hot-reload passes, and the
//! background workers all reach for it from places with no caller to thread it
//! down from.
//!
//! So the session holds it, and the `concinnity` binary [`open`]s it once at
//! startup from whatever directory it decided the project lives in. Nothing
//! here resolves a root: `open` is given one.

use std::sync::{Mutex, OnceLock};

use concinnity_host::store::paths::StateTree;

fn opened() -> &'static Mutex<Option<StateTree>> {
    static OPENED: OnceLock<Mutex<Option<StateTree>>> = OnceLock::new();
    OPENED.get_or_init(|| Mutex::new(None))
}

/// Open `tree` as this session's project, and point the build cache at the
/// segment it names. Until a host calls this the session has no project: every
/// build resolves no `assets/`, has nowhere to write blobs, and warms nothing.
pub fn open(tree: StateTree) {
    concinnity_cook::cache::anchor(&tree.build_cache_path());
    *opened().lock().unwrap() = Some(tree);
}

/// Close the session's project, leaving it with none.
pub fn close() {
    concinnity_cook::cache::clear_anchor();
    *opened().lock().unwrap() = None;
}

/// The session's project, or `None` when nothing opened one.
pub fn tree() -> Option<StateTree> {
    opened().lock().unwrap().clone()
}

/// An app that reads and writes under the open project: its blobs, settings,
/// saves and the caches it warms. Without a project the app still runs a world,
/// and everything it would persist does nothing.
pub(crate) fn app() -> concinnity_engine::App {
    let app = concinnity_engine::App::new();
    match tree() {
        Some(tree) => app.in_tree(tree),
        None => app,
    }
}

/// The open project, or the error a command reports when the session has none:
/// every build writes into a tree, so there is nothing sensible to do without
/// one.
pub(crate) fn require() -> std::io::Result<StateTree> {
    tree().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no project state directory to build into",
        )
    })
}

/// The `assets/` a bare source filename is resolved against: the open
/// project's, or `None` for a session with no project, which resolves nothing.
pub(crate) fn assets_dir() -> Option<std::path::PathBuf> {
    tree().map(|tree| tree.assets_dir())
}

/// The `data/` a build writes its blobs into, and a run reads them from.
pub(crate) fn data_dir() -> Option<std::path::PathBuf> {
    tree().map(|tree| tree.data_dir())
}

/// The world lock a build writes and a blob boot reads back.
pub(crate) fn world_lock_path() -> Option<std::path::PathBuf> {
    tree().map(|tree| tree.world_lock_path())
}

/// The `worlds/` a named world is looked up in.
pub(crate) fn worlds_dir() -> Option<std::path::PathBuf> {
    tree().map(|tree| tree.worlds_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Opening hands the session a tree and nothing else resolves one: the
    // directories below are the open project's, and closing leaves the session
    // with none rather than with a guess.
    #[test]
    fn opening_a_project_is_what_gives_the_session_its_directories() {
        // The one exclusive guard: this moves the session-wide project, which
        // every other test in this binary reads.
        let _guard = crate::test_support::lock();
        let dir = concinnity_testing::TempTree::new();

        close();
        assert_eq!(tree(), None);
        assert_eq!(assets_dir(), None);
        assert_eq!(data_dir(), None);
        assert_eq!(world_lock_path(), None);
        assert_eq!(worlds_dir(), None);
        assert_eq!(
            require().unwrap_err().kind(),
            std::io::ErrorKind::NotFound,
            "a build with no project reports it rather than writing somewhere"
        );

        open(StateTree::at(dir.path()));
        assert_eq!(
            tree().as_ref().map(StateTree::content_root),
            Some(dir.path())
        );
        assert_eq!(assets_dir(), Some(dir.path().join("assets")));
        assert_eq!(data_dir(), Some(dir.path().join("data")));
        assert_eq!(world_lock_path(), Some(dir.path().join("world-lock.json")));
        assert_eq!(worlds_dir(), Some(dir.path().join("worlds")));
        assert!(require().is_ok());

        // An app built for the session runs against that same tree.
        assert_eq!(app().state_tree(), tree().as_ref());

        // Leave the binary's other tests the project they expect.
        crate::test_support::isolate_state_dir();
    }
}
