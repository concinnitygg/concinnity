//! The world.jsonl path a dev host is running. concinnity-dev's `run_interpreted`,
//! its editor entry and editor world switching set it once the world path is
//! resolved; `GraphicsSystem::init` reads it (only in a dev-loop launch) so the
//! Prop-transform hot-reload watcher knows which file to subscribe to.
//! world.jsonl discovery is authoring I/O that lives in `concinnity-cook`, which
//! the runtime does not link, so the dev host resolves the path and hands it in
//! rather than the engine looking it up. Left None for `cn run` and embedded
//! preview.

use std::sync::Mutex;

// Path to the world.jsonl the dev host is running, or None outside a dev host.
static WORLD_JSONL_PATH: Mutex<Option<String>> = Mutex::new(None);

/// Record the world.jsonl path the dev host resolved, so the hot-reload watcher
/// can subscribe to it. Called by the editor's `cn debug` / `cn editor` entry
/// before world build; the library only reads it.
pub fn set_world_jsonl_path(path: Option<String>) {
    *WORLD_JSONL_PATH.lock().unwrap() = path;
}

// The world.jsonl path the dev host handed in, or None outside a dev host. Read
// by `GraphicsSystem::init` to seed the Prop-transform reload watcher.
pub(crate) fn world_jsonl_path() -> Option<String> {
    WORLD_JSONL_PATH.lock().unwrap().clone()
}
