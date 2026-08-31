//! Stable roots a test suite keeps between runs.
//!
//! Everything else a test writes belongs in a [`TempTree`](crate::TempTree)
//! that goes when the test does. Two things do not fit that:
//!
//! - A **cache** exists to be kept. The cook's store exists to avoid
//!   recompiling a shader that has not changed, and a root dying with the test
//!   recompiles it for every test instead of every change. Under coverage
//!   instrumentation that is the dominant cost, since what repeats is
//!   compilation.
//! - **State** a suite installs process-wide, which no single test owns and so
//!   no single test can drop. A blob a build wrote is not a cache: a persisted
//!   one makes "the build wrote one" assert nothing.
//!
//! So they are two roots, not one, and the difference between them is the
//! clearing: [`shared_cache_dir`] persists, [`shared_state_dir`] is emptied on
//! its first call in a process. A caller wanting both points a state tree's
//! cache root at the first and its content root at the second, which is what
//! "a warm cache and a fresh build" is.
//!
//! Both stay bounded -- one directory per name per machine -- and concurrent
//! test processes may share the cache: the worst case is two of them computing
//! the same entry.

use std::path::{Path, PathBuf};

// One parent for both kinds, so a machine has a single place to clear.
const PARENT: &str = "concinnity-test-cache";

// The parent every shared root lives under, so a machine clearing them has one
// place to look: `<system temp>/concinnity-test-cache`.
fn shared_parent() -> PathBuf {
    std::env::temp_dir().join(PARENT)
}

/// The stable cache root called `name`, created if it is not there yet and
/// never cleared.
///
/// `name` separates one suite's cache from another's; it is a single path
/// segment, not a path.
///
/// # Panics
///
/// If the directory cannot be created.
pub fn shared_cache_dir(name: &str) -> PathBuf {
    ensure(name)
}

/// The stable state root called `name`, emptied on the first call in a process
/// so a run inherits nothing from the last one.
///
/// Call it from every test that needs the root, not once: the clearing happens
/// on the first call and the rest are a lookup.
///
/// # Panics
///
/// If the directory cannot be created.
pub fn shared_state_dir(name: &str) -> PathBuf {
    let dir = ensure(name);

    // Once per process, so tests share what this run writes and inherit
    // nothing from the last.
    static CLEARED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CLEARED.get_or_init(|| clear(&dir));
    dir
}

// The named root under the shared parent, created if absent.
fn ensure(name: &str) -> PathBuf {
    assert!(
        !name.is_empty() && !name.contains(['/', '\\', '.']),
        "a shared root's name is one path segment, not a path: {name:?}"
    );
    let dir = shared_parent().join(name);
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("create the shared root {}: {e}", dir.display()));
    dir
}

// Remove every entry under `dir`, leaving the directory itself.
fn clear(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let _ = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A state root starts each process empty: a blob or a session file the last
    // run wrote must not make this run's assertions pass.
    #[test]
    fn a_state_root_keeps_nothing_across_a_process() {
        let dir = shared_state_dir("harness-state-test");
        std::fs::create_dir_all(dir.join("data")).expect("data dir");
        std::fs::write(dir.join("data").join("0"), b"blob").expect("blob");
        std::fs::write(dir.join("editor"), b"session").expect("session");

        // What the next process would do on its first call.
        clear(&dir);

        assert!(!dir.join("data").exists(), "build output is not a cache");
        assert!(!dir.join("editor").exists(), "session state is not a cache");
        assert!(dir.is_dir(), "the root itself survives");
    }

    // A cache root is never cleared: that is the whole reason it is a root of
    // its own rather than a subdirectory of the state.
    #[test]
    fn a_cache_root_survives() {
        let dir = shared_cache_dir("harness-cache-test");
        std::fs::create_dir_all(dir.join("cache")).expect("cache dir");
        std::fs::write(dir.join("cache").join("1"), b"warm").expect("cache entry");

        assert_eq!(
            std::fs::read(
                shared_cache_dir("harness-cache-test")
                    .join("cache")
                    .join("1")
            )
            .expect("cache kept"),
            b"warm"
        );
    }

    // The two are different directories, so pointing a state tree's cache root
    // at one and its content root at the other separates them completely.
    #[test]
    fn a_cache_root_and_a_state_root_are_different_places() {
        let cache = shared_cache_dir("harness-split-cache");
        let state = shared_state_dir("harness-split-state");

        assert_ne!(cache, state);
        assert_eq!(cache.parent(), state.parent());
        assert_eq!(cache.parent(), Some(shared_parent().as_path()));
    }

    #[test]
    fn the_same_name_is_the_same_directory_every_time() {
        let first = shared_cache_dir("harness-self-test");
        let second = shared_cache_dir("harness-self-test");

        assert_eq!(first, second, "a cache is reused, not recreated");
        assert!(first.is_dir());
        assert!(first.starts_with(shared_parent()));
    }

    #[test]
    #[should_panic(expected = "one path segment")]
    fn a_name_with_a_separator_is_rejected() {
        shared_cache_dir("escapes/upward");
    }

    #[test]
    #[should_panic(expected = "one path segment")]
    fn a_name_with_a_dot_is_rejected() {
        shared_state_dir("..");
    }
}
