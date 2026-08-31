//! A stable root for the caches a test suite wants to keep between runs.
//!
//! Everything else a test writes belongs in a [`TempTree`](crate::TempTree)
//! that goes when the test does. A cache is the exception: the cook's store
//! exists to avoid recompiling a shader that has not changed, and a root dying
//! with the test recompiles it for every test instead of every change. Under
//! coverage instrumentation that is the dominant cost, since what repeats is
//! compilation.
//!
//! Only the cache subdirectory survives, and the caller names it. A root that
//! persists is safe for entries keyed by their content and unsafe for anything
//! else, and both land here: the cook writes its build cache beside its blobs,
//! and a persisted blob makes "the build wrote one" assert nothing. So the
//! first call in a process clears everything else.
//!
//! It stays bounded because the store is content-addressed and
//! segment-budgeted, and concurrent test processes may share it: the worst
//! case is two of them computing the same entry.

use std::path::{Path, PathBuf};

// One parent for every named cache, so a machine has a single place to clear.
const PARENT: &str = "concinnity-test-cache";

// The parent every cache root lives under, so a machine clearing them has one
// place to look: `<system temp>/concinnity-test-cache`.
fn shared_cache_parent() -> PathBuf {
    std::env::temp_dir().join(PARENT)
}

/// The stable cache root called `name`, created if it is not there yet, with
/// everything except the `keep` subdirectory cleared once per process.
///
/// `name` separates one suite's cache from another's; it is a single path
/// segment, not a path. Call it from every test that needs the root, not once:
/// the clearing happens on the first call and the rest are a lookup.
///
/// `keep` names the subdirectory holding the cache. This crate does not know a
/// state tree's shape and should not: the caller passes the name from whichever
/// crate owns that layout.
///
/// # Panics
///
/// If the directory cannot be created.
pub fn shared_cache_dir(name: &str, keep: &str) -> PathBuf {
    assert!(
        !name.is_empty() && !name.contains(['/', '\\', '.']),
        "a cache name is one path segment, not a path: {name:?}"
    );
    let dir = shared_cache_parent().join(name);
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("create the cache root {}: {e}", dir.display()));

    // Once per process, so tests share what this run computes and inherit
    // nothing but the cache from the last one.
    static CLEARED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CLEARED.get_or_init(|| clear_all_but(&dir, keep));
    dir
}

// Remove every entry under `dir` except the `keep` subtree.
fn clear_all_but(dir: &Path, keep: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name() == keep {
            continue;
        }
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

    #[test]
    fn the_cache_subtree_survives_a_clear_and_nothing_else_does() {
        let dir = shared_cache_dir("harness-clear-test", "cache");
        std::fs::create_dir_all(dir.join("cache")).expect("cache dir");
        std::fs::write(dir.join("cache").join("1"), b"warm").expect("cache entry");
        std::fs::create_dir_all(dir.join("data")).expect("data dir");
        std::fs::write(dir.join("data").join("0"), b"blob").expect("blob");
        std::fs::write(dir.join("editor"), b"session").expect("session");

        // What the next process would do on its first call.
        clear_all_but(&dir, "cache");

        assert_eq!(
            std::fs::read(dir.join("cache").join("1")).expect("cache kept"),
            b"warm"
        );
        assert!(!dir.join("data").exists(), "build output is not a cache");
        assert!(!dir.join("editor").exists(), "session state is not a cache");
    }

    #[test]
    fn the_same_name_is_the_same_directory_every_time() {
        let first = shared_cache_dir("harness-self-test", "cache");
        let second = shared_cache_dir("harness-self-test", "cache");

        assert_eq!(first, second, "a cache is reused, not recreated");
        assert!(first.is_dir());
        assert!(first.starts_with(shared_cache_parent()));
    }

    #[test]
    fn different_names_are_different_directories() {
        let a = shared_cache_dir("harness-self-test-a", "cache");
        let b = shared_cache_dir("harness-self-test-b", "cache");

        assert_ne!(a, b);
        assert_eq!(a.parent(), b.parent());
        assert_eq!(a.parent(), Some(shared_cache_parent().as_path()));
    }

    #[test]
    #[should_panic(expected = "one path segment")]
    fn a_name_with_a_separator_is_rejected() {
        shared_cache_dir("escapes/upward", "cache");
    }

    #[test]
    #[should_panic(expected = "one path segment")]
    fn a_name_with_a_dot_is_rejected() {
        shared_cache_dir("..", "cache");
    }
}
