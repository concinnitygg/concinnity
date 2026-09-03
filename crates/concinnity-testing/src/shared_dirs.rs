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
//! So they are two roots, not one, and the difference between them is who they
//! belong to. [`shared_cache_dir`] is one directory per name per machine and
//! persists. Any number of processes may share it: an entry is
//! content-addressed, and the store inside it is written so that the loser of a
//! race loses entries rather than publishing the wrong bytes.
//! [`shared_state_dir`] is one directory per name **per process**, made new
//! when that process first asks for it. A caller wanting both points a state tree's
//! cache root at the first and its content root at the second, which is what
//! "a warm cache and a fresh build" is.
//!
//! State is per process because the harness's exclusion is per process: the
//! lock a test holds over the session's open project excludes the threads
//! beside it and nothing else. A runner that gives each test a process of its
//! own -- `cargo nextest` -- would otherwise have every one of them empty a
//! root its siblings are mid-write in.
//!
//! A process's state root outlives it, since nothing runs after the last test.
//! The next run sweeps the ones left by processes that are long gone, so the
//! parent stays bounded whether a run spends one process or seven thousand.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, SystemTime};

// One parent for both kinds, so a machine has a single place to clear.
const PARENT: &str = "concinnity-test-cache";

// How long a state root is left alone before another run may reclaim it. Well
// past any test's runtime, so a sweep never reaches a root still in use, and
// short enough that a day of runs does not accumulate.
const STALE_AFTER: Duration = Duration::from_secs(60 * 60);

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
    ensure(&shared_parent(), name)
}

/// This process's state root called `name`, made new on its first call here so
/// the process inherits nothing from whoever held the root before it.
///
/// Sibling processes get roots of their own, so tests that run one to a process
/// do not write over each other. Call it from every test that needs the root,
/// not once: the root is made on the first call and the rest are a lookup.
///
/// `name` separates one suite's state from another's; it is a single path
/// segment, not a path.
///
/// # Panics
///
/// If the directory cannot be created.
pub fn shared_state_dir(name: &str) -> PathBuf {
    let family = ensure(&shared_parent(), name);
    let pid = std::process::id().to_string();

    let mut prepared = prepared().lock().unwrap_or_else(PoisonError::into_inner);
    if prepared.insert(name.to_owned()) {
        // A pid comes round again on a machine that has been up a while.
        let mine = claim(&family, &pid);
        sweep(&family, &mine, SystemTime::now());
        return mine;
    }
    ensure(&family, &pid)
}

// The state names this process has already made new. Keyed by name rather than
// held in one flag, so a suite asking for two roots gets both.
fn prepared() -> &'static Mutex<BTreeSet<String>> {
    static PREPARED: OnceLock<Mutex<BTreeSet<String>>> = OnceLock::new();
    PREPARED.get_or_init(Mutex::default)
}

// `parent/name`, created if absent.
fn ensure(parent: &Path, name: &str) -> PathBuf {
    let dir = resolve(parent, name);
    create(&dir);
    dir
}

// `parent/name`, made new rather than emptied, so it dates from this call and
// not from whoever held the name last. Emptying an already empty root leaves
// that older date in place, which a sweep reads as an owner that is gone.
fn claim(parent: &Path, name: &str) -> PathBuf {
    let dir = resolve(parent, name);
    let _ = std::fs::remove_dir_all(&dir);
    create(&dir);
    dir
}

// `parent/name`, with `name` held to one path segment.
fn resolve(parent: &Path, name: &str) -> PathBuf {
    assert!(
        !name.is_empty() && !name.contains(['/', '\\', '.']),
        "a shared root's name is one path segment, not a path: {name:?}"
    );
    parent.join(name)
}

// Create `dir` and its parents.
fn create(dir: &Path) {
    std::fs::create_dir_all(dir)
        .unwrap_or_else(|e| panic!("create the shared root {}: {e}", dir.display()));
}

// Drop the state roots beside `keep` that nothing has touched in a while. A
// live sibling is a running test process, and its root is minutes old at most.
fn sweep(family: &Path, keep: &Path, now: SystemTime) {
    let Ok(entries) = std::fs::read_dir(family) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == keep || !stale(&path, now) {
            continue;
        }
        let _ = std::fs::remove_dir_all(&path);
    }
}

// Whether nothing has written to `path` for longer than a run could take. A
// path whose age cannot be read is left alone: unreadable is not gone.
fn stale(path: &Path, now: SystemTime) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| now.duration_since(t).ok())
        .is_some_and(|age| age > STALE_AFTER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TempTree;

    // A family of state roots the test owns outright. The real parent is shared
    // by every test process on the machine and outlives the run, so a test that
    // swept it would be judging roots other runs left there, at whatever age
    // they happened to be.
    fn family(tree: &TempTree) -> PathBuf {
        tree.dir("family")
    }

    // Backdate a directory. A sweep reads the age off the filesystem rather
    // than taking it as an argument, so a root left by an earlier run is one a
    // test has to make.
    fn age(dir: &Path, by: Duration) {
        let times = std::fs::FileTimes::new().set_modified(SystemTime::now() - by);
        open_dir(dir)
            .and_then(|handle| handle.set_times(times))
            .unwrap_or_else(|e| panic!("backdate {}: {e}", dir.display()));
    }

    #[cfg(not(windows))]
    fn open_dir(dir: &Path) -> std::io::Result<std::fs::File> {
        std::fs::File::open(dir)
    }

    // A directory needs backup semantics to open at all, and writing its times
    // needs the attribute right that a read handle does not carry.
    #[cfg(windows)]
    fn open_dir(dir: &Path) -> std::io::Result<std::fs::File> {
        use std::os::windows::fs::OpenOptionsExt;

        const FILE_WRITE_ATTRIBUTES: u32 = 0x0100;
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

        std::fs::OpenOptions::new()
            .access_mode(FILE_WRITE_ATTRIBUTES)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(dir)
    }

    // A state root starts each process empty: a blob or a session file the last
    // process to hold it wrote must not make this one's assertions pass.
    #[test]
    fn a_claimed_root_keeps_nothing_the_last_holder_left() {
        let tree = TempTree::new();
        let family = family(&tree);
        let left = ensure(&family, "1234");
        std::fs::create_dir_all(left.join("data")).expect("data dir");
        std::fs::write(left.join("data").join("0"), b"blob").expect("blob");
        std::fs::write(left.join("editor"), b"session").expect("session");

        let mine = claim(&family, "1234");

        assert_eq!(mine, left, "the pid names the root, whoever held it before");
        assert!(!mine.join("data").exists(), "build output is not a cache");
        assert!(
            !mine.join("editor").exists(),
            "session state is not a cache"
        );
        assert!(mine.is_dir(), "the root itself is there to be written to");
    }

    // The root a returning pid inherits carries the age of whoever left it, and
    // emptying an already empty one does not lift it. Claiming dates the root
    // from now, which is what a sibling's sweep reads as an owner still running.
    #[test]
    fn a_claimed_root_outlives_a_siblings_sweep() {
        let tree = TempTree::new();
        let family = family(&tree);
        age(&ensure(&family, "1234"), STALE_AFTER * 2);

        let mine = claim(&family, "1234");
        let sibling = ensure(&family, "5678");
        sweep(&family, &sibling, SystemTime::now());

        assert!(mine.is_dir(), "a root claimed just now has an owner");
    }

    // The root a process gets is its own, and a sibling that reclaimed one it
    // was writing is the bug this shape exists to rule out.
    #[test]
    fn a_state_root_belongs_to_one_process() {
        let dir = shared_state_dir("harness-state-owner");

        assert_eq!(
            dir.file_name().and_then(|s| s.to_str()),
            Some(std::process::id().to_string().as_str()),
            "the process owning the root names it"
        );
        assert_eq!(
            dir.parent().and_then(Path::file_name),
            Some(std::ffi::OsStr::new("harness-state-owner")),
            "sibling processes share the name, not the directory"
        );
    }

    // A cache root is never cleared: that is the whole reason it is a root of
    // its own rather than a subdirectory of the state.
    #[test]
    fn a_cache_root_survives() {
        // The root is shared by every process on the machine, so the entry is
        // named for this one. Two processes writing one path would have a
        // reader see the truncation rather than the bytes.
        let entry = Path::new("cache").join(std::process::id().to_string());
        let dir = shared_cache_dir("harness-cache-test");
        std::fs::create_dir_all(dir.join("cache")).expect("cache dir");
        std::fs::write(dir.join(&entry), b"warm").expect("cache entry");

        assert_eq!(
            std::fs::read(shared_cache_dir("harness-cache-test").join(&entry)).expect("cache kept"),
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
        assert_eq!(cache.parent(), Some(shared_parent().as_path()));
        assert_eq!(state.parent().and_then(Path::parent), cache.parent());
    }

    // Nothing runs after the last test, so a state root outlives its process.
    // The next run is what reclaims it, and only once no process could still be
    // writing there.
    #[test]
    fn a_sweep_reclaims_the_roots_a_run_left_and_spares_its_own() {
        let tree = TempTree::new();
        let family = family(&tree);
        let mine = ensure(&family, "1");
        let theirs = ensure(&family, "2");

        // A run an hour and more after the one that left these.
        sweep(&family, &mine, SystemTime::now() + STALE_AFTER * 2);

        assert!(
            mine.is_dir(),
            "a sweep never reclaims the caller's own root"
        );
        assert!(
            !theirs.exists(),
            "a root no run has touched in an hour is done"
        );
    }

    // A sibling being written to right now is a running test, and reclaiming
    // its root is the failure this whole shape exists to rule out.
    #[test]
    fn a_sweep_spares_a_root_a_running_process_owns() {
        let tree = TempTree::new();
        let family = family(&tree);
        let mine = ensure(&family, "1");
        let live = ensure(&family, "2");

        sweep(&family, &mine, SystemTime::now());

        assert!(mine.is_dir());
        assert!(
            live.is_dir(),
            "a root written to just now is a running test"
        );
    }

    // An unreadable age is not evidence the owner is gone, so the root stays.
    #[test]
    fn a_root_whose_age_cannot_be_read_is_left_alone() {
        let tree = TempTree::new();

        assert!(!stale(&tree.join("absent"), SystemTime::now()));
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
