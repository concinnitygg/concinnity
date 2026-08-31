// src/test_support.rs
//
// Tests that touch process-global state -- the working directory, the
// debug/hot-reload statics, and the engine's development flags -- must not run
// concurrently, because Cargo runs a binary's tests in parallel threads within
// one process.
//
// The guard is the workspace's one process-global lock, so a test here writing
// a development flag excludes the engine's own readers of that same flag
// rather than racing them under a second, private lock.

pub(crate) fn lock() -> concinnity_testing::ExclusiveAccess {
    prepare();
    concinnity_testing::exclusive()
}

// Exclusive access with the working directory moved into a temp tree, for a
// test that writes a cwd-relative path (the build's `world-lock.json`).
//
// A test takes this *instead of* `lock`, never as well: both are the one
// exclusive guard, and taking it twice on a thread deadlocks.
#[must_use]
pub(crate) fn lock_in_temp_cwd() -> concinnity_testing::GlobalState {
    prepare();
    concinnity_testing::GlobalState::acquire().with_cwd()
}

// What every guard here sets up before taking the lock.
fn prepare() {
    // Route tracing output through the harness's per-test capture so expected
    // ERROR logs stay hidden on pass but replay on failure. First install wins.
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    // Nothing in this crate's tests has a window to open, so a call that
    // reaches one is a regression: panic naming the backend rather than block.
    concinnity_testing::forbid_windows();
}

// Anchor both state roots away from the working directory.
//
// The content root is the suite's stable cache root, so a shader that has not
// changed is compiled once for the machine rather than once per test.
//
// The writable root is a subdirectory of it, not the same path, which is what
// `writable_state_dir` falls back to. It holds the editor's session store, the
// saves and the settings -- state, not cache -- and sharing it let a run
// inherit the last one's camera bookmarks. The clear leaves it empty.
pub(crate) fn isolate_state_dir() {
    let root = concinnity_testing::shared_cache_dir(
        "concinnity-dev-tests",
        concinnity_host::store::paths::CACHE_DIR,
    );
    concinnity_host::store::paths::set_writable_state_dir(root.join("run"));
    concinnity_host::store::paths::set_state_dir(root);
}
