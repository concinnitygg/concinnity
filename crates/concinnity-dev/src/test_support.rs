// src/test_support.rs
//
// Process-global test serialization lock. Tests that touch process-global state
// -- notably the current working directory and the debug/hot-reload statics --
// must not run concurrently, because Cargo runs a binary's tests in parallel
// threads within one process.

pub(crate) static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn lock() -> std::sync::MutexGuard<'static, ()> {
    // Route tracing output through the harness's per-test capture so expected
    // ERROR logs stay hidden on pass but replay on failure. First install wins.
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// Point the cook's content-addressed cache at a private temp dir for the test
// process, so an in-memory rebuild never touches the working directory. Installs
// on every call rather than once: a test that clears the root when it finishes
// would otherwise leave every later caller without one, which reads as an
// unrelated failure in whichever test the run happened to order next.
pub(crate) fn isolate_state_dir() {
    let dir = std::env::temp_dir().join(format!("cn-editor-tests-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    concinnity_host::store::paths::set_state_dir(dir);
}
