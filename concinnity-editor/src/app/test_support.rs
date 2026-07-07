// src/app/test_support.rs
//
// Process-global test serialization lock. Tests that touch process-global state
// -- notably the current working directory switched by the FFI build / add / rm
// entry points -- must not run concurrently, because Cargo runs a binary's
// tests in parallel threads within one process.

pub(crate) static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn lock() -> std::sync::MutexGuard<'static, ()> {
    // Route tracing output through the harness's per-test capture so expected
    // ERROR logs stay hidden on pass but replay on failure. First install wins,
    // so cn_init's real subscriber becomes a no-op under the test harness.
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
