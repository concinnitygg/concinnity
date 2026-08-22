//! Process-wide flags shared between the engine loop (library) and the
//! binary-only `cn debug` subsystem. Only the two flags the library itself
//! names live here; the world.jsonl / shader-stage "changed" flags and the
//! decal / emitter spawn queue moved fully into the binary-only debug tree
//! (`crate::debug`), since nothing in the library references them.
//!
//!   ENABLED              "are we running under a dev-loop entry point?" Set
//!                        once by main.rs's `Commands::Debug` / `Commands::Editor`
//!                        arms before world build; read by `GraphicsSystem::init`
//!                        / `AnimationSystem` / the draw list builder to enable
//!                        disk-first shader loading + the hot-reload source
//!                        capture. `cn run` leaves it false so production keeps
//!                        the static `include_str!`-baked path with no
//!                        filesystem dependency.
//!   PENDING_ANIMATIONS   "an Animation source changed." Set by the cn debug
//!                        watcher / WS `reload-assets` handler; consumed by the
//!                        editor crate's `anim_reload::reload_clips_if_pending`,
//!                        which the debug drive calls each frame to re-import
//!                        file-backed clips. The flag lives here (in the runtime
//!                        crate) because it bridges the runtime AnimationSystem,
//!                        which reads ENABLED, and the editor-driven hot-reload.
//!   VALIDATION           "did the launch request graphics validation?" Set by
//!                        the CLI `--validation` flag (`cn run` / `cn debug`).
//!                        Tri-state: unset defers to the build profile (on for
//!                        debug, off for release). `resolve_validation` settles
//!                        the two, and `GraphicsSystem::init` enables the
//!                        DirectX / Vulkan debug layers from the result. Metal's
//!                        validation layer cannot be toggled from a running
//!                        process, so the CLI re-execs with the env var instead;
//!                        this flag does not drive Metal.
//!   WORLD_JSONL_PATH     the world.jsonl the dev host is running. Set by the
//!                        editor's `cn debug` / `cn editor` entry once the world
//!                        path is resolved; read by `GraphicsSystem::init` (only
//!                        under ENABLED) so the Prop-transform hot-reload watcher
//!                        knows which file to subscribe to. world.jsonl discovery
//!                        is authoring I/O that lives in `concinnity-cook`, which
//!                        the runtime does not link, so the dev host resolves the
//!                        path and hands it in rather than the engine looking it
//!                        up. Left None for `cn run` and embedded preview.
//!
//! A static is the pragmatic shape here: the flags are process-wide because the
//! rendering backend is too (a single context per process owns the GPU), and
//! plumbing them through the public `App` / `run_interpreted` signatures would
//! touch far more code for the same observable behaviour.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static PENDING_ANIMATIONS: AtomicBool = AtomicBool::new(false);
// "keep the presented frame blit-readable for an exit screenshot." Set by
// `cn run --screenshot` before world build; read by `GraphicsSystem::init`.
// The dev loop's ENABLED implies capture without this flag.
static CAPTURE: AtomicBool = AtomicBool::new(false);

// Tri-state validation request: 0 = unset (use the build-profile default),
// 1 = explicitly off, 2 = explicitly on.
static VALIDATION: AtomicU8 = AtomicU8::new(0);

// Path to the world.jsonl the dev host is running, or None outside a dev host.
static WORLD_JSONL_PATH: Mutex<Option<String>> = Mutex::new(None);

// The harness runs a binary's tests on parallel threads, so a test that writes
// a flag races every test whose code reads one -- graphics init reads three.
// Writers take this exclusively, readers share it, so only the writers
// serialise.
#[cfg(test)]
static FLAG_ACCESS: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Mark this process as running under a dev-loop entry point (`cn debug` /
/// `cn editor`). Call once before world build.
///
/// `dead_code` allow: only the binary's `Commands::Debug` / `Commands::Editor`
/// arms call this; the library never sets the flag (it only reads it), so
/// `cargo check --lib` reports it unused.
pub fn set_enabled(v: bool) {
    ENABLED.store(v, Ordering::SeqCst);
}

// True when the process is running under a dev-loop entry point that wants
// shader hot-reload. False for `cn run` and any embedded preview.
pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::SeqCst)
}

// Arm frame capture for a production run that wants an exit screenshot.
// Called by `start_runtime` before world build when a screenshot path was
// requested.
pub(crate) fn set_capture(v: bool) {
    CAPTURE.store(v, Ordering::SeqCst);
}

// True when a production run armed frame capture (`cn run --screenshot`).
pub(crate) fn capture() -> bool {
    CAPTURE.load(Ordering::SeqCst)
}

/// Raise the "Animation source changed" flag. Called by the cn debug asset
/// hot-reload watcher and the WS `reload-assets` handler.
///
/// `dead_code` allow: only the binary-only debug subsystem sets this; the
/// library never does, so `cargo check --lib` reports it unused.
pub fn set_pending_animations() {
    PENDING_ANIMATIONS.store(true, Ordering::SeqCst);
}

/// Swap the "Animation source changed" flag to `false`, returning whether it
/// was set. The editor crate's `anim_reload::reload_clips_if_pending` calls
/// this; a `true` result kicks the per-clip re-import pass.
pub fn take_pending_animations() -> bool {
    PENDING_ANIMATIONS.swap(false, Ordering::SeqCst)
}

/// Record the CLI `--validation` request. `None` leaves the build-profile
/// default in effect; `Some` forces validation on or off.
///
/// `dead_code` allow: only the binary's `Commands::Run` / `Commands::Debug` arms
/// call this; the library only reads it, so `cargo check --lib` reports it unused.
pub fn set_validation(v: Option<bool>) {
    let encoded = match v {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    VALIDATION.store(encoded, Ordering::SeqCst);
}

// The CLI validation request, or `None` when the launch did not specify one.
pub(crate) fn validation() -> Option<bool> {
    match VALIDATION.load(Ordering::SeqCst) {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

// Settle the graphics-validation request: the CLI `--validation` flag if the
// launch passed one, otherwise the build profile. Running a debug layer is a
// launch concern, so no world can ask for it.
pub(crate) fn resolve_validation() -> bool {
    validation().unwrap_or(cfg!(debug_assertions))
}

/// Record the world.jsonl path the dev host resolved, so the hot-reload watcher
/// can subscribe to it. Called by the editor's `cn debug` / `cn editor` entry
/// before world build.
///
/// `dead_code` allow: only the editor crate sets this; the library just reads
/// it, so `cargo check --lib` reports it unused.
pub fn set_world_jsonl_path(path: Option<String>) {
    *WORLD_JSONL_PATH.lock().unwrap() = path;
}

// The world.jsonl path the dev host handed in, or None outside a dev host. Read
// by `GraphicsSystem::init` to seed the Prop-transform reload watcher.
pub(crate) fn world_jsonl_path() -> Option<String> {
    WORLD_JSONL_PATH.lock().unwrap().clone()
}

// Shared flag access for a test whose code path reads a flag. Held for as long
// as the read matters: for graphics init, across the whole `run_init`.
#[cfg(test)]
pub(crate) fn read_access() -> std::sync::RwLockReadGuard<'static, ()> {
    FLAG_ACCESS.read().unwrap_or_else(|e| e.into_inner())
}

// Exclusive flag access for a test that writes one. Restores every flag graphics
// init reads when it drops, so a panicking test cannot leak one into the rest of
// the binary. Poison is ignored: the test holding it has already failed, and
// erroring every later lock buries that failure under a cascade. Not reentrant,
// so a test holding this must not also take `read_access`.
#[cfg(test)]
pub(crate) struct WriteAccess {
    _guard: std::sync::RwLockWriteGuard<'static, ()>,
    enabled: bool,
    validation: Option<bool>,
    world_jsonl_path: Option<String>,
}

#[cfg(test)]
pub(crate) fn write_access() -> WriteAccess {
    WriteAccess {
        _guard: FLAG_ACCESS.write().unwrap_or_else(|e| e.into_inner()),
        enabled: enabled(),
        validation: validation(),
        world_jsonl_path: world_jsonl_path(),
    }
}

#[cfg(test)]
impl Drop for WriteAccess {
    fn drop(&mut self) {
        set_enabled(self.enabled);
        set_validation(self.validation);
        set_world_jsonl_path(self.world_jsonl_path.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_off_and_round_trips() {
        let _flags = write_access();
        set_enabled(false);
        assert!(!enabled());
        set_enabled(true);
        assert!(enabled());
    }

    #[test]
    fn validation_tristate_round_trips() {
        let _flags = write_access();
        set_validation(None);
        assert_eq!(validation(), None);
        set_validation(Some(true));
        assert_eq!(validation(), Some(true));
        set_validation(Some(false));
        assert_eq!(validation(), Some(false));
    }

    #[test]
    fn the_launch_flag_outranks_the_build_profile() {
        let _flags = write_access();

        // No flag: the build profile decides.
        set_validation(None);
        assert_eq!(resolve_validation(), cfg!(debug_assertions));

        // An explicit flag decides instead, either way.
        set_validation(Some(false));
        assert!(!resolve_validation());
        set_validation(Some(true));
        assert!(resolve_validation());
    }
}
