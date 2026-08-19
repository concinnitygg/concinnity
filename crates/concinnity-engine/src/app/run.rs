// src/app/run.rs
//
// The runtime player path. Loads compiled blob data and drives the system
// loop. Fully synchronous -- no Tokio runtime here. Systems that need async
// (HttpServerSystem, LlmSystem, etc.) spin up their own runtimes internally.
//
// On macOS the world loop is driven by CFRunLoopRunInMode so that AppKit
// (GLFW window creation, Metal pipeline compilation, event dispatch) can
// process its callbacks on the main thread each tick. On all other platforms
// a tight Rust loop is used, which is what VulkanRenderer expects.
//
// This is the `cn run` path only: no debug server, no WebSocket command
// channel, no in-memory rebuild. A shipped run is neither remotely inspectable
// nor remotely driven. The interpreted (`cn debug`) path with hot-reload and
// the command channel lives in the editor crate.

use crate::app::runloop;
use crate::app::startup_error::StartupError;
use crate::app::state::App;
use std::path::Path;
use tracing_subscriber::EnvFilter;

// Default tracing filter applied when RUST_LOG is unset: info for debug
// builds, warn for release builds. A RUST_LOG value always takes precedence.
fn default_log_directive() -> &'static str {
    if cfg!(debug_assertions) {
        "info"
    } else {
        "warn"
    }
}

// Build the tracing filter from RUST_LOG, falling back to the build-profile
// default when the variable is unset.
fn log_filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_log_directive()))
}

// Install the global tracing subscriber. The single place the log level is
// configured: the CLI entry points call it directly, and the FFI entry point
// (cn_init) calls it for the macOS app. Safe to call once per process. The
// crash ring layer rides along so crash reports carry the recent log lines.
pub fn init_logging() {
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let fmt = tracing_subscriber::fmt::layer().with_filter(log_filter());
    let _ = tracing_subscriber::registry()
        .with(fmt)
        .with(crate::crash::RingLayer)
        .try_init();

    // After the subscriber is up, so the warning is visible when it fires.
    crate::heap::verify_installed();
}

// Whether the runtime overlaps simulation and rendering on separate threads
// (the default) or steps both serially on the main thread (the editor's mode,
// and `cn run --serial` for A/B comparison and as an escape hatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelineMode {
    #[default]
    Pipelined,
    Serial,
}

// Runtime launch options beyond the world itself.
#[derive(Debug, Default)]
pub struct RunOptions {
    pub mode: PipelineMode,
    // Whether systems may fan their internal work across the job pool
    // (default) or keep everything on the stepping thread
    // (`cn run --serial-schedule`, the determinism oracle).
    pub schedule: crate::ecs::ScheduleMode,
    // Capture the last presented frame to this path when the run stops, for
    // headless verification of the runtime path (`cn run --screenshot`).
    pub screenshot: Option<String>,
    // Override the world's `GraphicsConfig.max_frames`, bounding the run.
    pub max_frames: Option<u64>,
}

// Production entry point (`cn run`). Reads the compiled binary blobs from
// data/, written by a prior `cn build`. No debug server, no WebSocket command
// channel: a shipped run is neither remotely inspectable nor remotely driven.
pub fn run(options: RunOptions) -> std::io::Result<()> {
    init_logging();

    let mut app = App::new();

    if let Err(e) = app.load_blob() {
        report_startup_error(StartupError::from_blob_failure(primary_blob_path(), e));
        return Ok(());
    }

    start_runtime(app, options)
}

// The primary blob's path, which is what a load failure is reported against.
fn primary_blob_path() -> std::path::PathBuf {
    std::path::PathBuf::from(concinnity_store::blob::blob_path(0))
}

// Report a fatal startup failure: always to the log, and on screen as well when
// a window can be stood up, so a packaged app that a user double-clicked says
// something rather than exiting silently. The screen blocks until dismissed.
fn report_startup_error(error: StartupError) {
    tracing::error!("{}", error.log_line());
    if !crate::error_screen::show("Concinnity", &error.user_message()) {
        // No window, so the log line above is the whole report; repeat it on
        // stderr, which a console user sees regardless of the tracing filter.
        eprintln!("{}", error.log_line());
    }
}

// Production entry point for a shipped app: like `run`, but with the state root
// pinned to `state_dir` (the flat tree beside the executable or inside an app
// bundle, holding `data/`, `saves/`, and `settings`), and a missing blob is a
// hard error rather than a silent no-op -- a packaged app without its data
// cannot do anything useful. The concinnity-runtime binary calls this.
pub fn run_from(state_dir: &Path) -> std::io::Result<()> {
    init_logging();
    concinnity_store::paths::set_state_dir(state_dir);

    let mut app = App::new();
    if let Err(e) = app.load_blob() {
        let error = StartupError::from_blob_failure(primary_blob_path(), e);
        report_startup_error(error.clone());
        // The process still exits non-zero: the screen is how the user learns
        // what happened, not a substitute for failing.
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            error.log_line(),
        ));
    }
    start_runtime(app, RunOptions::default())
}

// Startup and loop entry once the App's world is populated. Registers the
// CTRL+C handler, activates AppKit on macOS, starts the app, then drives
// frames -- pipelined (sim thread + render half) or serial (the
// single-threaded world loop) -- until the window closes, a system stops the
// world, or CTRL+C is received. External callers reach this through
// `App::run` / `App::run_with`.
pub(crate) fn start_runtime(mut app: App, options: RunOptions) -> std::io::Result<()> {
    tracing::info!("Running app...");
    runloop::install_ctrlc_handler(&app);

    // Resolved before `start()` (while the GraphicsConfig is still present) and
    // reused after, so the post-start loop choice doesn't depend on the config
    // component, which `start()` drains.
    let renders = app.world().renders();

    if let Some(max) = options.max_frames {
        for config in app.world_mut().query_mut::<crate::assets::GraphicsConfig>() {
            config.max_frames = Some(max);
        }
    }
    if options.screenshot.is_some() {
        // Before `start()`, so graphics init arms the blit-readable path.
        crate::app::dev_flags::set_capture(true);
    }
    app.world_mut().insert_resource(options.schedule);

    #[cfg(target_os = "macos")]
    if renders {
        runloop::activate_app_macos();
    }

    if let Err(e) = app.start() {
        eprintln!("Failed to start app: {}", e);
        std::process::exit(1);
    }

    match options.mode {
        PipelineMode::Pipelined if renders => {
            crate::app::pipeline::run_pipelined(app, options.screenshot.as_deref());
        }
        _ => {
            // The serial loop: no per-tick hook; a rendering macOS world pumps
            // the Cocoa run loop, every other case uses the tight loop.
            runloop::run_loop(&mut app, cfg!(target_os = "macos") && renders, |_| {});
            capture_exit_screenshot(&mut app, options.screenshot.as_deref());
        }
    }

    Ok(())
}

// Capture the last presented frame on the way out of a serial run, when
// requested. The backend is still parked in the world after the loop ends.
fn capture_exit_screenshot(app: &mut App, path: Option<&str>) {
    let Some(path) = path else { return };
    let Some(mut backend) = app.world_mut().take_render_backend() else {
        tracing::warn!("screenshot skipped: no live backend at exit");
        return;
    };
    match backend.screenshot(path) {
        Ok(saved) => tracing::info!("screenshot saved: {}", saved),
        Err(e) => tracing::warn!("screenshot failed: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_directive_matches_build_profile() {
        let expected = if cfg!(debug_assertions) {
            "info"
        } else {
            "warn"
        };
        assert_eq!(default_log_directive(), expected);
    }

    #[test]
    fn default_directive_is_a_valid_filter() {
        // The fallback string must parse as an EnvFilter, otherwise log_filter
        // would panic when RUST_LOG is unset.
        EnvFilter::new(default_log_directive());
    }
}
