//! The runtime player path. Loads compiled blob data and drives the system
//! loop. Fully synchronous -- no Tokio runtime here. Systems that need async
//! (HttpServerSystem, LlmSystem, etc.) spin up their own runtimes internally.
//!
//! On macOS the world loop is driven by CFRunLoopRunInMode so that AppKit
//! (GLFW window creation, Metal pipeline compilation, event dispatch) can
//! process its callbacks on the main thread each tick. On all other platforms
//! a tight Rust loop is used, which is what VulkanRenderer expects.
//!
//! This is the `cn run` path only: no debug server, no WebSocket command
//! channel, no in-memory rebuild. A shipped run is neither remotely inspectable
//! nor remotely driven. The interpreted (`cn debug`) path with hot-reload and
//! the command channel lives in the editor crate.

use crate::app::runloop;
use crate::app::startup_error::StartupError;
use crate::app::state::App;
use crate::result::CnResult;
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

/// Install the global tracing subscriber. The single place the log level is
/// configured: the CLI entry points call it directly, and the FFI entry point
/// (cn_init) calls it for the macOS app. Safe to call once per process. The
/// crash ring layer rides along so crash reports carry the recent log lines.
pub fn init_logging() {
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let fmt = tracing_subscriber::fmt::layer().with_filter(log_filter());
    let _ = tracing_subscriber::registry()
        .with(fmt)
        .with(crate::crash::RingLayer)
        .try_init();
}

/// Whether the runtime overlaps simulation and rendering on separate threads
/// (the default) or steps both serially on the main thread (the editor's mode,
/// and `cn run --serial` for A/B comparison and as an escape hatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelineMode {
    #[default]
    /// Simulation and rendering overlap on separate threads.
    Pipelined,
    /// Simulation and rendering step serially on the main thread.
    Serial,
}

/// Runtime launch options beyond the world itself.
#[derive(Debug, Default)]
pub struct RunOptions {
    /// Whether the runtime pipelines simulation and rendering.
    pub mode: PipelineMode,
    /// Whether systems may fan their internal work across the job pool
    /// (default) or keep everything on the stepping thread
    /// (`cn run --serial-schedule`, the determinism oracle).
    pub schedule: crate::ecs::ScheduleMode,
    /// Capture the last presented frame to this path when the run stops, for
    /// headless verification of the runtime path (`cn run --screenshot`).
    pub screenshot: Option<String>,
    /// Override the world's `GraphicsConfig.max_frames`, bounding the run.
    pub max_frames: Option<u64>,
}

/// Production entry point (`cn run`). Reads the compiled binary blobs from
/// data/, written by a prior `cn build`. No debug server, no WebSocket command
/// channel: a shipped run is neither remotely inspectable nor remotely driven.
pub fn run(options: RunOptions) -> std::io::Result<()> {
    init_logging();

    let mut app = App::new();

    if let Err(e) = app.load_blob() {
        report_startup_error(match primary_blob_path() {
            Some(blob) => StartupError::from_blob_failure(blob, e),
            None => StartupError::NoStateRoot,
        });
        return Ok(());
    }

    start_runtime(app, options).map_err(start_failure)
}

// A refused start, in the form a process exit status is built from.
fn start_failure(e: CnResult) -> std::io::Error {
    std::io::Error::other(format!("failed to start app: {e}"))
}

// The primary blob's path, which is what a load failure is reported against.
// `None` when nothing anchored the state tree, so there is no path to name.
fn primary_blob_path() -> Option<std::path::PathBuf> {
    concinnity_host::store::blob::blob_path(0).map(std::path::PathBuf::from)
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

/// Where a shipped app's compiled world sits. Both forms make the same file
/// blob 0; they differ in whether the world is allowed to spill into overflow
/// payload blobs, which are always siblings named by index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobSource<'a> {
    /// A directory holding blob 0 plus any overflow blobs beside it.
    Directory(&'a Path),
    /// A single self-contained blob file. A world that needs overflow blobs is
    /// refused rather than half-loaded, since its siblings would be written
    /// into whatever directory the file happens to sit in.
    File(&'a Path),
}

impl BlobSource<'_> {
    // The primary blob file: the named file itself, or blob 0 in the directory.
    fn primary(&self) -> std::path::PathBuf {
        match self {
            BlobSource::Directory(dir) => dir.join("0"),
            BlobSource::File(file) => file.to_path_buf(),
        }
    }

    // The refusal a single-file source owes a world that spans more blobs.
    fn check_span(&self, max_blob_index: u32) -> Option<StartupError> {
        match self {
            BlobSource::File(file) if max_blob_index > 0 => {
                Some(StartupError::OverflowUnsupported {
                    blob: file.to_path_buf(),
                    needed: max_blob_index,
                })
            }
            _ => None,
        }
    }
}

/// Production entry point for a shipped app: like `run`, but with the state root
/// pinned to `state_dir` (the tree beside the executable or inside an app
/// bundle, holding `saves/` and `settings`) and the world read from `blob`. A
/// missing blob is a hard error rather than a silent no-op -- a packaged app
/// without its data cannot do anything useful. The concinnity-run binary
/// calls this.
pub fn run_from(state_dir: &Path, blob: BlobSource<'_>) -> std::io::Result<()> {
    init_logging();
    concinnity_host::store::paths::set_state_dir(state_dir);

    let primary = blob.primary();
    let mut app = App::new();
    let failure = match app.load_blob_from(&primary) {
        Ok(max_blob_index) => blob.check_span(max_blob_index),
        Err(e) => Some(StartupError::from_blob_failure(primary, e)),
    };
    if let Some(error) = failure {
        report_startup_error(error.clone());
        // The process still exits non-zero: the screen is how the user learns
        // what happened, not a substitute for failing.
        return Err(std::io::Error::new(error.io_kind(), error.log_line()));
    }
    start_runtime(app, RunOptions::default()).map_err(start_failure)
}

// Startup and loop entry once the App's world is populated. Registers the
// CTRL+C handler, activates AppKit on macOS, starts the app, then drives
// frames -- pipelined (sim thread + render half) or serial (the
// single-threaded world loop) -- until the window closes, a system stops the
// world, or CTRL+C is received. External callers reach this through
// `App::run` / `App::run_with`.
pub(crate) fn start_runtime(mut app: App, options: RunOptions) -> Result<(), CnResult> {
    // A host that installed its own subscriber keeps it (`try_init` no-ops),
    // so an embedded app gets logs without wiring any up itself.
    init_logging();
    tracing::info!("Running app...");
    runloop::install_ctrlc_handler(&app);

    // Resolved before `start()` (while the GraphicsConfig is still present) and
    // reused after, so the post-start loop choice doesn't depend on the config
    // component, which `start()` drains.
    let renders = crate::ecs::renders(app.world());

    if let Some(max) = options.max_frames {
        for config in app
            .world_mut()
            .query_mut::<crate::components::GraphicsConfig>()
        {
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
        // Returned rather than exiting the process, so the world's systems
        // (and the GPU resources they hold) still drop on the way out.
        tracing::error!("failed to start app: {e}");
        return Err(e);
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
    let Some(mut backend) = crate::ecs::take_render_backend(app.world_mut()) else {
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

    // Both forms make the same file blob 0: the file itself, or `0` inside the
    // directory. This is what lets one runtime entry point serve both.
    #[test]
    fn each_blob_source_names_the_same_primary_file() {
        let dir = Path::new("/apps/MyGame/data");
        assert_eq!(
            BlobSource::Directory(dir).primary(),
            dir.join("0"),
            "a directory holds blob 0"
        );

        let file = Path::new("/apps/MyGame/data");
        assert_eq!(
            BlobSource::File(file).primary(),
            file.to_path_buf(),
            "a single file is blob 0"
        );
    }

    // Overflow blobs are siblings named by index, so only the directory form
    // has somewhere to hold them. A single file whose world spans more is
    // refused rather than half-loaded, and the message names the fix.
    #[test]
    fn a_single_file_source_refuses_a_world_that_overflows() {
        let file = Path::new("/apps/MyGame/data");

        assert_eq!(BlobSource::File(file).check_span(0), None);
        assert_eq!(
            BlobSource::File(file).check_span(2),
            Some(StartupError::OverflowUnsupported {
                blob: file.to_path_buf(),
                needed: 2,
            })
        );

        // The directory form carries any span, which is why export picks it.
        let dir = Path::new("/apps/MyGame/data");
        assert_eq!(BlobSource::Directory(dir).check_span(0), None);
        assert_eq!(BlobSource::Directory(dir).check_span(7), None);
    }

    // A world that refuses to start reports it through the return value. The
    // process stays alive, so the caller's cleanup and the world's own drops
    // still run; an already-started app is the reproducible refusal.
    #[test]
    fn a_refused_start_returns_instead_of_exiting_the_process() {
        let mut app = App::new();
        app.start().expect("the first start succeeds");

        assert_eq!(
            app.run_with(RunOptions::default()),
            Err(CnResult::InvalidState),
            "a second start is refused"
        );
    }
}
