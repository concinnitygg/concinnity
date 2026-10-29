//! The runtime player path. Loads compiled blob data and drives the system
//! loop. Fully synchronous -- no Tokio runtime here. Systems that need async
//! (`HttpServerSystem`, `LlmSystem`, etc.) spin up their own runtimes internally.
//!
//! On macOS the world loop is driven by `CFRunLoopRunInMode` so that `AppKit`
//! (GLFW window creation, Metal pipeline compilation, event dispatch) can
//! process its callbacks on the main thread each tick. On all other platforms
//! a tight Rust loop is used, which is what `VulkanRenderer` expects.
//!
//! This is the blob-playback path: no debug server, no remote command
//! channel, no in-memory rebuild. A shipped run is neither remotely inspectable
//! nor remotely driven. The paths with hot-reload and the command channel live
//! in the dev crate.

use concinnity_engine::error::WorldError;
use concinnity_engine::render::backend::error::RenderError;
use concinnity_host::store::paths::StateTree;
use tracing_subscriber::EnvFilter;

use crate::app::blob::BlobSource;
use crate::app::runloop;
use crate::app::runtime::Runtime;
use crate::app::startup_error::StartupError;
use concinnity_engine::launch::{LaunchArgs, LaunchRequest, PipelineMode};

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
/// (`cn_init`) calls it for the macOS app. Safe to call once per process. The
/// crash ring layer rides along so crash reports carry the recent log lines.
pub fn init_logging() {
    install_logging(false);
}

/// Install the global tracing subscriber, printing nothing when `quiet`. A
/// quiet process still keeps the crash ring, so a crash report carries the
/// recent log lines. The first call in a process wins.
#[expect(
    clippy::let_underscore_must_use,
    reason = "a subscriber the host already installed takes precedence"
)]
pub fn install_logging(quiet: bool) {
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let fmt = (!quiet).then(|| tracing_subscriber::fmt::layer().with_filter(log_filter()));
    let _ = tracing_subscriber::registry()
        .with(fmt)
        .with(crate::crash::RingLayer)
        .try_init();
}

/// Blob-playback entry point (`cn run` on a cooked world). Reads the compiled
/// binary blobs from `tree`'s `data/`, written by a prior `cn cook`, and runs
/// them as `launch` asks. No debug server, no command channel: a shipped run is
/// neither remotely inspectable nor remotely driven.
///
/// # Errors
///
/// Returns an error if the world in `data/` cannot be loaded or the runtime
/// fails to start.
pub fn run(tree: &StateTree, launch: LaunchRequest) -> std::io::Result<()> {
    install_logging(launch.args.quiet);

    let mut runtime = Runtime::new().in_tree(tree.clone()).with_launch(launch);
    let data_dir = tree.data_dir();
    if let Err(error) = load_world(&mut runtime, &BlobSource::Directory(data_dir)) {
        return Err(report_startup_error(&error));
    }
    start_runtime(runtime).map_err(|e| start_failure(&e))
}

// A refused start, in the form a process exit status is built from.
fn start_failure(e: &WorldError) -> std::io::Error {
    std::io::Error::other(format!("failed to start app: {e}"))
}

// Report a fatal startup failure: always to the log, and on screen as well when
// a window can be stood up, so a packaged app that a user double-clicked says
// something rather than exiting silently. The screen blocks until dismissed.
// The process still exits non-zero with the returned error: the screen is how
// the user learns what happened, not a substitute for failing.
fn report_startup_error(error: &StartupError) -> std::io::Error {
    tracing::error!("{error}");
    if !crate::render::error_screen::show("Concinnity", &error.user_message()) {
        // No window, so the log line above is the whole report; repeat it on
        // stderr, which a console user sees regardless of the tracing filter.
        eprintln!("{error}");
    }
    std::io::Error::new(error.io_kind(), error.to_string())
}

// Populate `runtime` with the world `source` holds, refusing a layout that cannot
// hold every blob the world spans.
fn load_world(runtime: &mut Runtime, source: &BlobSource) -> Result<(), StartupError> {
    let max_blob_index = runtime.load_blob_from(&source.primary())?;
    source.check_span(max_blob_index).map_or(Ok(()), Err)
}

/// Production entry point for a shipped app: like `run`, but with the world
/// read from `blob` rather than from `tree`'s own `data/`. The tree (beside the
/// executable or inside an app bundle) still holds `saves/` and `settings`, so
/// pointing the player at a blob elsewhere never relocates what it writes. A
/// missing blob is a hard error rather than a silent no-op -- a packaged app
/// without its data cannot do anything useful. The concinnity-run binary
/// calls this.
///
/// # Errors
///
/// Returns an error if the world in `blob` cannot be loaded or the runtime
/// fails to start.
pub fn run_from(tree: &StateTree, blob: &BlobSource) -> std::io::Result<()> {
    init_logging();

    let mut runtime = Runtime::new().in_tree(tree.clone());
    if let Err(error) = load_world(&mut runtime, blob) {
        return Err(report_startup_error(&error));
    }
    start_runtime(runtime).map_err(|e| start_failure(&e))
}

// Startup and loop entry once the Runtime's world is populated. Registers the
// CTRL+C handler, activates AppKit on macOS, starts the runtime, then drives
// frames -- pipelined (sim thread + render half), or serial for a world with
// nothing to render -- until the window closes, a system stops the world, or
// CTRL+C is received. External callers reach this through `Runtime::run`.
pub(crate) fn start_runtime(mut runtime: Runtime) -> Result<(), WorldError> {
    // A host that installed its own subscriber keeps it (`try_init` no-ops),
    // so an embedded app gets logs without wiring any up itself.
    let args = runtime.launch_args();
    install_logging(args.quiet);
    warn_unserved_debug_port(args);
    tracing::info!("Running app...");
    runloop::install_ctrlc_handler(&runtime);

    // Resolved before `start()`, which is where the columns the resolution
    // reads are drained. The runtime caches it, so `start()` publishes this
    // same answer rather than probing the GPU twice.
    let renders = runtime.render_mode().renders();

    #[cfg(target_os = "macos")]
    if renders {
        runloop::activate_app_macos();
    }

    if let Err(e) = runtime.start() {
        // Returned rather than exiting the process, so the world's systems
        // (and the GPU resources they hold) still drop on the way out. A
        // renderer that refused gets the error screen as well, since a
        // double-clicked app has no console to print to.
        if let WorldError::RenderUnavailable(cause) = &e {
            report_render_failure(cause);
        }
        tracing::error!("failed to start app: {e}");
        return Err(e);
    }

    match PipelineMode::Pipelined.effective(renders) {
        PipelineMode::Pipelined => crate::app::pipeline::run_pipelined(runtime),
        PipelineMode::Serial => {
            // The serial loop: no per-tick hook; a rendering macOS world pumps
            // the Cocoa run loop, every other case uses the tight loop.
            runloop::run_loop(&mut runtime, cfg!(target_os = "macos") && renders, |_| {});
            finish_frame_captures(&mut runtime);
        }
    }

    Ok(())
}

// This path carries no debug server, so a launch that asks for one is told.
fn warn_unserved_debug_port(args: &LaunchArgs) {
    if let Some(port) = args.debug_port {
        tracing::warn!("--debug-port {port}: this build has no debug server, so none is started");
    }
}

// Report a renderer that refused, or a world that needs one this machine
// cannot give it, on screen where one can be stood up. A machine with a GPU
// reaches here when its driver refused, so the driver is what is actionable.
//
// The screen draws through the same backend that just failed, so it may well
// fail too; that is the console fallback, matching the blob-load path.
fn report_render_failure(cause: &RenderError) {
    let message = format!(
        "This app could not start its renderer.\n\n{cause}\n\n\
         Updating your graphics driver is the usual fix."
    );
    if !crate::render::error_screen::show("Concinnity", &message) {
        eprintln!("{message}");
    }
}

// Let the frame captures a serial run asked for finish writing before the
// process can exit. The backend is still parked in the world.
fn finish_frame_captures(runtime: &mut Runtime) {
    if let Some(backend) = runtime
        .world_mut()
        .resource_mut::<concinnity_engine::ecs::ActiveRenderBackend>()
        .and_then(|slot| slot.0.as_mut())
    {
        backend.finish_captures();
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

    // A tree with no build behind it is a missing-data failure in either
    // layout, which is what makes `cn run` exit non-zero rather than start empty.
    #[test]
    fn loading_a_world_that_was_never_built_reports_missing_data() {
        let tmp = concinnity_testing::TempTree::new();
        let tree = StateTree::at(tmp.path());

        let data_dir = tree.data_dir();
        let mut runtime = Runtime::new().in_tree(tree.clone());
        let error = load_world(&mut runtime, &BlobSource::Directory(data_dir.clone()))
            .expect_err("an empty tree has no world");
        assert!(
            matches!(&error, StartupError::MissingData { blob } if *blob == data_dir.join("0")),
            "{error:?}"
        );
        assert_eq!(error.io_kind(), std::io::ErrorKind::NotFound);

        let file = tmp.join("missing.blob");
        let mut runtime = Runtime::new().in_tree(tree);
        let error = load_world(&mut runtime, &BlobSource::File(file.clone()))
            .expect_err("a named blob that is not there has no world");
        assert!(
            matches!(&error, StartupError::MissingData { blob } if *blob == file),
            "{error:?}"
        );
    }

    // A world that refuses to start reports it through the return value. The
    // process stays alive, so the caller's cleanup and the world's own drops
    // still run; an already-started runtime is the reproducible refusal.
    #[test]
    fn a_refused_start_returns_instead_of_exiting_the_process() {
        let mut runtime = Runtime::new();
        runtime.start().expect("the first start succeeds");

        assert!(
            matches!(runtime.run(), Err(WorldError::AlreadyStarted)),
            "a second start is refused"
        );
    }
}
