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

// Production entry point (`cn run`). Reads the compiled binary blobs from
// data/, written by a prior `cn build`. No debug server, no WebSocket command
// channel: a shipped run is neither remotely inspectable nor remotely driven.
pub fn run() -> std::io::Result<()> {
    init_logging();

    let mut app = App::new();

    if let Err(e) = app.load_blob() {
        tracing::info!(
            "No blob found (data/0): {:?} -- run `concinnity build` first",
            e
        );

        return Ok(());
    }

    start_runtime(app)
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
    app.load_blob().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "no compiled world data under {}: {e}",
                concinnity_store::paths::data_dir().display()
            ),
        )
    })?;
    start_runtime(app)
}

// Startup and loop entry once the App's world is populated from a compiled
// blob. Registers the CTRL+C handler, activates AppKit on macOS, starts the
// app, then runs the world loop until the window closes, a system stops the
// world, or CTRL+C is received.
pub fn start_runtime(mut app: App) -> std::io::Result<()> {
    tracing::info!("Running app...");
    runloop::install_ctrlc_handler(&app);

    // Resolved before `start()` (while the GraphicsConfig is still present) and
    // reused after, so the post-start render-loop choice doesn't depend on the
    // config component, which `start()` drains. Only the macOS path branches.
    #[cfg(target_os = "macos")]
    let renders = app.world().renders();

    #[cfg(target_os = "macos")]
    if renders {
        runloop::activate_app_macos();
    }

    if let Err(e) = app.start() {
        eprintln!("Failed to start app: {}", e);
        std::process::exit(1);
    }

    // The runtime has no per-tick hook; a rendering macOS world pumps the Cocoa
    // run loop, every other case uses the tight loop.
    #[cfg(target_os = "macos")]
    runloop::run_loop(&mut app, renders, |_| {});
    #[cfg(not(target_os = "macos"))]
    runloop::run_loop(&mut app, false, |_| {});

    Ok(())
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
