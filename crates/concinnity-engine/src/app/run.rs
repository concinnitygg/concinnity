//! The runtime player path. Loads compiled blob data and drives the system
//! loop. Fully synchronous -- no Tokio runtime here. Systems that need async
//! (HttpServerSystem, LlmSystem, etc.) spin up their own runtimes internally.
//!
//! On macOS the world loop is driven by CFRunLoopRunInMode so that AppKit
//! (GLFW window creation, Metal pipeline compilation, event dispatch) can
//! process its callbacks on the main thread each tick. On all other platforms
//! a tight Rust loop is used, which is what VulkanRenderer expects.
//!
//! This is the `cn run` path only: no debug server, no remote command
//! channel, no in-memory rebuild. A shipped run is neither remotely inspectable
//! nor remotely driven. The interpreted (`cn debug`) path with hot-reload and
//! the command channel lives in the editor crate.

use concinnity_core::components::GraphicsConfig;
use concinnity_core::ecs::ScheduleMode;
use concinnity_core::error::WorldError;
use concinnity_core::render::rt_geom::RtDynamicMode;
use concinnity_host::store::paths::StateTree;
use std::path::Path;
use tracing_subscriber::EnvFilter;

use crate::app::runloop;
use crate::app::runtime::Runtime;
use crate::app::startup_error::StartupError;
use crate::gfx::quality_preset::QualityPreset;

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
    pub schedule: ScheduleMode,
    /// Capture the last presented frame to this path when the run stops, for
    /// headless verification of the runtime path (`cn run --screenshot`).
    pub screenshot: Option<String>,
    /// Override the world's `GraphicsConfig.max_frames`, bounding the run.
    pub max_frames: Option<u64>,
    /// What the launch asks the engine to arm (`cn run` only; [`run_from`] and
    /// [`Runtime::run`] keep the runtime's own).
    pub launch: LaunchRequest,
}

/// What the launch asked the engine to arm, published as a world resource by
/// [`Runtime::start`]. Build one with [`Runtime::with_launch`]; a runtime built without one
/// runs the shipping behavior. Each `None` defers to that setting's default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaunchRequest {
    /// Keep the presented frame blit-readable for an exit screenshot.
    pub capture: bool,
    /// Run as a development session: capture hot-reload sources, load shaders
    /// disk-first, keep frame capture available, and show the debug HUD.
    pub dev_loop: bool,
    /// Force the DirectX / Vulkan debug layers on or off. `None` follows the
    /// build profile. Metal's layer is set by the process environment instead.
    pub validation: Option<bool>,
    /// Force the master quality preset over the persisted choice, unpersisted.
    pub quality_preset: Option<QualityPreset>,
    /// Force how the ray-tracing acceleration structure tracks moving props.
    /// `None` is [`RtDynamicMode::Auto`].
    pub rt_dynamic: Option<RtDynamicMode>,
    /// Whether skinned meshes join the ray-tracing acceleration structure.
    /// `None` leaves them in.
    pub rt_skinned_geometry: Option<bool>,
}

impl LaunchRequest {
    /// Whether the graphics debug layers run: the request, else the build profile.
    pub fn resolve_validation(&self) -> bool {
        self.validation.unwrap_or(cfg!(debug_assertions))
    }

    /// The master quality preset: the request, else `persisted`. `None` means
    /// neither exists, which is a first launch the caller seeds.
    pub fn resolve_quality_preset(
        &self,
        persisted: Option<QualityPreset>,
    ) -> Option<QualityPreset> {
        self.quality_preset.or(persisted)
    }

    /// How the acceleration structure tracks moving props: the request, else `Auto`.
    pub fn resolve_rt_dynamic(&self) -> RtDynamicMode {
        self.rt_dynamic.unwrap_or_default()
    }

    /// Whether skinned meshes join the acceleration structure: the request, else in.
    pub fn resolve_rt_skinned_geometry(&self) -> bool {
        self.rt_skinned_geometry.unwrap_or(true)
    }

    /// Whether the presented frame stays readable: always in a dev session, and
    /// for a launch that asked for an exit screenshot.
    pub fn frame_capture(&self) -> bool {
        self.dev_loop || self.capture
    }
}

/// Production entry point (`cn run`). Reads the compiled binary blobs from
/// `tree`'s `data/`, written by a prior `cn build`. No debug server, no
/// command channel: a shipped run is neither remotely inspectable nor
/// remotely driven.
pub fn run(tree: &StateTree, options: RunOptions) -> std::io::Result<()> {
    init_logging();

    let mut runtime = Runtime::new()
        .in_tree(tree.clone())
        .with_launch(options.launch);
    let data_dir = tree.data_dir();
    if let Err(error) = load_world(&mut runtime, BlobSource::Directory(&data_dir)) {
        return Err(report_startup_error(error));
    }
    start_runtime(runtime, options).map_err(start_failure)
}

// A refused start, in the form a process exit status is built from.
fn start_failure(e: WorldError) -> std::io::Error {
    std::io::Error::other(format!("failed to start app: {e}"))
}

// Report a fatal startup failure: always to the log, and on screen as well when
// a window can be stood up, so a packaged app that a user double-clicked says
// something rather than exiting silently. The screen blocks until dismissed.
// The process still exits non-zero with the returned error: the screen is how
// the user learns what happened, not a substitute for failing.
fn report_startup_error(error: StartupError) -> std::io::Error {
    tracing::error!("{error}");
    if !crate::error_screen::show("Concinnity", &error.user_message()) {
        // No window, so the log line above is the whole report; repeat it on
        // stderr, which a console user sees regardless of the tracing filter.
        eprintln!("{error}");
    }
    std::io::Error::new(error.io_kind(), error.to_string())
}

// Populate `runtime` with the world `source` holds, refusing a layout that cannot
// hold every blob the world spans.
fn load_world(runtime: &mut Runtime, source: BlobSource<'_>) -> Result<(), StartupError> {
    let max_blob_index = runtime.load_blob_from(&source.primary())?;
    source.check_span(max_blob_index).map_or(Ok(()), Err)
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

/// Production entry point for a shipped app: like `run`, but with the world
/// read from `blob` rather than from `tree`'s own `data/`. The tree (beside the
/// executable or inside an app bundle) still holds `saves/` and `settings`, so
/// pointing the player at a blob elsewhere never relocates what it writes. A
/// missing blob is a hard error rather than a silent no-op -- a packaged app
/// without its data cannot do anything useful. The concinnity-run binary
/// calls this.
pub fn run_from(tree: &StateTree, blob: BlobSource<'_>) -> std::io::Result<()> {
    init_logging();

    let mut runtime = Runtime::new().in_tree(tree.clone());
    if let Err(error) = load_world(&mut runtime, blob) {
        return Err(report_startup_error(error));
    }
    start_runtime(runtime, RunOptions::default()).map_err(start_failure)
}

// Startup and loop entry once the Runtime's world is populated. Registers the
// CTRL+C handler, activates AppKit on macOS, starts the runtime, then drives
// frames -- pipelined (sim thread + render half) or serial (the
// single-threaded world loop) -- until the window closes, a system stops the
// world, or CTRL+C is received. External callers reach this through
// `Runtime::run` / `Runtime::run_with`.
pub(crate) fn start_runtime(mut runtime: Runtime, options: RunOptions) -> Result<(), WorldError> {
    // A host that installed its own subscriber keeps it (`try_init` no-ops),
    // so an embedded app gets logs without wiring any up itself.
    init_logging();
    tracing::info!("Running app...");
    runloop::install_ctrlc_handler(&runtime);

    // Resolved before `start()`, which is where the columns the resolution
    // reads are drained. The runtime caches it, so `start()` publishes this
    // same answer rather than probing the GPU twice.
    let renders = runtime.render_mode().renders();

    if let Some(max) = options.max_frames {
        for config in runtime.world_mut().query_mut::<GraphicsConfig>() {
            config.max_frames = Some(max);
        }
    }
    if options.screenshot.is_some() {
        runtime.launch_mut().capture = true;
    }
    runtime.world_mut().insert_resource(options.schedule);

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

    match options.mode {
        PipelineMode::Pipelined if renders => {
            crate::app::pipeline::run_pipelined(runtime, options.screenshot.as_deref());
        }
        _ => {
            // The serial loop: no per-tick hook; a rendering macOS world pumps
            // the Cocoa run loop, every other case uses the tight loop.
            runloop::run_loop(&mut runtime, cfg!(target_os = "macos") && renders, |_| {});
            capture_exit_screenshot(&mut runtime, options.screenshot.as_deref());
        }
    }

    Ok(())
}

// Report a renderer that refused, on screen where one can be stood up. The
// machine has a GPU (a machine without one runs headless instead of reaching
// here), so what is actionable is the driver, not the hardware.
//
// The screen draws through the same backend that just failed, so it may well
// fail too; that is the console fallback, matching the blob-load path.
fn report_render_failure(cause: &concinnity_core::render::error::RenderError) {
    let message = format!(
        "This app could not start its renderer.\n\n{cause}\n\n\
         Updating your graphics driver is the usual fix."
    );
    if !crate::error_screen::show("Concinnity", &message) {
        eprintln!("{message}");
    }
}

// Capture the last presented frame on the way out of a serial run, when
// requested. The backend is still parked in the world after the loop ends.
fn capture_exit_screenshot(runtime: &mut Runtime, path: Option<&str>) {
    let Some(path) = path else { return };
    let Some(mut backend) = crate::ecs::take_render_backend(runtime.world_mut()) else {
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

    #[test]
    fn the_validation_request_outranks_the_build_profile() {
        let request = |validation| LaunchRequest {
            validation,
            ..Default::default()
        };
        assert_eq!(request(None).resolve_validation(), cfg!(debug_assertions));
        assert!(!request(Some(false)).resolve_validation());
        assert!(request(Some(true)).resolve_validation());
    }

    #[test]
    fn the_quality_preset_request_outranks_the_persisted_choice() {
        let unset = LaunchRequest::default();
        assert_eq!(unset.resolve_quality_preset(None), None);
        assert_eq!(
            unset.resolve_quality_preset(Some(QualityPreset::Auto)),
            Some(QualityPreset::Auto)
        );

        let forced = LaunchRequest {
            quality_preset: Some(QualityPreset::Ultra),
            ..Default::default()
        };
        assert_eq!(
            forced.resolve_quality_preset(Some(QualityPreset::Auto)),
            Some(QualityPreset::Ultra)
        );
        assert_eq!(
            forced.resolve_quality_preset(None),
            Some(QualityPreset::Ultra)
        );
    }

    #[test]
    fn an_unset_rt_dynamic_request_is_auto() {
        assert_eq!(
            LaunchRequest::default().resolve_rt_dynamic(),
            RtDynamicMode::Auto
        );
        for mode in [
            RtDynamicMode::Off,
            RtDynamicMode::Auto,
            RtDynamicMode::Rebuild,
            RtDynamicMode::Tlas,
        ] {
            let request = LaunchRequest {
                rt_dynamic: Some(mode),
                ..Default::default()
            };
            assert_eq!(request.resolve_rt_dynamic(), mode);
        }
    }

    #[test]
    fn skinned_rt_geometry_is_in_unless_the_request_clears_it() {
        let request = |rt_skinned_geometry| LaunchRequest {
            rt_skinned_geometry,
            ..Default::default()
        };
        assert!(request(None).resolve_rt_skinned_geometry());
        assert!(request(Some(true)).resolve_rt_skinned_geometry());
        assert!(!request(Some(false)).resolve_rt_skinned_geometry());
    }

    #[test]
    fn frame_capture_is_armed_by_a_dev_loop_or_a_capture_request() {
        assert!(!LaunchRequest::default().frame_capture());
        let dev_loop = LaunchRequest {
            dev_loop: true,
            ..Default::default()
        };
        assert!(dev_loop.frame_capture());
        let capture = LaunchRequest {
            capture: true,
            ..Default::default()
        };
        assert!(capture.frame_capture());
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

        assert!(BlobSource::File(file).check_span(0).is_none());
        assert!(matches!(
            BlobSource::File(file).check_span(2),
            Some(StartupError::OverflowUnsupported { blob, needed: 2 }) if blob == file
        ));

        // The directory form carries any span, which is why export picks it.
        let dir = Path::new("/apps/MyGame/data");
        assert!(BlobSource::Directory(dir).check_span(0).is_none());
        assert!(BlobSource::Directory(dir).check_span(7).is_none());
    }

    // A tree with no build behind it is a missing-data failure in either
    // layout, which is what makes `cn run` exit non-zero rather than start empty.
    #[test]
    fn loading_a_world_that_was_never_built_reports_missing_data() {
        let tmp = concinnity_testing::TempTree::new();
        let tree = StateTree::at(tmp.path());

        let data_dir = tree.data_dir();
        let mut runtime = Runtime::new().in_tree(tree.clone());
        let error = load_world(&mut runtime, BlobSource::Directory(&data_dir))
            .expect_err("an empty tree has no world");
        assert!(
            matches!(&error, StartupError::MissingData { blob } if *blob == data_dir.join("0")),
            "{error:?}"
        );
        assert_eq!(error.io_kind(), std::io::ErrorKind::NotFound);

        let file = tmp.join("missing.blob");
        let mut runtime = Runtime::new().in_tree(tree);
        let error = load_world(&mut runtime, BlobSource::File(&file))
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
            matches!(
                runtime.run_with(RunOptions::default()),
                Err(WorldError::AlreadyStarted)
            ),
            "a second start is refused"
        );
    }
}
