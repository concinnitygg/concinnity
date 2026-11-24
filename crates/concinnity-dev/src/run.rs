//! The `cn run` / `cn bench` paths: a file named with `-f` compiled fully in
//! memory and hot-reloaded, or the project's application cooked and its blobs
//! played back, each with the debug server as its per-frame hook when the
//! launch names a port.

use concinnity_engine::launch::{LaunchArgs, LaunchRequest};
use concinnity_host::store::paths::StateTree;
use concinnity_runtime::Runtime;

use crate::command::{Target, Verbosity};
use crate::debug::hot_reload::WorldPathHandle;
use crate::frame_hook::FrameHook;
use std::path::{Path, PathBuf};

// Where a run's assets come from.
#[derive(Debug, PartialEq, Eq)]
enum RunSource {
    // The named file, compiled in memory.
    Named(PathBuf),
    // The blobs a cook left in the project's `data/`, with no application to
    // cook.
    Cooked,
    // The project's application, cooked into `data/` before it plays.
    CookFirst(PathBuf),
}

impl RunSource {
    // An application always cooks first, so the run plays what it says. Blobs
    // play as they are only when there is no application to cook them from.
    fn of(target: Target<'_>, cooked: bool) -> std::io::Result<Self> {
        match target.find()? {
            Some(path) if target.file.is_some() => Ok(Self::Named(path)),
            Some(path) => Ok(Self::CookFirst(path)),
            None if cooked => Ok(Self::Cooked),
            None => Err(crate::command::no_application()),
        }
    }
}

// Whether a run from `source` is a development session: a file compiled from
// its source, or one a debug client drives.
fn dev_loop(source: &RunSource, args: &LaunchArgs) -> bool {
    matches!(source, RunSource::Named(_)) || args.debug_port.is_some()
}

/// Fix the graphics API this process renders with: the launch's, else the
/// player's settings-menu choice in `tree`. Runs before anything cooks or
/// probes the device, so an application's shaders are built for the API that
/// draws them.
pub(crate) fn select_graphics_api(args: &LaunchArgs, tree: &StateTree) {
    match args.graphics_api {
        Some(api) => {
            concinnity_runtime::platform::select(Some(api));
        }
        None => {
            concinnity_runtime::platform::select_persisted(Some(tree));
        }
    }
}

/// `cn run` and `cn bench`: run the file `-f` names, compiled in memory, else
/// cook the project's application and play its blobs, else play the blobs
/// already in place.
pub(crate) fn run_app(
    target: Target<'_>,
    args: LaunchArgs,
    tree: &StateTree,
) -> std::io::Result<()> {
    select_graphics_api(&args, tree);
    let source = RunSource::of(
        target,
        concinnity_host::store::blob::primary_in(&tree.data_dir()).is_file(),
    )?;
    let launch = LaunchRequest {
        dev_loop: dev_loop(&source, &args),
        args,
    };
    match source {
        RunSource::Named(path) => {
            concinnity_runtime::install_logging(launch.args.quiet);
            let hook = debug_hook(launch.args.debug_port, Some(&path))?;
            run_interpreted(launch, &path, hook)
        }
        RunSource::Cooked => play_cooked(launch, tree, None),
        RunSource::CookFirst(path) => {
            crate::command::cook_path(&path, Verbosity::Normal)?;
            play_cooked(launch, tree, Some(&path))
        }
    }
}

// Play the blobs in `tree`'s `data/`: the runtime's own player, or with the
// debug server as its hook, watching `source` for hot reload, when the launch
// names a port.
fn play_cooked(
    launch: LaunchRequest,
    tree: &StateTree,
    source: Option<&Path>,
) -> std::io::Result<()> {
    if launch.args.debug_port.is_none() {
        return concinnity_runtime::run(tree, launch);
    }
    concinnity_runtime::install_logging(launch.args.quiet);
    let hook = debug_hook(launch.args.debug_port, source)?;
    let mut runtime = Runtime::new().in_tree(tree.clone()).with_launch(launch);
    runtime.load_blob()?;
    start_app(runtime, hook)
}

// The localhost debug server on `port`, watching `world_path` for hot reload
// when there is one. `None` for a launch that names no port.
fn debug_hook(
    port: Option<u16>,
    world_path: Option<&Path>,
) -> std::io::Result<Option<Box<dyn FrameHook>>> {
    let Some(port) = port else {
        return Ok(None);
    };
    let server = crate::debug::DebugServer::start(port).inspect_err(|e| {
        eprintln!("error: could not start debug server: {e}");
    })?;
    let hook: Box<dyn FrameHook> = match world_path {
        Some(path) => Box::new(server.with_world_path(WorldPathHandle::new(path))),
        None => Box::new(server),
    };
    Ok(Some(hook))
}

// Interpreted entry point (`cn run -f`). Compiles the file fully in memory
// -- shaders, meshes, textures, and all -- then runs the app without reading
// or writing any binary blob files.
pub(crate) fn run_interpreted(
    launch: LaunchRequest,
    json_path: &Path,
    hook: Option<Box<dyn FrameHook>>,
) -> std::io::Result<()> {
    let mut runtime = crate::project::runtime().with_launch(launch);
    *runtime.world_mut() = crate::authoring::build_world_from_path(json_path).map_err(|e| {
        tracing::error!("Could not build {}: {e}", json_path.display());
        e
    })?;

    start_app(runtime, hook)
}

// Shared startup and loop entry once the Runtime's world is populated. The world
// loop itself (and the platform event-pump + window activation) is the shared
// `concinnity_runtime::runloop` driver; the interpreted path's only
// addition is ticking the frame hook each frame.
pub(crate) fn start_app(
    mut runtime: Runtime,
    mut hook: Option<Box<dyn FrameHook>>,
) -> std::io::Result<()> {
    use concinnity_runtime::runloop;

    let shutdown = runtime.shutdown_token();
    runloop::install_ctrlc_handler(&runtime);

    // Hand the shutdown token to the hook so a debug client can request a clean
    // exit (the `shutdown` debug tool call). No-op when no hook is present.
    if let Some(hook) = hook.as_mut() {
        hook.attach_shutdown(shutdown);
    }

    // Resolved before `start()`, which is where the columns the resolution
    // reads are drained. The runtime caches it, so `start()` publishes this
    // same answer. Only the macOS path branches on it.
    #[cfg(target_os = "macos")]
    let renders = runtime.render_mode().renders();

    // On macOS, NSApplication is a per-process singleton. Activate it once
    // before the first NSWindow is created.
    #[cfg(target_os = "macos")]
    if renders {
        runloop::activate_app_macos();
    }

    if let Err(e) = runtime.start() {
        // Returned rather than exiting the process, so the world's systems
        // (and the GPU resources they hold) still drop on the way out.
        tracing::error!("failed to start app: {e}");
        return Err(std::io::Error::other(format!("failed to start app: {e}")));
    }

    // The interpreted path ticks its hook each frame before the world step.
    // After the tick (which sees only `&mut World`), the hook is given the whole
    // Runtime so it can apply a pending world swap (the `cn edit` live SAVE).
    let on_tick = |runtime: &mut Runtime| {
        if let Some(hook) = hook.as_deref_mut() {
            hook.tick(runtime.world_mut());
            hook.apply_world_swap(runtime);
        }
    };

    #[cfg(target_os = "macos")]
    runloop::run_loop(&mut runtime, renders, on_tick);
    #[cfg(not(target_os = "macos"))]
    runloop::run_loop(&mut runtime, false, on_tick);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // A named file compiles in memory; an application cooks before it plays,
    // even over existing blobs; blobs play as they are only with no
    // application to cook, and with neither the run reports the missing one.
    #[test]
    fn a_run_is_named_cooked_first_or_cooked() {
        let _guard = crate::test_support::lock();
        let dir = concinnity_testing::TempTree::new();
        crate::project::open(StateTree::at(dir.path()));
        let file = dir.path().join("scene.jsonl");
        std::fs::write(&file, "").unwrap();
        let named = file.to_string_lossy().into_owned();
        let none = (
            RunSource::of(Target::default(), true).unwrap(),
            RunSource::of(Target::default(), false).unwrap_err().kind(),
        );
        let app = dir.path().join("app.jsonl");
        std::fs::write(&app, "[\"AppConfig\"]\n").unwrap();
        let sources = [
            RunSource::of(Target::file(&named), false).unwrap(),
            RunSource::of(Target::default(), true).unwrap(),
        ];
        crate::test_support::isolate_state_dir();

        assert_eq!(none, (RunSource::Cooked, std::io::ErrorKind::NotFound));
        assert_eq!(sources, [RunSource::Named(file), RunSource::CookFirst(app)]);
    }

    #[test]
    fn a_named_file_or_a_debug_port_makes_a_dev_session() {
        let plain = LaunchArgs::NONE;
        let served = LaunchArgs {
            debug_port: Some(8777),
            ..LaunchArgs::NONE
        };
        assert!(dev_loop(&RunSource::Named("w.jsonl".into()), &plain));
        assert!(!dev_loop(&RunSource::Cooked, &plain));
        assert!(!dev_loop(&RunSource::CookFirst("w.jsonl".into()), &plain));
        assert!(dev_loop(&RunSource::Cooked, &served));
    }

    #[test]
    fn no_port_starts_no_debug_server() {
        assert!(
            debug_hook(None, Some(Path::new("w.jsonl")))
                .unwrap()
                .is_none()
        );
    }
}
