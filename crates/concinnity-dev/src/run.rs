//! The `cn run` / `cn bench` paths: a world named with `-f` compiled fully in
//! memory and hot-reloaded, or the cooked blobs played back (cooked first when
//! there are none), each with the debug server as its per-frame hook when the
//! launch names a port.

use concinnity_engine::launch::{LaunchArgs, LaunchRequest};
use concinnity_host::store::paths::StateTree;
use concinnity_runtime::Runtime;

use crate::command::{Verbosity, discover_world_path, resolve_world_path};
use crate::debug::hot_reload::WorldPathHandle;
use crate::frame_hook::FrameHook;

// Where a run's world comes from.
#[derive(Debug, PartialEq, Eq)]
enum WorldSource<'a> {
    // The named world.jsonl, compiled in memory.
    Named(&'a str),
    // The blobs a cook left in the project's `data/`.
    Cooked,
    // The discovered world, cooked into `data/` before it plays.
    CookFirst,
}

impl<'a> WorldSource<'a> {
    fn of(file: Option<&'a str>, cooked: bool) -> Self {
        match (file, cooked) {
            (Some(file), _) => Self::Named(file),
            (None, true) => Self::Cooked,
            (None, false) => Self::CookFirst,
        }
    }
}

// Whether a run from `source` is a development session: a world compiled from
// its source, or one a debug client drives.
fn dev_loop(source: &WorldSource<'_>, args: &LaunchArgs) -> bool {
    matches!(source, WorldSource::Named(_)) || args.debug_port.is_some()
}

/// Fix the graphics API this process renders with: the launch's, else the
/// player's settings-menu choice in `tree`. Runs before anything cooks or
/// probes the device, so a world's shaders are built for the API that draws them.
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

/// `cn run` and `cn bench`: run the world `file` names, compiled in memory,
/// else the project's cooked blobs, cooking the discovered world first when
/// there are none.
pub(crate) fn run_world(
    file: Option<&str>,
    args: LaunchArgs,
    tree: &StateTree,
) -> std::io::Result<()> {
    select_graphics_api(&args, tree);
    let source = WorldSource::of(
        file,
        concinnity_host::store::blob::primary_in(&tree.data_dir()).is_file(),
    );
    let launch = LaunchRequest {
        dev_loop: dev_loop(&source, &args),
        args,
    };
    match source {
        WorldSource::Named(file) => {
            concinnity_runtime::install_logging(launch.args.quiet);
            let world_path = resolve_world_path(Some(file))?;
            let hook = debug_hook(launch.args.debug_port, Some(&world_path))?;
            run_interpreted(launch, &world_path, hook)
        }
        WorldSource::Cooked => play_cooked(launch, tree),
        WorldSource::CookFirst => {
            crate::command::cook(None, Verbosity::Normal)?;
            play_cooked(launch, tree)
        }
    }
}

// Play the blobs in `tree`'s `data/`: the runtime's own player, or with the
// debug server as its hook when the launch names a port.
fn play_cooked(launch: LaunchRequest, tree: &StateTree) -> std::io::Result<()> {
    if launch.args.debug_port.is_none() {
        return concinnity_runtime::run(tree, launch);
    }
    concinnity_runtime::install_logging(launch.args.quiet);
    let world_path = discover_world_path().ok();
    let hook = debug_hook(launch.args.debug_port, world_path.as_deref())?;
    let mut runtime = Runtime::new().in_tree(tree.clone()).with_launch(launch);
    runtime
        .load_blob()
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    start_app(runtime, hook)
}

// The localhost debug server on `port`, watching `world_path` for hot reload
// when there is one. `None` for a launch that names no port.
fn debug_hook(
    port: Option<u16>,
    world_path: Option<&str>,
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

// Interpreted entry point (`cn run -f`). Compiles world.jsonl fully in memory
// -- shaders, meshes, textures, and all -- then runs the app without reading
// or writing any binary blob files.
pub(crate) fn run_interpreted(
    launch: LaunchRequest,
    json_path: &str,
    hook: Option<Box<dyn FrameHook>>,
) -> std::io::Result<()> {
    let mut runtime = crate::project::runtime().with_launch(launch);
    *runtime.world_mut() = crate::authoring::build_world_from_path(json_path).map_err(|e| {
        tracing::error!("Could not build world from {json_path}: {e}");
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

    // A named world compiles in memory; without one the cooked blobs play,
    // cooked first when there are none.
    #[test]
    fn a_runs_world_is_named_cooked_or_cooked_first() {
        assert_eq!(
            WorldSource::of(Some("w.jsonl"), true),
            WorldSource::Named("w.jsonl")
        );
        assert_eq!(
            WorldSource::of(Some("w.jsonl"), false),
            WorldSource::Named("w.jsonl")
        );
        assert_eq!(WorldSource::of(None, true), WorldSource::Cooked);
        assert_eq!(WorldSource::of(None, false), WorldSource::CookFirst);
    }

    #[test]
    fn a_named_world_or_a_debug_port_makes_a_dev_session() {
        let plain = LaunchArgs::NONE;
        let served = LaunchArgs {
            debug_port: Some(8777),
            ..LaunchArgs::NONE
        };
        assert!(dev_loop(&WorldSource::Named("w.jsonl"), &plain));
        assert!(!dev_loop(&WorldSource::Cooked, &plain));
        assert!(!dev_loop(&WorldSource::CookFirst, &plain));
        assert!(dev_loop(&WorldSource::Cooked, &served));
    }

    #[test]
    fn no_port_starts_no_debug_server() {
        assert!(debug_hook(None, Some("w.jsonl")).unwrap().is_none());
    }
}
