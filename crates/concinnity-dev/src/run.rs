//! The interpreted (`cn debug`) run path: compiles world.jsonl fully in memory
//! and drives the system loop with a per-frame hook. The production
//! `cn run` path (compiled-blob playback) lives in the runtime crate's `app::run`.

use concinnity_engine::app::run::LaunchRequest;
use concinnity_engine::app::runtime::Runtime;

use crate::command::resolve_world_path;
use crate::debug::hot_reload::WorldPathHandle;
use crate::frame_hook::FrameHook;

/// The `cn debug` server path: start the localhost debug server on `port`,
/// then run interpreted with it as the per-frame hook. This is the entry point
/// the CLI binary calls; the hook assembly stays inside this crate.
pub fn run_debug(launch: LaunchRequest, json_path: Option<&str>, port: u16) -> std::io::Result<()> {
    concinnity_engine::app::run::init_logging();
    let world_path = resolve_world_path(json_path)?;
    let hook: Box<dyn FrameHook> = match crate::debug::DebugServer::start(port) {
        Ok(srv) => Box::new(srv.with_world_path(WorldPathHandle::new(world_path.as_str()))),
        Err(e) => {
            eprintln!("error: could not start debug server: {e}");
            return Err(e);
        }
    };
    run_interpreted(launch, &world_path, Some(hook))
}

// Interpreted entry point (`cn debug`). Compiles world.jsonl fully in memory
// -- shaders, meshes, textures, and all -- then runs the app without reading
// or writing any binary blob files. Always paired with the localhost debug
// server.
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
// `concinnity_engine::app::runloop` driver; the interpreted path's only
// addition is ticking the frame hook each frame.
pub(crate) fn start_app(
    mut runtime: Runtime,
    mut hook: Option<Box<dyn FrameHook>>,
) -> std::io::Result<()> {
    use concinnity_engine::app::runloop;

    let shutdown = runtime.shutdown_token();
    runloop::install_ctrlc_handler(&runtime);

    // Hand the shutdown token to the hook so a debug client can request a clean
    // exit (the `shutdown` debug tool call). No-op when no hook is present.
    if let Some(hook) = hook.as_mut() {
        hook.attach_shutdown(shutdown.clone());
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
    // Runtime so it can apply a pending world swap (the `cn editor` live SAVE).
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
