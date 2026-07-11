// src/run.rs
//
// The interpreted (`cn debug`) run path: compiles world.jsonl fully in memory
// and drives the system loop with the per-frame debug hook. The production
// `cn run` path (compiled-blob playback) lives in the runtime crate's `app::run`.

use crate::app::state::App;
use crate::debug_hook::DebugHook;
use crate::world::find_world_jsonl;

// Interpreted entry point (`cn debug`). Compiles world.jsonl fully in memory
// -- shaders, meshes, textures, and all -- then runs the app without reading
// or writing any binary blob files. Always paired with the localhost debug
// server.
pub(crate) fn run_interpreted(
    json_path: Option<&str>,
    debug: Option<Box<dyn DebugHook>>,
) -> std::io::Result<()> {
    concinnity_engine::app::run::init_logging();

    let resolved;
    let json_path = match json_path {
        Some(p) if std::path::Path::new(p).exists() => p,
        _ => {
            resolved = find_world_jsonl(None)?;
            resolved.as_str()
        }
    };

    // Hand the resolved world path to the engine so its hot-reload watcher can
    // subscribe to world.jsonl. The engine no longer discovers it (that lookup
    // is authoring I/O in concinnity-cook, which the runtime does not link).
    concinnity_engine::app::dev_flags::set_world_jsonl_path(Some(json_path.to_string()));

    let mut app = App::new();
    *app.world_mut() = crate::build_world_from_path(json_path).map_err(|e| {
        tracing::error!("Could not build world from {json_path}: {e}");
        e
    })?;

    start_app(app, debug)
}

// Shared startup and loop entry once the App's world is populated. The world
// loop itself (and the platform event-pump + window activation) is the shared
// `concinnity_engine::app::runloop` driver; the interpreted path's only
// addition is ticking the debug hook each frame.
pub(crate) fn start_app(
    mut app: App,
    mut debug: Option<Box<dyn DebugHook>>,
) -> std::io::Result<()> {
    use crate::app::runloop;

    let shutdown = app.shutdown_token();
    runloop::install_ctrlc_handler(&app);

    // Hand the shutdown token to the debug hook so a debug client can request
    // a clean exit (the `shutdown` WS command). No-op when no hook is present.
    if let Some(hook) = debug.as_mut() {
        hook.attach_shutdown(shutdown.clone());
    }

    // Resolved before `start()` (while the GraphicsConfig is still present),
    // so the render-loop choice doesn't depend on the config component, which
    // `start()` drains. Only the macOS path branches on it.
    #[cfg(target_os = "macos")]
    let renders = app.world().renders();

    // On macOS, NSApplication is a per-process singleton. Activate it once
    // before the first NSWindow is created.
    #[cfg(target_os = "macos")]
    if renders {
        runloop::activate_app_macos();
    }

    if let Err(e) = app.start() {
        eprintln!("Failed to start app: {}", e);
        std::process::exit(1);
    }

    // The interpreted path ticks its debug hook each frame before the world step.
    // After the tick (which sees only `&mut World`), the hook is given the whole
    // App so it can apply a pending world swap (the `cn editor` live SAVE).
    let on_tick = |app: &mut App| {
        if let Some(hook) = debug.as_deref_mut() {
            hook.tick(app.world_mut());
            hook.apply_world_swap(app);
        }
    };

    #[cfg(target_os = "macos")]
    runloop::run_loop(&mut app, renders, on_tick);
    #[cfg(not(target_os = "macos"))]
    runloop::run_loop(&mut app, false, on_tick);

    Ok(())
}
