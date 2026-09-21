//! The interpreted (`cn debug`) run path: compiles world.jsonl fully in memory
//! and drives the system loop with the per-frame debug hook. The production
//! `cn run` path (compiled-blob playback) lives in the runtime crate's `app::run`.

use concinnity_cook::authoring::world::find_world_jsonl;
use concinnity_engine::app::run::LaunchRequest;
use concinnity_engine::app::runtime::Runtime;

use crate::debug::hot_reload::WorldPathHandle;
use crate::debug_hook::DebugHook;

/// The `cn debug` server path: start the localhost debug server on `port`,
/// then run interpreted with it as the per-frame hook. This is the entry point
/// the CLI binary calls; the hook assembly stays inside this crate.
pub fn run_debug(launch: LaunchRequest, json_path: Option<&str>, port: u16) -> std::io::Result<()> {
    concinnity_engine::app::run::init_logging();
    let world_path = resolve_world_path(json_path)?;
    let debug_hook: Box<dyn DebugHook> = match crate::debug::DebugServer::start(port) {
        Ok(srv) => Box::new(srv.with_world_path(WorldPathHandle::new(world_path.as_str()))),
        Err(e) => {
            eprintln!("error: could not start debug server: {e}");
            return Err(e);
        }
    };
    run_interpreted(launch, &world_path, Some(debug_hook))
}

// The world an interpreted run should load: the `-f` path when the caller gave
// one, otherwise whatever discovery finds.
//
// A `-f` that does not exist is an error rather than a fall back to discovery.
// Falling back runs a different world under the name the caller asked for, and
// says nothing: the loop starts, frames advance, and every trust check passes
// against a scene nobody asked for.
fn resolve_world_path(json_path: Option<&str>) -> std::io::Result<String> {
    match json_path {
        Some(p) if std::path::Path::new(p).exists() => Ok(p.to_string()),
        Some(p) => Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("world file not found: {p}"),
        )),
        None => find_world_jsonl(crate::project::worlds_dir().as_deref(), None),
    }
}

// Interpreted entry point (`cn debug`). Compiles world.jsonl fully in memory
// -- shaders, meshes, textures, and all -- then runs the app without reading
// or writing any binary blob files. Always paired with the localhost debug
// server.
pub(crate) fn run_interpreted(
    launch: LaunchRequest,
    json_path: &str,
    debug: Option<Box<dyn DebugHook>>,
) -> std::io::Result<()> {
    let mut runtime = crate::project::runtime().with_launch(launch);
    *runtime.world_mut() = crate::authoring::build_world_from_path(json_path).map_err(|e| {
        tracing::error!("Could not build world from {json_path}: {e}");
        e
    })?;

    start_app(runtime, debug)
}

// Shared startup and loop entry once the Runtime's world is populated. The world
// loop itself (and the platform event-pump + window activation) is the shared
// `concinnity_engine::app::runloop` driver; the interpreted path's only
// addition is ticking the debug hook each frame.
pub(crate) fn start_app(
    mut runtime: Runtime,
    mut debug: Option<Box<dyn DebugHook>>,
) -> std::io::Result<()> {
    use concinnity_engine::app::runloop;

    let shutdown = runtime.shutdown_token();
    runloop::install_ctrlc_handler(&runtime);

    // Hand the shutdown token to the debug hook so a debug client can request
    // a clean exit (the `shutdown` debug tool call). No-op when no hook is present.
    if let Some(hook) = debug.as_mut() {
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

    // The interpreted path ticks its debug hook each frame before the world step.
    // After the tick (which sees only `&mut World`), the hook is given the whole
    // Runtime so it can apply a pending world swap (the `cn editor` live SAVE).
    let on_tick = |runtime: &mut Runtime| {
        if let Some(hook) = debug.as_deref_mut() {
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
    use super::resolve_world_path;

    #[test]
    fn an_existing_path_is_taken_as_given() {
        let dir = tempfile::tempdir().expect("temp dir");
        let world = dir.path().join("scene.jsonl");
        std::fs::write(&world, "").expect("write");
        let given = world.to_string_lossy().into_owned();
        assert_eq!(resolve_world_path(Some(&given)).expect("resolves"), given);
    }

    // The whole point of the arm: a named world that is not there fails loudly
    // instead of silently becoming whatever discovery turns up.
    #[test]
    fn a_missing_path_errors_rather_than_falling_back() {
        let dir = tempfile::tempdir().expect("temp dir");
        let missing = dir.path().join("absent.jsonl");
        let given = missing.to_string_lossy().into_owned();
        let err = resolve_world_path(Some(&given)).expect_err("missing world is an error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("absent.jsonl"), "{err}");
    }
}
