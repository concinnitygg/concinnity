// src/app/run.rs
//
// The interpreted (`cn debug`) run path: compiles world.jsonl fully in memory
// and drives the system loop with the per-frame debug hook. The production
// `cn run` path (compiled-blob playback) lives in the runtime crate's `app::run`.

// The interpreted run path is driven only by the binary's `cn debug` command;
// unreferenced in the FFI lib build.
#![allow(dead_code)]

use crate::app::DebugHook;
use crate::app::state::App;
use crate::world::find_world_jsonl;

// Interpreted entry point (`cn debug`). Compiles world.jsonl fully in memory
// -- shaders, meshes, textures, and all -- then runs the app without reading
// or writing any binary blob files. Always paired with the localhost debug
// server.
pub(crate) fn run_interpreted(
    json_path: Option<&str>,
    debug: Option<Box<dyn DebugHook>>,
) -> std::io::Result<()> {
    concinnity_client::app::run::init_logging();

    let resolved;
    let json_path = match json_path {
        Some(p) if std::path::Path::new(p).exists() => p,
        _ => {
            resolved = find_world_jsonl(None)?;
            resolved.as_str()
        }
    };

    let mut app = App::new();
    *app.world_mut() = super::build::build_world_from_path(json_path).map_err(|e| {
        tracing::error!("Could not build world from {json_path}: {e}");
        e
    })?;

    start_app(app, debug)
}

// Shared startup and loop entry once the App's world is populated.
fn start_app(mut app: App, mut debug: Option<Box<dyn DebugHook>>) -> std::io::Result<()> {
    let shutdown = app.shutdown_token();

    let token = shutdown.clone();
    ctrlc::set_handler(move || {
        tracing::info!("CTRL+C received, cancelling all subsystems");
        token.cancel();
    })
    .expect("failed to set CTRL+C handler");

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
        activate_app_macos();
    }

    if let Err(e) = app.start() {
        eprintln!("Failed to start app: {}", e);
        std::process::exit(1);
    }

    #[cfg(target_os = "macos")]
    if renders {
        run_loop_macos(&mut app, debug.as_mut());
    } else {
        run_loop_default(&mut app, debug.as_mut());
    }

    #[cfg(not(target_os = "macos"))]
    run_loop_default(&mut app, debug.as_mut());

    Ok(())
}

// Non-macOS (and headless macOS) world loop.
fn run_loop_default(app: &mut App, mut debug: Option<&mut Box<dyn DebugHook>>) {
    use crate::ecs::StepResult;

    let shutdown = app.shutdown_token();

    loop {
        if shutdown.is_cancelled() {
            return;
        }

        if let Some(hook) = debug.as_deref_mut() {
            hook.tick(app.world_mut());
        }

        match app.world_step() {
            StepResult::Continue => {}
            StepResult::Stop | StepResult::Done => return,
        }
    }
}

// Activate NSApplication so AppKit windows can be displayed. Must be called
// before any NSWindow is created (i.e. before GraphicsSystem::init()).
#[cfg(target_os = "macos")]
fn activate_app_macos() {
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
    let mtm = objc2::MainThreadMarker::new()
        .expect("activate_app_macos must be called from the main thread");
    let ns_app = NSApplication::sharedApplication(mtm);
    ns_app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    ns_app.activate();
}

// On macOS the Cocoa run loop must be pumped on the main thread each tick so
// AppKit can process window events, draw callbacks, and Metal presentation.
//
// CFRunLoopRunInMode is called with returnAfterSourceHandled=true so it returns
// as soon as one event is processed rather than blocking for the full timeout.
// The outer loop re-enters immediately, draining the event queue before handing
// control to the world step. This keeps the window responsive and allows Metal
// drawable callbacks to fire without delay.
//
// The shutdown token is checked each iteration so CTRL+C exits cleanly.
#[cfg(target_os = "macos")]
fn run_loop_macos(app: &mut App, mut debug: Option<&mut Box<dyn DebugHook>>) {
    use crate::ecs::StepResult;
    use core_foundation::runloop::{CFRunLoopRunInMode, kCFRunLoopDefaultMode};

    let shutdown = app.shutdown_token();

    loop {
        if shutdown.is_cancelled() {
            tracing::info!("Shutdown token cancelled, exiting loop");
            return;
        }

        // Drain all pending AppKit/CoreFoundation events before the world step.
        // returnAfterSourceHandled=true means the call returns after each event,
        // so we loop until the run loop reports nothing left to handle (result != 4).
        loop {
            // kCFRunLoopRunHandledSource = 4; any other value means the queue is empty
            let result = unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.0, true as u8) };
            if result != 4 {
                break;
            }
        }

        if let Some(hook) = debug.as_deref_mut() {
            hook.tick(app.world_mut());
        }

        match app.world_step() {
            StepResult::Continue => {}
            StepResult::Stop | StepResult::Done => return,
        }
    }
}
