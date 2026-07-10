// src/app/runloop.rs
//
// The shared render/event loop that drives a live `App`. Both the compiled
// `cn run` runtime (`run::start_runtime`) and the interpreted `cn debug` path
// (in the editor crate) pump the same loop; the only difference is the per-tick
// hook the debug path threads through to run its DebugHook. Keeping the platform
// event-pump and window-activation glue in one place means it is not duplicated
// per entry point.
//
// On macOS the world loop must pump the Cocoa run loop on the main thread each
// tick so AppKit (GLFW window creation, Metal pipeline compilation, event
// dispatch) can process its callbacks and Metal drawable presentation fires. On
// all other platforms a tight Rust loop is used, which is what the Vulkan /
// DirectX renderers expect.

use crate::app::state::App;
use crate::ecs::StepResult;

// Install the process CTRL+C handler that cancels the app's shutdown token, so
// the render loop exits cleanly. Panics if a handler is already installed; only
// one entry point installs it per process.
pub fn install_ctrlc_handler(app: &App) {
    let token = app.shutdown_token();
    ctrlc::set_handler(move || {
        tracing::info!("CTRL+C received, cancelling all subsystems");
        token.cancel();
    })
    .expect("failed to set CTRL+C handler");
}

// Activate NSApplication so AppKit windows can be displayed. Must be called
// before any NSWindow is created (i.e. before GraphicsSystem::init()).
#[cfg(target_os = "macos")]
pub fn activate_app_macos() {
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
    let mtm = objc2::MainThreadMarker::new()
        .expect("activate_app_macos must be called from the main thread");
    let ns_app = NSApplication::sharedApplication(mtm);
    ns_app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    ns_app.activate();
}

// Drive the world loop to completion. Each iteration: exit if the shutdown token
// is cancelled; on macOS, when `pump_events` is set (a window is present), drain
// the pending AppKit/CoreFoundation events so the window stays responsive and
// Metal drawable callbacks fire; run the per-tick `on_tick` hook; then step the
// world, stopping on Stop/Done. `on_tick` is where the interpreted debug path
// ticks its DebugHook; the runtime passes a no-op.
//
// `pump_events` is only meaningful on macOS (it gates the Cocoa pump): the
// caller sets it from whether the world actually renders, so a headless macOS
// world uses the same tight loop as every other platform.
pub fn run_loop(app: &mut App, pump_events: bool, mut on_tick: impl FnMut(&mut App)) {
    let shutdown = app.shutdown_token();

    loop {
        if shutdown.is_cancelled() {
            tracing::info!("Shutdown token cancelled, exiting loop");
            return;
        }

        #[cfg(target_os = "macos")]
        if pump_events {
            drain_cocoa_events();
        }
        #[cfg(not(target_os = "macos"))]
        let _ = pump_events;

        on_tick(app);

        match app.world_step() {
            StepResult::Continue => {}
            StepResult::Stop | StepResult::Done => return,
        }
    }
}

// Drain all currently-pending Cocoa events without blocking.
// CFRunLoopRunInMode is called with returnAfterSourceHandled=true so it returns
// as soon as one source is handled (result == kCFRunLoopRunHandledSource == 4);
// any other result means the queue is empty, so the drain stops and control
// returns to the world step.
#[cfg(target_os = "macos")]
fn drain_cocoa_events() {
    use core_foundation::runloop::{CFRunLoopRunInMode, kCFRunLoopDefaultMode};
    loop {
        let result = unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.0, true as u8) };
        if result != 4 {
            break;
        }
    }
}
