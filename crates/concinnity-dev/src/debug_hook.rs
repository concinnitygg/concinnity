// src/debug_hook.rs
// Per-frame injection point for the debug subsystem.
//
// The run loop (`crate::run`) owns the world loop but knows nothing about
// debugging. A `DebugHook` is an optional per-frame callback it invokes on the
// main thread; the only implementation is `crate::debug::DebugServer`. The
// trait stays `pub(crate)` so it is not part of any public surface.

use crate::app::state::App;
use crate::ecs::World;
use concinnity_engine::shutdown::ShutdownToken;

pub(crate) trait DebugHook: Send {
    // Called once per frame on the main thread, just before the world step.
    // Receives the live world so the hook can inspect (and later mutate) it.
    fn tick(&mut self, world: &mut World);

    // Called once per frame right after `tick`, handing the hook the whole app.
    // Lets a hook perform an App-level world swap -- replace the world and
    // re-`start` it -- which `tick`'s `&mut World` cannot reach. The `cn editor`
    // live SAVE uses it to install the recompiled world (carrying the render
    // backend transplanted out of the pre-edit world) without recreating the
    // OS window. Default: no swap.
    fn apply_world_swap(&mut self, _app: &mut App) {}

    // Called once before the run loop starts, handing the hook the app's
    // shutdown token. A hook can cancel it to ask the engine to exit cleanly
    // (the run loop checks the token every iteration), e.g. a debug client
    // issuing a `shutdown` command. Default: ignore the token.
    fn attach_shutdown(&mut self, _shutdown: ShutdownToken) {}
}
