//! The per-frame seam a dev session's run loop ticks on the main thread.
//!
//! Implemented by the editor's `EditorHook`, the debug server, and the
//! hot-reload driver; the editor's `MultiHook` runs several in order.

use concinnity_core::ecs::World;
use concinnity_engine::app::runtime::Runtime;
use concinnity_engine::shutdown::ShutdownToken;

pub(crate) trait FrameHook: Send {
    // Called once per frame on the main thread, just before the world step.
    // Receives the live world so the hook can inspect (and later mutate) it.
    fn tick(&mut self, world: &mut World);

    // Called once per frame right after `tick`, handing the hook the whole runtime.
    // Lets a hook perform a Runtime-level world swap -- replace the world and
    // re-`start` it -- which `tick`'s `&mut World` cannot reach. The `cn editor`
    // live SAVE uses it to install the recompiled world (carrying the render
    // backend transplanted out of the pre-edit world) without recreating the
    // OS window. Default: no swap.
    fn apply_world_swap(&mut self, _app: &mut Runtime) {}

    // Called once before the run loop starts, handing the hook the runtime's
    // shutdown token. A hook can cancel it to ask the engine to exit cleanly
    // (the run loop checks the token every iteration), e.g. a debug client
    // issuing a `shutdown` command. Default: ignore the token.
    fn attach_shutdown(&mut self, _shutdown: ShutdownToken) {}
}
