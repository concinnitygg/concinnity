// src/ecs/system.rs
//
// The runtime behavior trait every engine system implements, plus its per-step
// control signal. Renderer-free: `System` names only `PipelineContext` (which is
// core), so it lives here where the physics / audio subsystem crates can name it
// without depending on the renderer. The client `ecs` module re-exports both
// under the historical `crate::ecs::*` paths, and its `define_systems!` table
// generates the `SystemAsset` value enum that dispatches them.

use crate::ecs::PipelineContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    // Keep running.
    Continue,
    // This system is finished -- remove it from the active set.
    // The world exits naturally when no systems remain.
    Done,
    // Hard stop -- halt everything immediately.
    #[allow(dead_code)]
    Stop,
}

// System -- has behavior, receives a PipelineContext each tick. Every system
// is internal engine code: `World::build_internal_systems` constructs it from
// world components (via the system's own `new(..)`), so a system is never
// loaded from or written to a blob. `init` runs once at `World::start`; `step`
// runs every tick.
pub trait System: Sized + std::fmt::Debug + 'static {
    fn init(&mut self, _ctx: &mut PipelineContext) {}
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult;
}
