//! Client application runtime: the world loop and the runtime systems that drive
//! a compiled world. The build / edit / debug / preview paths live in the
//! editor crate.
//! `pub` so the editor crate (which drives a live App via the runtime API) can
//! reach these runtime app items through `concinnity_engine::app::*`.
pub mod anim_runtime;

/// Process-level thread + memory budgets computed at App start.
pub mod budget;
// Fixed-timestep accumulator advanced before each world step.
pub(crate) mod clock;
pub mod dev_flags;
/// Long-session drift of process memory against the tracked heap.
pub mod mem_drift;
pub(crate) mod pacing;
// Pipelined frame driver: sim thread + render half + the channel pair.
pub(crate) mod pipeline;
pub mod run;
pub mod runloop;
// Classification of fatal startup failures into a log line plus a sentence for
// the error screen. `pub` so a host binary can report one the same way.
pub(crate) mod startup_error;
/// The `App` value a host constructs, starts, and steps.
pub mod state;
/// Host-memory queries backing the memory budget + the live-usage readout.
pub mod syscpu;
pub mod sysmem;

// Run the app from compiled binary data: the `cn run` production path. `pub`
// so the editor crate's CLI can dispatch to it.
pub use run::run;
