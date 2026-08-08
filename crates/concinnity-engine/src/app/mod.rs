// src/app/mod.rs
// Client application runtime: the world loop and the runtime systems that drive
// a compiled world. The build / edit / debug / preview paths live in the
// editor crate.
// `pub` so the editor crate (which drives a live App via the runtime API) can
// reach these runtime app items through `concinnity_engine::app::*`.
pub mod anim_runtime;

// Process-level thread + memory budgets computed at App start.
pub mod budget;
// Fixed-timestep accumulator advanced before each world step.
pub(crate) mod clock;
pub mod dev_flags;
pub(crate) mod pacing;
pub mod run;
pub mod runloop;
pub mod state;
// Host-memory queries backing the memory budget + the live-usage readout.
pub mod syscpu;
pub mod sysmem;

// Run the app from compiled binary data: the `cn run` production path. `pub`
// so the editor crate's CLI can dispatch to it.
pub use run::run;
