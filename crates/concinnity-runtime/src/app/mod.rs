//! The run loop: the `Runtime` a host constructs, starts and steps, the serial
//! and pipelined frame drivers, and the `cn run` and shipped-player entry
//! points. The build / edit / debug / preview paths live in concinnity-dev.

pub(crate) mod blob;
// The role the frame threads take from the published thread budget.
pub(crate) mod budget;
// Fixed-timestep accumulator advanced before each world step.
pub(crate) mod clock;
pub(crate) mod frame_gate;
pub(crate) mod launch;
pub(crate) mod loop_choice;
// The windowed loop's implementation of the core driver trait.
mod driver;
pub(crate) mod pacing;
// Pipelined frame driver: sim thread + render half + the channel pair.
pub(crate) mod pipeline;
/// How a world runs -- windowed or headless -- resolved once before it starts.
pub(crate) mod render_mode;
pub(crate) mod run;
pub mod runloop;
pub(crate) mod runtime;
pub(crate) mod startup_error;
