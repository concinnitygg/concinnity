//! The contract a host's loop implements, and the headless loop's own
//! implementation of it.

use alloc::boxed::Box;

use crate::app::App;
use crate::ecs::World;
use crate::result::CnResult;

/// A loop that runs a [`World`].
///
/// [`App`] is the implementation with nothing underneath it: a fixed virtual
/// timestep, no window, and no wall clock. A host that has an operating system
/// implements this over its own loop, and what such a host adds -- pacing a
/// frame against a display, following real elapsed time, catching a signal --
/// stays on its side of the seam.
///
/// [`run`](Driver::run) reports whether the run failed rather than how it
/// ended. A [`StepResult`](crate::ecs::StepResult) is a system's verdict on its
/// own tick, and a host ends a run for reasons no system sees, so a driver that
/// reported one would owe readings for them.
///
/// `run` consumes the driver, since a run ends the world it was handed;
/// [`into_world`](Driver::into_world) is the other way out, for a caller that
/// wants the world back instead of run.
pub trait Driver {
    /// Build the world's systems and run their `init`.
    fn start(&mut self) -> Result<(), CnResult>;

    /// Run the world until it ends.
    fn run(self: Box<Self>) -> Result<(), CnResult>;

    /// Take the world back instead of running it.
    fn into_world(self: Box<Self>) -> World;
}

impl Driver for App {
    fn start(&mut self) -> Result<(), CnResult> {
        App::start(self)
    }

    fn run(mut self: Box<Self>) -> Result<(), CnResult> {
        App::run(&mut self).map(|_| ())
    }

    fn into_world(self: Box<Self>) -> World {
        self.world
    }
}
