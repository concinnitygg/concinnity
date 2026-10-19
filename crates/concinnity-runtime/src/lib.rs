//! The std runtime: what runs a compiled world on an operating system.
//! `concinnity_core` makes every decision; this crate holds the run loop and
//! host entry ([`Runtime`], [`run_from`]), the graphics driver that builds a
//! `Box<dyn RenderBackend>` through concinnity-device and submits each frame
//! without naming a concrete backend, and what core's systems are lent through
//! `SystemHost`: the job pool, the planet ground thread, the kira audio output,
//! gilrs gamepads and the file stores behind saves and settings. It also
//! installs crash capture and the allocator. Depends on concinnity-core,
//! concinnity-host and concinnity-device (no concinnity-cook, no image
//! decoders). A host that edits a running world (concinnity-dev) does it
//! through [`live_edit`]; everything else it drives is the ordinary host API.

// The run loop, its frame drivers and the `cn run` / shipped-player entry.
mod app;
// The system table and what this host lends core's systems.
mod lend;
// The file stores behind core's persistence seams.
mod persist;
// The graphics driver: GraphicsSystem and its streaming load threads.
mod render;

// The kira audio output core's AudioSystem plays through; the one module that
// names kira.
mod audio;
// The gilrs gamepads core's InputSystem merges; the one module that names gilrs.
mod gamepad;
mod heap;

/// Crash reporting: panic hook, native fault capture, local report files.
/// `pub` so the binaries install the hooks and the editor composes the
/// recent-log ring layer into its tracing subscriber.
pub mod crash;
pub mod live_edit;
/// The shader platform this build's rendering backend consumes. The backend is
/// this crate's own compile-time choice, so the crate that knows it is the one
/// that states it: the editor and the cook are handed the value rather than
/// resolving a backend of their own.
pub mod platform;

// The surface a host embeds: construct a Runtime, populate its world, and drive
// it with `Runtime::run`, or hand the run to `run` (`cn run`) or `run_from` (a
// shipped player, given the world's `BlobSource`).
pub use app::blob::{BlobSource, WorldLoadError, world_from};
pub use app::run::{RunOptions, init_logging, run, run_from};
pub use app::runloop;
pub use app::runtime::Runtime;
pub use app::startup_error::StartupError;
pub use lend::SYSTEMS;
