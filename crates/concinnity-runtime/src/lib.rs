//! The std runtime: what runs a compiled world on an operating system.
//! `concinnity_core` makes every decision; this crate holds the run loop and
//! host entry ([`Runtime`], [`run_from`]), the graphics driver that builds a
//! `Box<dyn RenderBackend>` through concinnity-device and submits each frame
//! without naming a concrete backend, and the adapters core's systems are lent
//! through `SystemHost`: the kira audio output, gilrs gamepads, the file stores
//! behind saves and settings, the planet ground thread and the job pool. It
//! also installs crash capture and the allocator. Depends on concinnity-core,
//! concinnity-host and concinnity-device (no concinnity-cook, no image
//! decoders). A host that edits a running world (concinnity-dev) does it
//! through [`live_edit`]; everything else it drives is the ordinary host API.
mod app;
mod blob;
mod ecs;
mod heap;

#[cfg(test)]
mod bench;

// The surface a host embeds: construct a Runtime, populate its world, and drive
// it with `Runtime::run`, or hand the run to `run` (`cn run`) or `run_from` (a
// shipped player, given the world's `BlobSource`).
pub use app::blob_source::BlobSource;
pub use app::run::{RunOptions, init_logging, run, run_from};
pub use app::runloop;
pub use app::runtime::Runtime;
pub use app::startup_error::StartupError;
pub use blob::{WorldLoadError, world_from};
pub use ecs::SYSTEMS;

/// The shader platform this build's rendering backend consumes. The backend is
/// this crate's own compile-time choice, so the crate that knows it is the one
/// that states it: the editor and the cook are handed the value rather than
/// resolving a backend of their own.
pub mod platform;

/// Crash reporting: panic hook, native fault capture, local report files.
/// `pub` so the binaries install the hooks and the editor composes the
/// recent-log ring layer into its tracing subscriber.
pub mod crash;
mod device;
// The kira audio output core's AudioSystem plays through; the one module that
// names kira.
pub(crate) mod audio;
// The standalone startup-error window, shown when a fatal startup failure
// happens before any world exists.
pub(crate) mod error_screen;
mod gfx;
pub(crate) mod input;
pub mod live_edit;
// The thread that builds core PlanetSystem's ground patches.
pub(crate) mod planet;
// The settings file store core's SettingsSystem writes through.
pub(crate) mod settings;
// Behavior state and story saves kept under the host's save directory.
pub(crate) mod behavior;
pub(crate) mod story;
