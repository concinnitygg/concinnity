//! concinnity-engine: the engine itself, `no_std`, everything that runs a
//! world short of the machine it runs on. The ECS storage and scheduler, the
//! component types and the registry built from them, every system (physics,
//! animation, behavior, audio mixing, UI, spawning, settings), the render
//! preparation and render graph the device backends draw from, the bake
//! kernels and payload codecs, and the `.cnb` blob container format a cooked
//! world travels in.
//!
//! It builds on `concinnity-core`, the shared foundation (math, the allocation
//! layer, thread fan-out). `concinnity-runtime` is the std host that runs it,
//! lending threads, audio output, input and file stores through `SystemHost`;
//! `concinnity-device` implements the `RenderBackend` seam in [`render`]. The
//! build side, `concinnity-cook`, depends on this crate for the component
//! vocabulary and this crate has no edge back into it.
#![cfg_attr(
    test,
    expect(
        clippy::float_cmp,
        reason = "tests assert exact, deterministic float results"
    )
)]
#![no_std]

extern crate alloc;
// The derive macros emit `::concinnity_engine::` paths, which resolve here too.
extern crate self as concinnity_engine;
#[cfg(test)]
extern crate std;

/// The version of everything a cooked blob's bytes depend on: the
/// postcard-visible component schema, the blob container's record shapes, and
/// the payload formats in [`bake`]. Stamped into every blob header and folded
/// into the cook's payload cache key.
///
/// A blob whose stored version differs was written by a different engine and
/// fails the load check instead of mis-decoding.
///
/// # Bumping
///
/// Bump this when a change makes previously cooked bytes unreadable or stale:
///
/// - reordering a serialized struct's fields, or an enum's variants
/// - swapping a serialized field's type for one that encodes to the same width
/// - changing what a `build` payload serializer writes
///
/// Adding or removing a serialized field needs no bump: a blob frame is
/// length-delimited and `codec::decode_exact` rejects one that does not decode
/// exactly, so a stale record fails the load on its own.
pub const SCHEMA_VERSION: u32 = 3;

pub mod access_ids;
mod app;
pub mod audio;
pub mod bake;
pub mod benchmark;
pub mod blob;
pub mod budget;
pub mod codec;
pub mod components;
pub mod defaults;
pub mod ecs;
pub mod environment;
pub mod error;
pub(crate) mod frame_report;
pub mod geometry;
pub mod gfx;
pub mod input;
pub mod launch;
pub mod presentation;
pub mod profile;
pub mod protocol;
pub mod render;
pub mod resource;
pub mod schedule;
pub mod settings;
pub mod shutdown;
pub mod simulation;
pub mod story;
#[cfg(test)]
mod test_support;
pub mod transform;

// The headless driver over a world and the trait any loop that runs one
// implements, named at the crate root because they are the counterpart to
// `ecs::World` rather than a corner of the module tree.
pub use app::{App, Driver, TickAccumulator};
