//! concinnity-core: the engine's RUNTIME vocabulary and the CPU compute over it.
//! The types the renderer, the cook pipeline, the subsystem crates, and the
//! editor all have to agree on and none of them owns: the backend-agnostic GPU
//! data layouts the CPU and the shaders both name, the transform and skeleton
//! math those layouts are expressed in, the ECS storage mechanism plus the
//! component definitions and the registry built from them, the post-process /
//! quality setting structs, the behavior virtual machine that evaluates
//! declarative logic, and the `.cnb` blob container format the cooked world
//! travels in.
//!
//! Above that vocabulary, the kernels that compute over it and belong to no
//! single consumer: skinning and pose blending, IK, LOD decimation,
//! rasterisation, IBL convolution, the procedural geometry generators, and the
//! payload codecs (`build`, `decode`). Each asset's AUTHORED schema (what a
//! world.jsonl declares) sits in `components` beside the runtime half it bakes
//! into; the build-only assets, which never reach a running world, live with
//! their registry group in concinnity-cook, the build-side crate that also owns
//! the asset COMPILE pipeline and that this crate has no edge into.

#![no_std]
// The blob container and the payload codecs parse bytes the process did not
// produce, so a panic here is a crash on a corrupt file rather than a bug.
// Invariants that genuinely cannot fail use `expect` with the invariant named;
// tests unwrap freely.
#![warn(clippy::unwrap_used)]
#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "tests unwrap freely; the crate-wide warn covers non-test code"
    )
)]

extern crate alloc;
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
/// - changing what a `build` payload serialiser writes
///
/// Adding or removing a serialized field needs no bump: a blob frame is
/// length-delimited and `blob::decode_exact` rejects one that does not decode
/// exactly, so a stale record fails the load on its own.
pub const SCHEMA_VERSION: u32 = 1;

mod app;
pub mod bake;
pub mod behavior;
pub mod blob;
pub mod components;
pub mod decode;
pub mod defaults;
pub mod ecs;
pub mod geometry;
pub mod gfx;
pub mod math;
pub mod memory;
pub mod physics;
pub mod platform;
pub mod render;
pub mod resource;
pub mod result;
pub mod spawn;
#[cfg(test)]
mod test_support;

// The headless driver over a world and the trait any loop that runs one
// implements, named at the crate root because they are the counterpart to
// `ecs::World` rather than a corner of the module tree.
pub use app::{App, Driver};
