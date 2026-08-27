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
//! payload codecs (`build`, `decode`). The AUTHORED vocabulary (what a
//! world.jsonl declares) is concinnity-asset, below. The asset COMPILE pipeline
//! is concinnity-cook, which this crate has no edge into.

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

include!(concat!(env!("OUT_DIR"), "/build_source_hash.rs"));
include!(concat!(env!("OUT_DIR"), "/component_schema_hash.rs"));
include!(concat!(env!("OUT_DIR"), "/runtime_asset_docs.rs"));

/// Hash of the postcard-visible schema this build was compiled against, stamped
/// into every blob header. A blob whose stored hash differs was written by a
/// different engine schema and fails the load check instead of mis-decoding.
///
/// Mixed from the three pieces of that schema: the authored asset types, this
/// crate's divergent runtime structs and component registry (whose list order
/// is the tag), and the blob container's record shapes. No manually maintained
/// version number, and no crate reaching into another's directory to compute it.
pub const SCHEMA_HASH: u32 = mix(&[
    concinnity_asset::SOURCE_HASH,
    COMPONENT_SCHEMA_HASH,
    blob::RECORD_SCHEMA_HASH,
]);

// FNV-1a over the parts, order-significant.
const fn mix(parts: &[u32]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    let mut i = 0;
    while i < parts.len() {
        let bytes = parts[i].to_le_bytes();
        let mut b = 0;
        while b < bytes.len() {
            hash ^= bytes[b] as u32;
            hash = hash.wrapping_mul(0x0100_0193);
            b += 1;
        }
        i += 1;
    }
    hash
}

mod app;
pub mod behavior;
pub mod blob;
pub mod build;
pub mod components;
pub mod decode;
pub mod ecs;
pub mod geometry;
pub mod gfx;
pub mod math;
pub mod physics;
pub mod platform;
pub mod resource;
pub mod result;
pub mod spawn;
#[cfg(test)]
mod test_support;

// The headless driver over a world, named at the crate root because it is the
// counterpart to `ecs::World` rather than a corner of the module tree.
pub use app::App;
