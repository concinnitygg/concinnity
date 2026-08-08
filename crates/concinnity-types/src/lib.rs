// src/lib.rs
//
// concinnity-types: the engine's RUNTIME vocabulary. The types the renderer,
// the cook pipeline, the subsystem crates, and the editor all have to agree on
// and none of them owns: the backend-agnostic GPU data layouts the CPU and the
// shaders both name, the transform and skeleton math those layouts are
// expressed in, the ECS component definitions and the registry built from them,
// and the post-process / quality setting structs.
//
// Data and the small total functions over it. CPU compute over that vocabulary
// -- skinning, IK, LOD decimation, rasterisation, IBL convolution -- lives in
// concinnity-core, which sits directly above this crate and re-exports it.
// The AUTHORED vocabulary (what a world.jsonl declares) is concinnity-asset,
// below.
//
// The crate is `#![no_std]` because nothing here needs an operating system, not
// as a portability goal in itself: it opens no file, spawns no thread, and
// reads no clock. `libm` supplies the f32 transcendentals `core` leaves out
// (see `math`).

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod assets;
pub mod ecs;
pub mod gfx;
pub mod math;
pub mod platform;
pub mod resource;
pub mod result;
#[cfg(test)]
mod test_support;
