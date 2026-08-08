// src/lib.rs
//
// concinnity-core: the renderer-free runtime foundation shared by the client
// runtime and the `concinnity-cook` compile pipeline. It holds the crate-wide
// result type, the backend-agnostic GPU data layouts plus CPU-side mesh /
// skinning / camera math (the `gfx` data layer), the asset type definitions and
// registry metadata, and the payload decoders (`build`). The asset COMPILE
// pipeline -- including world.jsonl parsing and expansion -- lives in
// `concinnity-cook`; core has no dependency on it. Core depends on no graphics
// backend, windowing, physics, or audio crate.
//
// Core knows where nothing lives: it names no path, opens no file, and reads no
// clock. Resolving the state tree and reading the compiled blob out of it is
// `concinnity-store`, which sits above this crate.

// Required by `concinnity_eas::define_component_storage!`, whose expansion names
// `::alloc::vec::Vec` so the macro stays usable from a no_std crate.
extern crate alloc;

pub mod assets;
pub mod build;
pub mod ecs;
pub mod geometry;
pub mod gfx;
pub mod platform;
pub mod resource;
pub mod result;
pub mod shutdown;
