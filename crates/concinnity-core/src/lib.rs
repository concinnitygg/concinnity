// src/lib.rs
//
// concinnity-core: the renderer-free runtime foundation shared by the client
// runtime and the `concinnity-cook` compile pipeline. It holds the crate-wide
// result type, the backend-agnostic GPU data layouts plus CPU-side mesh /
// skinning / camera math (the `gfx` data layer), the asset type definitions and
// registry metadata, the payload decoders (`build`), and blob reading. The asset
// COMPILE pipeline -- including world.jsonl parsing and expansion -- lives in
// `concinnity-cook`; core has no dependency on it. Core depends on no graphics
// backend, windowing, physics, or audio crate.
pub mod assets;
pub mod blob;
pub mod build;
pub mod ecs;
pub mod geometry;
pub mod gfx;
pub mod paths;
pub mod result;
