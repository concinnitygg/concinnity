// src/lib.rs
//
// concinnity-core: the CPU compute layer over the runtime vocabulary, shared by
// the client runtime and the `concinnity-cook` compile pipeline. Skinning and
// pose blending, IK, LOD decimation, rasterisation, IBL convolution, the
// procedural geometry generators, and the payload decoders (`build`). It takes
// threads (`rayon`) and is unapologetically std.
//
// The vocabulary all of that computes over -- the GPU data layouts, the
// transform and skeleton types, the assets, the ECS registry -- is
// `concinnity-types`, which this crate re-exports under its own module paths so
// a consumer reaches everything through `concinnity_core::*` as before. The
// asset COMPILE pipeline, including world.jsonl parsing and expansion, lives in
// `concinnity-cook`; core has no dependency on it. Core depends on no graphics
// backend, windowing, physics, or audio crate.
//
// Core knows where nothing lives: it names no path, opens no file, and reads no
// clock. Resolving the state tree and reading the compiled blob out of it is
// `concinnity-store`, which sits above this crate.

pub mod build;
pub mod ecs;
pub mod geometry;
pub mod gfx;

pub use concinnity_types::{assets, platform, resource, result};

// The component registry macros. Defined in concinnity-types alongside the
// component list they expand over; re-exported so the build crate keeps
// reaching them at `concinnity_core::*`.
pub use concinnity_types::{
    __define_asset_kind, define_components, for_each_component, for_each_resource_asset,
};
