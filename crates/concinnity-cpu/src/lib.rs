// src/lib.rs
//
// concinnity-cpu: the CPU compute layer over the runtime vocabulary, shared by
// the client runtime and the `concinnity-cook` compile pipeline. Skinning and
// pose blending, IK, LOD decimation, rasterisation, IBL convolution, the
// procedural geometry generators, and the payload decoders (`build`). It takes
// threads (`rayon`) and is unapologetically std.
//
// The vocabulary all of that computes over -- the GPU data layouts, the
// transform and skeleton types, the assets, the ECS registry -- is
// `concinnity-core`, which this crate sits directly above and consumers name
// for themselves. The asset COMPILE pipeline, including world.jsonl parsing and
// expansion, lives in `concinnity-cook`; this crate has no dependency on it,
// nor on any graphics backend, windowing, physics, or audio crate.
//
// It knows where nothing lives: it names no path, opens no file, and reads no
// clock. Resolving the state tree and reading the compiled blob out of it is
// `concinnity-store`, which sits above this crate.

pub mod build;
pub mod ecs;
pub mod geometry;
pub mod gfx;
pub mod jobs;
