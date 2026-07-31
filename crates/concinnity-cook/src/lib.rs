// src/lib.rs
//
// concinnity-cook: the asset compile pipeline, extracted from concinnity-core
// so the runtime foundation no longer carries the build-only dependencies
// (fbxcel, fontdue, shaderc, sha2, kira). This crate turns world.jsonl + source
// files into the binary blobs the runtime reads; it depends on concinnity-core
// and core has no edge back into it.
//
// The staying-in-core modules are re-exported so code moved here keeps resolving
// its `crate::{assets,ecs,gfx,geometry,result}` paths. The payload *decoders*
// and shared payload types live in `concinnity_core::build`; this crate's
// modules call back into them.
pub use concinnity_core::{assets, build, ecs, gfx, paths, result};

// The world front half -- the authored model, the type vocabulary
// (`ComponentType` / `ResourceAssetType`), and the pure semantic checks --
// lives in concinnity-world; re-exported so cook code and downstream consumers
// keep resolving `crate::{registry,template_spec}` paths. cook composes its
// compile-backed checks on top (crate::check) and owns expansion (crate::world).
pub use concinnity_world::{registry, template_spec, validate};

pub mod asset;
pub mod asset_api;
pub mod asset_impls;
pub mod audio_clip;
// Source-image format decoders (Targa, DDS, and the BCn block decompressors DDS
// needs). Build-only: they turn an authored `.tga` / `.dds` into RGBA the
// texture encoder packs; the runtime plays the compiled RGBA payload.
pub mod bcn;
pub mod blob;
pub mod cache;
pub mod check;
pub mod color_lut;
pub mod cubemap;
pub mod dds;
pub mod environment_map;
pub mod fbx;
pub mod file;
// Stat-based identity behind the cook's read memos, and the settle window that
// keeps a same-tick equal-length rewrite from being served stale.
mod file_stamp;
pub mod font;
// Build-time mesh generators + payload compilers. The runtime-side mesh helpers
// they share (tangents, the voxel mesher, chunk streaming) stay in
// `concinnity_core::geometry`; this module re-exports what cook code names.
pub mod geometry;
pub mod glb;
pub mod gltf;
pub mod gltf_source;
// Build-time HDR source primitives (Radiance decode, equirect->cube, cube
// payload format) shared by the CubemapTexture + EnvironmentMap compilers.
pub mod hdr;
pub mod import;
// KTX2 container decode: BCn block passthrough + Basis Universal (ETC1S / UASTC)
// transcode into the tagged compressed texture payload. Build-only.
pub mod ktx2;
pub mod mesh_compile;
pub mod mesh_reimport;
pub mod pipeline;
pub mod resource_handles;
pub mod root_motion;
pub mod scene_partition;
pub mod shader;
pub mod texture;
pub mod tga;
pub mod thumbnail;
pub mod wavefront;
pub mod world;

// Public build API: the entry points the CLI, the editor FFI, and the infra
// server call. The runtime-side decode + world parse API stays in
// concinnity-core.
pub use pipeline::{
    PipelineResult, TextureSourceInfo, build_compiled, build_from_path, build_pipeline_from_str,
    validate_asset, validate_world_jsonl, write_build_outputs,
};
pub use registry::ComponentType;
pub use world::prepare_world;
