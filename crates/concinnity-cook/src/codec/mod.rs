//! Container and image format decoders. Build-only: they turn an authored
//! `.tga` / `.dds` / `.ktx2` / `.hdr` into the pixels `crate::compile` encodes,
//! and the runtime plays the compiled payload rather than re-decoding a source.

/// BCn block decompression, which the DDS and KTX2 readers need to hand a
/// compressed source back as RGBA.
pub(crate) mod bcn;
pub(crate) mod dds;
/// Build-time HDR source primitives (Radiance decode, equirect->cube, cube
/// payload format) shared by the CubemapTexture + EnvironmentMap compilers.
pub(crate) mod hdr;
/// KTX2 container decode: BCn block passthrough + Basis Universal (ETC1S / UASTC)
/// transcode into the tagged compressed texture payload. Build-only.
pub(crate) mod ktx2;
pub(crate) mod tga;
