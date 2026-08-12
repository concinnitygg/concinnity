// src/build/mod.rs
//
// Asset payload format helpers + the shared build-time types. The asset COMPILE
// pipeline (importers, encoders, image/glTF decoders, source-image format
// decoders, shader compilation, the world expansion + check front-half, and
// blob writing) lives in the `concinnity-cook` crate; this module keeps only
// what a running engine needs: the pre-compiled payload `deserialise` family
// plus the payload-format types and consts. Submodules stay `pub` so both the
// client runtime and the build crate can reach them across the workspace split.
pub mod color_lut;
pub mod environment_map;
pub mod font;
pub mod texture;
