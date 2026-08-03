// src/build/mod.rs
//
// Asset payload format helpers + the shared build-time types. The asset COMPILE
// pipeline (importers, encoders, image/glTF decoders, source-image format
// decoders, shader compilation, the world expansion + check front-half, and
// blob writing) lives in the `concinnity-cook` crate; this module keeps only
// what a running engine needs: the pre-compiled payload `deserialise` family,
// the payload-format types and consts, and the built-in shader sources. The
// `Platform` selector is re-exported from `crate::platform`. Submodules stay
// `pub` so both the client runtime and the build crate can reach them across
// the workspace split.
pub mod color_lut;
pub mod environment_map;
pub mod font;
pub mod payload;
pub mod shader;
pub mod texture;

// `Platform` lives in `crate::platform` (pure, no build-time I/O); re-exported
// here so the build-side callers keep naming it `build::Platform`.
pub use crate::platform::Platform;
