//! Baking: the pure computation that turns an asset's args into the payload
//! the runtime plays -- payload formats, their `deserialise` family, and the
//! bake kernels that need no source file (IBL convolution, the built-in font's
//! SDF atlas, mesh payload packing). Everything here is no_std: schedulers and
//! caches are the caller's concern. The asset COMPILE pipeline (importers,
//! encoders, image/glTF decoders, shader compilation, the world expansion +
//! check front-half, and blob writing) lives in the `concinnity-cook` crate
//! and calls down into this module.
pub mod color_lut;
pub mod environment_map;
pub mod font;
pub mod mesh;
pub mod payload;
pub mod texture;
