//! The single-source engine shaders: the embedded HLSL, source assembly, the
//! program tables each backend compiles, hot reload, and the main-pass
//! pipelines of a world's material shaders.

#[cfg(test)]
pub(crate) mod consts;
pub mod embedded;
pub mod programs;
pub mod reload;
pub mod source;
pub mod world_pipelines;
