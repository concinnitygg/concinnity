//! The asset-streaming home: the load threads and payload sources behind
//! core's streaming pools (`concinnity_core::render::stream_pools`).

mod file_range;
pub(crate) mod loaders;
pub(crate) mod mesh;
pub(crate) mod planet;
pub(crate) mod shader;
pub(crate) mod texture;
