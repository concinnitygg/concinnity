//! The asset-streaming home: the load threads and payload sources behind
//! core's streaming pools (`concinnity_engine::render::streaming::pools`).

mod file_range;
pub(crate) mod loaders;
pub(crate) mod mesh;
pub(crate) mod planet;
pub(crate) mod shader;
// Streamed texture and deferred mesh payload sources over the blob files.
pub(crate) mod sources;
pub(crate) mod texture;
