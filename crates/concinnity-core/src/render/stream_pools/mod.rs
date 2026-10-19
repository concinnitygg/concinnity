//! The asset-streaming pools: which textures, meshes, voxel chunks and planet
//! tiles a frame loads, keeps and drops, and the GPU effects that follow. The
//! loads themselves run on the host's workers, reached through each pool's
//! [`Mailbox`]: the host delivers finished loads before a frame's drive and
//! sends the drive's requests after it.

mod accounting;
pub mod budget;
mod chunk_pool;
mod decoded_mesh;
mod deferred_payloads;
mod item_pool;
mod mailbox;
mod planet_tiles;
mod pressure;
mod procedural_chunks;
mod state;
mod stats_log;

pub use chunk_pool::{ChunkLoad, ChunkPool, ChunkRequest};
pub use decoded_mesh::{
    DecodedMesh, DeferredMeshPayload, decode_deferred_payload, decode_mesh, encode_mesh,
};
pub use deferred_payloads::deferred_mesh_payloads;
pub use item_pool::{ItemLoad, StreamPool, StreamedItem};
pub use mailbox::{LoadPool, Mailbox};
pub use planet_tiles::{PlanetTiles, TileCamera, TileMaterial, tile_mesh_size};
pub use procedural_chunks::ProceduralChunks;
pub use state::{
    ChunkStream, DriveView, PlanetView, ShaderPump, StreamingPools, StreamingPressure,
    StreamingState, StreamingStats, chunk_model_matrix,
};
