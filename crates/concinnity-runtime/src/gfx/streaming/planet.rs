// The threads that build a planet's terrain tile meshes for core's tile pool.

use concinnity_core::planet::{PlanetShape, TileId, TileMesh, tile_mesh};

use concinnity_host::thread::worker::Worker;

// Threads building tile meshes. A fresh camera wants a thousand or more.
const WORKERS: usize = 3;

// The threads that build `shape`'s tiles.
pub(crate) fn spawn_workers(
    shape: PlanetShape,
) -> std::io::Result<Worker<TileId, (TileId, TileMesh)>> {
    Worker::spawn("cn-planet-tiles", WORKERS, move |id| {
        (id, tile_mesh(&shape, id))
    })
}
