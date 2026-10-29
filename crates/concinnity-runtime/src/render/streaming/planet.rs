// The threads that build a planet's terrain tile meshes for core's tile pool.

use concinnity_engine::render::streaming::pools::{TileBuild, TileBuilt};

use concinnity_host::thread::worker::Worker;

// Threads building tile meshes. A fresh camera wants a thousand or more.
const WORKERS: usize = 3;

// The threads that build the tiles the pool asks for, each of the shape its
// request names.
pub(crate) fn spawn_workers() -> std::io::Result<Worker<TileBuild, TileBuilt>> {
    Worker::spawn_keyed(
        "cn-planet-tiles",
        WORKERS,
        TileBuild::clone,
        TileBuild::build,
    )
}
