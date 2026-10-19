//! The load threads behind core's streaming pools: each pool's worker and the
//! deferred shader builds, lent to core's `StreamingSystem` as its
//! `StreamLoaders`. Graphics init builds them and publishes them inside the
//! `StreamingRuntime` resource.

use concinnity_core::bake::texture::TextureImage;
use concinnity_core::planet::{TileId, TileMesh};
use concinnity_core::render::stream_pools::{
    ChunkLoad, ChunkRequest, DecodedMesh, ItemLoad, LoadPool, ShaderPump, StreamLoaders,
    StreamingState,
};

use crate::gfx::streaming::shader::ShaderWarmup;
use crate::worker::{Job, Worker};

// The load threads of each streaming pool, `Some` where that pool streams,
// and the deferred shader builds.
#[derive(Default)]
pub(crate) struct StreamWorkers {
    pub(crate) texture: Option<Worker<usize, ItemLoad<TextureImage>>>,
    pub(crate) mesh: Option<Worker<usize, ItemLoad<DecodedMesh>>>,
    pub(crate) chunk: Option<Worker<ChunkRequest, ChunkLoad>>,
    pub(crate) planet: Option<Worker<TileId, (TileId, TileMesh)>>,
    pub(crate) shaders: Option<ShaderWarmup>,
}

// Pair a pool with its worker, when both exist.
fn paired<'a, P, Req, Res>(
    worker: &'a mut Option<Worker<Req, Res>>,
    pool: Option<&'a mut P>,
) -> Option<(&'a mut Worker<Req, Res>, &'a mut P)>
where
    P: LoadPool<Request = Req, Result = Res>,
    Req: Job<Key = Req>,
    Res: Send + 'static,
{
    worker.as_mut().zip(pool)
}

impl StreamLoaders for StreamWorkers {
    fn deliver(&mut self, state: &mut StreamingState) {
        if let Some((w, p)) = paired(&mut self.texture, state.texture_mut()) {
            w.deliver(p);
        }
        if let Some((w, p)) = paired(&mut self.mesh, state.mesh_mut()) {
            w.deliver(p);
        }
        if let Some((w, p)) = paired(&mut self.chunk, state.chunk_pool_mut()) {
            w.deliver(p);
        }
        if let Some((w, p)) = paired(&mut self.planet, state.planet_mut()) {
            w.deliver(p);
        }
    }

    fn send(&mut self, state: &mut StreamingState) {
        if let Some((w, p)) = paired(&mut self.texture, state.texture_mut()) {
            w.send_requests(p);
        }
        if let Some((w, p)) = paired(&mut self.mesh, state.mesh_mut()) {
            w.send_requests(p);
        }
        if let Some((w, p)) = paired(&mut self.chunk, state.chunk_pool_mut()) {
            w.send_requests(p);
        }
        if let Some((w, p)) = paired(&mut self.planet, state.planet_mut()) {
            w.send_requests(p);
        }
    }

    fn shaders(&mut self) -> Option<&mut dyn ShaderPump> {
        self.shaders.as_mut().map(|s| s as &mut dyn ShaderPump)
    }
}

#[cfg(test)]
mod tests;
