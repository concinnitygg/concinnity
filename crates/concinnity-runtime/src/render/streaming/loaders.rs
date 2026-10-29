//! The load threads behind core's streaming pools: each pool's worker and the
//! deferred shader builds, lent to core's `StreamingSystem` as its
//! `StreamLoaders`. Graphics init builds them and publishes them inside the
//! `StreamingRuntime` resource.

use concinnity_engine::bake::texture::TextureImage;
use concinnity_engine::render::streaming::pools::{
    ChunkLoad, ChunkRequest, DecodedMesh, ItemLoad, LoadPool, ShaderPump, StreamLoaders,
    StreamingState, TileBuild, TileBuilt,
};

use crate::render::streaming::shader::ShaderWarmup;
use concinnity_host::thread::worker::Worker;

// The load threads of each streaming pool, `Some` where that pool streams,
// and the deferred shader builds.
#[derive(Default)]
pub(crate) struct StreamWorkers {
    pub(crate) texture: Option<Worker<usize, ItemLoad<TextureImage>>>,
    pub(crate) mesh: Option<Worker<usize, ItemLoad<DecodedMesh>>>,
    pub(crate) chunk: Option<Worker<ChunkRequest, ChunkLoad>>,
    pub(crate) planet: Option<Worker<TileBuild, TileBuilt>>,
    pub(crate) shaders: Option<ShaderWarmup>,
}

// Pair a pool with its worker, when both exist.
fn paired<'a, P, Req, Res>(
    worker: &'a mut Option<Worker<Req, Res>>,
    pool: Option<&'a mut P>,
) -> Option<(&'a mut Worker<Req, Res>, &'a mut P)>
where
    P: LoadPool<Request = Req, Result = Res>,
    Req: Send + 'static,
    Res: Send + 'static,
{
    worker.as_mut().zip(pool)
}

// Hand `pool` every finished load; a load that panicked is retired on the
// pool, which never asks for it again.
fn deliver<P: LoadPool>(worker: &Worker<P::Request, P::Result>, pool: &mut P)
where
    P::Request: Send + 'static,
    P::Result: Send + 'static,
{
    while let Some(answer) = worker.try_recv() {
        match answer {
            Ok(result) => pool.mailbox().deliver(result),
            Err(request) => pool.load_failed(request),
        }
    }
}

// Send the requests `pool`'s drive queued; one no thread takes is reverted on
// the pool.
fn send_requests<P: LoadPool>(worker: &mut Worker<P::Request, P::Result>, pool: &mut P)
where
    P::Request: Send + 'static,
    P::Result: Send + 'static,
{
    while let Some(request) = pool.mailbox().pop_request() {
        if let Err(request) = worker.send(request) {
            pool.request_refused(request);
        }
    }
}

impl StreamLoaders for StreamWorkers {
    fn deliver(&mut self, state: &mut StreamingState) {
        if let Some((w, p)) = paired(&mut self.texture, state.texture_mut()) {
            deliver(w, p);
        }
        if let Some((w, p)) = paired(&mut self.mesh, state.mesh_mut()) {
            deliver(w, p);
        }
        if let Some((w, p)) = paired(&mut self.chunk, state.chunk_pool_mut()) {
            deliver(w, p);
        }
        if let Some((w, p)) = paired(&mut self.planet, state.planet_mut()) {
            deliver(w, p);
        }
    }

    fn send(&mut self, state: &mut StreamingState) {
        if let Some((w, p)) = paired(&mut self.texture, state.texture_mut()) {
            send_requests(w, p);
        }
        if let Some((w, p)) = paired(&mut self.mesh, state.mesh_mut()) {
            send_requests(w, p);
        }
        if let Some((w, p)) = paired(&mut self.chunk, state.chunk_pool_mut()) {
            send_requests(w, p);
        }
        if let Some((w, p)) = paired(&mut self.planet, state.planet_mut()) {
            send_requests(w, p);
        }
    }

    fn shaders(&mut self) -> Option<&mut dyn ShaderPump> {
        self.shaders.as_mut().map(|s| s as &mut dyn ShaderPump)
    }
}

#[cfg(test)]
mod tests;
