//! An infinite voxel world's chunk pool: the [`ChunkWindow`] around the camera
//! decides which chunks load at which detail; this maps the camera to its chunk
//! and moves generations through the pool's [`Mailbox`].

use alloc::string::String;
use alloc::vec::Vec;

use super::decoded_mesh::DecodedMesh;
use super::{LoadPool, Mailbox};
use crate::geometry::chunk_coord::ChunkCoord;
use crate::gfx::mesh_payload::Vertex;
use crate::render::streaming::chunk_window::{ChunkDetail, ChunkWindow};

/// A requested chunk generation: where, and at what detail.
pub type ChunkRequest = (ChunkCoord, ChunkDetail);

/// A finished generation: the chunk and its geometry, or why it failed.
pub type ChunkLoad = (ChunkCoord, Result<DecodedMesh, String>);

// Retire `coord`, whose generation cannot succeed: it is never dispatched
// again.
fn retire(window: &mut ChunkWindow, coord: ChunkCoord, why: &str) {
    tracing::warn!(
        "chunk stream: generation of chunk ({},{}) failed for good: {}",
        coord.x,
        coord.z,
        why
    );
    window.mark_failed(coord);
}

/// Drives an infinite voxel world's chunks. Each frame the drive calls
/// [`plan`](Self::plan) then [`drain_completed`](Self::drain_completed).
pub struct ChunkPool {
    window: ChunkWindow,
    // World-space size of one chunk on X / Z, to map a camera position to its
    // chunk coordinate.
    chunk_w: f32,
    chunk_d: f32,
    mail: Mailbox<ChunkRequest, ChunkLoad>,
}

impl LoadPool for ChunkPool {
    type Request = ChunkRequest;
    type Result = ChunkLoad;

    fn mailbox(&mut self) -> &mut Mailbox<ChunkRequest, ChunkLoad> {
        &mut self.mail
    }

    fn request_refused(&mut self, request: ChunkRequest) {
        self.window.forget(request.0);
    }

    fn load_failed(&mut self, (coord, _): ChunkRequest) {
        retire(&mut self.window, coord, "the generation panicked");
    }
}

impl ChunkPool {
    /// A pool of full-detail chunks to `near_radius` and impostors to
    /// `far_radius` (equal radii disable impostors), requesting at most
    /// `load_budget` generations per frame, for chunks `chunk_w` by `chunk_d`
    /// world units.
    pub fn new(
        near_radius: i32,
        far_radius: i32,
        load_budget: usize,
        chunk_w: f32,
        chunk_d: f32,
    ) -> Self {
        Self {
            window: ChunkWindow::new(near_radius, far_radius, load_budget),
            chunk_w,
            chunk_d,
            mail: Mailbox::default(),
        }
    }

    /// The chunk a world-space camera position falls in.
    pub fn camera_chunk(&self, camera: [f32; 3]) -> ChunkCoord {
        ChunkCoord::from_world(camera[0], camera[2], self.chunk_w, self.chunk_d)
    }

    /// Run the window policy for a camera in chunk `camera`: request this
    /// frame's generations and return the chunks the caller must remove from
    /// the GPU (they have left the view window).
    pub fn plan(&mut self, camera: ChunkCoord) -> Vec<ChunkCoord> {
        let plan = self.window.plan(camera);
        for &request in &plan.to_load {
            self.mail.request(request);
        }
        plan.to_evict
    }

    /// Apply every finished generation through `upload`, which takes the
    /// chunk's geometry by value. A chunk evicted while its generation was in
    /// flight is discarded. A failed generation retires the chunk, as a
    /// panicked one does. Returns the chunks brought resident.
    pub fn drain_completed(
        &mut self,
        mut upload: impl FnMut(ChunkCoord, Vec<Vertex>, Vec<u16>),
    ) -> usize {
        let mut applied = 0;
        for (coord, decoded) in self.mail.take_results() {
            if !self.window.is_tracked(coord) {
                continue;
            }
            match decoded {
                Ok(mesh) => {
                    let bytes = mesh.byte_len();
                    upload(coord, mesh.vertices, mesh.indices);
                    self.window.mark_resident(coord, bytes);
                    applied += 1;
                }
                Err(e) => retire(&mut self.window, coord, &e),
            }
        }
        applied
    }

    /// `(resident, pending)` chunk counts.
    pub fn stats(&self) -> (usize, usize) {
        self.window.counts()
    }

    /// `(near_resident, far_resident)`: resident full chunks and resident
    /// distant impostors.
    pub fn detail_counts(&self) -> (usize, usize) {
        self.window.counts_by_detail()
    }

    /// Set (or clear with `None`) the resident-chunk-byte budget. When set, the
    /// window clamps its effective view radius to hold resident chunk bytes at
    /// or under it, shedding the far impostor band before the near band.
    pub fn set_byte_budget(&mut self, budget: Option<u64>) {
        self.window.set_byte_budget(budget);
    }

    /// Total resident chunk bytes.
    pub fn resident_bytes(&self) -> u64 {
        self.window.resident_bytes()
    }

    /// The active resident-byte budget, or `None` when byte accounting is off.
    pub fn byte_budget(&self) -> Option<u64> {
        self.window.byte_budget()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn tri() -> DecodedMesh {
        let v = Vertex {
            pos: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0; 3],
            uv: [0.0; 2],
        };
        DecodedMesh {
            vertices: alloc::vec![v; 3],
            indices: alloc::vec![0, 1, 2],
        }
    }

    fn serve(pool: &mut ChunkPool, ok: bool) {
        let requests: Vec<ChunkRequest> = pool.mailbox().take_requests().collect();
        for (coord, _) in requests {
            let load = if ok {
                Ok(tri())
            } else {
                Err("test".to_string())
            };
            pool.mailbox().deliver((coord, load));
        }
    }

    #[test]
    fn the_window_around_the_camera_loads_and_uploads() {
        let mut pool = ChunkPool::new(1, 1, 64, 16.0, 16.0);
        let camera = pool.camera_chunk([8.0, 0.0, 8.0]);
        assert_eq!(camera, ChunkCoord::new(0, 0));
        assert!(pool.plan(camera).is_empty());
        serve(&mut pool, true);
        let mut uploaded = 0;
        pool.drain_completed(|_, verts, idxs| {
            assert_eq!((verts.len(), idxs.len()), (3, 3));
            uploaded += 1;
        });
        assert_eq!(uploaded, 9, "a radius-1 window is 3 by 3");
        assert_eq!(pool.stats(), (9, 0));
        assert!(pool.resident_bytes() > 0);
        assert_eq!(pool.camera_chunk([20.0, 5.0, -1.0]), ChunkCoord::new(1, -1));
        // Jump far enough that the whole window leaves the evict band.
        let evicted = pool.plan(ChunkCoord::new(50, 0));
        assert!(evicted.contains(&ChunkCoord::new(0, 0)));
        assert_eq!(evicted.len(), 9);
    }

    // Within the near radius chunks carry full geometry; the band out to the
    // far radius streams as impostors.
    #[test]
    fn detail_counts_split_the_near_and_far_bands() {
        let mut pool = ChunkPool::new(0, 1, 100, 16.0, 16.0);
        pool.plan(ChunkCoord::new(0, 0));
        serve(&mut pool, true);
        pool.drain_completed(|_, _, _| {});
        assert_eq!(pool.detail_counts(), (1, 8));
    }

    // A chunk that left the window while generating is discarded on arrival.
    #[test]
    fn a_chunk_evicted_in_flight_is_discarded() {
        let mut pool = ChunkPool::new(0, 0, 64, 16.0, 16.0);
        pool.plan(ChunkCoord::new(0, 0));
        let requests: Vec<ChunkRequest> = pool.mailbox().take_requests().collect();
        pool.plan(ChunkCoord::new(5, 5));
        for (coord, _) in requests {
            pool.mailbox().deliver((coord, Ok(tri())));
        }
        assert_eq!(pool.drain_completed(|_, _, _| {}), 0);
    }

    // A failed generation retires the chunk: it holds no bytes and is never
    // dispatched again, even after the camera leaves and returns.
    #[test]
    fn a_failed_generation_is_retired() {
        let mut pool = ChunkPool::new(0, 0, 64, 16.0, 16.0);
        let home = ChunkCoord::new(0, 0);
        pool.plan(home);
        serve(&mut pool, false);
        assert_eq!(pool.drain_completed(|_, _, _| {}), 0);
        assert_eq!(pool.stats(), (0, 0));
        assert_eq!(pool.resident_bytes(), 0);
        pool.plan(home);
        assert!(!pool.mailbox().has_requests());
        pool.plan(ChunkCoord::new(40, 0));
        pool.mailbox().take_requests().for_each(drop);
        pool.plan(home);
        assert!(!pool.mailbox().has_requests());
    }

    #[test]
    fn a_refused_request_is_requested_again() {
        let mut pool = ChunkPool::new(0, 0, 64, 16.0, 16.0);
        pool.plan(ChunkCoord::new(0, 0));
        let refused: Vec<ChunkRequest> = pool.mailbox().take_requests().collect();
        for request in refused {
            pool.request_refused(request);
        }
        assert_eq!(pool.stats(), (0, 0));
        pool.plan(ChunkCoord::new(0, 0));
        assert!(pool.mailbox().has_requests());
    }
}
