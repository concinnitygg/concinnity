//! Where streamed mesh geometry comes from, and the thread that fetches it for
//! core's mesh pool.
//!
//! `MeshPayloadSource` is the seam. `MemMeshSource` serves geometry kept
//! resident in RAM (used by `cn debug`, which builds geometry in memory with no
//! disk artifacts); `DiskMeshSource` re-reads it from a scratch file written by
//! `write_mesh_scratch` (used by `cn run`, so the geometry never stays a second
//! RAM copy past GPU upload).

use concinnity_core::render::stream_pools::{
    DecodedMesh, DeferredMeshPayload, ItemLoad, decode_deferred_payload, decode_mesh, encode_mesh,
};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use concinnity_host::thread::worker::Worker;

// Fetches and decodes a streamable mesh payload by item id. `Send + Sync` so
// the load thread can own one; called off the main thread.
pub(crate) trait MeshPayloadSource: Send + Sync {
    // Decode item `id` into GPU-ready geometry, or a human-readable error.
    fn fetch(&self, id: usize) -> Result<DecodedMesh, String>;
}

// Geometry kept resident in RAM. `cn debug` has no disk artifacts to re-read,
// so this streams the GPU upload only.
pub(crate) struct MemMeshSource {
    meshes: Vec<DecodedMesh>,
}

impl MemMeshSource {
    // `meshes[id]` is the geometry for streamable mesh `id`.
    pub(crate) fn new(meshes: Vec<DecodedMesh>) -> Self {
        Self { meshes }
    }
}

impl MeshPayloadSource for MemMeshSource {
    fn fetch(&self, id: usize) -> Result<DecodedMesh, String> {
        self.meshes
            .get(id)
            .cloned()
            .ok_or_else(|| format!("no payload for streamed mesh {id}"))
    }
}

// Locates one streamed mesh's geometry record inside the scratch file.
#[derive(Clone)]
pub(crate) struct DiskMeshLocator {
    pub(crate) file_offset: u64,
    pub len: u64,
}

// Re-reads each mesh's geometry from a scratch file on disk, so the geometry
// never stays a second RAM copy. A streamed mesh's geometry only exists as a
// region of the assembled buffers, with no discrete on-disk payload, so
// `write_mesh_scratch` writes it once and this re-reads each record on demand.
// The file is removed when the source is dropped (world rebuild or shutdown).
pub(crate) struct DiskMeshSource {
    path: String,
    // locators[id] points streamed mesh `id` at its record in the scratch file
    locators: Vec<DiskMeshLocator>,
}

impl MeshPayloadSource for DiskMeshSource {
    fn fetch(&self, id: usize) -> Result<DecodedMesh, String> {
        let loc = self
            .locators
            .get(id)
            .ok_or_else(|| format!("no disk locator for streamed mesh {id}"))?;
        let bytes = super::file_range::read_at(&self.path, loc.file_offset, loc.len)?;
        decode_mesh(&bytes)
    }
}

impl Drop for DiskMeshSource {
    #[expect(
        clippy::let_underscore_must_use,
        reason = "the scratch file is regenerated on the next world build, so removing it is best effort"
    )]
    fn drop(&mut self) {
        // The scratch file is regenerated on the next world build/run, so a
        // dropped source has no reason to keep it.
        let _ = std::fs::remove_file(&self.path);
    }
}

// Wraps a base mesh source with per-id deferred payload overrides.
pub(crate) struct SceneDeferredMeshSource {
    base: Arc<dyn MeshPayloadSource>,
    deferred: hashbrown::HashMap<usize, DeferredMeshPayload>,
}

impl SceneDeferredMeshSource {
    pub(crate) fn new(
        base: Arc<dyn MeshPayloadSource>,
        deferred: hashbrown::HashMap<usize, DeferredMeshPayload>,
    ) -> Self {
        Self { base, deferred }
    }
}

impl MeshPayloadSource for SceneDeferredMeshSource {
    fn fetch(&self, id: usize) -> Result<DecodedMesh, String> {
        match self.deferred.get(&id) {
            None => self.base.fetch(id),
            Some(DeferredMeshPayload::Bytes(bytes)) => decode_deferred_payload(bytes),
            Some(DeferredMeshPayload::Disk { path, offset, len }) => {
                let bytes = super::file_range::read_at(path, *offset, *len)?;
                decode_deferred_payload(&bytes)
            }
        }
    }
}

// Write every streamed mesh's geometry to `path` and return a `DiskMeshSource`
// that re-reads each record on demand, so the caller can drop the RAM copy.
pub(crate) fn write_mesh_scratch(
    path: String,
    meshes: &[DecodedMesh],
) -> Result<DiskMeshSource, String> {
    let mut file = File::create(&path).map_err(|e| format!("create {path}: {e}"))?;
    let mut locators = Vec::with_capacity(meshes.len());
    let mut offset: u64 = 0;
    for mesh in meshes {
        let bytes = encode_mesh(mesh);
        file.write_all(&bytes)
            .map_err(|e| format!("write {path}: {e}"))?;
        locators.push(DiskMeshLocator {
            file_offset: offset,
            len: bytes.len() as u64,
        });
        offset += bytes.len() as u64;
    }
    file.flush().map_err(|e| format!("flush {path}: {e}"))?;
    Ok(DiskMeshSource { path, locators })
}

// A scratch-file path for one disk-backed source. Each call returns a
// distinct path so a world rebuild's new source does not collide with the old
// one's file; the source owns the file and removes it on drop.
pub(crate) fn default_scratch_path() -> String {
    concinnity_host::scratch::path("mesh-stream.bin")
        .to_string_lossy()
        .into_owned()
}

// The thread that answers the mesh pool's loads from `source`.
pub(crate) fn spawn_worker(
    source: Arc<dyn MeshPayloadSource>,
) -> std::io::Result<Worker<usize, ItemLoad<DecodedMesh>>> {
    Worker::spawn("cn-mesh-stream", 1, move |id| (id, source.fetch(id)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::mesh_payload::Vertex;

    fn mk_vertex(x: f32) -> Vertex {
        Vertex {
            pos: [x, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
        }
    }

    #[test]
    fn mem_mesh_source_serves_a_payload() {
        let source = MemMeshSource::new(vec![DecodedMesh {
            vertices: vec![mk_vertex(1.0), mk_vertex(2.0)],
            indices: vec![0, 1, 0],
        }]);
        let mesh = source.fetch(0).expect("fetch ok");
        assert_eq!(mesh.vertices.len(), 2);
        assert_eq!(mesh.indices, vec![0, 1, 0]);
    }

    #[test]
    fn mem_mesh_source_errors_on_unknown_id() {
        let source = MemMeshSource::new(vec![DecodedMesh {
            vertices: vec![mk_vertex(0.0)],
            indices: vec![0],
        }]);
        assert!(source.fetch(9).is_err());
    }

    #[test]
    fn disk_mesh_source_round_trips_multiple_meshes() {
        let meshes = vec![
            DecodedMesh {
                vertices: vec![mk_vertex(1.0)],
                indices: vec![0],
            },
            DecodedMesh {
                vertices: vec![mk_vertex(2.0), mk_vertex(3.0)],
                indices: vec![0, 1, 0],
            },
        ];
        let source = write_mesh_scratch(default_scratch_path(), &meshes).expect("write scratch");
        let m0 = source.fetch(0).expect("fetch 0");
        assert_eq!(m0.vertices[0].pos, [1.0, 0.0, 0.0]);
        assert_eq!(m0.indices, vec![0]);
        // mesh 1 lives at a non-zero offset, exercising the per-record seek
        let m1 = source.fetch(1).expect("fetch 1");
        assert_eq!(m1.vertices.len(), 2);
        assert_eq!(m1.vertices[1].pos, [3.0, 0.0, 0.0]);
        assert_eq!(m1.indices, vec![0, 1, 0]);
    }

    #[test]
    fn disk_mesh_source_errors_on_unknown_id() {
        let source = write_mesh_scratch(default_scratch_path(), &[]).expect("write scratch");
        assert!(source.fetch(0).is_err());
    }

    #[test]
    fn disk_mesh_source_removes_scratch_file_on_drop() {
        let path = default_scratch_path();
        let source = write_mesh_scratch(
            path.clone(),
            &[DecodedMesh {
                vertices: vec![mk_vertex(0.0)],
                indices: vec![0],
            }],
        )
        .expect("write scratch");
        assert!(std::path::Path::new(&path).exists());
        drop(source);
        assert!(!std::path::Path::new(&path).exists());
    }

    #[test]
    fn default_scratch_path_is_unique_per_call() {
        assert_ne!(default_scratch_path(), default_scratch_path());
    }
}
