//! Streamed mesh geometry: the decoded vertex and index data a mesh or chunk
//! load produces, and the record format a disk-backed mesh source keeps it in.

use crate::math::SaturatingFrom;
use alloc::string::String;
use alloc::vec::Vec;

use crate::decode::ByteReader;
use crate::gfx::mesh_payload::{self, Vertex};

/// A mesh payload decoded to GPU-ready vertex and index data.
///
/// Index values are mesh-relative (0-based): the renderer's sub-allocator
/// places the vertices at an arbitrary offset on upload, so the backend
/// rebases the indices onto that region rather than the payload baking in a
/// fixed base.
#[derive(Debug, Clone, Default)]
pub struct DecodedMesh {
    /// The vertices.
    pub vertices: Vec<Vertex>,
    /// Mesh-relative indices.
    pub indices: Vec<u16>,
}

impl DecodedMesh {
    /// The vertex + index buffer bytes it occupies once uploaded.
    pub fn byte_len(&self) -> u64 {
        (self.vertices.len() * size_of::<Vertex>() + self.indices.len() * size_of::<u16>()) as u64
    }
}

/// One deferred streamed mesh's compiled payload: init skipped its decode
/// (owned by a scene other than the start scene), so no geometry copy exists
/// in the mesh source and the worker decodes the blob payload instead.
pub enum DeferredMeshPayload {
    /// RAM-backed world: the raw compiled payload bytes.
    Bytes(Vec<u8>),
    /// Disk-backed world: the payload's absolute byte range in its blob file.
    Disk {
        /// The blob file.
        path: String,
        /// The payload's absolute offset in it.
        offset: u64,
        /// The payload's length.
        len: u64,
    },
}

/// Decode a compiled mesh payload into LOD0 geometry (streamed draws strip
/// their LOD alternates, so only LOD0 is ever uploaded).
pub fn decode_deferred_payload(bytes: &[u8]) -> Result<DecodedMesh, String> {
    let (vertices, indices, _) = mesh_payload::deserialize_with_lods(bytes)?;
    Ok(DecodedMesh { vertices, indices })
}

/// Serialize one mesh's geometry into the scratch record format:
/// `u32 vertex_count`, the vertices as 56-byte records, `u32 index_count`,
/// the indices as little-endian `u16`s.
pub fn encode_mesh(mesh: &DecodedMesh) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + mesh.vertices.len() * 56 + 4 + mesh.indices.len() * 2);
    buf.extend_from_slice(&u32::saturating_from(mesh.vertices.len()).to_le_bytes());
    for v in &mesh.vertices {
        for x in v
            .pos
            .iter()
            .chain(v.normal.iter())
            .chain(v.tangent.iter())
            .chain(v.color.iter())
            .chain(v.uv.iter())
        {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    buf.extend_from_slice(&u32::saturating_from(mesh.indices.len()).to_le_bytes());
    for i in &mesh.indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf
}

/// Inverse of [`encode_mesh`]: decode one scratch record. Errors rather than
/// panics on a truncated record, so a corrupt scratch file fails the load
/// loudly instead of reading out of bounds.
pub fn decode_mesh(bytes: &[u8]) -> Result<DecodedMesh, String> {
    const VERTEX_BYTES: usize = size_of::<Vertex>();

    let mut r = ByteReader::new(bytes, "mesh record");
    // Counts are read from the record, so each reservation is capped at what
    // the buffer could actually hold rather than what it claims.
    let vertex_count = r.u32()? as usize;
    let mut vertices = Vec::with_capacity(vertex_count.min(r.remaining() / VERTEX_BYTES));
    for _ in 0..vertex_count {
        vertices.push(Vertex {
            pos: [r.f32()?, r.f32()?, r.f32()?],
            normal: [r.f32()?, r.f32()?, r.f32()?],
            tangent: [r.f32()?, r.f32()?, r.f32()?],
            color: [r.f32()?, r.f32()?, r.f32()?],
            uv: [r.f32()?, r.f32()?],
        });
    }

    let index_count = r.u32()? as usize;
    let mut indices = Vec::with_capacity(index_count.min(r.remaining() / 2));
    for _ in 0..index_count {
        indices.push(r.u16()?);
    }

    Ok(DecodedMesh { vertices, indices })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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
    fn encode_mesh_round_trips_through_decode() {
        let mesh = DecodedMesh {
            vertices: vec![mk_vertex(1.0), mk_vertex(2.0), mk_vertex(3.0)],
            indices: vec![0, 1, 2, 2, 1, 0],
        };
        let decoded = decode_mesh(&encode_mesh(&mesh)).expect("decode ok");
        assert_eq!(decoded.vertices.len(), 3);
        assert_eq!(decoded.indices, vec![0, 1, 2, 2, 1, 0]);
        assert_eq!(decoded.vertices[1].pos, [2.0, 0.0, 0.0]);
        assert_eq!(decoded.vertices[2].normal, [0.0, 1.0, 0.0]);
        assert_eq!(decoded.vertices[0].tangent, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn decode_mesh_errors_on_truncated_record() {
        let bytes = encode_mesh(&DecodedMesh {
            vertices: vec![mk_vertex(0.0)],
            indices: vec![0],
        });
        // dropping the final index byte leaves an incomplete record
        assert!(decode_mesh(&bytes[..bytes.len() - 1]).is_err());
        // a header claiming more vertices than the buffer holds
        assert!(decode_mesh(&[9, 0, 0, 0]).is_err());
        // an empty buffer has not even a vertex count
        assert!(decode_mesh(&[]).is_err());
    }

    // A count near u32::MAX must fail on the missing data, without first
    // reserving one `Vertex` per claimed entry.
    #[test]
    fn decode_mesh_errors_on_an_absurd_vertex_count() {
        assert!(decode_mesh(&u32::MAX.to_le_bytes()).is_err());
    }

    #[test]
    fn decode_mesh_errors_on_an_absurd_index_count() {
        let mut bytes = 0u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_mesh(&bytes).is_err());
    }

    #[test]
    fn the_byte_length_counts_both_buffers() {
        let mesh = DecodedMesh {
            vertices: vec![mk_vertex(0.0); 3],
            indices: vec![0, 1, 2],
        };
        assert_eq!(mesh.byte_len(), (3 * size_of::<Vertex>() + 3 * 2) as u64);
    }
}
