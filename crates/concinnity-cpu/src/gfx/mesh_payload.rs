//! Canonical vertex type and the binary serialisation format shared between
//! the build step (build_mesh.rs writes) and GraphicsSystem (reads).
//!
//! The layout asserts below stay hand-written. A vertex payload reaches a shader
//! through a vertex descriptor or a raw pointer, never as a declared buffer
//! block, so slangc's reflection reports it as an attribute index with no byte
//! offset -- the reflection-driven check in concinnity-device's `shader_layout`
//! has nothing to compare against here.
//!
//! Format (little-endian):
//!   u32  vertex_count
//!   vertex_count * 56 bytes   float3 pos + float3 normal + float3 tangent + float3 color + float2 uv (14 x f32)
//!   u32  index_count                              // LOD0 indices
//!   index_count  * 2 bytes    u16 indices
//!   optional LOD trailer
//!   4 bytes                   ascii "LODS" magic (absent for legacy / single-LOD payloads)
//!   u32  alt_count            // number of additional LODs beyond LOD0
//!   alt_count × {
//!     f32  switch_distance    // camera-distance threshold (LOD i+1 applies at d >= switch_distance)
//!     u32  index_count
//!     index_count * 2 bytes   u16 indices
//!   }
//!
//! `deserialise` reads only the LOD0 indices and ignores any trailer, so old
//! readers keep working unchanged. `deserialise_with_lods` reads the trailer
//! when present and returns the additional LODs alongside LOD0.

use crate::decode::{ByteReader, checked_product};

// The little-endian f32 at byte offset `at` of a fixed-size chunk.
fn chunk_f32(chunk: &[u8], at: usize) -> f32 {
    f32::from_le_bytes([chunk[at], chunk[at + 1], chunk[at + 2], chunk[at + 3]])
}

// The little-endian u16 at byte offset `at` of a fixed-size chunk.
fn chunk_u16(chunk: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([chunk[at], chunk[at + 1]])
}

// Read a length-prefixed UTF-8 name, bounds-checking the whole block once.
fn read_name(cur: &mut ByteReader<'_>, len: usize, what: &str) -> Result<String, String> {
    core::str::from_utf8(cur.take(len)?)
        .map_err(|e| format!("{what} is not valid utf-8: {e}"))
        .map(str::to_string)
}

// Read `n` little-endian u16 indices, bounds-checking the whole block once.
fn read_indices(cur: &mut ByteReader<'_>, n: usize, what: &str) -> Result<Vec<u16>, String> {
    let block = cur.take(checked_product(what, &[n, 2])?)?;
    Ok(block.chunks_exact(2).map(|c| chunk_u16(c, 0)).collect())
}

/// Vertex layout shared by all mesh producers and both GPU backends.
/// Repr(C) so it can be cast directly to GPU buffer memory.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct Vertex {
    /// Object-space position.
    pub pos: [f32; 3],
    /// Object-space surface normal, normalised. Transformed to world space in
    /// the vertex shader. Used for diffuse lighting in the fragment shader.
    pub normal: [f32; 3],
    /// Object-space tangent vector (U direction of the normal map). Transformed
    /// to world space in the vertex shader. Used to build the TBN matrix for
    /// tangent-space normal mapping.
    pub tangent: [f32; 3],
    /// Linear RGB colour.
    pub color: [f32; 3],
    /// Texture coordinates in [0, 1] space.  (0,0) is top-left.
    pub uv: [f32; 2],
}

// Interleaved vertex tuple the payload format stores: position, normal,
// tangent, color, uv.
type VertTuple = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 2]);

// LOD alternates: (switch_distance, index buffer) pairs (LOD1..N).
type LodAlternates = Vec<(f32, Vec<u16>)>;

// Deserialised static mesh: vertices, LOD0 indices, and LOD alternates.
type DeserialisedStatic = (Vec<Vertex>, Vec<u16>, LodAlternates);

// Deserialised skinned mesh: vertices, indices, and the bind-pose skeleton.
type DeserialisedSkinned = (Vec<SkinnedVertex>, Vec<u16>, Vec<PayloadJoint>);

/// A fully deserialised skinned payload, including the optional morph and
/// LOD blocks (empty when the payload carries none).
#[derive(Clone, Debug, Default)]
pub struct SkinnedPayload {
    /// Skinned vertices.
    pub vertices: Vec<SkinnedVertex>,
    /// Triangle indices into `vertices`.
    pub indices: Vec<u16>,
    /// The bind-pose skeleton, parents before children.
    pub joints: Vec<PayloadJoint>,
    /// Morph-target block; empty when the mesh has no morphs.
    pub morphs: PayloadMorphs,
    /// LOD slices past LOD0; empty when the mesh declares one level.
    pub lods: LodAlternates,
}

/// Serialise vertex and index slices into the packed binary payload format.
/// Each vertex tuple is (pos, normal, tangent, color, uv).
pub fn serialise(vertices: &[VertTuple], indices: &[u16]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + vertices.len() * 56 + 4 + indices.len() * 2);
    buf.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
    for (pos, normal, tangent, color, uv) in vertices {
        for x in pos
            .iter()
            .chain(normal.iter())
            .chain(tangent.iter())
            .chain(color.iter())
            .chain(uv.iter())
        {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    buf.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    for i in indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf
}

// Magic header for the optional LOD trailer. Absent in legacy payloads so
// `deserialise` keeps working without changes.
const LODS_MAGIC: &[u8; 4] = b"LODS";

/// Serialise a multi-LOD mesh payload. `indices` is LOD0; `lod_alternates`
/// is the list of additional LODs (LOD1..N), each paired with the
/// camera-distance threshold that triggers a switch to it. When
/// `lod_alternates` is empty this is byte-identical to the single-LOD
/// `serialise` output, so the build can call this unconditionally.
pub fn serialise_with_lods(
    vertices: &[VertTuple],
    indices: &[u16],
    lod_alternates: &[(f32, Vec<u16>)],
) -> Vec<u8> {
    let mut buf = serialise(vertices, indices);
    if lod_alternates.is_empty() {
        return buf;
    }
    buf.extend_from_slice(LODS_MAGIC);
    buf.extend_from_slice(&(lod_alternates.len() as u32).to_le_bytes());
    for (distance, idx) in lod_alternates {
        buf.extend_from_slice(&distance.to_le_bytes());
        buf.extend_from_slice(&(idx.len() as u32).to_le_bytes());
        for i in idx {
            buf.extend_from_slice(&i.to_le_bytes());
        }
    }
    buf
}

// Magic header for the optional baked-heightfield collider trailer. Rides
// after the (optional) LOD trailer on a `heightfield`-generator ProceduralMesh
// payload so the physics terrain collider can read a ready-made height grid
// instead of decoding the source image at runtime. `deserialise` and
// `deserialise_with_lods` stop after the LOD block and ignore these bytes, so
// the render path is unaffected and legacy payloads keep loading unchanged.
const HFLD_MAGIC: &[u8; 4] = b"HFLD";

/// A baked heightfield collider grid: `rows` x `cols` world-space heights in
/// row-major order (row index increases along +Z, column index along +X),
/// matching the vertex order the heightfield mesh generator emits.
pub struct HeightfieldGrid {
    /// Grid rows, increasing along +Z.
    pub rows: usize,
    /// Grid columns, increasing along +X.
    pub cols: usize,
    /// World-space heights, row-major.
    pub heights: Vec<f32>,
}

/// Serialise a baked-heightfield collider trailer: `"HFLD"` magic, `u32 rows`,
/// `u32 cols`, then `rows * cols` little-endian f32 heights in row-major order.
/// Appended to a heightfield ProceduralMesh payload after the optional LOD
/// trailer.
pub fn serialise_heightfield_trailer(rows: usize, cols: usize, heights: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + 4 + 4 + heights.len() * 4);
    buf.extend_from_slice(HFLD_MAGIC);
    buf.extend_from_slice(&(rows as u32).to_le_bytes());
    buf.extend_from_slice(&(cols as u32).to_le_bytes());
    for h in heights {
        buf.extend_from_slice(&h.to_le_bytes());
    }
    buf
}

/// Decode the baked-heightfield trailer from a static mesh payload, if present.
/// The trailer rides at the very end, so this walks past the vertex, LOD0
/// index, and optional LOD blocks positionally before reading the `"HFLD"`
/// block. Returns `Ok(None)` for any payload without the trailer (i.e. every
/// non-heightfield mesh) so callers can treat absence as "no baked collider".
pub fn deserialise_heightfield(bytes: &[u8]) -> Result<Option<HeightfieldGrid>, String> {
    let mut cur = ByteReader::new(bytes, "mesh payload");

    // Vertex block (56 bytes each), then LOD0 indices (2 bytes each).
    let vertex_count = cur.u32()? as usize;
    cur.skip(checked_product("vertices", &[vertex_count, 56])?)?;
    let index_count = cur.u32()? as usize;
    cur.skip(checked_product("indices", &[index_count, 2])?)?;

    // Optional LOD trailer: skip the whole block when present so the cursor
    // lands on the HFLD trailer (if any) that follows it.
    if cur.peek(LODS_MAGIC) {
        cur.skip(4)?;
        let alt_count = cur.u32()? as usize;
        for _ in 0..alt_count {
            cur.skip(4)?; // switch distance (f32)
            let n = cur.u32()? as usize;
            cur.skip(checked_product("lod indices", &[n, 2])?)?;
        }
    }

    // Optional HFLD trailer.
    if !cur.peek(HFLD_MAGIC) {
        return Ok(None);
    }
    cur.skip(4)?;
    let rows = cur.u32()? as usize;
    let cols = cur.u32()? as usize;
    let count = checked_product("heightfield grid", &[rows, cols])?;
    let block = cur
        .take(checked_product("heightfield grid", &[count, 4])?)
        .map_err(|_| format!("heightfield trailer too short for {rows} x {cols} grid"))?;
    let heights = block.chunks_exact(4).map(|h| chunk_f32(h, 0)).collect();
    Ok(Some(HeightfieldGrid {
        rows,
        cols,
        heights,
    }))
}

/// Vertex layout for skeletally animated meshes. A superset of `Vertex`: the
/// same 56-byte static attributes plus four joint indices and four blend
/// weights. `repr(C)`, 80 bytes, so it casts directly to a GPU buffer.
///
/// The vertex shader skins `pos` / `normal` / `tangent` by blending up to four
/// joint matrices: `sum(weights[k] * joint[joints[k]] * v)`. Weights that sum
/// to less than 1 leave the remainder un-skinned; the build step normalises
/// them so this never happens for authored meshes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct SkinnedVertex {
    /// Object-space position.
    pub pos: [f32; 3],
    /// Unit-length normal.
    pub normal: [f32; 3],
    /// Object-space tangent, the normal map's U direction.
    pub tangent: [f32; 3],
    /// Linear RGB colour.
    pub color: [f32; 3],
    /// Texture coordinates.
    pub uv: [f32; 2],
    /// Indices into the skeleton's joint array, one per blend weight.
    pub joints: [u16; 4],
    /// Blend weights, parallel to `joints`. Normalised at build time.
    pub weights: [f32; 4],
}

// Magic header for the skinned-mesh binary payload. Distinguishes a skinned
// blob from the headerless static `Vertex` format so a mismatched payload
// fails loudly instead of being misread.
const SKINNED_MAGIC: &[u8; 4] = b"SKMV";

// Magic for the optional morph-target block after the joint block.
const MORPH_MAGIC: &[u8; 4] = b"MRPH";

/// One morph-target vertex delta as the GPU consumes it: position and normal
/// offsets added to the bind pose before skinning, scaled by the target's
/// weight. Plain tightly packed floats; the shader-side struct uses packed
/// types so the 24-byte stride matches.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct MorphDelta {
    /// World-space position.
    pub position: [f32; 3],
    /// Unit-length normal.
    pub normal: [f32; 3],
}

/// Morph-target block of a skinned payload: target names plus dense
/// target-major deltas (`deltas[t * vertex_count + v]`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PayloadMorphs {
    /// Morph-target names, in target order.
    pub names: Vec<String>,
    /// Dense target-major deltas, `deltas[t * vertex_count + v]`.
    pub deltas: Vec<MorphDelta>,
}

impl PayloadMorphs {
    /// Whether the mesh declares no morph targets.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Morph targets on the mesh.
    pub fn target_count(&self) -> usize {
        self.names.len()
    }
}

/// One joint of a skinned mesh's bind-pose skeleton, as stored in the
/// compiled payload. Mirrors `assets::skinned_mesh::SkeletonJoint` but lives in
/// `gfx` so the payload format stays self-contained: the build/runtime
/// boundaries convert between the two. Parents must appear before their
/// children, so the runtime can walk the array once when building the
/// `Skeleton`.
#[derive(Clone, Debug, PartialEq)]
pub struct PayloadJoint {
    /// The joint's authored name.
    pub name: String,
    /// Index of the parent joint, or -1 for a root.
    pub parent: i32,
    /// Bind-pose local translation.
    pub translation: [f32; 3],
    /// YXZ Euler rotation in degrees.
    pub rotation_deg: [f32; 3],
    /// Per-axis scale.
    pub scale: [f32; 3],
}

// Serialise skinned vertices, indices, and bind-pose skeleton into a packed
// binary payload.
//
// Format (little-endian): `"SKMV"` magic, `u32 vertex_count`,
// `vertex_count * 80` bytes of interleaved `SkinnedVertex` data,
// `u32 index_count`, `index_count * 2` bytes of u16 indices,
// `u32 joint_count`, then `joint_count` joint records, each:
// `u32 name_byte_len`, name UTF-8 bytes, `i32 parent`,
// `f32×3 translation`, `f32×3 rotation_deg`, `f32×3 scale`.
//
// The skeleton block is always present (possibly with `joint_count == 0`),
// so a payload deserialises into a self-contained runtime view, no need
// for the args JSON to carry the skeleton alongside.
//
// Calls [`serialise_skinned_with_lods`] with an empty alternates list, so
// the on-wire format is identical to the legacy single-LOD payload when
// no alternates are present.
#[cfg(test)]
pub(crate) fn serialise_skinned(
    vertices: &[SkinnedVertex],
    indices: &[u16],
    joints: &[PayloadJoint],
) -> Vec<u8> {
    serialise_skinned_with_lods(vertices, indices, joints, &PayloadMorphs::default(), &[])
}

/// Serialise a multi-LOD skinned mesh. Two optional blocks ride after the
/// joint block, each announced by a magic: `"MRPH"` (`u32 target_count`, per
/// target `u32 name_byte_len` + name UTF-8 bytes, then
/// `target_count * vertex_count * 24` bytes of dense f32 deltas) and `"LODS"`
/// (`u32 alt_count`, then per alternate `f32 switch_distance`,
/// `u32 index_count`, `index_count * 2` bytes of u16 indices). Empty morphs
/// and alternates match the legacy single-LOD payload byte-for-byte.
pub fn serialise_skinned_with_lods(
    vertices: &[SkinnedVertex],
    indices: &[u16],
    joints: &[PayloadJoint],
    morphs: &PayloadMorphs,
    lod_alternates: &[(f32, Vec<u16>)],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + 4 + vertices.len() * 80 + 4 + indices.len() * 2 + 4);
    buf.extend_from_slice(SKINNED_MAGIC);
    buf.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
    for v in vertices {
        for f in v
            .pos
            .iter()
            .chain(v.normal.iter())
            .chain(v.tangent.iter())
            .chain(v.color.iter())
            .chain(v.uv.iter())
        {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        for j in v.joints {
            buf.extend_from_slice(&j.to_le_bytes());
        }
        for w in v.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
    buf.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    for i in indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf.extend_from_slice(&(joints.len() as u32).to_le_bytes());
    for j in joints {
        let name_bytes = j.name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&j.parent.to_le_bytes());
        for x in j
            .translation
            .iter()
            .chain(j.rotation_deg.iter())
            .chain(j.scale.iter())
        {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    if !morphs.is_empty() {
        buf.extend_from_slice(MORPH_MAGIC);
        buf.extend_from_slice(&(morphs.names.len() as u32).to_le_bytes());
        for name in &morphs.names {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(name_bytes);
        }
        for d in &morphs.deltas {
            for x in d.position.iter().chain(d.normal.iter()) {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    if !lod_alternates.is_empty() {
        buf.extend_from_slice(LODS_MAGIC);
        buf.extend_from_slice(&(lod_alternates.len() as u32).to_le_bytes());
        for (distance, idx) in lod_alternates {
            buf.extend_from_slice(&distance.to_le_bytes());
            buf.extend_from_slice(&(idx.len() as u32).to_le_bytes());
            for i in idx {
                buf.extend_from_slice(&i.to_le_bytes());
            }
        }
    }
    buf
}

/// Deserialise a packed skinned-mesh payload produced by `serialise_skinned`.
/// The returned skeleton lives in the payload; the args JSON no longer needs
/// to carry it. The optional LOD trailer is parsed and discarded; callers
/// who need LOD alternates should use [`deserialise_skinned_with_lods`].
pub fn deserialise_skinned(bytes: &[u8]) -> Result<DeserialisedSkinned, String> {
    let p = deserialise_skinned_with_lods(bytes)?;
    Ok((p.vertices, p.indices, p.joints))
}

/// Deserialise a packed skinned-mesh payload, also returning any optional
/// LOD trailer. Mirrors [`deserialise_with_lods`] for static meshes:
/// legacy single-LOD payloads have no trailer and produce an empty
/// alternates vec.
pub fn deserialise_skinned_with_lods(bytes: &[u8]) -> Result<SkinnedPayload, String> {
    if bytes.len() < 8 || &bytes[0..4] != SKINNED_MAGIC {
        return Err("skinned mesh payload missing SKMV magic header".to_string());
    }
    let mut cur = ByteReader::new(bytes, "skinned mesh payload");
    cur.skip(4)?;

    let vertex_count = cur.u32()? as usize;
    let vertices = read_skinned_vertices(&mut cur, vertex_count)?;

    let index_count = cur.u32()? as usize;
    let indices = read_indices(&mut cur, index_count, "indices")?;

    let joint_count = cur.u32()? as usize;
    let mut joints_out = Vec::with_capacity(joint_count);
    for _ in 0..joint_count {
        let name_len = cur.u32()? as usize;
        let name = read_name(&mut cur, name_len, "joint name")?;
        let parent = cur.i32()?;
        let mut t = [0f32; 3];
        for x in &mut t {
            *x = cur.f32()?;
        }
        let mut r = [0f32; 3];
        for x in &mut r {
            *x = cur.f32()?;
        }
        let mut s = [0f32; 3];
        for x in &mut s {
            *x = cur.f32()?;
        }
        joints_out.push(PayloadJoint {
            name,
            parent,
            translation: t,
            rotation_deg: r,
            scale: s,
        });
    }

    // Optional morph-target block: names, then dense target-major deltas.
    let mut morphs = PayloadMorphs::default();
    if cur.peek(MORPH_MAGIC) {
        cur.skip(4)?;
        let target_count = cur.u32()? as usize;
        for _ in 0..target_count {
            let name_len = cur.u32()? as usize;
            morphs
                .names
                .push(read_name(&mut cur, name_len, "morph target name")?);
        }
        let delta_count = checked_product("morph deltas", &[target_count, vertex_count])?;
        let block = cur.take(checked_product("morph deltas", &[delta_count, 24])?)?;
        morphs.deltas.extend(block.chunks_exact(24).map(|d| {
            let f = |i: usize| chunk_f32(d, i * 4);
            MorphDelta {
                position: [f(0), f(1), f(2)],
                normal: [f(3), f(4), f(5)],
            }
        }));
    }

    // Optional LOD trailer (mirrors the static-mesh format): legacy
    // single-LOD payloads end at the joint block; if the next four bytes
    // are the `LODS` magic, the alternates follow.
    let mut alternates: Vec<(f32, Vec<u16>)> = Vec::new();
    if cur.peek(LODS_MAGIC) {
        cur.skip(4)?;
        let alt_count = cur.u32()? as usize;
        alternates.reserve(alt_count);
        for _ in 0..alt_count {
            let distance = cur.f32()?;
            let n = cur.u32()? as usize;
            let alt = read_indices(&mut cur, n, "LOD indices")?;
            alternates.push((distance, alt));
        }
    }

    Ok(SkinnedPayload {
        vertices,
        indices,
        joints: joints_out,
        morphs,
        lods: alternates,
    })
}

/// Deserialise a packed payload, also returning any optional LOD trailer.
/// Legacy single-LOD payloads have no trailer and produce an empty
/// alternates vec; multi-LOD payloads parse the `"LODS"` block after the
/// LOD0 indices and return one entry per additional level. The order is
/// preserved: `alternates[i]` is LOD `i + 1` and applies at camera
/// distance ≥ `alternates[i].0`.
pub fn deserialise_with_lods(bytes: &[u8]) -> Result<DeserialisedStatic, String> {
    let mut cur = ByteReader::new(bytes, "mesh payload");

    let vertex_count = cur.u32()? as usize;
    let vertices = read_vertices(&mut cur, vertex_count)?;

    let index_count = cur.u32()? as usize;
    let indices = read_indices(&mut cur, index_count, "indices")?;

    // Optional LOD trailer. The legacy single-LOD payload ends here; check
    // for the `LODS` magic before reading anything more.
    let mut alternates = Vec::new();
    if cur.peek(LODS_MAGIC) {
        cur.skip(4)?;
        let alt_count = cur.u32()? as usize;
        alternates.reserve(alt_count);
        for _ in 0..alt_count {
            let distance = cur.f32()?;
            let n = cur.u32()? as usize;
            let alt = read_indices(&mut cur, n, "LOD indices")?;
            alternates.push((distance, alt));
        }
    }

    Ok((vertices, indices, alternates))
}

// Read `count` interleaved skinned vertices (14 floats + 4 u16 joints + 4
// weights = 80 bytes), bounds-checking the whole block once.
fn read_skinned_vertices(
    cur: &mut ByteReader<'_>,
    count: usize,
) -> Result<Vec<SkinnedVertex>, String> {
    let block = cur.take(checked_product("skinned vertices", &[count, 80])?)?;
    Ok(block
        .chunks_exact(80)
        .map(|v| {
            let f = |i: usize| chunk_f32(v, i * 4);
            let j = |i: usize| chunk_u16(v, 56 + i * 2);
            let w = |i: usize| chunk_f32(v, 64 + i * 4);
            SkinnedVertex {
                pos: [f(0), f(1), f(2)],
                normal: [f(3), f(4), f(5)],
                tangent: [f(6), f(7), f(8)],
                color: [f(9), f(10), f(11)],
                uv: [f(12), f(13)],
                joints: [j(0), j(1), j(2), j(3)],
                weights: [w(0), w(1), w(2), w(3)],
            }
        })
        .collect())
}

// Read `count` interleaved 14-float vertices, bounds-checking the whole block
// once rather than per field.
fn read_vertices(cur: &mut ByteReader<'_>, count: usize) -> Result<Vec<Vertex>, String> {
    let block = cur.take(checked_product("vertices", &[count, 56])?)?;
    Ok(block
        .chunks_exact(56)
        .map(|v| {
            let f = |i: usize| chunk_f32(v, i * 4);
            Vertex {
                pos: [f(0), f(1), f(2)],
                normal: [f(3), f(4), f(5)],
                tangent: [f(6), f(7), f(8)],
                color: [f(9), f(10), f(11)],
                uv: [f(12), f(13)],
            }
        })
        .collect())
}
/// Deserialise a packed payload back into typed vertex and index vecs (static),
/// ignoring any LOD trailer.
#[cfg(test)]
pub fn deserialise(bytes: &[u8]) -> Result<(Vec<Vertex>, Vec<u16>), String> {
    let (vertices, indices, _) = deserialise_with_lods(bytes)?;
    Ok((vertices, indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_skinned() -> Vec<SkinnedVertex> {
        vec![
            SkinnedVertex {
                pos: [1.0, 2.0, 3.0],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                color: [0.5, 0.6, 0.7],
                uv: [0.25, 0.75],
                joints: [0, 1, 2, 3],
                weights: [0.5, 0.3, 0.2, 0.0],
            },
            SkinnedVertex {
                pos: [-4.0, 5.0, -6.0],
                normal: [0.0, 0.0, 1.0],
                tangent: [0.0, 1.0, 0.0],
                color: [1.0, 1.0, 1.0],
                uv: [0.0, 1.0],
                joints: [7, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
        ]
    }

    fn sample_skeleton() -> Vec<PayloadJoint> {
        vec![
            PayloadJoint {
                name: "root".to_string(),
                parent: -1,
                translation: [0.0, 0.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
            PayloadJoint {
                name: "tip".to_string(),
                parent: 0,
                translation: [0.0, 1.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
        ]
    }

    #[test]
    fn skinned_roundtrip_preserves_data() {
        let verts = sample_skinned();
        let idxs = vec![0u16, 1, 0];
        let skel = sample_skeleton();
        let bytes = serialise_skinned(&verts, &idxs, &skel);
        let (out_v, out_i, out_s) = deserialise_skinned(&bytes).expect("deserialise");
        assert_eq!(out_v, verts);
        assert_eq!(out_i, idxs);
        assert_eq!(out_s, skel);
    }

    #[test]
    fn skinned_roundtrip_with_empty_skeleton_keeps_trailer_present() {
        // joint_count == 0 still emits the u32 length prefix, so the format
        // is uniform regardless of whether the asset declared a skeleton.
        let verts = sample_skinned();
        let idxs = vec![0u16, 1, 0];
        let bytes = serialise_skinned(&verts, &idxs, &[]);
        let (out_v, out_i, out_s) = deserialise_skinned(&bytes).expect("deserialise");
        assert_eq!(out_v, verts);
        assert_eq!(out_i, idxs);
        assert!(out_s.is_empty());
    }

    #[test]
    fn skinned_payload_size_is_predictable() {
        // magic + vert_count + 2*vertex + idx_count + 3*idx + joint_count
        // + per-joint: name_len + name + parent + 3*vec3.
        let skel = sample_skeleton();
        let bytes = serialise_skinned(&sample_skinned(), &[0u16, 1, 0], &skel);
        let per_joint = skel
            .iter()
            .map(|j| 4 + j.name.len() + 4 + 12 + 12 + 12)
            .sum::<usize>();
        assert_eq!(bytes.len(), 4 + 4 + 2 * 80 + 4 + 3 * 2 + 4 + per_joint);
    }

    #[test]
    fn vertex_layout_matches_msl() {
        // `Vertex` is read through a pointer by the RT skinning kernel
        // (`VtxOut` in rt_skin.metal: five packed_float* fields, 56-byte
        // stride) and as the static RT vertex format, so the field offsets
        // must match exactly. The main/shadow passes consume it through a
        // vertex descriptor declaring the same 0/12/24/36/48 attribute offsets.
        use core::mem::{offset_of, size_of};
        assert_eq!(size_of::<Vertex>(), 56);
        assert_eq!(offset_of!(Vertex, pos), 0);
        assert_eq!(offset_of!(Vertex, normal), 12);
        assert_eq!(offset_of!(Vertex, tangent), 24);
        assert_eq!(offset_of!(Vertex, color), 36);
        assert_eq!(offset_of!(Vertex, uv), 48);
    }

    #[test]
    fn skinned_vertex_layout_matches_msl() {
        // `SkinnedVertex` is read through a pointer by the RT skinning kernel
        // (`SkinnedVtxIn` in rt_skin.metal), whose packed_float* + ushort[4] +
        // packed_float4 fields must line up byte-for-byte with this 80-byte
        // struct. The main/shadow skinned passes consume it through a vertex
        // descriptor declaring the same attribute offsets.
        use core::mem::{offset_of, size_of};
        assert_eq!(size_of::<SkinnedVertex>(), 80);
        assert_eq!(offset_of!(SkinnedVertex, pos), 0);
        assert_eq!(offset_of!(SkinnedVertex, normal), 12);
        assert_eq!(offset_of!(SkinnedVertex, tangent), 24);
        assert_eq!(offset_of!(SkinnedVertex, color), 36);
        assert_eq!(offset_of!(SkinnedVertex, uv), 48);
        assert_eq!(offset_of!(SkinnedVertex, joints), 56);
        assert_eq!(offset_of!(SkinnedVertex, weights), 64);
    }

    #[test]
    fn morph_delta_layout_matches_msl() {
        // `MorphDelta` is read through a pointer by the deform passes
        // (`MorphDelta` in rt_skin.metal, `VsMorphDelta` in main.metal),
        // both declaring two packed_float3 fields at a 24-byte stride.
        use core::mem::{offset_of, size_of};
        assert_eq!(size_of::<MorphDelta>(), 24);
        assert_eq!(offset_of!(MorphDelta, position), 0);
        assert_eq!(offset_of!(MorphDelta, normal), 12);
    }

    #[test]
    fn deserialise_skinned_rejects_missing_magic() {
        // The static payload format has no magic header, so feeding one in
        // must be rejected rather than silently misread.
        let static_bytes = serialise(&[([0.0; 3], [0.0; 3], [0.0; 3], [1.0; 3], [0.0; 2])], &[]);
        assert!(deserialise_skinned(&static_bytes).is_err());
    }

    fn sample_skinned_vertex(pos: [f32; 3]) -> SkinnedVertex {
        SkinnedVertex {
            pos,
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0; 3],
            uv: [0.0, 0.0],
            joints: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn skinned_payload_round_trips_the_morph_block() {
        let vertices = vec![
            sample_skinned_vertex([0.0, 0.0, 0.0]),
            sample_skinned_vertex([1.0, 0.0, 0.0]),
        ];
        let joints = vec![PayloadJoint {
            name: "root".to_string(),
            parent: -1,
            translation: [0.0; 3],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        }];
        let morphs = PayloadMorphs {
            names: vec!["smile".to_string(), "blink".to_string()],
            deltas: vec![
                MorphDelta {
                    position: [0.1, 0.2, 0.3],
                    normal: [0.0, 0.0, 1.0],
                },
                MorphDelta::default(),
                MorphDelta::default(),
                MorphDelta {
                    position: [-0.5, 0.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                },
            ],
        };
        let lods = vec![(9.0_f32, vec![0u16, 1, 0])];
        let bytes = serialise_skinned_with_lods(&vertices, &[0, 1, 0], &joints, &morphs, &lods);
        let p = deserialise_skinned_with_lods(&bytes).expect("deserialise");
        assert_eq!(p.vertices.len(), 2);
        assert_eq!(p.joints.len(), 1);
        assert_eq!(p.morphs, morphs, "morph block must round-trip exactly");
        assert_eq!(p.lods.len(), 1, "LOD trailer must survive after MRPH");
        assert_eq!(p.lods[0].1, vec![0u16, 1, 0]);
    }

    #[test]
    fn skinned_payload_without_morphs_is_byte_identical_to_legacy() {
        let vertices = vec![sample_skinned_vertex([0.0, 0.0, 0.0])];
        let legacy = serialise_skinned(&vertices, &[0, 0, 0], &[]);
        let with_empty =
            serialise_skinned_with_lods(&vertices, &[0, 0, 0], &[], &PayloadMorphs::default(), &[]);
        assert_eq!(legacy, with_empty, "empty morphs must add no bytes");
        let p = deserialise_skinned_with_lods(&legacy).expect("deserialise");
        assert!(p.morphs.is_empty());
    }

    fn sample_static_verts() -> Vec<VertTuple> {
        vec![
            (
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0; 3],
                [0.0, 0.0],
            ),
            (
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0; 3],
                [1.0, 0.0],
            ),
            (
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0; 3],
                [0.0, 1.0],
            ),
        ]
    }

    #[test]
    fn serialise_with_no_lods_matches_legacy_format() {
        let verts = sample_static_verts();
        let idx = vec![0u16, 1, 2];
        let legacy = serialise(&verts, &idx);
        let with_lods = serialise_with_lods(&verts, &idx, &[]);
        assert_eq!(legacy, with_lods, "no alternates → no trailer bytes");
    }

    #[test]
    fn lod_trailer_roundtrip_preserves_distances_and_indices() {
        let verts = sample_static_verts();
        let lod0 = vec![0u16, 1, 2];
        let alternates = vec![(8.0_f32, vec![0u16, 2, 1]), (25.0_f32, vec![0u16, 1, 2])];
        let bytes = serialise_with_lods(&verts, &lod0, &alternates);
        let (out_v, out_idx, out_alts) = deserialise_with_lods(&bytes).expect("deserialise");
        assert_eq!(out_v.len(), verts.len());
        assert_eq!(out_idx, lod0);
        assert_eq!(out_alts.len(), 2);
        assert_eq!(out_alts[0].0, 8.0);
        assert_eq!(out_alts[0].1, vec![0u16, 2, 1]);
        assert_eq!(out_alts[1].0, 25.0);
        assert_eq!(out_alts[1].1, vec![0u16, 1, 2]);
    }

    #[test]
    fn legacy_payload_has_no_alternates() {
        // A payload written by the single-LOD `serialise` must deserialise via
        // `deserialise_with_lods` with an empty alternates vec: backward
        // compatibility for every existing on-disk blob.
        let verts = sample_static_verts();
        let idx = vec![0u16, 1, 2];
        let bytes = serialise(&verts, &idx);
        let (_, _, alts) = deserialise_with_lods(&bytes).expect("deserialise");
        assert!(alts.is_empty());
    }

    #[test]
    fn heightfield_trailer_roundtrips_without_lods() {
        let verts = sample_static_verts();
        let idx = vec![0u16, 1, 2];
        let heights = vec![0.0f32, 1.0, 2.0, 3.0];
        let mut bytes = serialise_with_lods(&verts, &idx, &[]);
        bytes.extend_from_slice(&serialise_heightfield_trailer(2, 2, &heights));

        let grid = deserialise_heightfield(&bytes)
            .expect("parse")
            .expect("trailer present");
        assert_eq!(grid.rows, 2);
        assert_eq!(grid.cols, 2);
        assert_eq!(grid.heights, heights);

        // The render path ignores the trailer entirely.
        let (out_v, out_i, out_alts) = deserialise_with_lods(&bytes).expect("render path");
        assert_eq!(out_v.len(), verts.len());
        assert_eq!(out_i, idx);
        assert!(out_alts.is_empty());
    }

    #[test]
    fn heightfield_trailer_roundtrips_after_lod_trailer() {
        let verts = sample_static_verts();
        let lod0 = vec![0u16, 1, 2];
        let alternates = vec![(8.0_f32, vec![0u16, 2, 1]), (25.0_f32, vec![0u16, 1, 2])];
        let heights = vec![-1.0f32, 0.5, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0];
        let mut bytes = serialise_with_lods(&verts, &lod0, &alternates);
        bytes.extend_from_slice(&serialise_heightfield_trailer(3, 3, &heights));

        // Both trailers parse independently from the same payload.
        let (_, out_i, out_alts) = deserialise_with_lods(&bytes).expect("render path");
        assert_eq!(out_i, lod0);
        assert_eq!(out_alts.len(), 2);

        let grid = deserialise_heightfield(&bytes)
            .expect("parse")
            .expect("trailer present");
        assert_eq!((grid.rows, grid.cols), (3, 3));
        assert_eq!(grid.heights, heights);
    }

    #[test]
    fn a_heightfield_trailer_whose_footprint_overflows_is_rejected() {
        // `rows * cols` fits a usize while the byte footprint `* 4` does not, so
        // checking only the texel count leaves the multiply to wrap: the read
        // then succeeds against an empty slice and hands back a grid whose
        // declared extent has no heights behind it, which every consumer indexes
        // straight off the end.
        let verts = sample_static_verts();
        let mut bytes = serialise_with_lods(&verts, &[0u16, 1, 2], &[]);
        bytes.extend_from_slice(HFLD_MAGIC);
        bytes.extend_from_slice(&0x8000_0000u32.to_le_bytes());
        bytes.extend_from_slice(&0x8000_0000u32.to_le_bytes());

        let err = match deserialise_heightfield(&bytes) {
            Err(e) => e,
            Ok(_) => panic!("an overflowing grid must be rejected"),
        };
        assert!(err.contains("heightfield grid"), "{err}");
    }

    #[test]
    fn no_heightfield_trailer_returns_none() {
        let verts = sample_static_verts();
        let bytes = serialise_with_lods(&verts, &[0u16, 1, 2], &[(10.0, vec![0u16, 2, 1])]);
        assert!(deserialise_heightfield(&bytes).expect("parse").is_none());
    }

    #[test]
    fn legacy_deserialise_still_works_on_multi_lod_payload() {
        // The legacy `deserialise` reader must keep ignoring the LODS
        // trailer so any code path that didn't migrate yet still loads
        // LOD0 from a multi-LOD payload.
        let verts = sample_static_verts();
        let lod0 = vec![0u16, 1, 2];
        let bytes = serialise_with_lods(&verts, &lod0, &[(10.0, vec![0u16, 2, 1])]);
        let (out_v, out_idx) = deserialise(&bytes).expect("legacy reader");
        assert_eq!(out_v.len(), verts.len());
        assert_eq!(out_idx, lod0);
    }
}
