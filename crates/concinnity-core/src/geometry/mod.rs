//! The runtime geometry the engine builds on the fly: voxel-chunk streaming
//! (`build_chunk_mesh` / `build_chunk_impostor_mesh` regenerate a chunk's mesh as
//! it streams in), the glass/water quad generators the GPU backends call, and the
//! shared low-level mesh math (per-vertex tangents, face normals, the vertex
//! tuple type) that both this runtime path and the cook compile path use.
//!
//! The build-time payload compilers (`compile_mesh_payload` / `compile_room_payload`
//! / ... ) and the procedural generators they invoke (room, extrude, primitives,
//! terrain, skybox, heightfield) live in `concinnity-cook`; they call back into
//! the shared helpers exported here.

// Procedural voxel-chunk generation, consumed by the backends' chunk-streaming
// path.
mod chunk_gen;
pub mod glass_quad;
mod voxel;
pub mod water_grid;

use crate::math::vec3::{vec3_add, vec3_normalise};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

pub use chunk_gen::{ChunkBlockType, ChunkGenerator};
// Shared with the cook crate's `compile_voxel_chunk_payload`, which builds the
// same palette the runtime `build_chunk_mesh` consumes.
pub use voxel::{PaletteSlot, build_voxel_mesh};

/// Interleaved CPU vertex tuple the geometry generators produce before packing
/// into the GPU `Vertex`: position, normal, color, uv. Public so the cook crate's
/// generators and payload compilers can name the same shape the runtime tangent
/// pass consumes.
pub type Vert = ([f32; 3], [f32; 3], [f32; 3], [f32; 2]);

// Convert a payload-form joint back into the args-form `SkeletonJoint`.
fn payload_joint_to_def(
    j: crate::gfx::mesh_payload::PayloadJoint,
) -> crate::components::SkeletonJoint {
    crate::components::SkeletonJoint {
        name: j.name,
        parent: j.parent,
        translation: j.translation,
        rotation_deg: j.rotation_deg,
        scale: j.scale,
    }
}

/// Convert a payload-joint vec to the args-form vec the runtime
/// `build_skeleton_from_joint_defs` consumes. Public so the client runtime
/// init path can call it without re-implementing the field mapping.
pub fn payload_joints_to_defs(
    joints: Vec<crate::gfx::mesh_payload::PayloadJoint>,
) -> Vec<crate::components::SkeletonJoint> {
    joints.into_iter().map(payload_joint_to_def).collect()
}

/// Build a renderable mesh for one procedurally generated chunk.
///
/// The runtime counterpart of cook's `compile_voxel_chunk_payload`: it takes a
/// chunk's already-generated block array and resolved palette and returns
/// interleaved `Vertex` geometry directly, with no on-disk payload in between.
/// Chunk streaming (`app::chunk_stream`) calls this on its background thread.
pub fn build_chunk_mesh(
    dim: [u32; 3],
    block_size: f32,
    blocks: &[u32],
    palette: &[ChunkBlockType],
) -> Result<(Vec<crate::gfx::mesh_payload::Vertex>, Vec<u16>), String> {
    let slots: Vec<Option<PaletteSlot>> = palette
        .iter()
        .map(|b| {
            if b.solid {
                Some(PaletteSlot {
                    uv_top: b.uv_top,
                    uv_bottom: b.uv_bottom,
                    uv_side: b.uv_side,
                })
            } else {
                None
            }
        })
        .collect();
    let (verts, indices) = build_voxel_mesh(dim, block_size, blocks, &slots)?;
    let tangents = compute_tangents(&verts, &indices);
    let vertices = verts
        .into_iter()
        .zip(tangents)
        .map(
            |((pos, normal, color, uv), tangent)| crate::gfx::mesh_payload::Vertex {
                pos,
                normal,
                tangent,
                color,
                uv,
            },
        )
        .collect();
    Ok((vertices, indices))
}

/// Build a coarse "impostor" mesh for one distant chunk from its terrain
/// surface heights.
///
/// Where [`build_chunk_mesh`] emits every visible voxel face, this stands a
/// far-away chunk in for a fraction of the triangles: the surface height
/// sampled on a coarse `step`-block grid becomes a low-poly top surface (one
/// quad per coarse cell), wrapped by a perimeter skirt that drops to the chunk
/// floor to hide the gap against a nearer full-detail neighbour or the world
/// edge. Side and subsurface geometry are dropped: invisible at impostor
/// distance.
///
/// `heights[gz * (nx + 1) + gx]` is the surface block index at coarse corner
/// `(gx, gz)`, where `nx = ceil(dx / step)`, `nz = ceil(dz / step)`, and corner
/// `gx`'s local block column is `min(gx * step, dx)` (the last corner lands on
/// the chunk's far edge so adjacent impostors share it exactly). The caller
/// samples those heights from [`ChunkGenerator::surface_height_world`] at the
/// matching world columns, which keeps neighbouring impostors watertight.
/// `top_uv` / `side_uv` are the surface block's atlas rects.
pub fn build_chunk_impostor_mesh(
    dim: [u32; 3],
    block_size: f32,
    step: u32,
    heights: &[i32],
    top_uv: [f32; 4],
    side_uv: [f32; 4],
) -> Result<(Vec<crate::gfx::mesh_payload::Vertex>, Vec<u16>), String> {
    let step = step.max(1);
    let [dx, _dy, dz] = dim;
    let nx = dx.div_ceil(step);
    let nz = dz.div_ceil(step);
    let cols = (nx + 1) as usize;
    let expected = ((nx + 1) * (nz + 1)) as usize;
    if heights.len() != expected {
        return Err(format!(
            "impostor mesh: expected {} height samples for a {}x{} coarse grid, got {}",
            expected,
            nx + 1,
            nz + 1,
            heights.len()
        ));
    }
    let bs = block_size;
    // Local position of coarse corner gx / gz. The last corner clamps to the
    // chunk's far edge (dx / dz) so a non-dividing `step` still closes the mesh
    // exactly on the boundary shared with the next chunk.
    let cx = |gx: u32| ((gx * step).min(dx) as f32) * bs;
    let cz = |gz: u32| ((gz * step).min(dz) as f32) * bs;
    // Top of the surface block at corner (gx, gz): the +1 matches the full
    // mesher, whose top face of block `h` sits at `(h + 1) * block_size`.
    let surf_y = |gx: u32, gz: u32| ((heights[gz as usize * cols + gx as usize] + 1) as f32) * bs;

    type RawVerts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;
    let mut verts: RawVerts = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    let color = [0.75f32, 0.74, 0.72];

    // CCW-from-outside quad, matching `build_voxel_mesh`'s winding + UV mapping.
    let mut emit_quad = |corners: [[f32; 3]; 4], normal: [f32; 3], uv_rect: [f32; 4]| {
        if verts.len() + 4 > u16::MAX as usize {
            return;
        }
        let base = verts.len() as u16;
        let [u0, v0, u1, v1] = uv_rect;
        let uvs = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]];
        for (i, p) in corners.iter().enumerate() {
            verts.push((*p, normal, color, uvs[i]));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    };

    // Top surface: one up-facing quad per coarse cell. Each cell carries its
    // own 4 vertices; adjacent cells sample identical corner heights, so the
    // duplicated corner vertices coincide and the surface stays watertight.
    let n_up = [0.0, 1.0, 0.0];
    for gz in 0..nz {
        for gx in 0..nx {
            emit_quad(
                [
                    [cx(gx), surf_y(gx, gz + 1), cz(gz + 1)],
                    [cx(gx + 1), surf_y(gx + 1, gz + 1), cz(gz + 1)],
                    [cx(gx + 1), surf_y(gx + 1, gz), cz(gz)],
                    [cx(gx), surf_y(gx, gz), cz(gz)],
                ],
                n_up,
                top_uv,
            );
        }
    }

    // Perimeter skirt: vertical quads from the surface edge down to the chunk
    // floor (y = 0), one per boundary segment, facing outward. Hides the seam
    // where a coarse impostor abuts a nearer full chunk (or the world edge).
    let x_max = (dx as f32) * bs;
    let z_max = (dz as f32) * bs;
    for gx in 0..nx {
        // -Z edge (z = 0), outward normal -Z.
        emit_quad(
            [
                [cx(gx + 1), 0.0, 0.0],
                [cx(gx), 0.0, 0.0],
                [cx(gx), surf_y(gx, 0), 0.0],
                [cx(gx + 1), surf_y(gx + 1, 0), 0.0],
            ],
            [0.0, 0.0, -1.0],
            side_uv,
        );
        // +Z edge (z = z_max), outward normal +Z.
        emit_quad(
            [
                [cx(gx), 0.0, z_max],
                [cx(gx + 1), 0.0, z_max],
                [cx(gx + 1), surf_y(gx + 1, nz), z_max],
                [cx(gx), surf_y(gx, nz), z_max],
            ],
            [0.0, 0.0, 1.0],
            side_uv,
        );
    }
    for gz in 0..nz {
        // -X edge (x = 0), outward normal -X.
        emit_quad(
            [
                [0.0, 0.0, cz(gz)],
                [0.0, 0.0, cz(gz + 1)],
                [0.0, surf_y(0, gz + 1), cz(gz + 1)],
                [0.0, surf_y(0, gz), cz(gz)],
            ],
            [-1.0, 0.0, 0.0],
            side_uv,
        );
        // +X edge (x = x_max), outward normal +X.
        emit_quad(
            [
                [x_max, 0.0, cz(gz + 1)],
                [x_max, 0.0, cz(gz)],
                [x_max, surf_y(nx, gz), cz(gz)],
                [x_max, surf_y(nx, gz + 1), cz(gz + 1)],
            ],
            [1.0, 0.0, 0.0],
            side_uv,
        );
    }

    let tangents = compute_tangents(&verts, &indices);
    let vertices = verts
        .into_iter()
        .zip(tangents)
        .map(
            |((pos, normal, color, uv), tangent)| crate::gfx::mesh_payload::Vertex {
                pos,
                normal,
                tangent,
                color,
                uv,
            },
        )
        .collect();
    Ok((vertices, indices))
}

/// Compute a per-vertex tangent vector for every vertex in the mesh.
///
/// For each triangle the tangent is derived from the UV gradient. Contributions
/// are accumulated at each shared vertex and then Gram-Schmidt orthogonalized
/// against the existing normal. Degenerate UV triangles fall back to an
/// arbitrary perpendicular so the TBN matrix is always well-defined. Shared with
/// the cook payload compilers so baked meshes and streamed chunks derive
/// identical tangents.
pub fn compute_tangents(vertices: &[Vert], indices: &[u16]) -> Vec<[f32; 3]> {
    let n = vertices.len();
    let mut accum: Vec<[f32; 3]> = vec![[0.0; 3]; n];

    let tris = indices.len() / 3;
    for t in 0..tris {
        let ia = indices[t * 3] as usize;
        let ib = indices[t * 3 + 1] as usize;
        let ic = indices[t * 3 + 2] as usize;
        if ia >= n || ib >= n || ic >= n {
            continue;
        }
        let (pa, _, _, uva) = vertices[ia];
        let (pb, _, _, uvb) = vertices[ib];
        let (pc, _, _, uvc) = vertices[ic];

        let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let du1 = uvb[0] - uva[0];
        let dv1 = uvb[1] - uva[1];
        let du2 = uvc[0] - uva[0];
        let dv2 = uvc[1] - uva[1];

        let denom = du1 * dv2 - du2 * dv1;
        let tangent = if denom.abs() < 1e-8 {
            arbitrary_tangent(vertices[ia].1)
        } else {
            let r = 1.0 / denom;
            [
                (e1[0] * dv2 - e2[0] * dv1) * r,
                (e1[1] * dv2 - e2[1] * dv1) * r,
                (e1[2] * dv2 - e2[2] * dv1) * r,
            ]
        };

        vec3_add(&mut accum[ia], tangent);
        vec3_add(&mut accum[ib], tangent);
        vec3_add(&mut accum[ic], tangent);
    }

    vertices
        .iter()
        .zip(accum)
        .map(|((_, normal, _, _), raw)| {
            let dot = raw[0] * normal[0] + raw[1] * normal[1] + raw[2] * normal[2];
            let t = [
                raw[0] - dot * normal[0],
                raw[1] - dot * normal[1],
                raw[2] - dot * normal[2],
            ];
            vec3_normalise(t)
        })
        .collect()
}

// Returns an arbitrary unit vector perpendicular to `normal`.
fn arbitrary_tangent(normal: [f32; 3]) -> [f32; 3] {
    let up = if normal[0].abs() <= normal[1].abs() && normal[0].abs() <= normal[2].abs() {
        [1.0f32, 0.0, 0.0]
    } else if normal[1].abs() <= normal[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let t = [
        up[1] * normal[2] - up[2] * normal[1],
        up[2] * normal[0] - up[0] * normal[2],
        up[0] * normal[1] - up[1] * normal[0],
    ];
    vec3_normalise(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    // A flat coarse height grid of `(nx+1)*(nz+1)` corners all at height `h`.
    fn flat_heights(dim: [u32; 3], step: u32, h: i32) -> Vec<i32> {
        let nx = dim[0].div_ceil(step);
        let nz = dim[2].div_ceil(step);
        vec![h; ((nx + 1) * (nz + 1)) as usize]
    }

    #[test]
    fn impostor_mesh_counts_match_cells_plus_skirt() {
        let dim = [8, 8, 8];
        let step = 4;
        let heights = flat_heights(dim, step, 3);
        let uv = [0.0, 0.0, 1.0, 1.0];
        let (v, i) = build_chunk_impostor_mesh(dim, 1.0, step, &heights, uv, uv).expect("impostor");
        // 2x2 top cells + 2 skirt quads per edge * 4 edges = 4 + 8 = 12 quads.
        assert_eq!(v.len(), 12 * 4);
        assert_eq!(i.len(), 12 * 6);
    }

    #[test]
    fn impostor_top_surface_sits_above_the_surface_block() {
        let dim = [8, 4, 8];
        let bs = 2.0;
        let heights = flat_heights(dim, 4, 1);
        let uv = [0.0, 0.0, 1.0, 1.0];
        let (v, _) = build_chunk_impostor_mesh(dim, bs, 4, &heights, uv, uv).expect("impostor");
        // Surface block index 1: its top face sits at (1 + 1) * block_size.
        let want = 2.0 * bs;
        assert!(v.iter().any(|vert| (vert.pos[1] - want).abs() < 1e-4));
    }

    #[test]
    fn impostor_spans_the_full_chunk_footprint() {
        let dim = [8, 4, 8];
        let bs = 2.0;
        let heights = flat_heights(dim, 4, 1);
        let uv = [0.0, 0.0, 1.0, 1.0];
        let (v, _) = build_chunk_impostor_mesh(dim, bs, 4, &heights, uv, uv).expect("impostor");
        let max_x = v.iter().map(|vert| vert.pos[0]).fold(0.0f32, f32::max);
        let max_z = v.iter().map(|vert| vert.pos[2]).fold(0.0f32, f32::max);
        assert!((max_x - (dim[0] as f32 * bs)).abs() < 1e-4);
        assert!((max_z - (dim[2] as f32 * bs)).abs() < 1e-4);
    }

    #[test]
    fn impostor_rejects_a_mismatched_height_grid() {
        let dim = [8, 8, 8];
        let uv = [0.0, 0.0, 1.0, 1.0];
        let bad = vec![0; 3];
        assert!(build_chunk_impostor_mesh(dim, 1.0, 4, &bad, uv, uv).is_err());
    }

    #[test]
    fn impostor_with_step_exceeding_chunk_collapses_to_one_cell() {
        let dim = [8, 8, 8];
        let uv = [0.0, 0.0, 1.0, 1.0];
        let heights = flat_heights(dim, 32, 2);
        let (v, i) = build_chunk_impostor_mesh(dim, 1.0, 32, &heights, uv, uv).expect("impostor");
        // One coarse cell + 1 skirt quad per edge = 1 + 4 = 5 quads.
        assert_eq!(v.len(), 5 * 4);
        assert_eq!(i.len(), 5 * 6);
    }

    #[test]
    fn degenerate_uvs_still_produce_unit_tangents() {
        let verts: Vec<Vert> = vec![
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0; 3], [0.0, 0.0]),
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0; 3], [0.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0; 3], [0.0, 0.0]),
        ];
        let tangents = compute_tangents(&verts, &[0, 1, 2]);
        for (t, v) in tangents.iter().zip(&verts) {
            let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5);
            let dot = t[0] * v.1[0] + t[1] * v.1[1] + t[2] * v.1[2];
            assert!(dot.abs() < 1e-5);
        }
    }

    #[test]
    fn out_of_range_indices_are_skipped_by_the_tangent_pass() {
        let verts: Vec<Vert> = vec![([0.0; 3], [0.0, 0.0, 1.0], [1.0; 3], [0.0, 0.0])];
        // No triangle survives, so the accumulated tangent falls back to +Y.
        let tangents = compute_tangents(&verts, &[0, 1, 2]);
        assert_eq!(tangents, vec![[0.0, 1.0, 0.0]]);
    }

    #[test]
    fn arbitrary_tangent_is_unit_and_perpendicular() {
        for normal in [
            [1.0f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.6, 0.5, 0.3],
        ] {
            let t = arbitrary_tangent(normal);
            let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normal {normal:?}");
            let dot = t[0] * normal[0] + t[1] * normal[1] + t[2] * normal[2];
            assert!(dot.abs() < 1e-5, "normal {normal:?}");
        }
    }

    #[test]
    fn chunk_mesh_respects_solid_flags() {
        let bt = |solid: bool| ChunkBlockType {
            solid,
            uv_top: [0.0, 0.0, 1.0, 1.0],
            uv_bottom: [0.0, 0.0, 1.0, 1.0],
            uv_side: [0.0, 0.0, 1.0, 1.0],
        };
        let (verts, indices) = build_chunk_mesh([1, 1, 1], 1.0, &[0], &[bt(true)]).unwrap();
        assert_eq!(verts.len(), 24);
        assert_eq!(indices.len(), 36);
        let (verts, indices) = build_chunk_mesh([1, 1, 1], 1.0, &[0], &[bt(false)]).unwrap();
        assert!(verts.is_empty() && indices.is_empty());
        // Two adjacent solid blocks cull the two shared interior faces.
        let (verts, _) = build_chunk_mesh([2, 1, 1], 1.0, &[0, 0], &[bt(true)]).unwrap();
        assert_eq!(verts.len(), 10 * 4);
    }

    #[test]
    fn payload_joints_convert_back_to_joint_defs() {
        let pj = crate::gfx::mesh_payload::PayloadJoint {
            name: "hip".to_string(),
            parent: 2,
            translation: [1.0, 2.0, 3.0],
            rotation_deg: [4.0, 5.0, 6.0],
            scale: [7.0, 8.0, 9.0],
        };
        let defs = payload_joints_to_defs(vec![pj]);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "hip");
        assert_eq!(defs[0].parent, 2);
        assert_eq!(defs[0].translation, [1.0, 2.0, 3.0]);
        assert_eq!(defs[0].rotation_deg, [4.0, 5.0, 6.0]);
        assert_eq!(defs[0].scale, [7.0, 8.0, 9.0]);
    }
}
