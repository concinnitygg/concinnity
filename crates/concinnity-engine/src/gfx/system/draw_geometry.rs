// Draw-object world positions and the world-triangle gather for reflection-probe
// auto-seed.

use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::render_types;

// World-space position used to score a draw object for texture streaming:
// the AABB center when bounds are finite, otherwise the model-matrix
// translation (dynamic props carry a non-finite sentinel AABB).
pub(super) fn draw_object_position(obj: &render_types::DrawObject) -> [f32; 3] {
    let finite = obj
        .bb_min
        .iter()
        .chain(obj.bb_max.iter())
        .all(|v| v.is_finite());
    if finite {
        [
            0.5 * (obj.bb_min[0] + obj.bb_max[0]),
            0.5 * (obj.bb_min[1] + obj.bb_max[1]),
            0.5 * (obj.bb_min[2] + obj.bb_max[2]),
        ]
    } else {
        [obj.model[3][0], obj.model[3][1], obj.model[3][2]]
    }
}

// Above this triangle count the reflection-probe auto-seed skips the world-triangle
// gather and keeps coarse object-AABB occupancy, so a heavy import (Bistro is ~2.8M
// triangles) pays nothing extra at load. Small authored scenes stay well under it and
// get the finer surface-voxel interior detection (a watertight single-mesh room is then
// seen as hollow).
const AUTO_SEED_MAX_TRIANGLES: usize = 200_000;

// Gather world-space triangles from the static draw list for reflection-probe auto-seed
// interior detection (surface voxelization needs real geometry, not AABBs). Returns
// `None` when there is no cullable static geometry or the scene exceeds
// `AUTO_SEED_MAX_TRIANGLES` -- the caller then falls back to coarse AABB occupancy. Each
// cullable draw's indexed triangles are transformed to world space by its model matrix;
// `base_vertex` is honored so streamed (mesh-relative) chunks resolve too, and every
// fetch is bounds-checked against the shared vertex buffer (build-time offsets should be
// in range, but a bad offset is skipped rather than risking an out-of-bounds index).
pub(super) fn gather_auto_seed_triangles(
    draw_objects: &[render_types::DrawObject],
    all_vertices: &[Vertex],
    all_indices: &[u32],
) -> Option<Vec<[[f32; 3]; 3]>> {
    let eligible = |o: &render_types::DrawObject| o.cullable() && o.index_count >= 3;
    let total_tris: usize = draw_objects
        .iter()
        .filter(|o| eligible(o))
        .map(|o| o.index_count / 3)
        .sum();
    if total_tris == 0 || total_tris > AUTO_SEED_MAX_TRIANGLES {
        return None;
    }

    // Column-major model-to-world transform of a model-space point.
    let xf = |m: &[[f32; 4]; 4], p: [f32; 3]| {
        [
            m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
            m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
            m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
        ]
    };

    let mut tris = Vec::with_capacity(total_tris);
    for o in draw_objects.iter().filter(|o| eligible(o)) {
        let iend = o.index_offset + o.index_count;
        if iend > all_indices.len() {
            continue;
        }
        for t in all_indices[o.index_offset..iend].chunks_exact(3) {
            let vi = |k: usize| (t[k] as i64 + o.base_vertex as i64) as usize;
            let (a, b, c) = (vi(0), vi(1), vi(2));
            if a >= all_vertices.len() || b >= all_vertices.len() || c >= all_vertices.len() {
                continue;
            }
            tris.push([
                xf(&o.model, all_vertices[a].pos),
                xf(&o.model, all_vertices[b].pos),
                xf(&o.model, all_vertices[c].pos),
            ]);
        }
    }
    (!tris.is_empty()).then_some(tris)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::render_types::{DrawObject, MaterialUniforms, NO_NORMAL_MAP_SLOT};

    // A finite-bounds draw object at `model` covering `[index_offset, +count)`
    // of the shared index buffer. Culling stays enabled unless the caller
    // passes a non-finite AABB.
    fn draw(
        model: [[f32; 4]; 4],
        bb_min: [f32; 3],
        bb_max: [f32; 3],
        index_offset: usize,
        index_count: usize,
        base_vertex: i32,
    ) -> DrawObject {
        DrawObject {
            vertex_offset: 0,
            vertex_count: 0,
            index_offset,
            index_count,
            base_vertex,
            geometry_generation: 0,
            shader_bucket: 0,
            model,
            texture_slot: 0,
            normal_map_slot: NO_NORMAL_MAP_SLOT,
            material: MaterialUniforms::DEFAULT,
            visible: true,
            resident: true,
            bb_min,
            bb_max,
            cull_distance: 0.0,
            lod_alternates: Vec::new(),
        }
    }

    fn vert(pos: [f32; 3]) -> Vertex {
        Vertex {
            pos,
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
        }
    }

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // Finite bounds score at the AABB center.
    #[test]
    fn draw_object_position_uses_aabb_center_when_finite() {
        let obj = draw(IDENTITY, [-2.0, 0.0, 4.0], [4.0, 6.0, 8.0], 0, 0, 0);
        assert_eq!(draw_object_position(&obj), [1.0, 3.0, 6.0]);
    }

    // A non-finite (dynamic sentinel) AABB falls back to the model translation.
    #[test]
    fn draw_object_position_uses_model_translation_when_unbounded() {
        let model = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [7.0, 8.0, 9.0, 1.0],
        ];
        let obj = draw(model, [f32::NAN; 3], [f32::NAN; 3], 0, 0, 0);
        assert_eq!(draw_object_position(&obj), [7.0, 8.0, 9.0]);
    }

    // A cullable draw's indexed triangle is transformed to world space by its
    // model matrix, honoring base_vertex.
    #[test]
    fn gather_auto_seed_triangles_transforms_by_model_and_base_vertex() {
        // Translate the whole draw by +10 on X; base_vertex shifts the index
        // fetch by one so indices 0,1,2 read vertices 1,2,3.
        let model = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0, 1.0],
        ];
        let verts = vec![
            vert([0.0, 0.0, 0.0]),
            vert([0.0, 0.0, 0.0]),
            vert([1.0, 0.0, 0.0]),
            vert([0.0, 0.0, 1.0]),
        ];
        let idx = vec![0u32, 1, 2];
        let objs = vec![draw(model, [-1.0; 3], [1.0; 3], 0, 3, 1)];
        let tris = gather_auto_seed_triangles(&objs, &verts, &idx).expect("one triangle");
        assert_eq!(tris.len(), 1);
        assert_eq!(tris[0][0], [10.0, 0.0, 0.0]);
        assert_eq!(tris[0][1], [11.0, 0.0, 0.0]);
        assert_eq!(tris[0][2], [10.0, 0.0, 1.0]);
    }

    // With no cullable geometry (a non-finite AABB draw) the auto-seed gather
    // returns None so the caller falls back to coarse AABB occupancy.
    #[test]
    fn gather_auto_seed_triangles_none_without_cullable_geometry() {
        let objs = vec![draw(IDENTITY, [f32::NAN; 3], [f32::NAN; 3], 0, 3, 0)];
        let verts = vec![vert([0.0; 3]), vert([1.0, 0.0, 0.0]), vert([0.0, 0.0, 1.0])];
        assert!(gather_auto_seed_triangles(&objs, &verts, &[0, 1, 2]).is_none());
    }

    // An index range past the end of the shared buffer is skipped rather than
    // panicking; with nothing left to gather the result is None.
    #[test]
    fn gather_auto_seed_triangles_skips_out_of_range_index_span() {
        let objs = vec![draw(IDENTITY, [-1.0; 3], [1.0; 3], 0, 6, 0)];
        // index_count claims 6 indices but the buffer holds only 3.
        let verts = vec![vert([0.0; 3]), vert([1.0, 0.0, 0.0]), vert([0.0, 0.0, 1.0])];
        assert!(gather_auto_seed_triangles(&objs, &verts, &[0, 1, 2]).is_none());
    }

    // A triangle referencing a vertex past the end of the vertex buffer is
    // dropped; an all-dropped scene yields None.
    #[test]
    fn gather_auto_seed_triangles_skips_out_of_range_vertex_index() {
        let objs = vec![draw(IDENTITY, [-1.0; 3], [1.0; 3], 0, 3, 0)];
        let verts = vec![vert([0.0; 3]), vert([1.0, 0.0, 0.0]), vert([0.0, 0.0, 1.0])];
        // Index 9 is out of range for a 3-vertex buffer.
        assert!(gather_auto_seed_triangles(&objs, &verts, &[0, 1, 9]).is_none());
    }
}
