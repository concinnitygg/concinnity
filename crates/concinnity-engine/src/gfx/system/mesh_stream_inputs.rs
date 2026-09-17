// Streaming inputs captured from the built draw list before it moves into the
// backend: texture scoring centers, per-streamed-mesh geometry copies, and the
// draw-to-mesh-handle mapping.

use std::collections::{HashMap, HashSet};

use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::render_types;

use super::draw_geometry::draw_object_position;
use crate::gfx::draw_list::DeferredMeshSeed;
use crate::gfx::streaming::mesh::DecodedMesh;

// Per-streamed-mesh data from `mesh_stream_data`: the draw-object index of each
// streamed mesh, its scoring center, and its decoded per-mesh geometry copy.
// The three vecs are column-aligned.
pub(super) struct MeshStreamData {
    pub(super) draw_indices: Vec<usize>,
    pub(super) centers: Vec<Vec<[f32; 3]>>,
    pub(super) payloads: Vec<DecodedMesh>,
}

// Per-texture-slot draw positions for the streaming scorer, which ranks each
// texture by the camera's distance to the nearest draw that samples it. Albedo
// and normal maps share one pool, so a draw contributes its position to both the
// slot it samples as albedo and the one it samples as a normal map
// (`NO_NORMAL_MAP_SLOT` = no normal map, scored by neither). `texture_count`
// sizes the outer vec so every pool slot has an entry.
pub(super) fn texture_stream_centers(
    draw_objects: &[render_types::DrawObject],
    texture_count: usize,
) -> Vec<Vec<[f32; 3]>> {
    let mut centers = vec![Vec::new(); texture_count];
    for obj in draw_objects {
        let pos = draw_object_position(obj);
        if let Some(slot) = centers.get_mut(obj.texture_slot) {
            slot.push(pos);
        }
        if obj.normal_map_slot != render_types::NO_NORMAL_MAP_SLOT
            && let Some(slot) = centers.get_mut(obj.normal_map_slot)
        {
            slot.push(pos);
        }
    }
    centers
}

// Per-streamed-mesh data captured before `draw_objects` moves into the backend.
// Only static, frustum-cullable draws stream; skybox, rooms, and dynamic props
// (sentinel AABB) stay resident so structural geometry never pops in. Each
// payload copies the draw's region of the shared vertex/index buffers, scored by
// its AABB center; indices are stored mesh-relative and narrowed to u16 (each
// per-mesh region fits in u16 by the build-time splitter). Draws whose
// build-time offsets fall out of range are skipped defensively.
pub(super) fn mesh_stream_data(
    draw_objects: &[render_types::DrawObject],
    all_vertices: &[Vertex],
    all_indices: &[u32],
    deferred_draws: &HashSet<usize>,
) -> MeshStreamData {
    let mut draw_indices: Vec<usize> = Vec::new();
    let mut centers: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut payloads: Vec<DecodedMesh> = Vec::new();
    for (draw_idx, obj) in draw_objects.iter().enumerate() {
        if !obj.cullable() {
            continue;
        }
        // A deferred draw appended no geometry (its record carries baked
        // counts over an empty region): stream it with an empty payload copy;
        // the deferred source decodes the blob payload instead.
        if deferred_draws.contains(&draw_idx) {
            draw_indices.push(draw_idx);
            centers.push(vec![draw_object_position(obj)]);
            payloads.push(DecodedMesh {
                vertices: Vec::new(),
                indices: Vec::new(),
            });
            continue;
        }
        let vstart = obj.vertex_offset / std::mem::size_of::<Vertex>();
        let vend = vstart + obj.vertex_count;
        let iend = obj.index_offset + obj.index_count;
        if vend > all_vertices.len() || iend > all_indices.len() {
            continue;
        }
        draw_indices.push(draw_idx);
        centers.push(vec![draw_object_position(obj)]);
        let vbase = vstart as u32;
        payloads.push(DecodedMesh {
            vertices: all_vertices[vstart..vend].to_vec(),
            indices: all_indices[obj.index_offset..iend]
                .iter()
                .map(|&i| (i - vbase) as u16)
                .collect(),
        });
    }
    MeshStreamData {
        draw_indices,
        centers,
        payloads,
    }
}

// The mesh-source handle each draw object was built from.
pub(super) fn draw_to_handle(
    mesh_handle_to_draws: &HashMap<usize, Vec<usize>>,
) -> HashMap<usize, usize> {
    mesh_handle_to_draws
        .iter()
        .flat_map(|(h, draws)| draws.iter().map(move |&d| (d, *h)))
        .collect()
}

// The draw objects built from a mesh whose payload decode was deferred.
pub(super) fn deferred_draws(
    deferred_mesh_seeds: &HashMap<usize, DeferredMeshSeed>,
    mesh_handle_to_draws: &HashMap<usize, Vec<usize>>,
) -> HashSet<usize> {
    deferred_mesh_seeds
        .keys()
        .filter_map(|h| mesh_handle_to_draws.get(h))
        .flatten()
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::PayloadLocator;
    use concinnity_core::gfx::render_types::{DrawObject, MaterialUniforms, NO_NORMAL_MAP_SLOT};

    // A draw over `[vertex_offset (bytes), +vertex_count]` / `[index_offset,
    // +index_count]` sampling `texture_slot` (+ `normal_map_slot`). A non-cullable
    // draw carries the NaN sentinel AABB, matching the skybox / dynamic path.
    fn draw(
        vertex_offset: usize,
        vertex_count: usize,
        index_offset: usize,
        index_count: usize,
        texture_slot: usize,
        normal_map_slot: usize,
        cullable: bool,
    ) -> DrawObject {
        let (bb_min, bb_max) = if cullable {
            ([0.0; 3], [1.0; 3])
        } else {
            ([f32::NAN; 3], [f32::NAN; 3])
        };
        DrawObject {
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            base_vertex: 0,
            geometry_generation: 0,
            shader_bucket: 0,
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            texture_slot,
            normal_map_slot,
            material: MaterialUniforms::DEFAULT,
            visible: true,
            resident: true,
            bb_min,
            bb_max,
            cull_distance: 0.0,
            lod_alternates: Vec::new(),
        }
    }

    fn vert(x: f32) -> Vertex {
        Vertex {
            pos: [x, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
        }
    }

    fn seed() -> DeferredMeshSeed {
        DeferredMeshSeed {
            locator: PayloadLocator {
                blob_index: 0,
                offset: 0,
                len: 0,
            },
            bytes: None,
        }
    }

    #[test]
    fn texture_stream_centers_scores_albedo_and_normal_slots() {
        // One draw sampling slot 0 as albedo and slot 2 as its normal map.
        let objs = vec![draw(0, 1, 0, 1, 0, 2, true)];
        let centers = texture_stream_centers(&objs, 4);
        assert_eq!(centers.len(), 4);
        assert_eq!(centers[0].len(), 1);
        assert_eq!(centers[2].len(), 1);
        assert!(centers[1].is_empty());
        assert!(centers[3].is_empty());
    }

    #[test]
    fn texture_stream_centers_skips_absent_normal_map() {
        let objs = vec![draw(0, 1, 0, 1, 1, NO_NORMAL_MAP_SLOT, true)];
        let centers = texture_stream_centers(&objs, 2);
        assert_eq!(centers[1].len(), 1);
        assert!(centers[0].is_empty());
    }

    #[test]
    fn mesh_stream_data_includes_cullable_and_narrows_indices_to_u16() {
        let verts: Vec<Vertex> = (0..4).map(|i| vert(i as f32)).collect();
        // Global indices into a mesh whose vertex region starts at vertex 2.
        let indices: Vec<u32> = vec![2, 3, 2];
        // vertex_offset is a BYTE offset; vertex 2 => 2 * size_of::<Vertex>().
        let vbyte = 2 * std::mem::size_of::<Vertex>();
        let objs = vec![draw(vbyte, 2, 0, 3, 0, NO_NORMAL_MAP_SLOT, true)];
        let data = mesh_stream_data(&objs, &verts, &indices, &Default::default());
        assert_eq!(data.draw_indices, vec![0]);
        assert_eq!(data.payloads.len(), 1);
        assert_eq!(data.payloads[0].vertices.len(), 2);
        // Global indices 2,3,2 rebased mesh-relative (minus vbase 2): 0,1,0.
        assert_eq!(data.payloads[0].indices, vec![0u16, 1, 0]);
    }

    #[test]
    fn mesh_stream_data_skips_non_cullable_and_out_of_range() {
        let verts: Vec<Vertex> = (0..2).map(|i| vert(i as f32)).collect();
        let indices: Vec<u32> = vec![0, 1];
        let objs = vec![
            // Non-cullable (NaN AABB): skybox / dynamic, stays resident.
            draw(0, 2, 0, 2, 0, NO_NORMAL_MAP_SLOT, false),
            // Cullable but vertex_count overruns the 2-vertex buffer: skipped.
            draw(0, 5, 0, 2, 0, NO_NORMAL_MAP_SLOT, true),
        ];
        let data = mesh_stream_data(&objs, &verts, &indices, &Default::default());
        assert!(data.draw_indices.is_empty());
        assert!(data.payloads.is_empty());
    }

    #[test]
    fn draw_to_handle_maps_every_draw_of_a_handle_back_to_it() {
        let by_handle = HashMap::from([(7, vec![0, 3, 5]), (2, vec![1])]);
        let inverted = draw_to_handle(&by_handle);
        assert_eq!(inverted.len(), 4);
        for draw in [0, 3, 5] {
            assert_eq!(inverted[&draw], 7);
        }
        assert_eq!(inverted[&1], 2);
    }

    #[test]
    fn deferred_draws_collects_the_draws_of_deferred_handles_only() {
        let by_handle = HashMap::from([(7, vec![0, 3]), (2, vec![1]), (9, vec![4])]);
        let seeds = HashMap::from([(7, seed()), (8, seed())]);
        assert_eq!(deferred_draws(&seeds, &by_handle), HashSet::from([0, 3]));
    }
}
