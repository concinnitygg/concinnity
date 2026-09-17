// Shrinkable seed VRAM (Metal + DirectX + Vulkan). By default `build_draw_list`
// bakes every streamed mesh into the shared vertex/index buffers, sizing them for
// the whole streamed set, so streaming reuses space but never shrinks GPU memory.
// When the residency cap is smaller than the streamed set, init compacts the
// resident geometry and reserves a smaller seed headroom, sized to the cap-many
// largest meshes, that the streamed meshes are placed into on upload. It runs
// before the backend build so the GPU buffers are born small and the RT
// acceleration structure sees the final offsets.

use std::collections::HashMap;

use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::mesh_seed::{self, MeshSeedRegion};
use concinnity_core::gfx::render_types::{DrawObject, InstancedCluster};

use crate::gfx::streaming::mesh::DecodedMesh;

// Mesh streaming and LOD alternates don't yet cooperate: upload_mesh writes only
// LOD0 to its newly-allocated region, but `lod_alternates` still carries the
// build-time offsets for LOD1..N. Once another stream upload reuses those byte
// ranges, active_lod() returns offsets into unrelated geometry and the draw
// renders garbage or nothing. Until upload_mesh streams every LOD, strip the
// alternates from every streamed draw so active_lod() always returns LOD0.
pub(super) fn strip_streamed_lod_alternates(
    draw_objects: &mut [DrawObject],
    stream_draw_indices: &[usize],
) {
    for &draw_idx in stream_draw_indices {
        if let Some(obj) = draw_objects.get_mut(draw_idx) {
            obj.lod_alternates.clear();
        }
    }
}

// The (vertex, index) seed headroom in bytes for the streamed meshes, or None
// when the cap already holds the whole set and nothing is deferred. A deferred
// mesh's payload copy is empty (its decode was skipped), so its size comes from
// the baked `deferred_counts` keyed by mesh-source handle instead.
pub(super) fn plan_mesh_seed_bytes(
    mesh_payloads: &[DecodedMesh],
    stream_draw_indices: &[usize],
    draw_to_handle: &HashMap<usize, usize>,
    deferred_counts: &HashMap<u32, (u32, u32)>,
    mesh_cap: usize,
    has_deferred_seeds: bool,
) -> Option<(u64, u64)> {
    let vertex_bytes = std::mem::size_of::<Vertex>() as u64;
    let index_bytes = std::mem::size_of::<u32>() as u64;
    let sizes: Vec<(u64, u64)> = mesh_payloads
        .iter()
        .zip(stream_draw_indices)
        .map(|(m, draw_idx)| {
            if !m.vertices.is_empty() {
                return (
                    m.vertices.len() as u64 * vertex_bytes,
                    m.indices.len() as u64 * index_bytes,
                );
            }
            draw_to_handle
                .get(draw_idx)
                .and_then(|h| deferred_counts.get(&(*h as u32)))
                .map(|&(vc, ic)| (vc as u64 * vertex_bytes, ic as u64 * index_bytes))
                .unwrap_or((0, 0))
        })
        .collect();
    // Deferred meshes have no baked region for the full-set evict path to free,
    // so force compaction with a whole-set headroom when the cap alone would not
    // shrink.
    mesh_seed::plan_seed_bytes(&sizes, mesh_cap).or_else(|| {
        has_deferred_seeds.then(|| {
            (
                sizes.iter().map(|s| s.0).sum(),
                sizes.iter().map(|s| s.1).sum(),
            )
        })
    })
}

// The shared geometry buffers the compaction rewrites, and the streamed draws
// it leaves out of them.
pub(super) struct MeshSeedCompaction<'a> {
    pub(super) vertices: &'a mut Vec<Vertex>,
    pub(super) indices: &'a mut Vec<u32>,
    pub(super) draw_objects: &'a mut Vec<DrawObject>,
    pub(super) instanced_clusters: &'a mut Vec<InstancedCluster>,
    pub(super) stream_draw_indices: &'a [usize],
}

// Compact the resident geometry and append the planned `seed` headroom,
// returning the region to seed the mesh sub-allocators with. A stream index
// past the draw list is ignored.
pub(super) fn compact_streamed_geometry(
    inputs: MeshSeedCompaction<'_>,
    seed: (u64, u64),
    mesh_cap: usize,
) -> MeshSeedRegion {
    let MeshSeedCompaction {
        vertices,
        indices,
        draw_objects,
        instanced_clusters,
        stream_draw_indices,
    } = inputs;
    let (seed_vtx, seed_idx) = seed;
    let mut streamed = vec![false; draw_objects.len()];
    for &idx in stream_draw_indices {
        if let Some(s) = streamed.get_mut(idx) {
            *s = true;
        }
    }
    let region = mesh_seed::compact_for_streaming(
        vertices,
        indices,
        draw_objects,
        instanced_clusters,
        &streamed,
        seed_vtx,
        seed_idx,
    );
    tracing::info!(
        "GraphicsSystem: shrinkable seed VRAM -- {} streamed mesh(es), cap {}, seed headroom {} KiB vtx + {} KiB idx",
        stream_draw_indices.len(),
        mesh_cap,
        seed_vtx / 1024,
        seed_idx / 1024,
    );
    region
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::render_types::{LodSlice, MaterialUniforms, NO_NORMAL_MAP_SLOT};

    const VERTEX_BYTES: u64 = std::mem::size_of::<Vertex>() as u64;
    const INDEX_BYTES: u64 = std::mem::size_of::<u32>() as u64;

    // A draw over `vertex_count` vertices starting at vertex `first_vertex` and
    // `index_count` indices starting at `index_offset`.
    fn draw(
        first_vertex: usize,
        vertex_count: usize,
        index_offset: usize,
        index_count: usize,
    ) -> DrawObject {
        DrawObject {
            vertex_offset: first_vertex * VERTEX_BYTES as usize,
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
            texture_slot: 0,
            normal_map_slot: NO_NORMAL_MAP_SLOT,
            material: MaterialUniforms::DEFAULT,
            visible: true,
            resident: true,
            bb_min: [0.0; 3],
            bb_max: [1.0; 3],
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

    fn mesh(vertices: usize, indices: usize) -> DecodedMesh {
        DecodedMesh {
            vertices: (0..vertices).map(|i| vert(i as f32)).collect(),
            indices: vec![0; indices],
        }
    }

    #[test]
    fn only_streamed_draws_lose_their_lod_alternates() {
        let mut draws = vec![draw(0, 3, 0, 3), draw(3, 3, 3, 3), draw(6, 3, 6, 3)];
        for d in &mut draws {
            d.lod_alternates.push(LodSlice {
                index_offset: d.index_offset,
                index_count: d.index_count,
                switch_distance: 10.0,
            });
        }
        strip_streamed_lod_alternates(&mut draws, &[1, 9]);
        assert_eq!(draws[0].lod_alternates.len(), 1);
        assert!(draws[1].lod_alternates.is_empty());
        assert_eq!(draws[2].lod_alternates.len(), 1);
    }

    #[test]
    fn a_deferred_mesh_with_an_empty_payload_contributes_its_baked_counts() {
        // Stream 0 is decoded, stream 1 is a deferred draw of mesh handle 5.
        let payloads = [mesh(4, 6), mesh(0, 0)];
        let draw_to_handle = HashMap::from([(0, 2), (1, 5)]);
        let counts = HashMap::from([(5, (10, 30))]);
        let seed = plan_mesh_seed_bytes(&payloads, &[0, 1], &draw_to_handle, &counts, 8, true);
        assert_eq!(
            seed,
            Some((14 * VERTEX_BYTES, 36 * INDEX_BYTES)),
            "a cap covering the set still reserves the whole set when a mesh is deferred"
        );
    }

    #[test]
    fn a_cap_covering_the_set_with_nothing_deferred_plans_no_seed() {
        let payloads = [mesh(4, 6), mesh(3, 3)];
        let draw_to_handle = HashMap::from([(0, 0), (1, 1)]);
        let seed = plan_mesh_seed_bytes(
            &payloads,
            &[0, 1],
            &draw_to_handle,
            &HashMap::new(),
            8,
            false,
        );
        assert_eq!(seed, None);
    }

    #[test]
    fn compaction_ignores_an_out_of_range_stream_index_and_returns_the_seed() {
        // Two three-vertex draws; draw 1 streams, and index 7 names no draw.
        let mut vertices: Vec<Vertex> = (0..6).map(|i| vert(i as f32)).collect();
        let mut indices: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
        let mut draw_objects = vec![draw(0, 3, 0, 3), draw(3, 3, 3, 3)];
        let mut instanced_clusters = Vec::new();
        let seed = (3 * VERTEX_BYTES, 3 * INDEX_BYTES);
        let region = compact_streamed_geometry(
            MeshSeedCompaction {
                vertices: &mut vertices,
                indices: &mut indices,
                draw_objects: &mut draw_objects,
                instanced_clusters: &mut instanced_clusters,
                stream_draw_indices: &[1, 7],
            },
            seed,
            1,
        );
        assert_eq!(
            region,
            MeshSeedRegion {
                vtx_offset: 3 * VERTEX_BYTES,
                vtx_bytes: seed.0,
                idx_offset: 3 * INDEX_BYTES,
                idx_bytes: seed.1,
            }
        );
        assert!(draw_objects[0].resident);
        assert!(!draw_objects[1].resident);
        assert_eq!(
            vertices.len(),
            6,
            "resident geometry plus the seed headroom"
        );
        assert_eq!(indices.len(), 6);
    }
}
