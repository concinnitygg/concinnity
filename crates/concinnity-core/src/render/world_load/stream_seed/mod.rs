//! What the streaming pools take from the built draw list before it moves into
//! the backend: texture scoring centers, per-streamed-mesh geometry copies, the
//! draw-to-mesh-handle map, and (when the residency cap is below the streamed
//! set) the compacted seed geometry the mesh pool uploads into.

use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::components::StreamingConfig;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::mesh_seed::MeshSeedRegion;
use crate::gfx::render_types::{DrawIndex, DrawObject, InstancedCluster};
use crate::render::draw_list::DeferredMeshSeed;

mod compaction;
mod inputs;

pub use inputs::MeshStreamData;

use compaction::{
    MeshSeedCompaction, compact_streamed_geometry, plan_mesh_seed_bytes,
    strip_streamed_lod_alternates,
};
use inputs::{deferred_draws, draw_to_handle, mesh_stream_data, texture_stream_centers};

/// The built draw list and the deferral state the seed is taken from.
pub struct StreamGeometry<'a> {
    /// The shared vertex buffer.
    pub vertices: &'a mut Vec<Vertex>,
    /// The shared index buffer.
    pub indices: &'a mut Vec<u32>,
    /// The draw objects.
    pub draw_objects: &'a mut Vec<DrawObject>,
    /// The instanced clusters.
    pub instanced_clusters: &'a mut Vec<InstancedCluster>,
    /// The draws built from each mesh source, by handle.
    pub mesh_handle_to_draws: &'a HashMap<usize, Vec<DrawIndex>>,
    /// The mesh sources whose decode init deferred, by handle.
    pub deferred_mesh_seeds: &'a HashMap<usize, DeferredMeshSeed>,
    /// Baked (vertex, index) counts of the deferred mesh sources, by handle.
    pub deferred_mesh_counts: &'a HashMap<u32, (u32, u32)>,
    /// Texture slots in the pool.
    pub texture_count: usize,
    /// The world's streaming settings, when it streams.
    pub config: Option<&'a StreamingConfig>,
}

/// What the streaming pools need once the draw list has moved into the
/// backend.
pub struct StreamSeed {
    /// Each texture slot's scoring centers.
    pub texture_centers: Vec<Vec<[f32; 3]>>,
    /// The streamed meshes.
    pub mesh: MeshStreamData,
    /// The mesh source each draw was built from.
    pub draw_to_handle: HashMap<DrawIndex, usize>,
    /// The compacted seed region the mesh pool uploads into, when compaction
    /// ran.
    pub seed_region: Option<MeshSeedRegion>,
}

/// Capture the streaming scores and per-mesh geometry, then shrink the seed
/// geometry when the residency cap is below the streamed set, so the backend
/// build sizes its buffers from the compacted draw list.
pub fn plan_stream_seed(geometry: StreamGeometry<'_>) -> StreamSeed {
    let StreamGeometry {
        vertices,
        indices,
        draw_objects,
        instanced_clusters,
        mesh_handle_to_draws,
        deferred_mesh_seeds,
        deferred_mesh_counts,
        texture_count,
        config,
    } = geometry;
    let texture_centers = texture_stream_centers(draw_objects, texture_count);
    let mesh = mesh_stream_data(
        draw_objects,
        vertices,
        indices,
        &deferred_draws(deferred_mesh_seeds, mesh_handle_to_draws),
    );
    let draw_to_handle = draw_to_handle(mesh_handle_to_draws);
    if config.is_some() && !mesh.payloads.is_empty() {
        strip_streamed_lod_alternates(draw_objects, &mesh.draw_indices);
    }
    let seed_region = match config {
        Some(cfg) if !mesh.payloads.is_empty() => plan_mesh_seed_bytes(
            &mesh.payloads,
            &mesh.draw_indices,
            &draw_to_handle,
            deferred_mesh_counts,
            cfg.mesh_cap(),
            !deferred_mesh_seeds.is_empty(),
        )
        .map(|seed| {
            compact_streamed_geometry(
                MeshSeedCompaction {
                    vertices,
                    indices,
                    draw_objects,
                    instanced_clusters,
                    stream_draw_indices: &mesh.draw_indices,
                },
                seed,
                cfg.mesh_cap(),
            )
        }),
        _ => None,
    };
    StreamSeed {
        texture_centers,
        mesh,
        draw_to_handle,
        seed_region,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::{LodSlice, MaterialUniforms, NO_NORMAL_MAP_SLOT};
    use alloc::vec;

    // One cullable triangle draw over the first three vertices, with a LOD alternate.
    fn triangle() -> (Vec<Vertex>, Vec<u32>, Vec<DrawObject>) {
        let vert = |x: f32| Vertex {
            pos: [x, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0; 3],
            uv: [0.0; 2],
        };
        let draw = DrawObject {
            vertex_offset: 0,
            vertex_count: 3,
            index_offset: 0,
            index_count: 3,
            planet_tile: None,
            base_vertex: 0,
            geometry_generation: 0,
            shader_bucket: 0,
            model: crate::transform::IDENTITY,
            texture_slot: 0,
            normal_map_slot: NO_NORMAL_MAP_SLOT,
            material: MaterialUniforms::DEFAULT,
            visible: true,
            resident: true,
            bb_min: [0.0; 3],
            bb_max: [1.0; 3],
            local_bounds: None,
            cull_distance: 0.0,
            lod_alternates: vec![LodSlice {
                index_offset: 0,
                index_count: 3,
                switch_distance: 10.0,
            }],
        };
        (
            vec![vert(0.0), vert(1.0), vert(2.0)],
            vec![0, 1, 2],
            vec![draw],
        )
    }

    fn plan(
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u32>,
        draw_objects: &mut Vec<DrawObject>,
        config: Option<&StreamingConfig>,
    ) -> StreamSeed {
        plan_stream_seed(StreamGeometry {
            vertices,
            indices,
            draw_objects,
            instanced_clusters: &mut Vec::new(),
            mesh_handle_to_draws: &HashMap::from([(0, vec![DrawIndex(0)])]),
            deferred_mesh_seeds: &HashMap::new(),
            deferred_mesh_counts: &HashMap::new(),
            texture_count: 2,
            config,
        })
    }

    // Without streaming the draw list keeps its LOD alternates and its geometry.
    #[test]
    fn no_streaming_config_leaves_the_draw_list_untouched() {
        let (mut vertices, mut indices, mut draws) = triangle();
        let seed = plan(&mut vertices, &mut indices, &mut draws, None);
        assert!(seed.seed_region.is_none());
        assert_eq!(seed.texture_centers.len(), 2);
        assert_eq!(seed.mesh.draw_indices, vec![DrawIndex(0)]);
        assert_eq!(seed.draw_to_handle, HashMap::from([(DrawIndex(0), 0)]));
        assert_eq!(draws[0].lod_alternates.len(), 1);
        assert_eq!(vertices.len(), 3);
    }

    // A streamed mesh drops its LOD alternates: the streamer owns its slices.
    #[test]
    fn streaming_strips_the_lod_alternates_of_streamed_meshes() {
        let (mut vertices, mut indices, mut draws) = triangle();
        let config = StreamingConfig::default();
        let seed = plan(&mut vertices, &mut indices, &mut draws, Some(&config));
        assert_eq!(seed.mesh.payloads.len(), 1);
        assert!(draws[0].lod_alternates.is_empty());
    }
}
