// Asset streaming across the backend build: the inputs captured from the draw
// list before it moves into the backend, and the pool wiring that consumes them.

use std::collections::HashMap;

use concinnity_core::components::{BlockType, StreamingConfig, VoxelWorld};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{MaterialHandle, PayloadLocator};
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::mesh_seed::MeshSeedRegion;
use concinnity_core::gfx::render_types::{DrawObject, InstancedCluster};

use super::GraphicsSystem;
use super::mesh_seed_compaction::{
    MeshSeedCompaction, compact_streamed_geometry, plan_mesh_seed_bytes,
    strip_streamed_lod_alternates,
};
use super::mesh_stream_inputs::{
    MeshStreamData, deferred_draws, draw_to_handle, mesh_stream_data, texture_stream_centers,
};
use super::stream_sources::{deferred_mesh_payloads, disk_mesh_payload};
use super::streaming::MeshStreamSetup;
use crate::gfx::draw_list::DeferredMeshSeed;
use crate::gfx::material_entry::MaterialEntry;

// The built draw list and the deferral state the streaming plan reads.
pub(super) struct StreamGeometry<'a> {
    pub(super) vertices: &'a mut Vec<Vertex>,
    pub(super) indices: &'a mut Vec<u32>,
    pub(super) draw_objects: &'a mut Vec<DrawObject>,
    pub(super) instanced_clusters: &'a mut Vec<InstancedCluster>,
    pub(super) mesh_handle_to_draws: &'a HashMap<usize, Vec<usize>>,
    pub(super) deferred_mesh_seeds: &'a HashMap<usize, DeferredMeshSeed>,
    // Baked (vertex, index) counts of the deferred mesh sources, by handle.
    pub(super) deferred_mesh_counts: &'a HashMap<u32, (u32, u32)>,
    pub(super) texture_count: usize,
    pub(super) config: Option<&'a StreamingConfig>,
}

// What the streaming pools need once the draw list has moved into the backend.
pub(super) struct StreamPlan {
    pub(super) texture_centers: Vec<Vec<[f32; 3]>>,
    pub(super) mesh: MeshStreamData,
    pub(super) draw_to_handle: HashMap<usize, usize>,
    pub(super) seed_region: Option<MeshSeedRegion>,
}

// Capture the streaming scores and per-mesh geometry, then shrink the seed
// geometry when the residency cap is below the streamed set, so the backend
// build sizes its buffers from the compacted draw list.
pub(super) fn plan_stream_geometry(geometry: StreamGeometry<'_>) -> StreamPlan {
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
    StreamPlan {
        texture_centers,
        mesh,
        draw_to_handle,
        seed_region,
    }
}

// The streaming plan plus the payloads and voxel content the pools stream from.
pub(super) struct StreamingSetup<'a> {
    pub(super) config: Option<StreamingConfig>,
    pub(super) plan: StreamPlan,
    pub(super) texture_payloads: Vec<Vec<u8>>,
    pub(super) texture_locators: &'a [PayloadLocator],
    pub(super) disk_backed: bool,
    pub(super) deferred_mesh_seeds: &'a HashMap<usize, DeferredMeshSeed>,
    pub(super) voxel_world: Option<VoxelWorld>,
    pub(super) block_types: &'a HashMap<AssetId, BlockType>,
    pub(super) material_map: &'a HashMap<MaterialHandle, MaterialEntry>,
}

impl GraphicsSystem {
    // Wire the texture, mesh and voxel-world streaming pools onto the backend.
    pub(super) fn setup_streaming(&mut self, setup: StreamingSetup<'_>) {
        let StreamingSetup {
            config,
            plan,
            texture_payloads,
            texture_locators,
            disk_backed,
            deferred_mesh_seeds,
            voxel_world,
            block_types,
            material_map,
        } = setup;
        self.setup_texture_streaming(
            config.clone(),
            texture_payloads,
            texture_locators,
            disk_backed,
            plan.texture_centers,
        );
        // Per-stream-id payload refs for the deferred meshes, so the worker
        // can decode them from the blob payload when their scene pins.
        let deferred_payloads = deferred_mesh_payloads(
            deferred_mesh_seeds,
            &plan.draw_to_handle,
            &plan.mesh.draw_indices,
            disk_mesh_payload,
        );
        if !deferred_payloads.is_empty() {
            tracing::info!(
                "GraphicsSystem: deferred {} scene-owned mesh payload(s) past init",
                deferred_payloads.len()
            );
        }
        self.setup_mesh_streaming(
            config,
            MeshStreamSetup {
                payloads: plan.mesh.payloads,
                centers: plan.mesh.centers,
                draw_indices: plan.mesh.draw_indices,
                disk_backed,
                seed_region: plan.seed_region,
                deferred_payloads,
            },
        );
        self.setup_voxel_world_streaming(voxel_world, block_types, material_map);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::render_types::{LodSlice, MaterialUniforms, NO_NORMAL_MAP_SLOT};

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
            base_vertex: 0,
            geometry_generation: 0,
            shader_bucket: 0,
            model: concinnity_core::transform::IDENTITY,
            texture_slot: 0,
            normal_map_slot: NO_NORMAL_MAP_SLOT,
            material: MaterialUniforms::DEFAULT,
            visible: true,
            resident: true,
            bb_min: [0.0; 3],
            bb_max: [1.0; 3],
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
    ) -> StreamPlan {
        plan_stream_geometry(StreamGeometry {
            vertices,
            indices,
            draw_objects,
            instanced_clusters: &mut Vec::new(),
            mesh_handle_to_draws: &HashMap::from([(0, vec![0])]),
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
        let plan = plan(&mut vertices, &mut indices, &mut draws, None);
        assert!(plan.seed_region.is_none());
        assert_eq!(plan.texture_centers.len(), 2);
        assert_eq!(plan.mesh.draw_indices, vec![0]);
        assert_eq!(plan.draw_to_handle, HashMap::from([(0, 0)]));
        assert_eq!(draws[0].lod_alternates.len(), 1);
        assert_eq!(vertices.len(), 3);
    }

    // A streamed mesh drops its LOD alternates: the streamer owns its slices.
    #[test]
    fn streaming_strips_the_lod_alternates_of_streamed_meshes() {
        let (mut vertices, mut indices, mut draws) = triangle();
        let config = StreamingConfig::default();
        let plan = plan(&mut vertices, &mut indices, &mut draws, Some(&config));
        assert_eq!(plan.mesh.payloads.len(), 1);
        assert!(draws[0].lod_alternates.is_empty());
    }
}
