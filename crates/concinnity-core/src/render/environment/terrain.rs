//! Every `Terrain`'s cooked payload, decoded once at graphics init: its grid
//! becomes the terrain's static draws, and with its masks the ground its grass
//! layers grow on and the far field its surface is shaded as past the blades.
//! The components stay in place for the physics system, which inits after
//! graphics and builds each terrain's collider from the same payload.

use crate::math::SaturatingFrom;
use alloc::string::ToString;
use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::components::{Grass, Identity, Terrain};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::frustum;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{DrawObject, FarFieldTint};
use crate::render::environment::grass::GrassTerrain;
use crate::render::environment::grass::tint::{FarField, far_fields};
use crate::terrain::payload::TerrainPayload;
use crate::terrain::{terrain_chunks, terrain_chunks_colored};
use crate::transform::trs_matrix;

use crate::render::material_entry::MaterialEntry;

/// One terrain and what it cooked into.
pub struct LoadedTerrain {
    /// The terrain as authored.
    pub terrain: Terrain,
    /// Its decoded height grid and density masks.
    pub payload: TerrainPayload,
    /// What its surface is shaded as past the blades, when it grows grass.
    pub far_field: Option<FarField>,
}

/// Decode every terrain's payload, plus the blobs they live in for the
/// release step. `None` (with the error logged) when one is missing or
/// malformed, which only a failed build leaves behind.
pub fn load_terrains(ctx: &mut PipelineContext<'_>) -> Option<(Vec<LoadedTerrain>, Vec<u32>)> {
    let terrains: Vec<(Option<AssetId>, Terrain)> = ctx
        .query_with_entity::<Terrain>()
        .map(|(entity, t)| (ctx.get::<Identity>(entity).map(|i| i.id()), t.clone()))
        .collect();
    let mut loaded = Vec::with_capacity(terrains.len());
    let mut blobs = Vec::with_capacity(terrains.len());
    for (id, terrain) in terrains {
        let Some(locator) = terrain.locator.clone() else {
            tracing::error!(
                "graphics: Terrain {id:?} has no compiled payload; did the build succeed?"
            );
            return None;
        };
        blobs.push(locator.blob_index);
        let payload = match ctx
            .read_payload(&locator)
            .map_err(|e| e.to_string())
            .and_then(|bytes| TerrainPayload::decode(bytes, terrain.extent))
        {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("graphics: Terrain {id:?} payload: {e}");
                return None;
            }
        };
        loaded.push(LoadedTerrain {
            terrain,
            payload,
            far_field: None,
        });
    }
    let fields = far_fields(&grass_terrains(&loaded, &grass_by_id(ctx)));
    for (t, far) in loaded.iter_mut().zip(fields) {
        t.far_field = far;
    }
    Some((loaded, blobs))
}

/// Every `Grass` in the world by id, read in place: the effects drain them
/// later.
pub fn grass_by_id(ctx: &PipelineContext<'_>) -> HashMap<AssetId, Grass> {
    ctx.query_with_entity::<Grass>()
        .filter_map(|(entity, g)| Some((ctx.get::<Identity>(entity)?.id(), g.clone())))
        .collect()
}

/// The terrains as the grass resolves them: each layer's `Grass` looked up in
/// `grass` by id. A layer naming no known `Grass` grows nothing.
pub fn grass_terrains(
    terrains: &[LoadedTerrain],
    grass: &HashMap<AssetId, Grass>,
) -> Vec<GrassTerrain> {
    terrains.iter().map(|t| grass_terrain(t, grass)).collect()
}

fn grass_terrain(t: &LoadedTerrain, grass: &HashMap<AssetId, Grass>) -> GrassTerrain {
    GrassTerrain {
        center: t.terrain.center,
        grid: t.payload.grid.clone(),
        layers: t
            .terrain
            .layers
            .iter()
            .zip(t.payload.masks.iter())
            .filter_map(|(layer, mask)| Some((grass.get(&layer.grass.id())?.clone(), mask.clone())))
            .collect(),
    }
}

/// Append `terrain`'s chunks to the shared buffers as static, frustum-culled
/// draws under `material`. A terrain growing grass carries its far-field
/// colors in its vertex colors and the band its surface blends over.
pub fn append_terrain_draws(
    terrain: &LoadedTerrain,
    mut material: MaterialEntry,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    draws: &mut Vec<DrawObject>,
) {
    let model = trs_matrix(terrain.terrain.center, [0.0; 3], [1.0; 3]);
    let grid = &terrain.payload.grid;
    let chunks = match &terrain.far_field {
        Some(far) => {
            material.uniforms.far_field = FarFieldTint::of(far.canopy.packed());
            terrain_chunks_colored(grid, &far.shares)
        }
        None => terrain_chunks(grid),
    };
    for chunk in chunks {
        let vertex_offset = vertices.len() * size_of::<Vertex>();
        let index_offset = indices.len();
        let base = u32::saturating_from(vertices.len());
        vertices.extend_from_slice(&chunk.vertices);
        indices.extend(chunk.indices.iter().map(|&i| u32::from(i) + base));
        let (bb_min, bb_max) = frustum::transform_aabb(chunk.bounds.0, chunk.bounds.1, model);
        draws.push(DrawObject {
            vertex_offset,
            vertex_count: chunk.vertices.len(),
            index_offset,
            index_count: chunk.indices.len(),
            planet_tile: None,
            base_vertex: 0,
            geometry_generation: 0,
            model,
            texture_slot: material.albedo_slot,
            normal_map_slot: material.normal_map_slot,
            material: material.uniforms,
            shader_bucket: material.shader_bucket,
            visible: true,
            resident: true,
            bb_min,
            bb_max,
            local_bounds: Some(chunk.bounds),
            cull_distance: 0.0,
            lod_alternates: Vec::new(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::TerrainLayer;
    use crate::ecs::Ref;
    use crate::render::environment::grass::CanopyRange;
    use crate::terrain::{DensityMask, TERRAIN_CHUNK_CELLS, TerrainGrid};
    use alloc::vec;

    fn loaded(resolution: u32, layers: Vec<TerrainLayer>) -> LoadedTerrain {
        let side = resolution as usize + 1;
        let heights = (0..side * side).map(|i| (i % 5) as f32 * 0.1).collect();
        let masks = layers
            .iter()
            .enumerate()
            .map(|(i, _)| (i == 1).then(|| DensityMask::new(1, 1, vec![128]).unwrap()))
            .collect();
        LoadedTerrain {
            terrain: Terrain {
                center: [3.0, 1.0, -2.0],
                extent: [20.0, 10.0],
                resolution,
                layers,
                ..Terrain::default()
            },
            payload: TerrainPayload {
                grid: TerrainGrid::new(resolution, [20.0, 10.0], heights).unwrap(),
                masks,
            },
            far_field: None,
        }
    }

    // Every chunk is one static, culled draw over its own slice of the shared
    // buffers, placed at the terrain's center: a terrain always contributes
    // draw records, which is what keeps the GPU-driven pass its grass draws
    // on running.
    #[test]
    fn a_terrain_appends_one_culled_draw_per_chunk() {
        let t = loaded(u32::saturating_from(TERRAIN_CHUNK_CELLS) + 2, Vec::new());
        let (mut vertices, mut indices, mut draws) = (Vec::new(), Vec::new(), Vec::new());
        vertices.push(Vertex {
            pos: [0.0; 3],
            normal: [0.0; 3],
            tangent: [0.0; 3],
            color: [0.0; 3],
            uv: [0.0; 2],
        });
        append_terrain_draws(
            &t,
            MaterialEntry::UNTEXTURED,
            &mut vertices,
            &mut indices,
            &mut draws,
        );
        assert_eq!(draws.len(), 4);
        for d in &draws {
            assert!(d.cullable(), "terrain chunks are frustum-culled");
            assert_eq!(d.model[3][..3], [3.0, 1.0, -2.0]);
            let first = d.vertex_offset / size_of::<Vertex>();
            let slice = &indices[d.index_offset..d.index_offset + d.index_count];
            assert!(slice.iter().all(|&i| (i as usize) >= first));
            assert!(slice.iter().all(|&i| (i as usize) < first + d.vertex_count));
            assert!(d.bb_min[0] >= 3.0 - 20.0 && d.bb_max[0] <= 3.0 + 20.0);
        }
        let cells = (TERRAIN_CHUNK_CELLS + 2) * (TERRAIN_CHUNK_CELLS + 2);
        assert_eq!(indices.len(), cells * 6);
        // No grass: the vertex colors stay a white multiplier, with no band.
        assert!(vertices[1..].iter().all(|v| v.color == [1.0; 3]));
        assert!(
            draws
                .iter()
                .all(|d| d.material.far_field == FarFieldTint::NONE)
        );
    }

    // A terrain under grass carries its layers' densities per corner, and the
    // blade fade's band and its canopy run in every chunk's material.
    #[test]
    fn a_grassy_terrain_carries_its_far_field() {
        let mut t = loaded(4, Vec::new());
        let side = t.payload.grid.side();
        let colors: Vec<[f32; 3]> = (0..side * side)
            .map(|i| [0.01 * i as f32, 0.1, 0.0])
            .collect();
        let canopy = CanopyRange {
            first: 2,
            count: 1,
            planet: false,
        };
        t.far_field = Some(FarField {
            shares: colors.clone(),
            canopy,
        });
        let (mut vertices, mut indices, mut draws) = (Vec::new(), Vec::new(), Vec::new());
        append_terrain_draws(
            &t,
            MaterialEntry::UNTEXTURED,
            &mut vertices,
            &mut indices,
            &mut draws,
        );
        for d in &draws {
            assert_eq!(d.material.far_field, FarFieldTint::of(canopy.packed()));
            assert!(d.material.far_field.grows());
        }
        assert_eq!(vertices.len(), colors.len());
        for (v, c) in vertices.iter().zip(&colors) {
            assert_eq!(v.color, *c);
        }
    }

    #[test]
    fn layers_take_their_grass_and_mask() {
        let meadow = AssetId(4);
        let layers = vec![
            TerrainLayer {
                grass: Ref::new(meadow),
                density_mask: None,
            },
            TerrainLayer {
                grass: Ref::new(meadow),
                density_mask: None,
            },
            TerrainLayer {
                grass: Ref::new(AssetId(99)),
                density_mask: None,
            },
        ];
        let t = loaded(4, layers);
        let grass = HashMap::from([(
            meadow,
            Grass {
                height: 0.9,
                ..Grass::default()
            },
        )]);
        let out = grass_terrains(&[t], &grass);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].center, [3.0, 1.0, -2.0]);
        assert_eq!(out[0].layers.len(), 2, "the unknown grass grows nothing");
        assert_eq!(out[0].layers[0].0.height, 0.9);
        assert!(out[0].layers[0].1.is_none());
        assert!(out[0].layers[1].1.is_some());
    }
}
