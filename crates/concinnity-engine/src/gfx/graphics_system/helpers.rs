// Free-function helpers shared by GraphicsSystem's init / frame / streaming
// code: chunk conversion, camera-relative chunk placement, draw-object
// position extraction, streaming payload sources, and backend construction.

use crate::components::BlockType;
use crate::gfx::mesh_payload::Vertex;

// Resolve a `BlockType` asset into the chunk mesher's palette entry. Per-face
// UV overrides fall back to the uv_min/uv_max rectangle, mirroring the
// build-time `geometry::resolve_block_type`. Used by every backend's
// chunk-streaming setup.
pub(super) fn block_type_to_chunk(bt: &BlockType) -> crate::geometry::ChunkBlockType {
    let default_rect = [bt.uv_min[0], bt.uv_min[1], bt.uv_max[0], bt.uv_max[1]];
    crate::geometry::ChunkBlockType {
        solid: bt.solid,
        uv_top: bt.uv_top.unwrap_or(default_rect),
        uv_bottom: bt.uv_bottom.unwrap_or(default_rect),
        uv_side: bt.uv_side.unwrap_or(default_rect),
    }
}

// World-space position used to score a draw object for texture streaming:
// the AABB centre when bounds are finite, otherwise the model-matrix
// translation (dynamic props carry a non-finite sentinel AABB).
pub(super) fn draw_object_position(obj: &crate::gfx::render_types::DrawObject) -> [f32; 3] {
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
pub(super) const AUTO_SEED_MAX_TRIANGLES: usize = 200_000;

// Gather world-space triangles from the static draw list for reflection-probe auto-seed
// interior detection (surface voxelisation needs real geometry, not AABBs). Returns
// `None` when there is no cullable static geometry or the scene exceeds
// `AUTO_SEED_MAX_TRIANGLES` -- the caller then falls back to coarse AABB occupancy. Each
// cullable draw's indexed triangles are transformed to world space by its model matrix;
// `base_vertex` is honoured so streamed (mesh-relative) chunks resolve too, and every
// fetch is bounds-checked against the shared vertex buffer (build-time offsets should be
// in range, but a bad offset is skipped rather than risking an out-of-bounds index).
pub(super) fn gather_auto_seed_triangles(
    draw_objects: &[crate::gfx::render_types::DrawObject],
    all_vertices: &[Vertex],
    all_indices: &[u32],
) -> Option<Vec<[[f32; 3]; 3]>> {
    let eligible = |o: &crate::gfx::render_types::DrawObject| o.cullable() && o.index_count >= 3;
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

// Build the payload source for a streamed texture pool (albedo or normal-map).
//
// When `disk_backed`, each locator's payload-section offset is turned into an
// absolute file offset (the blob file's payload section starts past its header
// and defs) so the streamer can re-read payloads from disk without a RAM copy.
// Otherwise the retained `payloads` are wrapped RAM-resident. Used by the
// Metal, Vulkan, and DirectX texture-streaming paths.
pub(super) fn build_texture_payload_source(
    payloads: Vec<Vec<u8>>,
    locators: &[crate::ecs::PayloadLocator],
    disk_backed: bool,
) -> Result<std::sync::Arc<dyn crate::gfx::streaming::texture::PayloadSource>, String> {
    if !disk_backed {
        return Ok(std::sync::Arc::new(
            crate::gfx::streaming::texture::MemPayloadSource::new(payloads),
        ));
    }
    let mut section_starts: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut disk_locators = Vec::with_capacity(locators.len());
    for loc in locators {
        let path = crate::blob::blob_path(loc.blob_index);
        let start = match section_starts.get(&loc.blob_index) {
            Some(&s) => s,
            None => {
                let s = crate::blob::payload_section_start(&path)
                    .map_err(|e| format!("blob {}: {:?}", loc.blob_index, e))?;
                section_starts.insert(loc.blob_index, s);
                s
            }
        };
        disk_locators.push(crate::gfx::streaming::texture::DiskTextureLocator {
            path,
            file_offset: start + loc.offset,
            len: loc.len,
        });
    }
    Ok(std::sync::Arc::new(
        crate::gfx::streaming::texture::DiskPayloadSource::new(disk_locators),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::{DrawObject, MaterialUniforms, NO_NORMAL_MAP_SLOT};

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

    // A per-face override wins where set; the remaining faces fall back to the
    // uv_min/uv_max rectangle. Mirrors build-time resolve_block_type.
    #[test]
    fn block_type_to_chunk_overrides_then_falls_back() {
        let bt = BlockType {
            solid: true,
            uv_min: [0.1, 0.2],
            uv_max: [0.6, 0.7],
            uv_top: Some([0.0, 0.0, 0.25, 0.25]),
            uv_bottom: None,
            uv_side: None,
            ..Default::default()
        };
        let chunk = block_type_to_chunk(&bt);
        assert!(chunk.solid);
        assert_eq!(chunk.uv_top, [0.0, 0.0, 0.25, 0.25], "override kept");
        let fallback = [0.1, 0.2, 0.6, 0.7];
        assert_eq!(
            chunk.uv_bottom, fallback,
            "unset face -> uv_min/uv_max rect"
        );
        assert_eq!(chunk.uv_side, fallback);
    }

    #[test]
    fn block_type_to_chunk_air_is_not_solid() {
        let bt = BlockType {
            solid: false,
            ..Default::default()
        };
        assert!(!block_type_to_chunk(&bt).solid);
    }

    // Finite bounds score at the AABB centre.
    #[test]
    fn draw_object_position_uses_aabb_centre_when_finite() {
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
    // model matrix, honouring base_vertex.
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

    // The RAM-resident source decodes a compiled texture payload on fetch.
    #[test]
    fn build_texture_payload_source_mem_backed_decodes_payload() {
        // 1x1 RGBA tagged payload via the shared serialiser.
        let payload = crate::build::texture::serialise(
            &crate::build::texture::TextureImage::rgba8(1, 1, vec![0x11, 0x22, 0x33, 0xFF]),
        );

        let src = build_texture_payload_source(vec![payload], &[], false).expect("mem source");
        let decoded = src.fetch(0).expect("decodes item 0");
        assert_eq!((decoded.image.width(), decoded.image.height()), (1, 1));
        assert_eq!(decoded.image.mips[0].data, vec![0x11, 0x22, 0x33, 0xFF]);
        // Out-of-range item id surfaces an error rather than panicking.
        assert!(src.fetch(1).is_err());
    }
}
