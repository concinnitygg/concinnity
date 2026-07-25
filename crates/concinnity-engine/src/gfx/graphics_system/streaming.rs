// GraphicsSystem asset-streaming setup: wires the texture, normal-map, mesh,
// and voxel-world streaming pools onto the backend, plus the stats accessor.

use crate::assets::{BlockType, StreamingConfig, VoxelWorld};
use crate::ecs::asset_id::AssetId;
use crate::gfx::draw_list::MaterialEntry;
use crate::gfx::mesh_payload::Vertex;

use super::helpers::*;
use super::*;

// Default fractions of the GPU's reported memory each streaming pool may hold
// resident when a world sets no explicit byte budget. The remainder is left for
// render targets, the chunk pool, and slack.
const TEXTURE_VRAM_FRACTION: f64 = 0.50;
const MESH_VRAM_FRACTION: f64 = 0.30;
// Infinite voxel-world chunk residency. Chunk streaming is not
// StreamingConfig-gated, so there is no user override: the budget is always
// derived from the GPU's reported memory, keeping this pool schema-free.
const CHUNK_VRAM_FRACTION: f64 = 0.15;

// Resolve a streaming pool's resident-byte budget. A non-zero `override_mb`
// wins, converted from mebibytes. Otherwise the budget is `fraction` of the
// GPU's reported memory; a GPU that reports 0 (no figure available) yields
// `None`, leaving the pool on its count-only residency policy.
fn derive_byte_budget(override_mb: u32, gpu_memory_bytes: u64, fraction: f64) -> Option<u64> {
    if override_mb != 0 {
        return Some(override_mb as u64 * 1024 * 1024);
    }
    if gpu_memory_bytes == 0 {
        return None;
    }
    Some((gpu_memory_bytes as f64 * fraction) as u64)
}

// Worst-case resident chunk count for a streaming VoxelWorld: the bound the
// GPU-cull buffers reserve at init so every resident chunk gets a `GpuObjectData`
// record (chunks fold into the indirect path). The streamer RETAINS a
// chunk until its Chebyshev distance exceeds the evict radius =
// `far_radius + EVICT_HYSTERESIS(=2)` (gfx::chunk_window), where
// `far_radius = impostor_radius()` -- which `VoxelWorld::impostor_radius()` floors
// at `view_radius()`, so this evict-window span is correct whether or not impostors
// are enabled (with impostors off, `far_radius == view_radius`). Peak residency =
// `(2*(far_radius+2)+1)^2`, the SAME `total_chunks` `setup_voxel_world_streaming`
// sizes the geometry headroom for. Capped so a typo radius cannot demand gigabytes
// of record memory (the geometry headroom is the real residency limit anyway).
pub(super) fn chunk_reserve_count(vw: &VoxelWorld) -> usize {
    const MAX_CHUNK_RECORDS: u64 = 65536;
    let far_radius = vw.impostor_radius() as u64;
    let evict_span = 2 * (far_radius + 2) + 1;
    (evict_span * evict_span).min(MAX_CHUNK_RECORDS) as usize
}

// Texture slots whose payloads init skips: exclusively owned by a scene other
// than the start scene, gated behind streaming (the streamer is the runtime
// load path that brings them in when their scene pins). Disk-backed, a skipped
// payload -- and any scene blob holding only skipped payloads -- is never read
// at init. Empty when streaming is off or the world declares no scenes.
pub(super) fn deferred_texture_slots(
    ctx: &crate::ecs::PipelineContext,
    streaming: bool,
    slot_count: usize,
) -> std::collections::HashSet<usize> {
    let mut deferred = std::collections::HashSet::new();
    if !streaming {
        return deferred;
    }
    // Scenes are still undrained at this point; the first declared is the
    // start scene (the one setup_scene_flow pins).
    let Some(start) = ctx
        .query::<crate::assets::Scene>()
        .next()
        .map(|s| s.asset_id)
    else {
        return deferred;
    };
    let Some(groups) = ctx.resource::<crate::ecs::BlobSceneGroups>() else {
        return deferred;
    };
    let texture_kind = concinnity_core::ecs::ResourceKind::Texture as u8;
    for group in &groups.0 {
        if group.scene == start {
            continue;
        }
        for &(kind, handle) in &group.resources {
            if kind == texture_kind && (handle as usize) < slot_count {
                deferred.insert(handle as usize);
            }
        }
    }
    deferred
}

// Mesh sources whose payload decode init defers: exclusively owned by a scene
// other than the start scene, gated behind streaming like the texture path.
// Members without a baked bounds record fall out (they decode eagerly).
pub(super) fn deferred_mesh_sources(
    ctx: &crate::ecs::PipelineContext,
    streaming: bool,
) -> crate::gfx::draw_list::DeferredMeshSources {
    let mut out = crate::gfx::draw_list::DeferredMeshSources::default();
    if !streaming {
        return out;
    }
    let Some(start) = ctx
        .query::<crate::assets::Scene>()
        .next()
        .map(|s| s.asset_id)
    else {
        return out;
    };
    let Some(groups) = ctx.resource::<crate::ecs::BlobSceneGroups>() else {
        return out;
    };
    let Some(baked) = ctx.resource::<crate::ecs::BlobMeshBounds>() else {
        return out;
    };
    for record in &baked.0 {
        out.bounds.insert(record.handle, (record.min, record.max));
        out.counts
            .insert(record.handle, (record.vertex_count, record.index_count));
    }
    let mesh_kind = concinnity_core::ecs::ResourceKind::Mesh as u8;
    for group in &groups.0 {
        if group.scene == start {
            continue;
        }
        for &(kind, handle) in &group.resources {
            if kind == mesh_kind {
                out.by_handle.insert(handle);
            }
        }
        for &def in &group.defs {
            out.by_def.insert(def);
        }
    }
    out
}

// The mesh-streaming inputs init assembles: per-stream geometry copies,
// scoring centers, draw mapping, the seed headroom, and the deferred payload
// refs the worker decodes on demand.
pub(super) struct MeshStreamSetup {
    pub payloads: Vec<crate::gfx::streaming::mesh::DecodedMesh>,
    pub centers: Vec<Vec<[f32; 3]>>,
    pub draw_indices: Vec<usize>,
    pub disk_backed: bool,
    pub seed_region: Option<crate::gfx::mesh_seed::MeshSeedRegion>,
    pub deferred_payloads:
        std::collections::HashMap<usize, crate::gfx::streaming::mesh::DeferredMeshPayload>,
}

impl GraphicsSystem {
    // Build scene residency over the streaming pools: texture members come
    // from the blob's baked per-scene groups (a streamed texture's slot is its
    // resource handle), mesh members from each streamed draw's SceneMember
    // entity. Every owned member starts blocked; pinning the start scene
    // unblocks its set, so only it and the global set stream.
    pub(super) fn build_scene_residency(
        &mut self,
        ctx: &crate::ecs::PipelineContext,
    ) -> Option<crate::gfx::scene_residency::SceneResidency> {
        use crate::gfx::scene_residency::{CHANNEL_MESH, CHANNEL_TEXTURE, SceneResidency};

        let (scenes, current) = {
            let flow = self.scene_flow.as_ref()?;
            (flow.scenes.clone(), flow.current)
        };
        if self.texture_streamer.is_none() && self.mesh_streamer.is_none() {
            return None;
        }
        let scene_idx: std::collections::HashMap<AssetId, usize> =
            scenes.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let mut members: Vec<Vec<(u8, u32)>> = vec![Vec::new(); scenes.len()];

        if let Some(streamer) = &self.texture_streamer
            && let Some(groups) = ctx.resource::<crate::ecs::BlobSceneGroups>()
        {
            let texture_kind = concinnity_core::ecs::ResourceKind::Texture as u8;
            for group in &groups.0 {
                let Some(&idx) = scene_idx.get(&group.scene) else {
                    continue;
                };
                for &(kind, handle) in &group.resources {
                    if kind == texture_kind && (handle as usize) < streamer.len() {
                        members[idx].push((CHANNEL_TEXTURE, handle));
                    }
                }
            }
        }

        if self.mesh_streamer.is_some() {
            let mut scene_of_draw = std::collections::HashMap::new();
            for (_, member, handle) in
                ctx.join2::<crate::assets::SceneMember, crate::assets::RenderHandle>()
            {
                for &slot in &handle.draws {
                    scene_of_draw.insert(slot as usize, member.0);
                }
            }
            for (stream_id, draw_idx) in self.mesh_stream_draw_indices.iter().enumerate() {
                if let Some(scene) = scene_of_draw.get(draw_idx)
                    && let Some(&idx) = scene_idx.get(scene)
                {
                    members[idx].push((CHANNEL_MESH, stream_id as u32));
                }
            }
        }

        let mut residency = SceneResidency::new(scenes.iter().copied().zip(members).collect());
        let blocked: Vec<(u8, u32)> = residency.all_members().collect();
        let unblocked = residency.sync_pins(&[current]).unblocked;
        for &(channel, id) in &blocked {
            self.set_stream_blocked(channel, id, true);
        }
        for &(channel, id) in &unblocked {
            self.set_stream_blocked(channel, id, false);
        }
        tracing::info!(
            "GraphicsSystem: scene residency enabled ({} scenes, {} owned members, start scene {})",
            scenes.len(),
            blocked.len(),
            current,
        );
        Some(residency)
    }

    fn set_stream_blocked(&mut self, channel: u8, id: u32, blocked: bool) {
        use crate::gfx::scene_residency::{CHANNEL_MESH, CHANNEL_TEXTURE};
        match channel {
            CHANNEL_TEXTURE => {
                if let Some(s) = &mut self.texture_streamer {
                    s.set_blocked(id as usize, blocked);
                }
            }
            CHANNEL_MESH => {
                if let Some(s) = &mut self.mesh_streamer {
                    s.set_blocked(id as usize, blocked);
                }
            }
            _ => {}
        }
    }

    pub(super) fn setup_texture_streaming(
        &mut self,
        config: Option<StreamingConfig>,
        texture_payloads: Vec<Vec<u8>>,
        texture_locators: &[crate::ecs::PayloadLocator],
        disk_backed: bool,
        texture_centers: Vec<Vec<[f32; 3]>>,
    ) {
        let Some(config) = config else { return };
        // When disk-backed the payloads were not retained, so the streamed
        // slot count comes from the locators instead.
        let slot_count = if disk_backed {
            texture_locators.len()
        } else {
            texture_payloads.len()
        };
        if slot_count == 0 {
            return;
        }
        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        // The GPU's reported memory (discrete VRAM, or the unified-memory
        // working set on Apple); 0 when the driver cannot report it.
        let gpu_memory_bytes = backend.gpu_profile().memory_budget_bytes;
        // Each backend's update_texture_slot rewrites whichever descriptors,
        // argument-buffers, or per-cluster SRVs sample that slot.
        for slot in 0..slot_count {
            if let Err(e) = backend.evict_texture_slot(slot) {
                tracing::warn!("GraphicsSystem: texture evict slot {}: {}", slot, e);
            }
        }
        let source =
            match build_texture_payload_source(texture_payloads, texture_locators, disk_backed) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("GraphicsSystem: texture streaming source: {}", e);
                    return;
                }
            };
        let mut streamer = crate::gfx::streaming::texture::TextureStreamer::new(
            source,
            texture_centers,
            config.budget(),
            config.cap(),
        );
        let byte_budget = derive_byte_budget(
            config.texture_budget_mb,
            gpu_memory_bytes,
            TEXTURE_VRAM_FRACTION,
        );
        streamer.set_byte_budget(byte_budget);
        tracing::info!(
            "GraphicsSystem: texture streaming enabled ({} textures, {} source, budget {}/frame, cap {}, byte budget {})",
            streamer.len(),
            if disk_backed { "disk" } else { "ram" },
            config.budget(),
            config.cap(),
            byte_budget.map_or_else(
                || "count-only".to_string(),
                |b| format!("{} MB", b / (1024 * 1024))
            ),
        );
        self.texture_streamer = Some(streamer);
    }

    // Stand up the mesh-geometry streaming subsystem when a StreamingConfig
    // was declared. Every streamed draw's geometry region is zeroed now (via
    // evict_mesh); the streamer brings them back resident over the next
    // frames, nearest first.
    //
    // The payload source depends on where the world came from: a disk-backed
    // `cn run` world writes the streamed geometry to a scratch file and
    // re-reads it from there (no persistent RAM copy), an in-memory `cn debug`
    // world keeps the geometry RAM-resident.
    //
    // The args are consumed unconditionally so they never warn as unused on a
    // backend that does not yet stream.
    pub(super) fn setup_mesh_streaming(
        &mut self,
        config: Option<StreamingConfig>,
        setup: MeshStreamSetup,
    ) {
        let MeshStreamSetup {
            payloads: mesh_payloads,
            centers: mesh_centers,
            draw_indices: mesh_draw_indices,
            disk_backed,
            seed_region,
            deferred_payloads,
        } = setup;
        let Some(config) = config else { return };
        if mesh_payloads.is_empty() {
            return;
        }
        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        // The GPU's reported memory (discrete VRAM, or the unified-memory
        // working set on Apple); 0 when the driver cannot report it.
        let gpu_memory_bytes = backend.gpu_profile().memory_budget_bytes;
        // Init residency. Two paths:
        //  - Shrinkable seed (`seed_region` present): the streamed geometry was
        //    never baked into the buffers -- compaction already marked each
        //    streamed draw non-resident and reserved one headroom block -- so
        //    seed the sub-allocators with that block. Calling `evict_mesh` here
        //    would zero/free the placeholder offset-0 region and corrupt the
        //    first resident draw, so it must be skipped on this path.
        //  - Full-set seed (`seed_region` absent: a backend without the
        //    shrinkable seed, or no shrink possible): free each streamed mesh's
        //    build-time region into the sub-allocators. retire_frame 0 -- nothing
        //    has been drawn, so the space is reusable immediately.
        match seed_region {
            Some(r) => {
                backend.seed_mesh_streaming(r.vtx_offset, r.vtx_bytes, r.idx_offset, r.idx_bytes);
            }
            None => {
                for &draw_idx in &mesh_draw_indices {
                    if let Err(e) = backend.evict_mesh(draw_idx, 0) {
                        tracing::warn!("GraphicsSystem: mesh evict draw {}: {}", draw_idx, e);
                    }
                }
            }
        }
        // A disk-backed world spills the geometry to a scratch file so the
        // `mesh_payloads` RAM copy can be dropped; `cn debug` keeps it
        // resident since it has no disk artifacts to re-read.
        let source: std::sync::Arc<dyn crate::gfx::streaming::mesh::MeshPayloadSource> =
            if disk_backed {
                let path = crate::gfx::streaming::mesh::default_scratch_path();
                match crate::gfx::streaming::mesh::write_mesh_scratch(path, &mesh_payloads) {
                    Ok(s) => std::sync::Arc::new(s),
                    Err(e) => {
                        tracing::error!("GraphicsSystem: mesh streaming scratch file: {}", e);
                        return;
                    }
                }
            } else {
                std::sync::Arc::new(crate::gfx::streaming::mesh::MemMeshSource::new(
                    mesh_payloads,
                ))
            };
        // Deferred meshes have no geometry copy in the base source; their
        // fetch decodes the blob payload instead.
        let source: std::sync::Arc<dyn crate::gfx::streaming::mesh::MeshPayloadSource> =
            if deferred_payloads.is_empty() {
                source
            } else {
                std::sync::Arc::new(crate::gfx::streaming::mesh::SceneDeferredMeshSource::new(
                    source,
                    deferred_payloads,
                ))
            };
        let mut streamer = crate::gfx::streaming::mesh::MeshStreamer::new(
            source,
            mesh_centers,
            config.mesh_budget(),
            config.mesh_cap(),
        );
        let byte_budget =
            derive_byte_budget(config.mesh_budget_mb, gpu_memory_bytes, MESH_VRAM_FRACTION);
        streamer.set_byte_budget(byte_budget);
        tracing::info!(
            "GraphicsSystem: mesh streaming enabled ({} meshes, {} source, budget {}/frame, cap {}, byte budget {})",
            streamer.len(),
            if disk_backed { "disk" } else { "ram" },
            config.mesh_budget(),
            config.mesh_cap(),
            byte_budget.map_or_else(
                || "count-only".to_string(),
                |b| format!("{} MB", b / (1024 * 1024))
            ),
        );
        self.mesh_streamer = Some(streamer);
        self.mesh_stream_draw_indices = mesh_draw_indices;
    }

    // Stand up the infinite-world chunk-streaming subsystem when a VoxelWorld
    // was declared. Resolves the block palette and shared material, grows the
    // GPU buffers by a chunk-headroom region, and builds the ChunkStreamer;
    // `step` then generates and uploads chunks around the camera each frame.
    // The buffer-growth + SRV/descriptor setup differs per backend (the
    // `setup_chunk_streaming` match below); the palette/material resolution,
    // headroom sizing, and streamer build are backend-agnostic.
    pub(super) fn setup_voxel_world_streaming(
        &mut self,
        voxel_world: Option<VoxelWorld>,
        block_types: &std::collections::HashMap<AssetId, BlockType>,
        material_map: &std::collections::HashMap<crate::ecs::MaterialHandle, MaterialEntry>,
    ) {
        let Some(vw) = voxel_world else { return };

        // Resolve the palette: each id is a BlockType; index 0 is air. A
        // missing entry degrades to air rather than failing the world.
        let palette: Vec<crate::geometry::ChunkBlockType> = vw
            .palette
            .iter()
            .map(|id| match block_types.get(id) {
                Some(bt) => block_type_to_chunk(bt),
                None => {
                    tracing::warn!(
                        "GraphicsSystem: VoxelWorld palette entry {} is not a known BlockType",
                        id
                    );
                    crate::geometry::ChunkBlockType {
                        solid: false,
                        uv_top: [0.0; 4],
                        uv_bottom: [0.0; 4],
                        uv_side: [0.0; 4],
                    }
                }
            })
            .collect();

        // Resolve the shared material to texture-pool slots + scalars.
        let (texture_slot, normal_map_slot, material) = vw
            .material
            .and_then(|id| material_map.get(&id).copied())
            .unwrap_or((
                0,
                crate::gfx::render_types::NO_NORMAL_MAP_SLOT,
                crate::gfx::render_types::MaterialUniforms::DEFAULT,
            ));

        let chunk_blocks = vw.chunk_blocks();
        let block_size = vw.block_size();
        let (chunk_w, chunk_d) = vw.chunk_world_size();
        let near_radius = vw.view_radius();
        let far_radius = vw.impostor_radius();
        let impostor_step = vw.impostor_step();

        // Size the chunk buffer headroom for the worst-case resident window.
        // The near band (full voxel meshes) reaches one ring past `near_radius`
        // (the detail-hysteresis transient where a receding chunk is still
        // full); the far band fills the rest of the evict window with cheap
        // impostors. Sizing the two bands separately keeps the impostor radius
        // from demanding full-chunk headroom for hundreds of distant chunks.
        let near_span = 2 * (near_radius as u64 + 1) + 1;
        let near_chunks = near_span * near_span;
        let evict_span = 2 * (far_radius as u64 + 2) + 1;
        let total_chunks = evict_span * evict_span;
        let far_chunks = total_chunks.saturating_sub(near_chunks);

        // Full-detail per-chunk budget: generous face count for rolling terrain;
        // an over-budget chunk fails its add and is logged rather than
        // corrupting GPU memory.
        let faces_per_chunk = (chunk_blocks[0] as u64) * (chunk_blocks[2] as u64) * 4;
        let full_vtx =
            (faces_per_chunk * 4).min(u16::MAX as u64) * std::mem::size_of::<Vertex>() as u64;
        // Shared index buffer is u32-typed; per-mesh indices are widened on
        // upload, so size the chunk headroom for u32 elements.
        let full_idx = faces_per_chunk * 6 * std::mem::size_of::<u32>() as u64;

        // Impostor per-chunk budget: one quad per coarse cell + a perimeter
        // skirt, 4 verts / 6 indices per quad (matches `build_chunk_impostor_mesh`).
        let nx = (chunk_blocks[0] as u64).div_ceil(impostor_step as u64);
        let nz = (chunk_blocks[2] as u64).div_ceil(impostor_step as u64);
        let impostor_quads = nx * nz + 2 * (nx + nz);
        let impostor_vtx = impostor_quads * 4 * std::mem::size_of::<Vertex>() as u64;
        let impostor_idx = impostor_quads * 6 * std::mem::size_of::<u32>() as u64;

        // Cap total headroom so a typo in the radii cannot demand gigabytes of
        // GPU memory.
        const MAX_HEADROOM: u64 = 512 * 1024 * 1024;
        let chunk_vtx_bytes =
            (near_chunks * full_vtx + far_chunks * impostor_vtx).min(MAX_HEADROOM) as usize;
        let chunk_idx_bytes =
            (near_chunks * full_idx + far_chunks * impostor_idx).min(MAX_HEADROOM) as usize;

        // The GPU's reported memory (discrete VRAM, or the unified-memory
        // working set on Apple); 0 when the driver cannot report it.
        let gpu_memory_bytes = self
            .backend
            .as_deref()
            .map(|b| b.gpu_profile().memory_budget_bytes)
            .unwrap_or(0);

        // Backend-specific buffer growth + SRV/descriptor setup. Metal binds
        // chunk textures per draw and ignores the slot args (its impl drops
        // them); DirectX and Vulkan bake one shared (albedo, normal)
        // descriptor from the chunk material.
        let setup_result = match self.backend.as_deref_mut() {
            Some(backend) => backend.setup_chunk_streaming(
                chunk_vtx_bytes,
                chunk_idx_bytes,
                texture_slot,
                normal_map_slot,
            ),
            None => return,
        };
        if let Err(e) = setup_result {
            tracing::error!("GraphicsSystem: VoxelWorld chunk streaming: {}", e);
            return;
        }

        let source = std::sync::Arc::new(crate::gfx::streaming::chunk::ProceduralChunkSource::new(
            vw.seed,
            chunk_blocks,
            block_size,
            palette,
            impostor_step,
        ));
        let mut streamer = crate::gfx::streaming::chunk::ChunkStreamer::new(
            source,
            near_radius,
            far_radius,
            vw.load_budget(),
            chunk_w,
            chunk_d,
        );
        // Chunk streaming has no user override (it is not StreamingConfig-gated),
        // so the byte budget is always the VRAM-derived fraction; a GPU that
        // reports nothing leaves the pool on its pure radius-only policy.
        let byte_budget = derive_byte_budget(0, gpu_memory_bytes, CHUNK_VRAM_FRACTION);
        streamer.set_byte_budget(byte_budget);
        tracing::info!(
            "GraphicsSystem: VoxelWorld streaming enabled (seed {}, {}x{}x{} blocks, near-radius {}, impostor-radius {} (step {}), budget {}/frame, byte budget {}, {} KiB chunk headroom)",
            vw.seed,
            chunk_blocks[0],
            chunk_blocks[1],
            chunk_blocks[2],
            near_radius,
            if vw.impostors_enabled() {
                far_radius
            } else {
                0
            },
            impostor_step,
            vw.load_budget(),
            byte_budget.map_or_else(
                || "count-only".to_string(),
                |b| format!("{} MB", b / (1024 * 1024))
            ),
            (chunk_vtx_bytes + chunk_idx_bytes) / 1024,
        );
        self.chunk_stream = Some(crate::gfx::streaming_system::ChunkStreamState {
            streamer,
            draws: std::collections::BTreeMap::new(),
            chunk_w,
            chunk_d,
            // Seeded at the world origin; the first `step` rebases onto the
            // camera's actual chunk before any chunk is resident, so the
            // seed value never places geometry.
            origin_chunk: crate::gfx::chunk_coord::ChunkCoord::new(0, 0),
            texture_slot,
            normal_map_slot,
            material,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::gfx::mock_backend::{Call, MockState, recording_backend};
    use crate::gfx::render_types::{MaterialUniforms, NO_NORMAL_MAP_SLOT};

    const MIB: u64 = 1024 * 1024;

    // Owns the storage a PipelineContext borrows from, for the boot-set
    // deferral tests: the declared Scenes plus the blob's baked scene groups
    // and mesh-bounds records.
    struct ResidencyWorld {
        components: crate::ecs::ComponentStorage,
        blob: crate::blob::BlobData,
        profile: crate::gfx::profile::FrameProfile,
        resources: crate::ecs::Resources,
    }

    impl ResidencyWorld {
        // `scenes` are pushed in declaration order, so the first is the start
        // scene the deferral spares.
        fn new(scenes: &[AssetId]) -> Self {
            let mut components = crate::ecs::ComponentStorage::default();
            for &asset_id in scenes {
                components.push_typed(crate::assets::Scene {
                    asset_id,
                    camera_shot: None,
                });
            }
            Self {
                components,
                blob: crate::blob::BlobData::empty(),
                profile: Default::default(),
                resources: crate::ecs::Resources::new(),
            }
        }

        fn with_groups(mut self, groups: Vec<crate::ecs::SceneGroup>) -> Self {
            self.resources.insert(crate::ecs::BlobSceneGroups(groups));
            self
        }

        fn with_mesh_bounds(mut self, records: Vec<crate::ecs::MeshBoundsRecord>) -> Self {
            self.resources.insert(crate::ecs::BlobMeshBounds(records));
            self
        }

        fn ctx(&mut self) -> crate::ecs::PipelineContext<'_> {
            crate::ecs::PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
            }
        }
    }

    fn group(
        scene: AssetId,
        resources: Vec<(u8, u32)>,
        defs: Vec<AssetId>,
    ) -> crate::ecs::SceneGroup {
        crate::ecs::SceneGroup {
            scene,
            resources,
            defs,
        }
    }

    fn bounds_record(
        handle: u32,
        vertex_count: u32,
        index_count: u32,
    ) -> crate::ecs::MeshBoundsRecord {
        crate::ecs::MeshBoundsRecord {
            handle,
            min: [-1.0; 3],
            max: [1.0; 3],
            vertex_count,
            index_count,
        }
    }

    const TEXTURE_KIND: u8 = concinnity_core::ecs::ResourceKind::Texture as u8;
    const MESH_KIND: u8 = concinnity_core::ecs::ResourceKind::Mesh as u8;

    const START: AssetId = AssetId(10);
    const LATER: AssetId = AssetId(11);

    // Only a non-start scene's exclusively-owned texture payloads are skipped at
    // init: the start scene renders on the first frame, so its slots must decode
    // eagerly.
    #[test]
    fn texture_deferral_spares_the_start_scene() {
        let mut world = ResidencyWorld::new(&[START, LATER]).with_groups(vec![
            group(START, vec![(TEXTURE_KIND, 0)], Vec::new()),
            group(
                LATER,
                vec![(TEXTURE_KIND, 1), (TEXTURE_KIND, 2)],
                Vec::new(),
            ),
        ]);
        let deferred = deferred_texture_slots(&world.ctx(), true, 8);
        assert_eq!(deferred, std::collections::HashSet::from([1, 2]));
    }

    // A group entry that is not a Texture, or whose handle is past the streamed
    // slot count, is not a texture slot this path can defer.
    #[test]
    fn texture_deferral_ignores_other_kinds_and_out_of_range_handles() {
        let mut world = ResidencyWorld::new(&[START, LATER]).with_groups(vec![group(
            LATER,
            vec![(TEXTURE_KIND, 1), (MESH_KIND, 2), (TEXTURE_KIND, 9)],
            Vec::new(),
        )]);
        let deferred = deferred_texture_slots(&world.ctx(), true, 3);
        assert_eq!(deferred, std::collections::HashSet::from([1]));
    }

    // The streamer is the runtime load path that brings a deferred payload in,
    // so with no StreamingConfig -- or no Scene to own the content, or no baked
    // groups -- everything decodes at init as before.
    #[test]
    fn texture_deferral_is_empty_without_streaming_scenes_or_groups() {
        let groups = vec![group(LATER, vec![(TEXTURE_KIND, 1)], Vec::new())];

        let mut unstreamed = ResidencyWorld::new(&[START, LATER]).with_groups(groups.clone());
        assert!(deferred_texture_slots(&unstreamed.ctx(), false, 8).is_empty());

        let mut sceneless = ResidencyWorld::new(&[]).with_groups(groups);
        assert!(deferred_texture_slots(&sceneless.ctx(), true, 8).is_empty());

        let mut ungrouped = ResidencyWorld::new(&[START, LATER]);
        assert!(deferred_texture_slots(&ungrouped.ctx(), true, 8).is_empty());
    }

    // Mesh deferral mirrors the texture path over both mesh-source forms
    // (resource-stream handles and payload-carrying defs), while the baked
    // bounds + counts are taken from every record: they are looked up by handle
    // for whichever sources end up deferred.
    #[test]
    fn mesh_deferral_marks_later_scene_sources_and_keeps_every_baked_record() {
        let start_def = AssetId(20);
        let later_def = AssetId(21);
        let mut world = ResidencyWorld::new(&[START, LATER])
            .with_groups(vec![
                group(START, vec![(MESH_KIND, 0)], vec![start_def]),
                group(
                    LATER,
                    vec![(MESH_KIND, 1), (TEXTURE_KIND, 5)],
                    vec![later_def],
                ),
            ])
            .with_mesh_bounds(vec![bounds_record(0, 24, 36), bounds_record(1, 8, 12)]);

        let sources = deferred_mesh_sources(&world.ctx(), true);
        assert_eq!(sources.by_handle, std::collections::HashSet::from([1]));
        assert_eq!(
            sources.by_def,
            std::collections::HashSet::from([later_def]),
            "only the later scene's defs defer"
        );
        // Draw records for a deferred mesh are built from these, so both the
        // spared and the deferred record must be readable.
        assert_eq!(sources.counts.get(&0), Some(&(24, 36)));
        assert_eq!(sources.counts.get(&1), Some(&(8, 12)));
        assert_eq!(sources.bounds.get(&1), Some(&([-1.0; 3], [1.0; 3])));
    }

    // A mesh source with no baked bounds record cannot be deferred (init has
    // nothing to size its draw record from), and neither can any source in a
    // world whose blob predates the baked table.
    #[test]
    fn mesh_deferral_is_empty_without_streaming_or_baked_bounds() {
        let groups = vec![group(LATER, vec![(MESH_KIND, 1)], Vec::new())];

        let mut unstreamed = ResidencyWorld::new(&[START, LATER])
            .with_groups(groups.clone())
            .with_mesh_bounds(vec![bounds_record(1, 8, 12)]);
        assert!(
            deferred_mesh_sources(&unstreamed.ctx(), false)
                .by_handle
                .is_empty()
        );

        let mut unbaked = ResidencyWorld::new(&[START, LATER]).with_groups(groups);
        let sources = deferred_mesh_sources(&unbaked.ctx(), true);
        assert!(sources.by_handle.is_empty());
        assert!(sources.counts.is_empty());
    }

    // A GraphicsSystem carrying the recording backend, as init has it while the
    // setup routines wire the pools onto it. The mock reports the UNKNOWN GPU
    // profile (0 bytes of memory), so a pool with no explicit override lands on
    // its count-only policy.
    fn system_with_backend() -> (Arc<Mutex<MockState>>, GraphicsSystem) {
        let (recorded, backend) = recording_backend();
        let mut gs = GraphicsSystem::new();
        gs.backend = Some(Box::new(backend));
        (recorded, gs)
    }

    // The payload bytes are never decoded at setup (the streamer's worker does
    // that later), so any bytes stand in for a compiled texture.
    fn texture_payloads(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| vec![i as u8; 8]).collect()
    }

    fn centers(n: usize) -> Vec<Vec<[f32; 3]>> {
        (0..n).map(|i| vec![[i as f32, 0.0, 0.0]]).collect()
    }

    fn mesh_payloads(n: usize) -> Vec<crate::gfx::streaming::mesh::DecodedMesh> {
        (0..n)
            .map(|_| crate::gfx::streaming::mesh::DecodedMesh {
                vertices: Vec::new(),
                indices: Vec::new(),
            })
            .collect()
    }

    // A streamable world's texture slots are all evicted to placeholders at
    // setup; the streamer then brings them back nearest-first.
    #[test]
    fn texture_setup_evicts_every_slot_and_builds_the_pool() {
        let (recorded, mut gs) = system_with_backend();
        gs.setup_texture_streaming(
            Some(StreamingConfig::default()),
            texture_payloads(2),
            &[],
            false,
            centers(2),
        );

        assert_eq!(gs.texture_streamer.as_ref().map(|s| s.len()), Some(2));
        let s = recorded.lock().unwrap();
        assert!(s.saw(&Call::EvictTextureSlot(0)));
        assert!(s.saw(&Call::EvictTextureSlot(1)));
    }

    // The three guards that leave a world unstreamed: no config declared, no
    // streamable slot, and no backend to wire the pool onto.
    #[test]
    fn texture_setup_is_skipped_without_a_config_slots_or_backend() {
        let (recorded, mut gs) = system_with_backend();
        gs.setup_texture_streaming(None, texture_payloads(2), &[], false, centers(2));
        assert!(gs.texture_streamer.is_none(), "no StreamingConfig declared");

        gs.setup_texture_streaming(
            Some(StreamingConfig::default()),
            Vec::new(),
            &[],
            false,
            Vec::new(),
        );
        assert!(gs.texture_streamer.is_none(), "no slot to stream");
        assert!(recorded.lock().unwrap().calls.is_empty());

        let mut headless = GraphicsSystem::new();
        headless.setup_texture_streaming(
            Some(StreamingConfig::default()),
            texture_payloads(1),
            &[],
            false,
            centers(1),
        );
        assert!(headless.texture_streamer.is_none(), "no backend");
    }

    // An explicit texture budget wins over the VRAM-derived fraction; with no
    // override the mock's unreporting GPU leaves the pool count-only.
    #[test]
    fn texture_setup_honors_an_explicit_byte_budget() {
        let (_recorded, mut gs) = system_with_backend();
        gs.setup_texture_streaming(
            Some(StreamingConfig {
                texture_budget_mb: 8,
                ..Default::default()
            }),
            texture_payloads(1),
            &[],
            false,
            centers(1),
        );
        assert_eq!(
            gs.texture_streamer.as_ref().unwrap().byte_budget(),
            Some(8 * MIB)
        );

        let (_r2, mut count_only) = system_with_backend();
        count_only.setup_texture_streaming(
            Some(StreamingConfig::default()),
            texture_payloads(1),
            &[],
            false,
            centers(1),
        );
        assert_eq!(
            count_only.texture_streamer.as_ref().unwrap().byte_budget(),
            None
        );
    }

    // A disk-backed world drops the payload copies, so the streamed slot count
    // comes from the locators instead.
    #[test]
    fn disk_backed_texture_setup_counts_slots_from_the_locators() {
        let (recorded, mut gs) = system_with_backend();
        let locators = vec![
            crate::ecs::PayloadLocator {
                blob_index: 0,
                offset: 0,
                len: 8,
            },
            crate::ecs::PayloadLocator {
                blob_index: 0,
                offset: 8,
                len: 8,
            },
        ];
        // Payloads were not retained on this path: the locators alone decide
        // how many slots are swept.
        gs.setup_texture_streaming(
            Some(StreamingConfig::default()),
            Vec::new(),
            &locators,
            true,
            centers(2),
        );

        let s = recorded.lock().unwrap();
        assert!(s.saw(&Call::EvictTextureSlot(0)));
        assert!(s.saw(&Call::EvictTextureSlot(1)));
    }

    // With a shrinkable seed reserved, the streamed geometry was never baked in:
    // the sub-allocators are seeded with the headroom block, and evicting here
    // would free the placeholder region out from under the first resident draw.
    #[test]
    fn mesh_setup_seeds_the_reserved_region_instead_of_evicting() {
        let (recorded, mut gs) = system_with_backend();
        gs.setup_mesh_streaming(
            Some(StreamingConfig::default()),
            MeshStreamSetup {
                payloads: mesh_payloads(2),
                centers: centers(2),
                draw_indices: vec![4, 7],
                disk_backed: false,
                seed_region: Some(crate::gfx::mesh_seed::MeshSeedRegion {
                    vtx_offset: 0,
                    vtx_bytes: 1024,
                    idx_offset: 0,
                    idx_bytes: 512,
                }),
                deferred_payloads: Default::default(),
            },
        );

        let s = recorded.lock().unwrap();
        assert!(s.saw(&Call::SeedMeshStreaming));
        assert!(
            !s.calls.iter().any(|c| matches!(c, Call::EvictMesh(_))),
            "evicting would corrupt the placeholder region"
        );
    }

    // Without a seed region each streamed mesh's build-time region is freed into
    // the sub-allocators instead, and the id -> draw map is retained.
    #[test]
    fn mesh_setup_frees_each_build_time_region_without_a_seed() {
        let (recorded, mut gs) = system_with_backend();
        gs.setup_mesh_streaming(
            Some(StreamingConfig::default()),
            MeshStreamSetup {
                payloads: mesh_payloads(2),
                centers: centers(2),
                draw_indices: vec![4, 7],
                disk_backed: false,
                seed_region: None,
                deferred_payloads: Default::default(),
            },
        );

        assert_eq!(gs.mesh_streamer.as_ref().map(|s| s.len()), Some(2));
        assert_eq!(gs.mesh_stream_draw_indices, vec![4, 7]);
        let s = recorded.lock().unwrap();
        assert!(s.saw(&Call::EvictMesh(4)));
        assert!(s.saw(&Call::EvictMesh(7)));
        assert!(!s.saw(&Call::SeedMeshStreaming));
    }

    #[test]
    fn mesh_setup_is_skipped_without_a_config_payloads_or_backend() {
        let (recorded, mut gs) = system_with_backend();
        gs.setup_mesh_streaming(
            None,
            MeshStreamSetup {
                payloads: mesh_payloads(1),
                centers: centers(1),
                draw_indices: vec![0],
                disk_backed: false,
                seed_region: None,
                deferred_payloads: Default::default(),
            },
        );
        assert!(gs.mesh_streamer.is_none(), "no StreamingConfig declared");

        gs.setup_mesh_streaming(
            Some(StreamingConfig::default()),
            MeshStreamSetup {
                payloads: Vec::new(),
                centers: Vec::new(),
                draw_indices: Vec::new(),
                disk_backed: false,
                seed_region: None,
                deferred_payloads: Default::default(),
            },
        );
        assert!(gs.mesh_streamer.is_none(), "no mesh to stream");
        assert!(recorded.lock().unwrap().calls.is_empty());

        let mut headless = GraphicsSystem::new();
        headless.setup_mesh_streaming(
            Some(StreamingConfig::default()),
            MeshStreamSetup {
                payloads: mesh_payloads(1),
                centers: centers(1),
                draw_indices: vec![0],
                disk_backed: false,
                seed_region: None,
                deferred_payloads: Default::default(),
            },
        );
        assert!(headless.mesh_streamer.is_none(), "no backend");
    }

    #[test]
    fn mesh_setup_honors_an_explicit_byte_budget() {
        let (_recorded, mut gs) = system_with_backend();
        gs.setup_mesh_streaming(
            Some(StreamingConfig {
                mesh_budget_mb: 16,
                ..Default::default()
            }),
            MeshStreamSetup {
                payloads: mesh_payloads(1),
                centers: centers(1),
                draw_indices: vec![0],
                disk_backed: false,
                seed_region: None,
                deferred_payloads: Default::default(),
            },
        );
        assert_eq!(
            gs.mesh_streamer.as_ref().unwrap().byte_budget(),
            Some(16 * MIB)
        );
    }

    fn solid_block() -> BlockType {
        BlockType {
            solid: true,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            ..Default::default()
        }
    }

    // A VoxelWorld resolves its shared material to texture-pool slots + scalars
    // and stands the chunk pool up on the backend at the world's chunk size.
    #[test]
    fn voxel_setup_resolves_the_material_and_builds_the_chunk_pool() {
        let (recorded, mut gs) = system_with_backend();
        let air = AssetId(1);
        let ground = AssetId(2);
        let handle = crate::ecs::MaterialHandle(0);
        let mut material = MaterialUniforms::DEFAULT;
        material.roughness = 0.25;

        let block_types = std::collections::HashMap::from([
            (
                air,
                BlockType {
                    solid: false,
                    ..Default::default()
                },
            ),
            (ground, solid_block()),
        ]);
        let material_map = std::collections::HashMap::from([(handle, (5usize, 6usize, material))]);

        gs.setup_voxel_world_streaming(
            Some(VoxelWorld {
                chunk_blocks: [8, 16, 8],
                block_size: 2.0,
                view_radius: 1,
                palette: vec![air, ground],
                material: Some(handle),
                ..Default::default()
            }),
            &block_types,
            &material_map,
        );

        let cs = gs.chunk_stream.as_ref().expect("chunk pool built");
        assert_eq!(cs.texture_slot, 5);
        assert_eq!(cs.normal_map_slot, 6);
        assert_eq!(cs.material.roughness, 0.25);
        // 8 blocks of 2 world units on each of X / Z.
        assert_eq!((cs.chunk_w, cs.chunk_d), (16.0, 16.0));
        assert!(recorded.lock().unwrap().saw(&Call::SetupChunkStreaming));
    }

    // A VoxelWorld naming no material renders its chunks with the engine
    // defaults rather than failing the world.
    #[test]
    fn voxel_setup_without_a_material_falls_back_to_the_defaults() {
        let (_recorded, mut gs) = system_with_backend();
        gs.setup_voxel_world_streaming(
            Some(VoxelWorld {
                view_radius: 1,
                ..Default::default()
            }),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        );

        let cs = gs.chunk_stream.as_ref().expect("chunk pool built");
        assert_eq!(cs.texture_slot, 0);
        assert_eq!(cs.normal_map_slot, NO_NORMAL_MAP_SLOT);
        assert_eq!(
            cs.origin_chunk,
            crate::gfx::chunk_coord::ChunkCoord::new(0, 0)
        );
    }

    // A palette naming an id that is not a BlockType degrades that entry to air:
    // the world still streams rather than failing over one bad name.
    #[test]
    fn an_unknown_palette_entry_degrades_to_air() {
        let (_recorded, mut gs) = system_with_backend();
        let known = AssetId(1);
        gs.setup_voxel_world_streaming(
            Some(VoxelWorld {
                view_radius: 1,
                palette: vec![AssetId(99), known],
                ..Default::default()
            }),
            &std::collections::HashMap::from([(known, solid_block())]),
            &std::collections::HashMap::new(),
        );
        assert!(gs.chunk_stream.is_some(), "the world still streams");
    }

    #[test]
    fn voxel_setup_is_skipped_without_a_voxel_world_or_backend() {
        let (recorded, mut gs) = system_with_backend();
        gs.setup_voxel_world_streaming(
            None,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        );
        assert!(gs.chunk_stream.is_none(), "no VoxelWorld declared");
        assert!(recorded.lock().unwrap().calls.is_empty());

        let mut headless = GraphicsSystem::new();
        headless.setup_voxel_world_streaming(
            Some(VoxelWorld::default()),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        );
        assert!(headless.chunk_stream.is_none(), "no backend");
    }

    // Chunk streaming has no user override, so an unreporting GPU leaves the
    // pool on its pure radius-only policy.
    #[test]
    fn chunk_streaming_is_count_only_when_the_gpu_reports_no_memory() {
        let (_recorded, mut gs) = system_with_backend();
        gs.setup_voxel_world_streaming(
            Some(VoxelWorld {
                view_radius: 1,
                ..Default::default()
            }),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        );
        assert_eq!(
            gs.chunk_stream.as_ref().unwrap().streamer.byte_budget(),
            None
        );
    }

    // An explicit non-zero override is taken verbatim (converted MiB -> bytes),
    // regardless of what the GPU reports.
    #[test]
    fn derive_byte_budget_uses_explicit_override() {
        // 256 MiB override, GPU memory irrelevant.
        assert_eq!(
            derive_byte_budget(256, 8 * 1024 * 1024 * 1024, TEXTURE_VRAM_FRACTION),
            Some(256 * 1024 * 1024)
        );
        // The override wins even when the GPU reports nothing.
        assert_eq!(
            derive_byte_budget(64, 0, MESH_VRAM_FRACTION),
            Some(64 * 1024 * 1024)
        );
    }

    // With no override, the budget is the fraction of the reported GPU memory.
    #[test]
    fn derive_byte_budget_derives_fraction_of_gpu_memory() {
        let vram = 8 * 1024 * 1024 * 1024u64; // 8 GiB
        assert_eq!(
            derive_byte_budget(0, vram, TEXTURE_VRAM_FRACTION),
            Some((vram as f64 * 0.50) as u64)
        );
        assert_eq!(
            derive_byte_budget(0, vram, MESH_VRAM_FRACTION),
            Some((vram as f64 * 0.30) as u64)
        );
        // Chunk streaming derives 15% of the GPU's reported memory.
        assert_eq!(
            derive_byte_budget(0, vram, CHUNK_VRAM_FRACTION),
            Some((vram as f64 * 0.15) as u64)
        );
    }

    // An unreporting GPU (0 bytes) with no override degrades to count-only.
    #[test]
    fn derive_byte_budget_is_none_when_gpu_reports_nothing() {
        assert_eq!(derive_byte_budget(0, 0, TEXTURE_VRAM_FRACTION), None);
        assert_eq!(derive_byte_budget(0, 0, MESH_VRAM_FRACTION), None);
    }

    // The chunk record reserve must cover the streamer's worst-case
    // residency, or resident chunks past the reserve get no GPU-driven draw record
    // and render invisibly. The streamer retains a chunk until its Chebyshev
    // distance exceeds `evict_radius = far_radius + EVICT_HYSTERESIS(=2)` (see
    // gfx::chunk_window), with `far_radius = impostor_radius()` (floored at
    // view_radius), so peak residency = `(2*(far_radius+2)+1)^2`. This must hold for
    // impostors-on AND impostors-off worlds (the default is impostor_radius = 0).
    #[test]
    fn chunk_reserve_covers_streamer_evict_window() {
        for (view, impostor) in [(5u32, 0u32), (2, 6), (8, 0), (3, 10), (0, 0), (32, 96)] {
            let vw = VoxelWorld {
                view_radius: view,
                impostor_radius: impostor,
                ..Default::default()
            };
            // `impostor_radius()` floors at `view_radius()`, so this is the real
            // far radius the streamer uses whether or not impostors are enabled.
            let far = vw.impostor_radius() as usize;
            let evict_span = 2 * (far + 2) + 1;
            let bound = (evict_span * evict_span).min(65536);
            assert!(
                chunk_reserve_count(&vw) >= bound,
                "view={view} impostor={impostor}: reserve {} < streamer evict window {}",
                chunk_reserve_count(&vw),
                bound,
            );
        }
    }
}
