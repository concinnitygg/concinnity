//! The file-backed sources a world's graphics load captures for a host that
//! reloads them: each source with the GPU slot or draw indices it owns. Plain
//! data; the watcher, decode and reload passes are the host's.

use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::components::ProceduralMesh;
use crate::ecs::asset_id::AssetId;
use crate::gfx::render_types::{DrawIndex, SkinnedIndex};
use crate::render::draw_list::MeshSourceMeta;
use crate::render::world_load::skinned::{SkinnedAssembly, SkinnedGeometry};

/// A catalog of reloadable sources, one entry each.
#[derive(Debug, Clone)]
pub struct SourceMap<E> {
    /// One entry per reloadable source.
    pub entries: Vec<E>,
}

impl<E> Default for SourceMap<E> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<E> SourceMap<E> {
    /// An empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries in the catalog.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// An entry read from a file, which a host watches for changes.
pub trait FileSourced {
    /// The source path as the asset declares it.
    fn source(&self) -> &str;
}

/// Reloadable textures.
pub type TextureSourceMap = SourceMap<TextureSourceEntry>;
/// Reloadable static meshes.
pub type MeshSourceMap = SourceMap<MeshSourceEntry>;
/// Reloadable procedural meshes, keyed by their generator args.
pub type ProceduralMeshSourceMap = SourceMap<ProceduralMeshSourceEntry>;
/// Reloadable skinned meshes.
pub type SkinnedMeshSourceMap = SourceMap<SkinnedMeshSourceEntry>;

/// One reload entry: a file-backed source and the GPU slot it owns. Built once
/// at graphics init from the live `Texture` assets and consulted on
/// every reload event. Procedural textures (sky / plaster / etc.) carry no
/// source file and are absent from the map.
#[derive(Debug, Clone)]
pub struct TextureSourceEntry {
    /// The `source` field from the original `Texture` asset, identical to the
    /// path the build pipeline read at compile time, relative to the working
    /// directory.
    pub source: String,
    /// `image_index` for `.glb`-image sources; 0 (ignored) for plain PNGs.
    pub image_index: u32,
    /// Slot in the shared texture pool (`textures[slot]`), regardless of whether
    /// the texture is sampled as an albedo, a normal map, or an optional map.
    pub slot: usize,
}

/// Singleton `ColorLut` reload entry. The 3D grading LUT has no slot (the
/// composite pass binds `self.color_lut` directly), so we only need the
/// resolved source path (the raw asset source string is resolved once at init
/// via `concinnity_host::store::source::resolve_source_path` so the watcher knows
/// where to subscribe and the per-frame reload knows what to re-read).
#[derive(Debug, Clone)]
pub struct ColorLutSource {
    /// Resolved on-disk path the build pipeline read at compile time. Stored
    /// resolved rather than raw so the watcher can subscribe to a real parent
    /// directory even when the asset declaration used a bare filename.
    pub resolved_path: String,
}

/// One file-backed `Mesh` reload entry. A single `Mesh` asset can be
/// referenced by many `Prop`s, each of which received an independent copy of
/// the mesh's geometry in the shared vertex / index buffer, so a reload has
/// to overwrite N draw slots, not one. `draw_indices` lists every slot that
/// carries this Mesh's geometry; the reload helper walks them all per entry.
#[derive(Debug, Clone)]
pub struct MeshSourceEntry {
    /// Path string from the asset declaration. Used as-is by
    /// the glTF parser in concinnity-cook, which resolves
    /// bare filenames internally. For the watcher's directory subscription a separate
    /// resolved path is held on the [`MeshSourceMap`].
    pub source: String,
    /// Which primitive (flattened across glTF meshes) to import; mirrors the
    /// asset declaration so the runtime decode matches the build pass.
    pub primitive_index: u32,
    /// Total LOD count from the asset declaration (`1` for no LODs).
    /// Re-applied at decode time so the recomputed payload's LOD trailer
    /// matches the slot's init-time layout.
    pub lod_levels: u32,
    /// Per-LOD switch distances from the asset declaration. Empty means the
    /// build derived a doubling sequence from the mesh's bounding radius;
    /// reload reproduces the same defaults by passing through empty.
    pub lod_distances: Vec<f32>,
    /// Every draw slot that received this mesh's geometry at init.
    pub draw_indices: Vec<DrawIndex>,
}

/// One `ProceduralMesh` reload entry. Procedural meshes have no source file,
/// so their hot-reload trigger is a `world.jsonl` save (or the
/// `reload-assets` debug tool call): the renderer captures each mesh's args at init
/// and re-runs the generator when the on-disk args change. `draw_indices`
/// mirrors [`MeshSourceEntry`]: one ProceduralMesh asset can map to many
/// draw slots when several `Prop`s share it.
#[derive(Debug, Clone)]
pub struct ProceduralMeshSourceEntry {
    /// The asset's name as declared in `world.jsonl`. The reload pass joins
    /// the on-disk JSONL's `ProceduralMesh` entries by name so a Prop's
    /// renamed-or-replaced mesh trips the same "unknown" log as any other
    /// add; we never have to round-trip AssetIds through the interner here.
    pub name: String,
    /// Last-applied generator args as the parsed component, so default-filled
    /// fields match what a reload-time parse of `world.jsonl` produces.
    /// Typed equality classifies whether to regenerate.
    pub args: ProceduralMesh,
    /// Every draw slot that received this mesh's geometry at init.
    pub draw_indices: Vec<DrawIndex>,
}

/// One file-backed `SkinnedMesh` reload entry. Unlike static `Mesh`, a
/// `SkinnedMesh` is 1:1 with its draw slot (there's no shared-instance
/// fan-out across Props), so a single `skinned_index` identifies the slot to
/// update. The vertex region is at `[vertex_base, vertex_base + vertex_count)`
/// in the shared skinned vertex buffer; `joint_count` is snapshotted at init
/// so the reload can reject skeleton-shape changes, which would need the
/// skinned pipeline rebuilt.
#[derive(Debug, Clone)]
pub struct SkinnedMeshSourceEntry {
    /// Path string from the asset declaration. Used as-is by
    /// the glTF parser in concinnity-cook, which resolves
    /// bare filenames internally.
    pub source: String,
    /// Mirrors `SkinnedMesh::skin_index`: which skinned mesh of `source` the
    /// reload re-imports.
    pub skin_index: u32,
    /// Index into `MtlContext.skinned_draw_objects` (and the corresponding
    /// `SkinnedDrawObject` slot on every backend) of the draw this entry
    /// owns.
    pub skinned_index: SkinnedIndex,
    /// Vertex offset (in vertex units, not bytes) into the shared skinned
    /// vertex buffer where this slot's geometry starts.
    pub vertex_base: u32,
    /// Number of vertices in this slot. Used to reject size-changing
    /// reloads before pushing through to the backend.
    pub vertex_count: usize,
    /// Number of indices in this slot, matches
    /// `SkinnedDrawObject.index_count`. Kept here too so the size check
    /// runs without indirecting through the backend.
    pub index_count: usize,
    /// Init-time bind-pose joint count. Reload is rejected if the re-imported
    /// skeleton has a different joint count; a different shape would need
    /// a full pipeline rebuild, which `upload_skinned` does not support
    /// post-init.
    pub joint_count: usize,
}

/// Singleton `EnvironmentMap` reload entry. The two IBL cubemaps have no slot
/// (the fragment shader binds `self.env_map.irradiance` and
/// `self.env_map.prefilter` directly), so we only need the resolved HDR path
/// plus the three sizing knobs from the asset declaration. The face sizes /
/// sample count are captured at init so the runtime re-decode produces the
/// same texture dimensions as the build pass (a size change would invalidate
/// fragment-shader assumptions about the prefilter mip chain).
#[derive(Debug, Clone)]
pub struct EnvironmentMapSource {
    /// Resolved on-disk path to the `.hdr` equirectangular. Stored resolved
    /// (not raw) so the watcher can subscribe to a real parent directory even
    /// when the asset declaration used a bare filename.
    pub resolved_path: String,
    /// Mip-0 face size of the prefiltered radiance cubemap.
    pub prefilter_face_size: u32,
    /// Face size of the irradiance cubemap.
    pub irradiance_face_size: u32,
    /// Hammersley sample count for the GGX prefilter convolution.
    pub prefilter_samples: u32,
    /// Per-texel brightness cap for the glossy reflection mips (firefly clamp).
    pub prefilter_clamp: f32,
}

impl FileSourced for TextureSourceEntry {
    fn source(&self) -> &str {
        &self.source
    }
}

impl FileSourced for MeshSourceEntry {
    fn source(&self) -> &str {
        &self.source
    }
}

impl FileSourced for SkinnedMeshSourceEntry {
    fn source(&self) -> &str {
        &self.source
    }
}

impl TextureSourceMap {
    /// Add a texture-pool entry. Source-less textures are the caller's to
    /// filter; every entry has a non-empty `source`.
    pub fn push_texture(&mut self, source: String, image_index: u32, slot: usize) {
        self.entries.push(TextureSourceEntry {
            source,
            image_index,
            slot,
        });
    }
}

/// The file-backed textures of a pool of `count` slots, with each texture's
/// name -> slot index, from the dev source catalog.
pub fn texture_catalog(
    sources: Option<&crate::resource::TextureSources>,
    count: usize,
) -> (TextureSourceMap, HashMap<AssetId, usize>) {
    let mut map = TextureSourceMap::new();
    let mut name_to_slot = HashMap::new();
    for (slot, info) in sources
        .into_iter()
        .flat_map(|s| s.0.iter())
        .take(count)
        .enumerate()
    {
        name_to_slot.insert(AssetId(info.name_id), slot);
        if !info.source.is_empty() {
            map.push_texture(info.source.clone(), info.image_index, slot);
        }
    }
    (map, name_to_slot)
}

/// The file-backed skinned meshes among `geometry`, each with the region its
/// template slot holds in `assembly`.
pub fn skinned_mesh_source_map(
    geometry: &[SkinnedGeometry],
    assembly: &SkinnedAssembly,
) -> SkinnedMeshSourceMap {
    let entries = geometry
        .iter()
        .zip(&assembly.templates)
        .filter(|(g, _)| !g.mesh.source.is_empty())
        .filter_map(|(g, t)| {
            let slot = assembly.draw_objects.get(t.template_index.index())?;
            Some(SkinnedMeshSourceEntry {
                source: g.mesh.source.clone(),
                skin_index: g.mesh.skin_index,
                skinned_index: t.template_index,
                vertex_base: slot.vertex_base,
                vertex_count: slot.vertex_count,
                index_count: slot.index_count,
                joint_count: slot.joint_count,
            })
        })
        .collect();
    SkinnedMeshSourceMap { entries }
}

/// The file-backed mesh sources captured at drain time, each with the draw
/// slots the draw list built for it. A mesh with no draws has nothing to
/// reload into, so it is omitted.
pub fn mesh_source_map(
    sources: &HashMap<usize, MeshSourceMeta>,
    mesh_handle_to_draws: &HashMap<usize, Vec<DrawIndex>>,
) -> MeshSourceMap {
    let entries = sources
        .iter()
        .filter_map(|(handle, meta)| {
            let draws = mesh_handle_to_draws.get(handle).filter(|d| !d.is_empty())?;
            Some(MeshSourceEntry {
                source: meta.source.clone(),
                primitive_index: meta.primitive_index,
                lod_levels: meta.lod_levels,
                lod_distances: meta.lod_distances.clone(),
                draw_indices: draws.clone(),
            })
        })
        .collect();
    MeshSourceMap { entries }
}

/// The same for procedural meshes, whose "source" is the args snapshot taken
/// before the drain. A mesh no prop draws is omitted.
pub fn procedural_mesh_source_map(
    snapshot: &HashMap<AssetId, (String, ProceduralMesh)>,
    component_handles: &HashMap<AssetId, usize>,
    mesh_handle_to_draws: &HashMap<usize, Vec<DrawIndex>>,
) -> ProceduralMeshSourceMap {
    let entries = snapshot
        .iter()
        .filter_map(|(asset_id, (name, args))| {
            let handle = component_handles.get(asset_id)?;
            let draws = mesh_handle_to_draws.get(handle).filter(|d| !d.is_empty())?;
            Some(ProceduralMeshSourceEntry {
                name: name.clone(),
                args: args.clone(),
                draw_indices: draws.clone(),
            })
        })
        .collect();
    ProceduralMeshSourceMap { entries }
}

#[cfg(test)]
mod tests;
