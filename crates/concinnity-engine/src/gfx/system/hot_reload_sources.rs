//! Passive source catalogs captured at `GraphicsSystem::init` (under
//! `cn debug`) describing every file-backed asset the renderer can hot-reload:
//! the on-disk source path plus the GPU slot / draw indices it owns. These are
//! plain data: the filesystem watcher, off-thread decode, and reload passes
//! that consume them live in the dev tooling crate (`concinnity_dev::debug::hot_reload`),
//! out of the library. `init` fills these maps and parks them as one
//! `HotReloadSources` world resource, which the dev drive takes once.

use concinnity_core::components::ProceduralMesh;
use concinnity_core::components::ShaderStage;
use concinnity_core::ecs::PipelineContext;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_host::thread::asset_id;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use super::parked::TextureNameSlots;
use crate::gfx::draw_list::MeshSourceMeta;

// Every unique parent directory across `paths`. The watcher subscribes to
// these; a bare-filename source (no parent) is skipped and only reachable via
// the `reload-assets` debug tool call. Callers pass resolved paths.
fn watch_dirs_of<'a>(paths: impl Iterator<Item = &'a str>) -> Vec<PathBuf> {
    let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
    for path in paths {
        if let Some(parent) = Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            dirs.insert(parent.to_path_buf());
        }
    }
    dirs.into_iter().collect()
}

/// One reload entry: a file-backed source and the GPU slot it owns. Built once
/// at `GraphicsSystem::init` from the live `Texture` assets and consulted on
/// every reload event. Procedural textures (sky / plaster / etc.) carry no
/// source file and are absent from the map.
#[derive(Debug, Clone)]
pub struct TextureSourceEntry {
    /// The `source` field from the original `Texture` asset, identical to the
    /// path the build pipeline read at compile time. Resolved relative to CWD:
    /// `cn debug` runs from the client checkout root, so the path is valid
    /// as-is.
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
    pub draw_indices: Vec<usize>,
}

/// Catalog of every file-backed `Mesh` asset the renderer can hot-reload.
/// Owned by `GraphicsSystem` under `cn debug` only. Sourced from
/// `build_draw_list` extending its return tuple with `(asset_id ->
/// draw_indices)` and cross-referenced against the source / `primitive_index`
/// / LOD metadata captured before drains in `load_mesh_geometry`.
#[derive(Debug, Clone, Default)]
pub struct MeshSourceMap {
    /// One entry per reloadable mesh.
    pub entries: Vec<MeshSourceEntry>,
}

impl MeshSourceMap {
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

    /// Directories the watcher must subscribe to for these sources.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        watch_dirs_of(self.entries.iter().map(|e| e.source.as_str()))
    }
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
    pub draw_indices: Vec<usize>,
}

/// Catalog of every `ProceduralMesh` asset whose generator args the
/// renderer can hot-reload from a live `world.jsonl`. Owned by
/// `GraphicsSystem` under `cn debug` only.
#[derive(Debug, Clone, Default)]
pub struct ProceduralMeshSourceMap {
    /// One entry per reloadable procedural mesh.
    pub entries: Vec<ProceduralMeshSourceEntry>,
}

impl ProceduralMeshSourceMap {
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

/// One of the world default Shader's files, as a reload entry: which hook it
/// defines plus the resolved on-disk path the build pipeline read, so the
/// hot-reload helper can recompile the Shader through the cook's own compile
/// and hand the fresh programs back to the backend for a pipeline rebuild.
#[derive(Debug, Clone)]
pub struct ShaderStageSourceEntry {
    /// Which of the Shader's files this is.
    pub stage: ShaderStage,
    /// Resolved on-disk path the build pipeline read at compile time. Stored
    /// resolved (not raw) so the watcher can subscribe to a real parent
    /// directory even when the asset declaration used a bare filename.
    pub resolved_path: String,
}

/// Catalog of the world default Shader's files, which the renderer can
/// hot-reload. Owned by `GraphicsSystem` under `cn debug` only; consumed by
/// `reload_shader_stages` when the asset hot-reload watcher fires on one of
/// them. At most one entry per [`concinnity_core::components::ShaderStage`].
#[derive(Debug, Clone, Default)]
pub struct ShaderStageSourceMap {
    /// One entry per reloadable shader stage.
    pub entries: Vec<ShaderStageSourceEntry>,
}

impl ShaderStageSourceMap {
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

    /// Directories the watcher must subscribe to for these sources.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        watch_dirs_of(self.entries.iter().map(|e| e.resolved_path.as_str()))
    }
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
    pub skinned_index: usize,
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

/// Catalog of every file-backed `SkinnedMesh` asset the renderer can
/// hot-reload. Owned by `GraphicsSystem` under `cn debug` only.
#[derive(Debug, Clone, Default)]
pub struct SkinnedMeshSourceMap {
    /// One entry per reloadable skinned mesh.
    pub entries: Vec<SkinnedMeshSourceEntry>,
}

impl SkinnedMeshSourceMap {
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

    /// Directories the watcher must subscribe to for these sources.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        watch_dirs_of(self.entries.iter().map(|e| e.source.as_str()))
    }
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

/// Catalog of every file-backed `Texture` slot the renderer can hot-reload.
/// Owned by `GraphicsSystem` under `cn debug` only.
#[derive(Debug, Clone, Default)]
pub struct TextureSourceMap {
    /// One entry per reloadable texture.
    pub entries: Vec<TextureSourceEntry>,
}

impl TextureSourceMap {
    /// An empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a texture-pool entry. Procedural / source-less textures should be
    /// filtered by the caller before calling this; every entry must have a
    /// non-empty `source`.
    pub fn push_texture(&mut self, source: String, image_index: u32, slot: usize) {
        self.entries.push(TextureSourceEntry {
            source,
            image_index,
            slot,
        });
    }

    /// Directories the watcher must subscribe to for these sources.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        watch_dirs_of(self.entries.iter().map(|e| e.source.as_str()))
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

/// Bundle of every captured source catalog, parked as a world resource by
/// `GraphicsSystem` init and taken once by concinnity-dev's hot-reload drive,
/// which builds the filesystem watcher + `AssetHotReloadState` from it. Never
/// parked under `cn run`, which captures no sources.
#[derive(Default)]
pub struct HotReloadSources {
    /// Reloadable textures.
    pub map: TextureSourceMap,
    /// The reloadable color LUT, when the world declares one.
    pub color_lut: Option<ColorLutSource>,
    /// The reloadable environment map, when the world declares one.
    pub environment_map: Option<EnvironmentMapSource>,
    /// Reloadable static meshes.
    pub meshes: MeshSourceMap,
    /// Reloadable skinned meshes.
    pub skinned_meshes: SkinnedMeshSourceMap,
    /// Reloadable procedural meshes.
    pub procedural_meshes: ProceduralMeshSourceMap,
    /// Reloadable shader stages.
    pub shader_stages: ShaderStageSourceMap,
    /// Path to the world's `world.jsonl`, when it was loaded from one.
    pub world_jsonl_path: Option<String>,
}

impl HotReloadSources {
    /// Whether nothing reloadable was captured, so no watcher is worth starting.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
            && self.color_lut.is_none()
            && self.environment_map.is_none()
            && self.meshes.is_empty()
            && self.skinned_meshes.is_empty()
            && self.procedural_meshes.is_empty()
            && self.shader_stages.is_empty()
            && self.world_jsonl_path.is_none()
    }
}

// Snapshot each ProceduralMesh, with its interned name, before the mesh load
// drains them. A world.jsonl reload diffs a freshly parsed entry against this
// and regenerates the mesh when they differ, logging it by name.
pub(super) fn procedural_mesh_snapshot(
    ctx: &PipelineContext,
) -> HashMap<AssetId, (String, ProceduralMesh)> {
    ctx.query::<ProceduralMesh>()
        .filter_map(|pm| Some((pm.asset_id, (asset_id::name_of(pm.asset_id)?, pm.clone()))))
        .collect()
}

// Cross-reference the file-backed mesh sources captured at drain time with the
// draw slots the draw list built. A mesh with no draws has nothing to reload
// into, so it is omitted.
pub(super) fn mesh_source_map(
    sources: &HashMap<usize, MeshSourceMeta>,
    mesh_handle_to_draws: &HashMap<usize, Vec<usize>>,
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

// The same cross-reference for procedural meshes, whose "source" is the args
// snapshot taken before the drain. A mesh no prop draws is omitted.
pub(super) fn procedural_mesh_source_map(
    snapshot: &HashMap<AssetId, (String, ProceduralMesh)>,
    component_handles: &HashMap<AssetId, usize>,
    mesh_handle_to_draws: &HashMap<usize, Vec<usize>>,
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

// Keep the captured sources, and the texture-name map beside them, only when
// something is reloadable. The dev drive's watcher subscribes to the parent
// directory of every captured source path.
pub(super) fn capture_hot_reload_sources(
    sources: HotReloadSources,
    texture_name_to_slot: HashMap<AssetId, usize>,
) -> (Option<HotReloadSources>, Option<TextureNameSlots>) {
    if sources.is_empty() {
        return (None, None);
    }
    tracing::info!(
        "asset hot-reload: captured {} file-backed texture source(s), {} \
         ColorLut source(s), {} EnvironmentMap source(s), {} Mesh \
         source(s), {} SkinnedMesh source(s), {} ProceduralMesh source(s), \
         {} shader stage source(s), and world.jsonl path = {:?}",
        sources.map.len(),
        usize::from(sources.color_lut.is_some()),
        usize::from(sources.environment_map.is_some()),
        sources.meshes.len(),
        sources.skinned_meshes.len(),
        sources.procedural_meshes.len(),
        sources.shader_stages.len(),
        sources.world_jsonl_path
    );
    (Some(sources), Some(TextureNameSlots(texture_name_to_slot)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh_entry(source: &str) -> MeshSourceEntry {
        MeshSourceEntry {
            source: source.to_string(),
            primitive_index: 0,
            lod_levels: 1,
            lod_distances: Vec::new(),
            draw_indices: vec![0],
        }
    }

    fn skinned_entry(source: &str) -> SkinnedMeshSourceEntry {
        SkinnedMeshSourceEntry {
            source: source.to_string(),
            skin_index: 0,
            skinned_index: 0,
            vertex_base: 0,
            vertex_count: 3,
            index_count: 3,
            joint_count: 1,
        }
    }

    fn dirs(paths: &[PathBuf]) -> Vec<String> {
        paths.iter().map(|p| p.display().to_string()).collect()
    }

    // A fresh map is empty, and pushing an entry is what gives it a length.
    #[test]
    fn texture_map_starts_empty_and_counts_pushed_entries() {
        let mut map = TextureSourceMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        map.push_texture("assets/wall.png".to_string(), 0, 4);
        map.push_texture("assets/scene.glb".to_string(), 2, 7);

        assert!(!map.is_empty());
        assert_eq!(map.len(), 2);
        // Each entry keeps the slot + image index it was pushed with, so a
        // reload rewrites the right pool slot.
        assert_eq!(map.entries[0].slot, 4);
        assert_eq!(map.entries[0].image_index, 0);
        assert_eq!(map.entries[1].slot, 7);
        assert_eq!(map.entries[1].image_index, 2);
    }

    // Entries sharing a directory collapse to one subscription, and the result
    // is sorted (the watcher subscribes once per directory).
    #[test]
    fn texture_watch_dirs_dedups_shared_parents() {
        let mut map = TextureSourceMap::new();
        map.push_texture("assets/textures/wall.png".to_string(), 0, 0);
        map.push_texture("assets/textures/floor.png".to_string(), 0, 1);
        map.push_texture("assets/models/scene.glb".to_string(), 0, 2);

        assert_eq!(
            dirs(&map.watch_dirs()),
            ["assets/models", "assets/textures"],
            "one entry per unique parent, sorted"
        );
    }

    // A bare filename has no parent directory to subscribe to, so it is skipped
    // rather than watching the process CWD.
    #[test]
    fn texture_watch_dirs_skips_bare_filenames() {
        let mut map = TextureSourceMap::new();
        map.push_texture("wall.png".to_string(), 0, 0);
        assert!(map.watch_dirs().is_empty());

        // A rooted sibling still contributes its own directory.
        map.push_texture("assets/floor.png".to_string(), 0, 1);
        assert_eq!(dirs(&map.watch_dirs()), ["assets"]);
    }

    // The Mesh catalog watches the same way, and one Mesh can own several
    // draw slots (a mesh shared by many Props).
    #[test]
    fn mesh_map_watches_parents_and_keeps_every_draw_slot() {
        let mut map = MeshSourceMap::new();
        assert!(map.is_empty());

        let mut shared = mesh_entry("assets/models/prop.glb");
        shared.draw_indices = vec![3, 9, 12];
        map.entries.push(shared);
        map.entries.push(mesh_entry("assets/models/tree.glb"));
        map.entries.push(mesh_entry("bare.glb"));

        assert_eq!(map.len(), 3);
        assert_eq!(
            dirs(&map.watch_dirs()),
            ["assets/models"],
            "the two rooted entries share one dir, the bare one is skipped"
        );
        assert_eq!(
            map.entries[0].draw_indices,
            vec![3, 9, 12],
            "a reload has to rewrite every slot carrying this mesh"
        );
    }

    // The skinned catalog watches like the static one; a skinned mesh is 1:1
    // with its draw slot.
    #[test]
    fn skinned_mesh_map_watches_parents() {
        let mut map = SkinnedMeshSourceMap::new();
        assert!(map.is_empty());
        map.entries.push(skinned_entry("assets/chars/fox.glb"));
        map.entries.push(skinned_entry("assets/chars/wolf.glb"));
        map.entries.push(skinned_entry("fox.glb"));

        assert_eq!(map.len(), 3);
        assert_eq!(dirs(&map.watch_dirs()), ["assets/chars"]);
    }

    // Shader files watch their resolved paths, one entry per stage.
    #[test]
    fn shader_stage_map_watches_resolved_parents() {
        use concinnity_core::components::ShaderStage;

        let mut map = ShaderStageSourceMap::new();
        assert!(map.is_empty());
        for (stage, path) in [
            (ShaderStage::Vertex, "shaders/sway.slang"),
            (ShaderStage::Fragment, "shaders/surface/scene.slang"),
        ] {
            map.entries.push(ShaderStageSourceEntry {
                stage,
                resolved_path: path.to_string(),
            });
        }

        assert_eq!(map.len(), 2);
        assert_eq!(dirs(&map.watch_dirs()), ["shaders", "shaders/surface"]);
    }

    // An entry with no on-disk file contributes no subscription.
    #[test]
    fn shader_stage_map_skips_an_empty_resolved_path() {
        let mut map = ShaderStageSourceMap::new();
        map.entries.push(ShaderStageSourceEntry {
            stage: ShaderStage::Fragment,
            resolved_path: String::new(),
        });
        assert!(map.watch_dirs().is_empty());
    }

    // Procedural meshes have no source file, so the map only counts entries;
    // their reload trigger is a world.jsonl save, not a watched directory.
    #[test]
    fn procedural_mesh_map_counts_entries() {
        let mut map = ProceduralMeshSourceMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        map.entries.push(ProceduralMeshSourceEntry {
            name: "ground".to_string(),
            args: Default::default(),
            draw_indices: vec![0, 1],
        });

        assert!(!map.is_empty());
        assert_eq!(map.len(), 1);
    }

    // The handoff bundle defaults to nothing captured, which is the `cn run`
    // shape: no watcher, no reloadable sources.
    #[test]
    fn bundle_defaults_to_nothing_captured() {
        let sources = HotReloadSources::default();
        assert!(sources.map.is_empty());
        assert!(sources.meshes.is_empty());
        assert!(sources.skinned_meshes.is_empty());
        assert!(sources.procedural_meshes.is_empty());
        assert!(sources.shader_stages.is_empty());
        assert!(sources.color_lut.is_none());
        assert!(sources.environment_map.is_none());
        assert!(sources.world_jsonl_path.is_none());
        assert!(sources.is_empty());
    }

    // Any one captured catalog is enough to make the bundle worth parking.
    #[test]
    fn bundle_with_any_single_capture_is_not_empty() {
        let populated: [fn(&mut HotReloadSources); 8] = [
            |s| s.map.push_texture("assets/wall.png".to_string(), 0, 0),
            |s| {
                s.color_lut = Some(ColorLutSource {
                    resolved_path: "assets/grade.cube".to_string(),
                })
            },
            |s| {
                s.environment_map = Some(EnvironmentMapSource {
                    resolved_path: "assets/sky.hdr".to_string(),
                    prefilter_face_size: 64,
                    irradiance_face_size: 16,
                    prefilter_samples: 32,
                    prefilter_clamp: 10.0,
                })
            },
            |s| s.meshes.entries.push(mesh_entry("assets/prop.glb")),
            |s| {
                s.skinned_meshes
                    .entries
                    .push(skinned_entry("assets/fox.glb"))
            },
            |s| {
                s.procedural_meshes.entries.push(ProceduralMeshSourceEntry {
                    name: "ground".to_string(),
                    args: Default::default(),
                    draw_indices: vec![0],
                })
            },
            |s| {
                s.shader_stages.entries.push(ShaderStageSourceEntry {
                    stage: ShaderStage::Fragment,
                    resolved_path: "shaders/scene.slang".to_string(),
                })
            },
            |s| s.world_jsonl_path = Some("world.jsonl".to_string()),
        ];
        for (i, populate) in populated.iter().enumerate() {
            let mut sources = HotReloadSources::default();
            populate(&mut sources);
            assert!(!sources.is_empty(), "field {i} alone is a capture");
        }
    }

    // An empty bundle parks nothing, not even the texture-name map.
    #[test]
    fn capturing_nothing_parks_nothing() {
        let names = HashMap::from([(AssetId(7), 0)]);
        let (sources, slots) = capture_hot_reload_sources(HotReloadSources::default(), names);
        assert!(sources.is_none());
        assert!(slots.is_none());
    }

    fn meta(source: &str) -> MeshSourceMeta {
        MeshSourceMeta {
            source: source.to_string(),
            primitive_index: 1,
            lod_levels: 2,
            lod_distances: vec![10.0],
        }
    }

    // A mesh handle with no draw entry, or an empty one, has nowhere to reload into.
    #[test]
    fn mesh_source_map_omits_handles_without_draws() {
        let sources = HashMap::from([
            (0, meta("assets/drawn.glb")),
            (1, meta("assets/unreferenced.glb")),
            (2, meta("assets/emptied.glb")),
        ]);
        let draws = HashMap::from([(0, vec![3, 4]), (2, Vec::new())]);
        let map = mesh_source_map(&sources, &draws);
        assert_eq!(map.len(), 1);
        let entry = &map.entries[0];
        assert_eq!(entry.source, "assets/drawn.glb");
        assert_eq!(entry.draw_indices, vec![3, 4]);
        assert_eq!((entry.primitive_index, entry.lod_levels), (1, 2));
    }

    // A procedural mesh with no handle, no draw entry, or an empty one is omitted.
    #[test]
    fn procedural_mesh_source_map_omits_meshes_without_draws() {
        let args = ProceduralMesh::default;
        let snapshot = HashMap::from([
            (AssetId(1), ("drawn".to_string(), args())),
            (AssetId(2), ("no_handle".to_string(), args())),
            (AssetId(3), ("unreferenced".to_string(), args())),
            (AssetId(4), ("emptied".to_string(), args())),
        ]);
        let handles = HashMap::from([(AssetId(1), 0), (AssetId(3), 1), (AssetId(4), 2)]);
        let draws = HashMap::from([(0, vec![5]), (2, Vec::new())]);
        let map = procedural_mesh_source_map(&snapshot, &handles, &draws);
        assert_eq!(map.len(), 1);
        assert_eq!(map.entries[0].name, "drawn");
        assert_eq!(map.entries[0].draw_indices, vec![5]);
    }
}
