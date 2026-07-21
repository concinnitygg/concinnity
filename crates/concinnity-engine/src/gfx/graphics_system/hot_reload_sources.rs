// Passive source catalogues captured at `GraphicsSystem::init` (under
// `cn debug`) describing every file-backed asset the renderer can hot-reload:
// the on-disk source path plus the GPU slot / draw indices it owns. These are
// plain data: the filesystem watcher, off-thread decode, and reload passes
// that consume them live in the `cn debug` binary (`crate::debug::hot_reload`),
// out of the library. `init` fills these maps and hands them off as a
// `HotReloadSources` bundle through `GraphicsSystem::take_hot_reload_sources`.
//
// These are filled by the library (init) and read by the `cn debug` binary, so
// from `cargo check --lib`'s view every field / `watch_dirs` is write-only.
// Allow dead code module-wide: the whole module is a binary-consumed handoff.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

// One reload entry: a file-backed source and the GPU slot it owns. Built once
// at `GraphicsSystem::init` from the live `Texture` assets and consulted on
// every reload event. Procedural textures (sky / plaster / etc.) carry no
// source file and are absent from the map.
#[derive(Debug, Clone)]
pub struct TextureSourceEntry {
    // The `source` field from the original `Texture` asset, identical to the
    // path the build pipeline read at compile time. Resolved relative to CWD:
    // `cn debug` runs from the client checkout root, so the path is valid
    // as-is.
    pub source: String,
    // `image_index` for `.glb`-image sources; 0 (ignored) for plain PNGs.
    pub image_index: u32,
    // Slot in the shared texture pool (`textures[slot]`), regardless of whether
    // the texture is sampled as an albedo, a normal map, or an optional map.
    pub slot: usize,
}

// Singleton `ColorLut` reload entry. The 3D grading LUT has no slot (the
// composite pass binds `self.color_lut` directly), so we only need the
// resolved source path (the raw asset source string is resolved once at init
// via `crate::build::color_lut::resolve_source_path` so the watcher knows
// where to subscribe and the per-frame reload knows what to re-read).
#[derive(Debug, Clone)]
pub struct ColorLutSource {
    // Resolved on-disk path the build pipeline read at compile time. Stored
    // resolved rather than raw so the watcher can subscribe to a real parent
    // directory even when the asset declaration used a bare filename.
    pub resolved_path: String,
}

// One file-backed `Mesh` reload entry. A single `Mesh` asset can be
// referenced by many `Prop`s, each of which received an independent copy of
// the mesh's geometry in the shared vertex / index buffer, so a reload has
// to overwrite N draw slots, not one. `draw_indices` lists every slot that
// carries this Mesh's geometry; the reload helper walks them all per entry.
#[derive(Debug, Clone)]
pub struct MeshSourceEntry {
    // Path string from the asset declaration. Used as-is by
    // the glTF parser in concinnity-cook, which resolves
    // bare filenames internally. For the watcher's directory subscription a separate
    // resolved path is held on the [`MeshSourceMap`].
    pub source: String,
    // Which primitive (flattened across glTF meshes) to import; mirrors the
    // asset declaration so the runtime decode matches the build pass.
    pub primitive_index: u32,
    // Total LOD count from the asset declaration (`1` for no LODs).
    // Re-applied at decode time so the recomputed payload's LOD trailer
    // matches the slot's init-time layout.
    pub lod_levels: u32,
    // Per-LOD switch distances from the asset declaration. Empty means the
    // build derived a doubling sequence from the mesh's bounding radius;
    // reload reproduces the same defaults by passing through empty.
    pub lod_distances: Vec<f32>,
    // Every draw slot that received this mesh's geometry at init.
    pub draw_indices: Vec<usize>,
}

// Catalogue of every file-backed `Mesh` asset the renderer can hot-reload.
// Owned by `GraphicsSystem` under `cn debug` only. Sourced from
// `build_draw_list` extending its return tuple with `(asset_id ->
// draw_indices)` and cross-referenced against the source / `primitive_index`
// / LOD metadata captured before drains in `load_mesh_geometry`.
#[derive(Debug, Clone, Default)]
pub struct MeshSourceMap {
    pub entries: Vec<MeshSourceEntry>,
}

impl MeshSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    // Every unique parent directory across all entries. The watcher uses
    // these to know what to subscribe to; bare-filename sources (no parent)
    // are skipped here and only reachable via the debug-WS `reload-assets`
    // command. The caller should pass *resolved* paths via the resolved
    // field in [`MeshSourceEntry`]; for now resolution lives at the call
    // site (init.rs).
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for e in &self.entries {
            if let Some(parent) = Path::new(&e.source).parent()
                && !parent.as_os_str().is_empty()
            {
                dirs.insert(parent.to_path_buf());
            }
        }
        dirs.into_iter().collect()
    }
}

// One `ProceduralMesh` reload entry. Procedural meshes have no source file,
// so their hot-reload trigger is a `world.jsonl` save (or the debug-WS
// `reload-assets` command): the renderer captures each mesh's args at init
// and re-runs the generator when the on-disk args change. `draw_indices`
// mirrors [`MeshSourceEntry`]: one ProceduralMesh asset can map to many
// draw slots when several `Prop`s share it.
#[derive(Debug, Clone)]
pub struct ProceduralMeshSourceEntry {
    // The asset's name as declared in `world.jsonl`. The reload pass joins
    // the on-disk JSONL's `ProceduralMesh` entries by name so a Prop's
    // renamed-or-replaced mesh trips the same "unknown" log as any other
    // add; we never have to round-trip AssetIds through the interner here.
    pub name: String,
    // Last-applied generator args as the parsed component, so default-filled
    // fields match what a reload-time parse of `world.jsonl` produces.
    // Typed equality classifies whether to regenerate.
    pub args: crate::assets::ProceduralMesh,
    // Every draw slot that received this mesh's geometry at init.
    pub draw_indices: Vec<usize>,
}

// Catalogue of every `ProceduralMesh` asset whose generator args the
// renderer can hot-reload from a live `world.jsonl`. Owned by
// `GraphicsSystem` under `cn debug` only.
#[derive(Debug, Clone, Default)]
pub struct ProceduralMeshSourceMap {
    pub entries: Vec<ProceduralMeshSourceEntry>,
}

impl ProceduralMeshSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

// Resolve a `ShaderStage`'s declared source to the on-disk path the watcher
// subscribes to and the runtime recompile reads, applying the same
// bare-filename fallback the build pipeline runs: bare filenames are searched
// in `.concinnity/assets/` (recursively), then fall back to a direct path under
// that directory. Paths already carrying a directory component are returned
// unchanged. Fills a [`ShaderStageSourceEntry`]'s `resolved_path`.
pub fn resolve_runtime_source_path(raw: &str) -> String {
    let p = Path::new(raw);
    if p.parent().map(|d| d.as_os_str().is_empty()).unwrap_or(true) {
        if let Some(path) = concinnity_core::paths::find_in_assets(raw) {
            return path;
        }
        return concinnity_core::paths::assets_dir()
            .join(raw)
            .to_string_lossy()
            .into_owned();
    }
    raw.to_string()
}

// One world-loaded [`crate::assets::ShaderStage`] reload entry. Captures
// the stage's kind + the resolved on-disk source path that the build
// pipeline read at compile time, so the hot-reload helper can rerun
// [`concinnity_cook::shader::compile_shader`] on the same file and feed the
// fresh metallib / SPIR-V / DXBC bytes back to the backend for a pipeline
// rebuild. Stages whose source is the embedded GLSL fallback (Vulkan-only,
// no on-disk file) have an empty `resolved_path` and are filtered by the
// caller before reaching this map.
#[derive(Debug, Clone)]
pub struct ShaderStageSourceEntry {
    pub kind: crate::assets::shader_stage::ShaderKind,
    // Resolved on-disk path the build pipeline read at compile time. Stored
    // resolved (not raw) so the watcher can subscribe to a real parent
    // directory even when the asset declaration used a bare filename.
    pub resolved_path: String,
}

// Catalogue of every world-loaded `ShaderStage` whose source the renderer
// can hot-reload. Owned by `GraphicsSystem` under `cn debug` only; consumed
// by [`reload_shader_stages`] when the asset hot-reload watcher fires on a
// captured shader-source file. The map holds at most one entry per
// [`crate::assets::shader_stage::ShaderKind`] (vertex, fragment, shadow,
// vertex_instanced): the runtime drains one stage per kind at init.
#[derive(Debug, Clone, Default)]
pub struct ShaderStageSourceMap {
    pub entries: Vec<ShaderStageSourceEntry>,
}

impl ShaderStageSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    // Every unique parent directory across all entries. The watcher uses
    // these to know what to subscribe to alongside the texture / mesh /
    // LUT / envmap / world directories. Bare filenames (no parent) are
    // skipped; those are only reachable via the debug-WS `reload-assets`
    // command, mirroring the static-Mesh + texture maps.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for e in &self.entries {
            if let Some(parent) = Path::new(&e.resolved_path).parent()
                && !parent.as_os_str().is_empty()
            {
                dirs.insert(parent.to_path_buf());
            }
        }
        dirs.into_iter().collect()
    }
}

// One file-backed `SkinnedMesh` reload entry. Unlike static `Mesh`, a
// `SkinnedMesh` is 1:1 with its draw slot (there's no shared-instance
// fan-out across Props), so a single `skinned_index` identifies the slot to
// update. The vertex region is at `[vertex_base, vertex_base + vertex_count)`
// in the shared skinned vertex buffer; `joint_count` is snapshotted at init
// so the reload can reject skeleton-shape changes (which would require
// rebuilding the skinned pipeline state from shader-library bytes that
// `upload_skinned` consumes and drops).
#[derive(Debug, Clone)]
pub struct SkinnedMeshSourceEntry {
    // Path string from the asset declaration. Used as-is by
    // the glTF parser in concinnity-cook, which resolves
    // bare filenames internally.
    pub source: String,
    // Mirrors [`SkinnedMesh::skin_index`]: which skinned mesh of `source` the
    // reload re-imports.
    pub skin_index: u32,
    // Index into `MtlContext.skinned_draw_objects` (and the corresponding
    // `SkinnedDrawObject` slot on every backend) of the draw this entry
    // owns.
    pub skinned_index: usize,
    // Vertex offset (in vertex units, not bytes) into the shared skinned
    // vertex buffer where this slot's geometry starts.
    pub vertex_base: u16,
    // Number of vertices in this slot. Used to reject size-changing
    // reloads before pushing through to the backend.
    pub vertex_count: usize,
    // Number of indices in this slot, matches
    // `SkinnedDrawObject.index_count`. Kept here too so the size check
    // runs without indirecting through the backend.
    pub index_count: usize,
    // Init-time bind-pose joint count. Reload is rejected if the re-imported
    // skeleton has a different joint count; a different shape would need
    // a full pipeline rebuild, which `upload_skinned` does not support
    // post-init.
    pub joint_count: usize,
}

// Catalogue of every file-backed `SkinnedMesh` asset the renderer can
// hot-reload. Owned by `GraphicsSystem` under `cn debug` only.
#[derive(Debug, Clone, Default)]
pub struct SkinnedMeshSourceMap {
    pub entries: Vec<SkinnedMeshSourceEntry>,
}

impl SkinnedMeshSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    // Every unique parent directory across all entries. The watcher uses
    // these alongside the static-Mesh watch dirs.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for e in &self.entries {
            if let Some(parent) = Path::new(&e.source).parent()
                && !parent.as_os_str().is_empty()
            {
                dirs.insert(parent.to_path_buf());
            }
        }
        dirs.into_iter().collect()
    }
}

// Singleton `EnvironmentMap` reload entry. The two IBL cubemaps have no slot
// (the fragment shader binds `self.env_map.irradiance` and
// `self.env_map.prefilter` directly), so we only need the resolved HDR path
// plus the three sizing knobs from the asset declaration. The face sizes /
// sample count are captured at init so the runtime re-decode produces the
// same texture dimensions as the build pass (a size change would invalidate
// fragment-shader assumptions about the prefilter mip chain).
#[derive(Debug, Clone)]
pub struct EnvironmentMapSource {
    // Resolved on-disk path to the `.hdr` equirectangular. Stored resolved
    // (not raw) so the watcher can subscribe to a real parent directory even
    // when the asset declaration used a bare filename.
    pub resolved_path: String,
    // Mip-0 face size of the prefiltered radiance cubemap.
    pub prefilter_face_size: u32,
    // Face size of the irradiance cubemap.
    pub irradiance_face_size: u32,
    // Hammersley sample count for the GGX prefilter convolution.
    pub prefilter_samples: u32,
    // Per-texel brightness cap for the glossy reflection mips (firefly clamp).
    pub prefilter_clamp: f32,
}

// Catalogue of every file-backed `Texture` slot the renderer can hot-reload.
// Owned by `GraphicsSystem` under `cn debug` only.
#[derive(Debug, Clone, Default)]
pub struct TextureSourceMap {
    pub entries: Vec<TextureSourceEntry>,
}

impl TextureSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    // Add a texture-pool entry. Procedural / source-less textures should be
    // filtered by the caller before calling this; every entry must have a
    // non-empty `source`.
    pub fn push_texture(&mut self, source: String, image_index: u32, slot: usize) {
        self.entries.push(TextureSourceEntry {
            source,
            image_index,
            slot,
        });
    }

    // Every unique parent directory across all entries. Used by the
    // filesystem watcher to know what to subscribe to. A `.glb` source has
    // its containing directory watched too; the whole file shows up as
    // "modified" when the user re-exports it.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for e in &self.entries {
            if let Some(parent) = Path::new(&e.source).parent()
                && !parent.as_os_str().is_empty()
            {
                dirs.insert(parent.to_path_buf());
            }
        }
        dirs.into_iter().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

// Bundle of every captured source catalogue, handed from `GraphicsSystem`
// init to the `cn debug` binary's hot-reload drive, which builds the
// filesystem watcher + `AssetHotReloadState` from it. Empty / `None` under
// `cn run`, which never captures sources.
#[derive(Default)]
pub struct HotReloadSources {
    pub map: TextureSourceMap,
    pub color_lut: Option<ColorLutSource>,
    pub environment_map: Option<EnvironmentMapSource>,
    pub meshes: MeshSourceMap,
    pub skinned_meshes: SkinnedMeshSourceMap,
    pub procedural_meshes: ProceduralMeshSourceMap,
    pub shader_stages: ShaderStageSourceMap,
    pub world_jsonl_path: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_keeps_paths_with_a_directory_component() {
        // A path that already contains a directory is returned verbatim; the
        // bare-filename branch consults process-global asset anchors and is left
        // to integration coverage. The build-side `resolve_source_path_for`
        // (which takes a `BuildCtx`) is covered in concinnity-cook.
        assert_eq!(
            resolve_runtime_source_path("shaders/x.metal"),
            "shaders/x.metal"
        );
    }

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

    // The Mesh catalogue watches the same way, and one Mesh can own several
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

    // The skinned catalogue watches like the static one; a skinned mesh is 1:1
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

    // Shader stages watch their resolved paths, one entry per kind.
    #[test]
    fn shader_stage_map_watches_resolved_parents() {
        use crate::assets::shader_stage::ShaderKind;

        let mut map = ShaderStageSourceMap::new();
        assert!(map.is_empty());
        for (kind, path) in [
            (ShaderKind::Vertex, "shaders/scene.metal"),
            (ShaderKind::Fragment, "shaders/scene.metal"),
            (ShaderKind::VertexInstanced, "shaders/instanced/pass.metal"),
        ] {
            map.entries.push(ShaderStageSourceEntry {
                kind,
                resolved_path: path.to_string(),
            });
        }

        assert_eq!(map.len(), 3);
        assert_eq!(
            dirs(&map.watch_dirs()),
            ["shaders", "shaders/instanced"],
            "the two stages sharing a file collapse to one dir"
        );
    }

    // A stage compiled from the embedded fallback has no on-disk file, so it
    // contributes no subscription.
    #[test]
    fn shader_stage_map_skips_an_empty_resolved_path() {
        let mut map = ShaderStageSourceMap::new();
        map.entries.push(ShaderStageSourceEntry {
            kind: crate::assets::shader_stage::ShaderKind::Fragment,
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
    }
}
