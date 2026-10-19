//! The source catalogs captured at `GraphicsSystem::init` (under `cn debug`),
//! bundled as the one `HotReloadSources` world resource the dev drive takes
//! once, and the directories its watcher subscribes to. The catalogs
//! themselves are core's `render::world_load::source_maps`; the shader and SDF
//! field catalogs, which resolve paths on disk, are this crate's.

use concinnity_core::components::{Identity, ProceduralMesh};
use concinnity_core::ecs::PipelineContext;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_host::thread::asset_id;
use hashbrown::HashMap;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::parked::{ShaderOverrides, TextureNameSlots};
use super::sdf_field_sources::SdfFieldMap;
use super::shader_sources::ShaderSourceMap;
use concinnity_core::render::world_load::source_maps::{
    ColorLutSource, EnvironmentMapSource, FileSourced, MeshSourceMap, ProceduralMeshSourceMap,
    SkinnedMeshSourceMap, SourceMap, TextureSourceMap,
};

// Every unique parent directory across `paths`. The watcher subscribes to
// these; a bare-filename source (no parent) is skipped and only reachable via
// the `reload-assets` debug tool call. Callers pass resolved paths.
pub(super) fn watch_dirs_of<'a>(paths: impl Iterator<Item = &'a str>) -> Vec<PathBuf> {
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

/// The directories a watcher subscribes to for a catalog's file sources.
pub trait WatchDirs {
    /// Every unique parent directory of the catalog's sources.
    fn watch_dirs(&self) -> Vec<PathBuf>;
}

impl<E: FileSourced> WatchDirs for SourceMap<E> {
    fn watch_dirs(&self) -> Vec<PathBuf> {
        watch_dirs_of(self.entries.iter().map(FileSourced::source))
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
    /// Reloadable world Shaders.
    pub shaders: ShaderSourceMap,
    /// Hot-reloaded Shader programs, shared with the streaming pump so a
    /// Shader whose scene loads later installs the edit.
    pub shader_overrides: ShaderOverrides,
    /// Reloadable `SdfVolume` fields.
    pub sdf_fields: SdfFieldMap,
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
            && self.shaders.is_empty()
            && self.sdf_fields.is_empty()
    }
}

// Snapshot each ProceduralMesh, with its interned name, before the mesh load
// drains them. A world.jsonl reload diffs a freshly parsed entry against this
// and regenerates the mesh when they differ, logging it by name.
pub(super) fn procedural_mesh_snapshot(
    ctx: &PipelineContext,
) -> HashMap<AssetId, (String, ProceduralMesh)> {
    ctx.join2::<ProceduralMesh, Identity>()
        .filter_map(|(_, pm, identity)| {
            let id = identity.id();
            Some((id, (asset_id::name_of(id)?, pm.clone())))
        })
        .collect()
}

// Park the captured sources and the texture-name map beside them, even when
// empty: the dev drive's watcher still subscribes to the host's world.jsonl.
pub(super) fn capture_hot_reload_sources(
    sources: HotReloadSources,
    texture_name_to_slot: HashMap<AssetId, usize>,
) -> (HotReloadSources, TextureNameSlots) {
    tracing::info!(
        "asset hot-reload: captured {} file-backed texture source(s), {} \
         ColorLut source(s), {} EnvironmentMap source(s), {} Mesh \
         source(s), {} SkinnedMesh source(s), {} ProceduralMesh source(s), \
         {} Shader source(s), and {} SdfVolume field(s)",
        sources.map.len(),
        usize::from(sources.color_lut.is_some()),
        usize::from(sources.environment_map.is_some()),
        sources.meshes.len(),
        sources.skinned_meshes.len(),
        sources.procedural_meshes.len(),
        sources.shaders.len(),
        sources.sdf_fields.len(),
    );
    (sources, TextureNameSlots(texture_name_to_slot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::render_types::{DrawIndex, SkinnedIndex};
    use concinnity_core::render::world_load::source_maps::{
        MeshSourceEntry, ProceduralMeshSourceEntry, SkinnedMeshSourceEntry,
    };

    fn mesh_entry(source: &str) -> MeshSourceEntry {
        MeshSourceEntry {
            source: source.to_string(),
            primitive_index: 0,
            lod_levels: 1,
            lod_distances: Vec::new(),
            draw_indices: vec![DrawIndex(0)],
        }
    }

    fn skinned_entry(source: &str) -> SkinnedMeshSourceEntry {
        SkinnedMeshSourceEntry {
            source: source.to_string(),
            skin_index: 0,
            skinned_index: SkinnedIndex(0),
            vertex_base: 0,
            vertex_count: 3,
            index_count: 3,
            joint_count: 1,
        }
    }

    fn dirs(paths: &[PathBuf]) -> Vec<String> {
        paths.iter().map(|p| p.display().to_string()).collect()
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
        shared.draw_indices = vec![DrawIndex(3), DrawIndex(9), DrawIndex(12)];
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
            vec![DrawIndex(3), DrawIndex(9), DrawIndex(12)],
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

    // The handoff bundle defaults to nothing captured, which is the `cn run`
    // shape: no watcher, no reloadable sources.
    #[test]
    fn bundle_defaults_to_nothing_captured() {
        let sources = HotReloadSources::default();
        assert!(sources.map.is_empty());
        assert!(sources.meshes.is_empty());
        assert!(sources.skinned_meshes.is_empty());
        assert!(sources.procedural_meshes.is_empty());
        assert!(sources.shaders.is_empty());
        assert!(sources.color_lut.is_none());
        assert!(sources.environment_map.is_none());
        assert!(sources.is_empty());
    }

    // Any one captured catalog is enough to make the bundle worth parking.
    #[test]
    fn bundle_with_any_single_capture_is_not_empty() {
        let populated: [fn(&mut HotReloadSources); 7] = [
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
                    draw_indices: vec![DrawIndex(0)],
                })
            },
            |s| {
                s.shaders
                    .entries
                    .push(super::super::shader_sources::ShaderSourceEntry {
                        id: AssetId(1),
                        name: "scene".to_string(),
                        bucket: 0,
                        files: Vec::new(),
                    })
            },
        ];
        for (i, populate) in populated.iter().enumerate() {
            let mut sources = HotReloadSources::default();
            populate(&mut sources);
            assert!(!sources.is_empty(), "field {i} alone is a capture");
        }
    }

    // An empty bundle still parks an empty catalog plus the texture-name map.
    #[test]
    fn capturing_nothing_parks_an_empty_catalog() {
        let names = HashMap::from([(AssetId(7), 0)]);
        let (sources, slots) = capture_hot_reload_sources(HotReloadSources::default(), names);
        assert!(sources.is_empty());
        assert_eq!(slots.0, HashMap::from([(AssetId(7), 0)]));
    }
}
