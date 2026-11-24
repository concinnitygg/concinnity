//! The directories the watcher subscribes to for each captured catalog: every
//! unique parent directory of its sources, each resolved under the build's
//! root, so a file named with no folder is watched in the root itself.

use concinnity_engine::render::world_load::source_maps::{
    FileSourced, SdfFieldMap, ShaderSourceMap, SourceMap,
};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

// Every unique parent directory across `paths`, each resolved under `root`,
// sorted. A parent spelled as nothing is the working directory, `.`.
fn watch_dirs_of<'a>(paths: impl Iterator<Item = &'a str>, root: &Path) -> Vec<PathBuf> {
    let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
    for path in paths {
        let path = concinnity_host::store::source::resolve(path, Some(root));
        let parent = path.parent().unwrap_or_else(|| Path::new(""));
        dirs.insert(if parent.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            parent.to_path_buf()
        });
    }
    dirs.into_iter().collect()
}

/// The directories a watcher subscribes to for a catalog's file sources.
pub(super) trait WatchDirs {
    /// Every unique parent directory of the catalog's sources, resolved under
    /// the build's `root`.
    fn watch_dirs(&self, root: &Path) -> Vec<PathBuf>;
}

impl<E: FileSourced> WatchDirs for SourceMap<E> {
    fn watch_dirs(&self, root: &Path) -> Vec<PathBuf> {
        watch_dirs_of(self.entries.iter().map(FileSourced::source), root)
    }
}

impl WatchDirs for ShaderSourceMap {
    fn watch_dirs(&self, root: &Path) -> Vec<PathBuf> {
        watch_dirs_of(self.files().map(|(path, _)| path), root)
    }
}

impl WatchDirs for SdfFieldMap {
    fn watch_dirs(&self, root: &Path) -> Vec<PathBuf> {
        watch_dirs_of(self.files().map(|(path, _)| path), root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::{SdfVolume, Shader};
    use concinnity_engine::ecs::asset_id::AssetId;
    use concinnity_engine::gfx::render_types::{DrawIndex, SkinnedIndex};
    use concinnity_engine::render::backend::init::SdfVolumeSource;
    use concinnity_engine::render::world_load::source_maps::{
        MeshSourceEntry, MeshSourceMap, SkinnedMeshSourceEntry, SkinnedMeshSourceMap,
        TextureSourceMap,
    };

    // Each directory with `/` separators, so one expectation holds on every host.
    fn dirs(paths: &[PathBuf]) -> Vec<String> {
        paths
            .iter()
            .map(|p| p.display().to_string().replace('\\', "/"))
            .collect()
    }

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

    fn shader(fragment: &str, vertex: Option<&str>) -> Shader {
        Shader {
            fragment: fragment.to_string(),
            vertex: vertex.map(str::to_string),
            locator: None,
        }
    }

    fn volume(label: &str, field: &str) -> SdfVolumeSource {
        SdfVolumeSource {
            volume: SdfVolume {
                fragment_shader: field.to_string(),
                ..SdfVolume::default()
            },
            fragment_source: Vec::new(),
            label: label.to_string(),
        }
    }

    // Entries sharing a directory collapse to one subscription, sorted.
    #[test]
    fn texture_watch_dirs_dedups_shared_parents() {
        let mut map = TextureSourceMap::new();
        map.push_texture("assets/textures/wall.png".to_string(), 0, 0);
        map.push_texture("assets/textures/floor.png".to_string(), 0, 1);
        map.push_texture("assets/models/scene.glb".to_string(), 0, 2);
        assert_eq!(
            dirs(&map.watch_dirs(Path::new(""))),
            ["assets/models", "assets/textures"]
        );
    }

    // A file named with no folder sits in the root, which is watched.
    #[test]
    fn a_file_at_the_root_watches_the_root() {
        let mut map = TextureSourceMap::new();
        map.push_texture("wall.png".to_string(), 0, 0);
        assert_eq!(dirs(&map.watch_dirs(Path::new("/p"))), ["/p"]);
        assert_eq!(dirs(&map.watch_dirs(Path::new(""))), ["."]);
        map.push_texture("assets/floor.png".to_string(), 0, 1);
        assert_eq!(dirs(&map.watch_dirs(Path::new("/p"))), ["/p", "/p/assets"]);
    }

    #[test]
    fn mesh_and_skinned_maps_watch_their_parents() {
        let mut meshes = MeshSourceMap::new();
        meshes.entries.push(mesh_entry("assets/models/prop.glb"));
        meshes.entries.push(mesh_entry("assets/models/tree.glb"));
        meshes.entries.push(mesh_entry("bare.glb"));
        assert_eq!(
            dirs(&meshes.watch_dirs(Path::new(""))),
            [".", "assets/models"]
        );

        let mut skinned = SkinnedMeshSourceMap::new();
        skinned.entries.push(skinned_entry("assets/chars/fox.glb"));
        skinned.entries.push(skinned_entry("fox.glb"));
        assert_eq!(
            dirs(&skinned.watch_dirs(Path::new(""))),
            [".", "assets/chars"]
        );
    }

    // A file two Shaders share is watched once.
    #[test]
    fn shader_watch_dirs_are_the_unique_parents_of_every_file() {
        let map = ShaderSourceMap::build(
            [
                (
                    Some(AssetId(1)),
                    &shader("shaders/lit.hlsl", Some("shaders/sway.hlsl")),
                ),
                (
                    Some(AssetId(2)),
                    &shader("shaders/surface/reeds.hlsl", Some("shaders/sway.hlsl")),
                ),
                (Some(AssetId(3)), &shader("bare.hlsl", None)),
            ],
            str::to_string,
            |_| None,
        );
        assert_eq!(
            dirs(&map.watch_dirs(Path::new(""))),
            [".", "shaders", "shaders/surface"]
        );
        assert!(
            ShaderSourceMap::default()
                .watch_dirs(Path::new(""))
                .is_empty()
        );
    }

    // A field two volumes share is watched once.
    #[test]
    fn sdf_watch_dirs_are_the_unique_parents_of_every_field() {
        let map = SdfFieldMap::build(
            &[
                volume("cloud_a", "fields/cloud.hlsl"),
                volume("cloud_b", "fields/cloud.hlsl"),
                volume("blob", "fields/blob/blob.hlsl"),
            ],
            |raw| Some(raw.to_string()),
        );
        assert_eq!(
            dirs(&map.watch_dirs(Path::new(""))),
            ["fields", "fields/blob"]
        );
        assert!(SdfFieldMap::default().watch_dirs(Path::new("")).is_empty());
    }
}
