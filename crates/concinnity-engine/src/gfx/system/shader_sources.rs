//! The hot-reload catalog of every world `Shader`'s files, captured at
//! `GraphicsSystem::init` under hot-reload capture and carried to the dev
//! tooling inside [`super::hot_reload_sources::HotReloadSources`]. Plain data:
//! the watcher and the recompile that consume it live in
//! `concinnity_dev::debug::hot_reload`.

use concinnity_core::components::{Shader, ShaderStage};
use concinnity_core::ecs::asset_id::AssetId;
use std::path::PathBuf;

/// One of a Shader's files: which hook it defines and the resolved on-disk path
/// the build read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderFile {
    /// Which of the Shader's files this is.
    pub stage: ShaderStage,
    /// Resolved on-disk path. Stored resolved (not as declared) so the watcher
    /// can subscribe to a real parent directory even when the declaration used
    /// a bare filename.
    pub resolved_path: String,
}

/// One world Shader as a reload entry: its identity, its shader bucket, and
/// the files it compiles from.
#[derive(Debug, Clone)]
pub struct ShaderSourceEntry {
    /// The Shader's asset id, which keys a reload request.
    pub id: AssetId,
    /// The Shader's asset name, for diagnostics and the compile.
    pub name: String,
    /// The shader bucket its pipeline occupies; 0 is the world default.
    pub bucket: u32,
    /// The Shader's files, the fragment always among them.
    pub files: Vec<ShaderFile>,
}

impl ShaderSourceEntry {
    /// The resolved path of `stage`'s file, if the Shader declares one.
    pub fn path(&self, stage: ShaderStage) -> Option<&str> {
        self.files
            .iter()
            .find(|f| f.stage == stage)
            .map(|f| f.resolved_path.as_str())
    }
}

/// Catalog of every world Shader the renderer can hot-reload.
#[derive(Debug, Clone, Default)]
pub struct ShaderSourceMap {
    /// One entry per declared Shader, in bucket order.
    pub entries: Vec<ShaderSourceEntry>,
}

impl ShaderSourceMap {
    /// Build the catalog from the world's Shaders in bucket order, each with
    /// its asset id. `resolve` maps a declared path to the on-disk one and
    /// `name_of` names an id. A Shader with no id cannot be addressed by a
    /// reload request, so it is left out.
    pub fn build<'a>(
        shaders: impl IntoIterator<Item = (Option<AssetId>, &'a Shader)>,
        resolve: impl Fn(&str) -> String,
        name_of: impl Fn(AssetId) -> Option<String>,
    ) -> Self {
        let entries = shaders
            .into_iter()
            .enumerate()
            .filter_map(|(bucket, (id, shader))| {
                let id = id?;
                let files = [ShaderStage::Vertex, ShaderStage::Fragment]
                    .into_iter()
                    .filter_map(|stage| {
                        Some(ShaderFile {
                            stage,
                            resolved_path: resolve(shader.stage(stage)?),
                        })
                    })
                    .collect();
                Some(ShaderSourceEntry {
                    id,
                    name: name_of(id).unwrap_or_else(|| format!("shader bucket {bucket}")),
                    bucket: bucket as u32,
                    files,
                })
            })
            .collect();
        Self { entries }
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Shaders in the catalog.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The entry for the Shader `id`.
    pub fn get(&self, id: AssetId) -> Option<&ShaderSourceEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Every file path in the catalog with the Shader that reads it. A file
    /// two Shaders share appears once per Shader.
    pub fn files(&self) -> impl Iterator<Item = (&str, AssetId)> {
        self.entries.iter().flat_map(|e| {
            e.files
                .iter()
                .map(move |f| (f.resolved_path.as_str(), e.id))
        })
    }

    /// Directories the watcher must subscribe to for these sources.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        super::hot_reload_sources::watch_dirs_of(self.files().map(|(path, _)| path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader(fragment: &str, vertex: Option<&str>) -> Shader {
        Shader {
            fragment: fragment.to_string(),
            vertex: vertex.map(str::to_string),
            locator: None,
        }
    }

    fn catalog(shaders: &[(Option<AssetId>, Shader)]) -> ShaderSourceMap {
        ShaderSourceMap::build(
            shaders.iter().map(|(id, s)| (*id, s)),
            |p| format!("assets/{p}"),
            |id| Some(format!("shader{}", id.0)),
        )
    }

    // Every Shader gets an entry carrying its bucket, not just the default.
    #[test]
    fn every_shader_is_cataloged_with_its_bucket() {
        let map = catalog(&[
            (Some(AssetId(10)), shader("lit.hlsl", None)),
            (Some(AssetId(11)), shader("water.hlsl", Some("waves.hlsl"))),
        ]);
        assert_eq!(map.len(), 2);
        let default = map.get(AssetId(10)).unwrap();
        assert_eq!((default.bucket, default.name.as_str()), (0, "shader10"));
        assert_eq!(default.path(ShaderStage::Fragment), Some("assets/lit.hlsl"));
        assert_eq!(default.path(ShaderStage::Vertex), None);
        let water = map.get(AssetId(11)).unwrap();
        assert_eq!(water.bucket, 1);
        assert_eq!(water.path(ShaderStage::Vertex), Some("assets/waves.hlsl"));
        assert_eq!(water.path(ShaderStage::Fragment), Some("assets/water.hlsl"));
    }

    // A file two Shaders share is listed once per Shader reading it.
    #[test]
    fn a_shared_file_is_listed_for_every_shader_reading_it() {
        let map = catalog(&[
            (Some(AssetId(1)), shader("lit.hlsl", Some("sway.hlsl"))),
            (Some(AssetId(2)), shader("reeds.hlsl", Some("sway.hlsl"))),
            (Some(AssetId(3)), shader("rock.hlsl", None)),
        ]);
        let readers: Vec<AssetId> = map
            .files()
            .filter(|(path, _)| *path == "assets/sway.hlsl")
            .map(|(_, id)| id)
            .collect();
        assert_eq!(readers, [AssetId(1), AssetId(2)]);
        assert_eq!(map.files().count(), 5);
    }

    // A Shader without an id is skipped, but the buckets after it keep their
    // positions.
    #[test]
    fn an_anonymous_shader_is_skipped_without_shifting_buckets() {
        let map = catalog(&[
            (None, shader("lit.hlsl", None)),
            (Some(AssetId(5)), shader("water.hlsl", None)),
        ]);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(AssetId(5)).unwrap().bucket, 1);
    }

    // An unnamed id still gets a readable name.
    #[test]
    fn an_unnamed_shader_is_named_by_its_bucket() {
        let map = ShaderSourceMap::build(
            [(Some(AssetId(4)), &shader("lit.hlsl", None))],
            str::to_string,
            |_| None,
        );
        assert_eq!(map.entries[0].name, "shader bucket 0");
    }

    // The watcher subscribes to each parent directory once, shared files
    // included.
    #[test]
    fn watch_dirs_are_the_unique_parents_of_every_file() {
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
        let dirs: Vec<String> = map
            .watch_dirs()
            .iter()
            .map(|d| d.to_string_lossy().into_owned())
            .collect();
        assert_eq!(dirs, ["shaders", "shaders/surface"]);
    }

    #[test]
    fn an_empty_catalog_watches_nothing() {
        let map = ShaderSourceMap::default();
        assert!(map.is_empty());
        assert!(map.watch_dirs().is_empty());
        assert!(map.get(AssetId(1)).is_none());
    }
}
