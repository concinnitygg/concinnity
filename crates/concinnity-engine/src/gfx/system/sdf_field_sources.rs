//! The hot-reload catalog of every raymarched `SdfVolume`'s field file,
//! captured at `GraphicsSystem::init` under hot-reload capture and carried to
//! the dev tooling inside [`super::hot_reload_sources::HotReloadSources`].
//! Plain data: the watcher and the recompile that consume it live in
//! `concinnity_dev::debug::hot_reload`.

use concinnity_core::render::backend_init::SdfVolumeSource;
use std::path::{Path, PathBuf};

/// One volume as a reload entry: its name, where the backend holds it, the
/// flags that decide which entries its field compiles, and the field's file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdfFieldEntry {
    /// The volume's asset name, for diagnostics and reports.
    pub name: String,
    /// The volume's position among the volumes the backend was built with,
    /// which is how a pipeline swap addresses it.
    pub volume: usize,
    /// Whether the volume is a participating medium.
    pub volumetric: bool,
    /// Whether the volume casts shadows.
    pub cast_shadows: bool,
    /// Resolved on-disk path of the field the build read. Stored resolved so
    /// the watcher can subscribe to a real parent directory.
    pub resolved_path: String,
}

/// Catalog of every `SdfVolume` whose field the renderer can hot-reload. A file
/// several volumes share appears once per volume.
#[derive(Debug, Clone, Default)]
pub struct SdfFieldMap {
    /// One entry per volume, in the order the backend was built with them.
    pub entries: Vec<SdfFieldEntry>,
}

impl SdfFieldMap {
    /// Build the catalog from the volumes the backend is built with. `resolve`
    /// maps a declared field path to the file that exists on disk; a volume
    /// whose field resolves to nothing has no file to watch and is left out.
    pub fn build(volumes: &[SdfVolumeSource], resolve: impl Fn(&str) -> Option<String>) -> Self {
        let entries = volumes
            .iter()
            .enumerate()
            .filter_map(|(volume, source)| {
                let resolved_path = resolve(&source.volume.fragment_shader)?;
                Some(SdfFieldEntry {
                    name: source.label.clone(),
                    volume,
                    volumetric: source.volume.volumetric,
                    cast_shadows: source.volume.cast_shadows,
                    resolved_path,
                })
            })
            .collect();
        Self { entries }
    }

    /// Build the catalog with each field resolved the way the cook found the
    /// file it compiled, under the build's asset root `assets_dir`.
    pub fn resolve(volumes: &[SdfVolumeSource], assets_dir: Option<&Path>) -> Self {
        Self::build(volumes, |raw| {
            concinnity_host::store::source::find_existing(raw, assets_dir)
        })
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Volumes in the catalog.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The entry for the volume named `name`.
    pub fn get(&self, name: &str) -> Option<&SdfFieldEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Every field path in the catalog with the volume that reads it.
    pub fn files(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries
            .iter()
            .map(|e| (e.resolved_path.as_str(), e.name.as_str()))
    }

    /// Directories the watcher must subscribe to for these fields.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        super::hot_reload_sources::watch_dirs_of(self.files().map(|(path, _)| path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::SdfVolume;

    fn source(label: &str, field: &str, volumetric: bool, cast_shadows: bool) -> SdfVolumeSource {
        SdfVolumeSource {
            volume: SdfVolume {
                fragment_shader: field.to_string(),
                volumetric,
                cast_shadows,
                ..SdfVolume::default()
            },
            fragment_source: Vec::new(),
            label: label.to_string(),
        }
    }

    // Each volume keeps its position among the backend's volumes, its flags and
    // its resolved field; a volume whose field resolves to nothing is skipped
    // without shifting the positions after it.
    #[test]
    fn every_resolvable_volume_is_cataloged_at_its_position() {
        let map = SdfFieldMap::build(
            &[
                source("blob", "shaders/blob.hlsl", false, true),
                source("lost", "shaders/lost.hlsl", false, false),
                source("cloud", "shaders/cloud.hlsl", true, false),
            ],
            |raw| (!raw.contains("lost")).then(|| format!("assets/{raw}")),
        );
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get("blob"),
            Some(&SdfFieldEntry {
                name: "blob".to_string(),
                volume: 0,
                volumetric: false,
                cast_shadows: true,
                resolved_path: "assets/shaders/blob.hlsl".to_string(),
            })
        );
        let cloud = map.get("cloud").unwrap();
        assert_eq!((cloud.volume, cloud.volumetric), (2, true));
        assert!(map.get("lost").is_none());
    }

    // A field two volumes share is listed once per volume, and watched once.
    #[test]
    fn a_shared_field_is_listed_for_every_volume_reading_it() {
        let map = SdfFieldMap::build(
            &[
                source("cloud_a", "fields/cloud.hlsl", true, false),
                source("cloud_b", "fields/cloud.hlsl", true, false),
                source("blob", "fields/blob/blob.hlsl", false, false),
            ],
            |raw| Some(raw.to_string()),
        );
        let readers: Vec<&str> = map
            .files()
            .filter(|(path, _)| *path == "fields/cloud.hlsl")
            .map(|(_, name)| name)
            .collect();
        assert_eq!(readers, ["cloud_a", "cloud_b"]);
        let dirs: Vec<String> = map
            .watch_dirs()
            .iter()
            .map(|d| d.to_string_lossy().into_owned())
            .collect();
        assert_eq!(dirs, ["fields", "fields/blob"]);
    }

    // A bare field name is found anywhere under the asset root, as the cook
    // finds it, so a field the build compiled from a nested directory is
    // watched there.
    #[test]
    fn a_field_resolves_where_the_cook_found_it() {
        let assets = tempfile::tempdir().unwrap();
        let fields = assets.path().join("fields");
        std::fs::create_dir_all(&fields).unwrap();
        std::fs::write(fields.join("blob.hlsl"), "").unwrap();
        let map = SdfFieldMap::resolve(
            &[
                source("blob", "blob.hlsl", false, true),
                source("lost", "cn_no_such_field.hlsl", false, false),
            ],
            Some(assets.path()),
        );
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get("blob").unwrap().resolved_path,
            fields.join("blob.hlsl").to_string_lossy()
        );
    }

    #[test]
    fn an_empty_catalog_watches_nothing() {
        let map = SdfFieldMap::default();
        assert!(map.is_empty());
        assert!(map.watch_dirs().is_empty());
        assert!(map.get("blob").is_none());
    }
}
