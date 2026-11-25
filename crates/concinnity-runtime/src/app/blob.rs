//! Where the blob file format meets a world: where a shipped app's compiled
//! world sits ([`BlobSource`]), and the decode of its primary blob into world
//! content. The reader and the lazy payload residency are
//! `concinnity_host::store::blob`'s.

use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{ComponentAsset, World};
use concinnity_engine::error::AssetError;
use concinnity_host::store::blob::{
    BlobData, BlobLoadError, BlobMeta, ResourceRecord, WorldManifest,
};
use std::path::{Path, PathBuf};

use crate::app::startup_error::StartupError;

/// Why a world could not be loaded from its blob files.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WorldLoadError {
    /// A blob file could not be read.
    #[error(transparent)]
    Blob(#[from] BlobLoadError),
    /// A record in the blob did not reconstruct its component.
    #[error(transparent)]
    Asset(#[from] AssetError),
}

/// A world that reads its compiled payloads from `blob`. The world names the
/// payload store only through its access seam, so this is where the blob file
/// format meets it.
pub fn world_from(blob: BlobData) -> World {
    World::from_payloads(Box::new(blob))
}

// Load the primary blob, resolve every stored def to a `ComponentAsset`, and
// return the resource stream, the world manifest, and `BlobData` alongside
// them. The blob carries a component stream (systems are internal and
// constructed at runtime), a resource stream (compiled resources addressed by
// a per-kind handle, which the caller loads into per-kind tables), and the
// manifest summarizing both (the caller pre-sizes ECS columns from its
// per-type counts).
//
// Each component that has a compiled payload carries its `PayloadLocator`
// injected into it (see `ComponentAsset::inject_locator`). Only blob 0's payload
// section is read into memory by `load_raw_at`; overflow blobs are read from disk
// lazily on first access.
// A loaded component paired with its def's name id, so the caller can index
// the entity it mints for it (the world's name -> entity map).
type NamedComponent = (Option<AssetId>, ComponentAsset);

// The decoded primary blob: resolved components, the resource stream, the
// baked per-scene groups, the physics reservation, the manifest, and the lazy
// payload reader.
pub(crate) struct LoadedBlob {
    pub(crate) components: Vec<NamedComponent>,
    pub(crate) resources: Vec<ResourceRecord>,
    pub(crate) scene_groups: Vec<concinnity_engine::blob::SceneGroup>,
    pub(crate) mesh_bounds: Vec<concinnity_engine::blob::MeshBoundsRecord>,
    pub(crate) physics_budget: Option<concinnity_engine::blob::PhysicsBudgetRecord>,
    pub(crate) manifest: WorldManifest,
    pub(crate) blob: BlobData,
}

// `load` against a primary blob file named directly, rather than the
// state root's `data/` layout. Overflow blobs are its siblings by index.
pub(crate) fn load_at(primary: &Path) -> Result<LoadedBlob, WorldLoadError> {
    resolve(concinnity_host::store::blob::load_raw_at(primary)?)
}

fn resolve((meta, blob_data): (BlobMeta, BlobData)) -> Result<LoadedBlob, WorldLoadError> {
    let components = meta
        .defs
        .iter()
        .map(|def| {
            // Every record is baked: the bytes are the serialized runtime
            // component (cook already ran the asset -> component translation).
            let mut component = def.component()?;
            if let Some(locator) = &def.payload {
                component.inject_locator(locator.clone());
            }
            Ok((def.name, component))
        })
        .collect::<Result<Vec<_>, AssetError>>()?;

    Ok(LoadedBlob {
        components,
        resources: meta.resources,
        scene_groups: meta.scene_groups,
        mesh_bounds: meta.mesh_bounds,
        physics_budget: meta.physics_budget,
        manifest: meta.manifest,
        blob: blob_data,
    })
}

/// Where an app's compiled world sits: the `data/` directory an export ships,
/// or whatever a positional argument names. Both forms make the same file
/// blob 0; they differ in whether the world is allowed to spill into overflow
/// payload blobs, which are always siblings named by index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlobSource {
    /// A directory holding blob 0 plus any overflow blobs beside it.
    Directory(PathBuf),
    /// A single self-contained blob file named as an argument. A world that
    /// needs overflow blobs is refused rather than half-loaded, since its
    /// siblings would be read from whatever directory the file sits in.
    File(PathBuf),
}

impl BlobSource {
    /// Classify `path` as a world: a directory holds blob 0, a file is blob 0
    /// itself. `None` when nothing is there, which is the same answer for a
    /// missing `data/` and a mistyped argument.
    pub fn classify(path: &Path) -> Option<Self> {
        if path.is_dir() {
            return Some(Self::Directory(path.to_path_buf()));
        }
        if path.is_file() {
            return Some(Self::File(path.to_path_buf()));
        }
        None
    }

    // The primary blob file: the named file itself, or blob 0 in the directory.
    pub(crate) fn primary(&self) -> PathBuf {
        match self {
            Self::Directory(dir) => dir.join("0"),
            Self::File(file) => file.clone(),
        }
    }

    // The refusal a single-file source owes a world that spans more blobs.
    pub(crate) fn check_span(&self, max_blob_index: u32) -> Option<StartupError> {
        match self {
            Self::File(file) if max_blob_index > 0 => Some(StartupError::OverflowUnsupported {
                blob: file.clone(),
                needed: max_blob_index,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A `data/` directory is the overflow-capable layout; a blob file named as
    // an argument is one self-contained blob.
    #[test]
    fn a_data_directory_and_a_blob_file_classify_as_their_own_forms() {
        let tmp = tempfile::tempdir().unwrap();

        let dir = tmp.path().join("data");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("0"), b"blob").unwrap();
        assert_eq!(
            BlobSource::classify(&dir),
            Some(BlobSource::Directory(dir.clone()))
        );

        let single = tmp.path().join("world.blob");
        std::fs::write(&single, b"blob").unwrap();
        assert_eq!(
            BlobSource::classify(&single),
            Some(BlobSource::File(single.clone()))
        );
    }

    // Nothing there is not an empty world: the caller turns this into a
    // not-found naming the path it looked at.
    #[test]
    fn a_missing_path_classifies_as_no_world() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(BlobSource::classify(&tmp.path().join("data")), None);
    }

    // Both forms make the same file blob 0: the file itself, or `0` inside the
    // directory. This is what lets one runtime entry point serve both.
    #[test]
    fn each_blob_source_names_the_same_primary_file() {
        let dir = PathBuf::from("/apps/MyGame/data");
        assert_eq!(
            BlobSource::Directory(dir.clone()).primary(),
            dir.join("0"),
            "a directory holds blob 0"
        );
        let file = PathBuf::from("/worlds/world.blob");
        assert_eq!(
            BlobSource::File(file.clone()).primary(),
            file,
            "a single file is blob 0"
        );
    }

    // Overflow blobs are siblings named by index, so only the directory form
    // has somewhere to hold them. A single file whose world spans more is
    // refused rather than half-loaded.
    #[test]
    fn a_single_file_source_refuses_a_world_that_overflows() {
        let path = PathBuf::from("/worlds/world.blob");
        let file = BlobSource::File(path.clone());
        assert!(file.check_span(0).is_none());
        assert!(matches!(
            file.check_span(2),
            Some(StartupError::OverflowUnsupported { blob, needed: 2 }) if blob == path
        ));
        let dir = BlobSource::Directory(PathBuf::from("/apps/MyGame/data"));
        assert!(dir.check_span(0).is_none());
        assert!(dir.check_span(7).is_none());
    }
}
