//! The blob WRITE side (build output): pack compiled payloads + the def table
//! into .cnb files and emit world-lock.json. The byte format itself -- the
//! header, the record schema, and the `encode_cnb` image builder -- is owned by
//! the I/O-free concinnity-blob crate; this file owns the packing POLICY
//! (payload distribution across overflow blobs, the size ceiling), the lock, and
//! the writes themselves.

use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use concinnity_core::blob::{
    MeshBoundsRecord, PhysicsBudgetRecord, SceneGroup, WorldManifest, encode_cnb, payload_section,
};
use concinnity_core::ecs::{BlobAssetDef, BlobMeta, PayloadLocator, ResourceRecord};

// `blob_path` is shared with the read side in `concinnity_host::store`: the
// build names blob 0 through it, and readers resolve the same layout.
pub(crate) use concinnity_host::store::blob::blob_path;
#[cfg(test)]
pub(crate) use concinnity_host::store::blob::read_cnb;

use serde::{Deserialize, Serialize};

/// The build record written beside the blobs. Provenance metadata, not part of
/// the .cnb format, so it lives here rather than in concinnity-blob.
pub const LOCK_PATH: &str = "world-lock.json";

/// Per-blob entry in the lock file
#[derive(Debug, Serialize, Deserialize)]
pub struct BlobEntry {
    /// The blob file's path, relative to the build output.
    pub path: String,
    /// sha-256 of the blob file.
    pub checksum: String,
    /// Bytes the blob's payload section occupies.
    pub payload_bytes: u64,
}

/// The resolved build record written alongside the binary blobs
/// Human-readable; owned by the build, not the user
#[derive(Debug, Serialize, Deserialize)]
pub struct BlobLock {
    /// Engine version the blob was built with; injected defaults come from the
    /// engine, so their content can change across versions.
    pub engine_version: String,
    /// When the build ran, as an RFC 3339 timestamp.
    pub built_at: String,
    /// One entry per blob file written.
    pub blobs: Vec<BlobEntry>,
    /// One entry per asset compiled into the def table.
    pub assets: Vec<LockedAsset>,
    /// Assets compiled into the blob's resource stream (addressed by a per-kind
    /// handle) rather than the component def table.
    #[serde(default)]
    pub resources: Vec<LockedResource>,
    /// Assets the build added that have no world.jsonl line (companions and
    /// engine defaults). Each entry carries its full args so it can be copied
    /// into world.jsonl verbatim as an override.
    pub injected: Vec<LockedInjection>,
    /// Generated assets the world declares its own copy of. The copy won and the
    /// generated entry was dropped, so the source file no longer drives these;
    /// recorded to make that override visible rather than silent.
    #[serde(default)]
    pub shadowed: Vec<LockedShadow>,
}

/// One asset as recorded in the lock file
#[derive(Debug, Serialize, Deserialize)]
pub struct LockedAsset {
    /// The asset's declared name.
    pub name: String,
    /// The dense interned id the build assigned this name. Lets a process that
    /// loads the prebuilt blobs (the editor booting without an in-process cook)
    /// rebuild the name table exactly as the build interned it.
    #[serde(default)]
    pub id: Option<u32>,
    /// The asset's registry type name.
    pub kind: String,
    /// The component type's registry tag.
    pub discriminant: u8,
    /// sha-256 of the asset's serialized args_bytes
    pub args_hash: String,
    /// which blob holds this asset's payload, if any
    pub payload_blob: Option<u32>,
}

/// One injected asset as recorded in the lock file
#[derive(Debug, Serialize, Deserialize)]
pub struct LockedInjection {
    /// The injected asset's name.
    pub name: String,
    #[serde(rename = "type")]
    /// The asset's registry type name.
    pub asset_type: String,
    /// The args the injection supplied.
    pub args: serde_json::Value,
    /// Which expander injected it.
    pub injected_by: String,
}

/// One generated asset the world overrides with its own copy. Carries no args:
/// the copy that won is the world.jsonl line of the same name.
#[derive(Debug, Serialize, Deserialize)]
pub struct LockedShadow {
    /// The shadowed asset's name.
    pub name: String,
    #[serde(rename = "type")]
    /// The asset's registry type name.
    pub asset_type: String,
    /// Which expander generated the copy the world overrode.
    pub generated_by: String,
}

/// One resource-stream asset as recorded in the lock file. Resource assets have
/// left the component def table, so they are recorded with their per-kind handle
/// instead of a component discriminant. `payload_blob` is None for a data
/// resource (bytes ride inline in the record).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LockedResource {
    /// The resource's declared name.
    pub name: String,
    /// The dense interned id the build assigned this name (the same id space
    /// as `LockedAsset.id`; `handle` is the per-kind index).
    #[serde(default)]
    pub id: Option<u32>,
    /// The resource kind's name.
    pub kind: String,
    /// The resource's dense per-kind handle.
    pub handle: u32,
    /// sha-256 of the asset's authored args JSON
    pub args_hash: String,
    /// which blob holds this resource's payload, if any
    pub payload_blob: Option<u32>,
    /// Dev source info mirrored from the build's hot-reload catalogues, so a
    /// blob boot can reconstruct them without the asset's args (a SceneImport
    /// product has none the boot can see). Present for every Texture / Mesh
    /// resource; an empty `source` means nothing to watch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture_source: Option<LockedTextureSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The mesh's re-import inputs, for a mesh resource.
    pub mesh_source: Option<LockedMeshSource>,
}

/// A texture resource's watchable file source as recorded in the lock file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LockedTextureSource {
    /// Authored source path; empty when there is nothing to watch.
    pub source: String,
    /// Index of the image within the source document.
    pub image_index: u32,
}

/// A mesh resource's re-import inputs as recorded in the lock file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LockedMeshSource {
    /// Authored source path; empty when there is nothing to watch.
    pub source: String,
    /// Index of the primitive within the source document.
    pub primitive_index: u32,
    /// How many LODs the mesh declares, including LOD0.
    pub lod_levels: u32,
    /// Camera distance at which each LOD past 0 takes over.
    pub lod_distances: Vec<f32>,
}

/// The result of a build pack: the blobs written and the path of each
pub struct PackResult {
    /// Path of each blob file written.
    pub blob_paths: Vec<String>,
}

// Encode one blob image and write it out. The format crate performs no I/O, so
// the write lives here alongside the rest of the build's file output.
fn write_cnb(meta: &BlobMeta, payload: &[u8], path: &str) -> std::io::Result<()> {
    let image = encode_cnb(concinnity_core::SCHEMA_VERSION, meta, payload)
        .map_err(|e| std::io::Error::other(format!("encoding {}: {:?}", path, e)))?;
    fs::write(path, image)
}

// The record streams one build produces, packed into blob 0's metadata block.
// Grouped because they travel together: every one of them is derived from the
// same compiled asset list and is written exactly once.
pub(crate) struct BlobStreams<'a> {
    pub(crate) defs: &'a [BlobAssetDef],
    pub(crate) resources: &'a [ResourceRecord],
    pub(crate) scene_groups: &'a [SceneGroup],
    pub(crate) mesh_bounds: &'a [MeshBoundsRecord],
    pub(crate) physics_budget: Option<PhysicsBudgetRecord>,
}

// Pack the metadata (component defs + resource records) and their payloads into
// one or more blobs. The full metadata rides in blob 0; overflow blobs carry an
// empty metadata section and pure payload bytes.
pub(crate) fn write_blobs(
    streams: BlobStreams<'_>,
    blob_payloads: &[Vec<u8>],
    primary: &Path,
) -> std::io::Result<PackResult> {
    if let Some(dir) = primary.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    // Blob 0 is the file named; every overflow blob is its sibling named by
    // index, which is the layout the runtime reads back.
    let blob_file = |index: u32| {
        let path = if index == 0 {
            primary.to_path_buf()
        } else {
            primary
                .parent()
                .map_or_else(|| Path::new(".").to_path_buf(), Path::to_path_buf)
                .join(index.to_string())
        };
        path.to_string_lossy().into_owned()
    };

    // The manifest is derived from the very streams it summarizes, so the
    // shipped copy is consistent by construction; the runtime re-derives it and
    // debug-asserts it matches.
    let primary_meta = || BlobMeta {
        defs: streams.defs.to_vec(),
        resources: streams.resources.to_vec(),
        manifest: WorldManifest::from_records(streams.defs, streams.resources),
        scene_groups: streams.scene_groups.to_vec(),
        mesh_bounds: streams.mesh_bounds.to_vec(),
        physics_budget: streams.physics_budget,
    };

    let mut blob_paths = Vec::new();

    for (idx, payload) in blob_payloads.iter().enumerate() {
        let path = blob_file(idx as u32);
        let meta = if idx == 0 {
            primary_meta()
        } else {
            BlobMeta::default()
        };
        write_cnb(&meta, payload, &path)?;
        blob_paths.push(path);
    }

    if blob_payloads.is_empty() {
        let primary = blob_file(0);
        write_cnb(&primary_meta(), &[], &primary)?;
        blob_paths.push(primary);
    }

    // Remove stale overflow blobs left by a previous, larger build so the
    // directory matches the manifest exactly.
    let mut stale = blob_paths.len() as u32;
    while fs::remove_file(blob_file(stale)).is_ok() {
        stale += 1;
    }

    Ok(PackResult { blob_paths })
}

// Size at which the packer rolls over to a fresh blob.
pub(crate) const DEFAULT_MAX_BLOB_BYTES: u64 = 1 << 30;

// PayloadPacker (build step)
pub(crate) struct PayloadPacker {
    max_blob_bytes: u64,
    blobs: Vec<Vec<u8>>,
    current_blob: u32,
    current_offset: u64,
    // A group boundary was requested; the next push starts a fresh blob.
    pending_group: bool,
}

impl PayloadPacker {
    pub(crate) fn new(max_blob_bytes: u64) -> Self {
        Self {
            max_blob_bytes,
            blobs: vec![Vec::new()],
            current_blob: 0,
            current_offset: 0,
            pending_group: false,
        }
    }

    // Start a payload group: the next push lands at the start of a blob that
    // holds no earlier content and is never blob 0 (whose payload section is
    // read eagerly at startup), so the group's payloads are contiguous and
    // separately loadable. A group with no pushes produces no blob.
    pub(crate) fn start_group(&mut self) {
        self.pending_group = true;
    }

    pub(crate) fn push(&mut self, data: &[u8]) -> PayloadLocator {
        let len = data.len() as u64;

        let group_roll = self.pending_group && (self.current_offset > 0 || self.current_blob == 0);
        let size_roll = self.current_offset > 0 && self.current_offset + len > self.max_blob_bytes;
        if group_roll || size_roll {
            self.blobs.push(Vec::new());
            self.current_blob += 1;
            self.current_offset = 0;
        }
        self.pending_group = false;

        let offset = self.current_offset;
        self.blobs[self.current_blob as usize].extend_from_slice(data);
        self.current_offset += len;

        PayloadLocator {
            blob_index: self.current_blob,
            offset,
            len,
        }
    }

    pub(crate) fn finish(self) -> Vec<Vec<u8>> {
        self.blobs
    }
}

// Lock file
pub(crate) fn write_lock(
    named_defs: &[(&str, &BlobAssetDef)],
    resources: &[LockedResource],
    injected: &[crate::build_only::InjectedAsset],
    shadowed: &[crate::build_only::ShadowedAsset],
    blob_paths: &[String],
) -> std::io::Result<()> {
    let mut blobs = Vec::new();
    for path in blob_paths {
        let data = fs::read(path).unwrap_or_default();
        let payload_bytes = payload_section(&data).len() as u64;
        blobs.push(BlobEntry {
            path: path.clone(),
            checksum: checksum(&data),
            payload_bytes,
        });
    }

    let assets = named_defs
        .iter()
        .map(|(name, def)| LockedAsset {
            name: name.to_string(),
            id: def.name.map(|n| n.0),
            kind: format!("{:?}", def.kind),
            discriminant: def.discriminant,
            args_hash: checksum(&def.args_bytes),
            payload_blob: def.payload.as_ref().map(|p| p.blob_index),
        })
        .collect();

    let lock = BlobLock {
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        built_at: now_iso8601(),
        blobs,
        assets,
        resources: resources.to_vec(),
        injected: injected
            .iter()
            .map(|i| LockedInjection {
                name: i.name.clone(),
                asset_type: i.asset_type.clone(),
                args: i.args.clone(),
                injected_by: i.injected_by.to_string(),
            })
            .collect(),
        shadowed: shadowed
            .iter()
            .map(|s| LockedShadow {
                name: s.name.clone(),
                asset_type: s.asset_type.clone(),
                generated_by: s.generated_by.clone(),
            })
            .collect(),
    };

    fs::write(LOCK_PATH, serde_json::to_string_pretty(&lock)?)
}

// Helpers
pub(crate) fn checksum(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

fn now_iso8601() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .expect("time format")
}

// Shared harness for tests that drive the build's file output. Both writes
// target process-global locations -- the blobs go under the installed state
// root, the lock file is written relative to the working directory -- so the
// lock serialises them and the guards restore the process afterwards.
#[cfg(test)]
pub(crate) mod test_output {
    pub(crate) static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // Anchors the state root at a temp tree for the life of the guard, so
    // `write_blobs` lands its files there instead of under the cwd.
    pub(crate) struct StateDir(tempfile::TempDir);

    impl StateDir {
        pub(crate) fn new() -> Self {
            let dir = tempfile::tempdir().expect("tempdir");
            concinnity_host::store::paths::set_state_dir(dir.path());
            Self(dir)
        }

        pub(crate) fn data_dir(&self) -> std::path::PathBuf {
            self.0.path().join("data")
        }
    }

    impl Drop for StateDir {
        fn drop(&mut self) {
            concinnity_host::store::paths::clear_state_dir();
        }
    }

    // Clears the cwd-relative lock path when the test ends, however it ends.
    // A test may leave a directory there instead of a file to make the write
    // fail, so both are removed.
    pub(crate) struct LockFile;

    impl Drop for LockFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(super::LOCK_PATH);
            let _ = std::fs::remove_dir_all(super::LOCK_PATH);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{AssetKind, asset_id::AssetId};
    use test_output::{LockFile, StateDir};

    fn locator(blob_index: u32, offset: u64, len: u64) -> PayloadLocator {
        PayloadLocator {
            blob_index,
            offset,
            len,
        }
    }

    fn component_def(discriminant: u8, payload: Option<PayloadLocator>) -> BlobAssetDef {
        BlobAssetDef {
            name: Some(AssetId(discriminant as u32)),
            kind: AssetKind::Component,
            discriminant,
            args_bytes: vec![discriminant, 0xAA],
            payload,
        }
    }

    // The default primary blob path: blob 0 under the anchored state dir, which
    // is what the build writes when nothing names a file.
    fn primary() -> std::path::PathBuf {
        std::path::PathBuf::from(blob_path(0).expect("the scoped state dir is installed"))
    }

    // The record streams of a build that only has components and resources.
    fn streams<'a>(defs: &'a [BlobAssetDef], resources: &'a [ResourceRecord]) -> BlobStreams<'a> {
        BlobStreams {
            defs,
            resources,
            scene_groups: &[],
            mesh_bounds: &[],
            physics_budget: None,
        }
    }

    #[test]
    fn write_blobs_keeps_metadata_in_blob_zero_and_splits_payload_bytes() {
        let _guard = test_output::LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let state = StateDir::new();

        let defs = vec![
            component_def(3, Some(locator(0, 0, 3))),
            component_def(4, None),
        ];
        let resources = vec![ResourceRecord {
            resource_kind: 2,
            handle: 0,
            payload: Some(locator(1, 0, 4)),
            data_bytes: vec![9, 9],
        }];
        let payloads = vec![vec![1, 2, 3], vec![4, 5, 6, 7]];
        let data_dir = state.data_dir();
        let paths = write_blobs(streams(&defs, &resources), &payloads, &primary())
            .expect("write_blobs")
            .blob_paths;
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], data_dir.join("0").to_string_lossy());
        assert_eq!(paths[1], data_dir.join("1").to_string_lossy());

        // Blob 0 carries the whole metadata block plus its own payload section.
        let (meta, payload_start) = read_cnb(&paths[0]).expect("blob 0 parses");
        assert_eq!(meta.defs.len(), 2);
        assert_eq!(meta.defs[0].discriminant, 3);
        assert_eq!(meta.resources.len(), 1);
        assert_eq!(meta.resources[0].data_bytes, vec![9, 9]);
        assert_eq!(
            meta.manifest,
            WorldManifest::from_records(&defs, &resources),
            "the shipped manifest is derived from the streams it summarizes"
        );
        assert_eq!(&fs::read(&paths[0]).unwrap()[payload_start..], &[1, 2, 3]);

        // An overflow blob is payload only: no defs, no resources.
        let (overflow_meta, overflow_start) = read_cnb(&paths[1]).expect("blob 1 parses");
        assert!(overflow_meta.defs.is_empty());
        assert!(overflow_meta.resources.is_empty());
        assert_eq!(
            &fs::read(&paths[1]).unwrap()[overflow_start..],
            &[4, 5, 6, 7]
        );
    }

    // The physics reservation rides in blob 0's metadata beside the manifest,
    // so the runtime reads what cook counted without re-deriving it.
    #[test]
    fn write_blobs_ships_the_physics_budget_in_blob_zero() {
        let _guard = test_output::LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _state = StateDir::new();

        let defs = vec![component_def(3, None)];
        let budget = PhysicsBudgetRecord {
            fixed: 3,
            dynamic: 2,
            kinematic: 1,
            sensors: 1,
            joints: 2,
            anchors: 1,
            spawn_headroom: 8,
        };
        let paths = write_blobs(
            BlobStreams {
                physics_budget: Some(budget),
                ..streams(&defs, &[])
            },
            &[],
            &primary(),
        )
        .expect("write_blobs")
        .blob_paths;
        let (meta, _) = read_cnb(&paths[0]).expect("blob 0 parses");
        assert_eq!(meta.physics_budget, Some(budget));

        // A world with no physics ships no reservation at all.
        let paths = write_blobs(streams(&defs, &[]), &[], &primary())
            .expect("write_blobs")
            .blob_paths;
        let (meta, _) = read_cnb(&paths[0]).expect("blob 0 parses");
        assert_eq!(meta.physics_budget, None);
    }

    #[test]
    fn write_blobs_removes_stale_overflow_blobs_from_a_larger_build() {
        let _guard = test_output::LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _state = StateDir::new();

        let defs = vec![component_def(3, None)];
        let payloads = vec![vec![1], vec![2], vec![3]];
        let first = write_blobs(streams(&defs, &[]), &payloads, &primary())
            .expect("write_blobs")
            .blob_paths;
        assert_eq!(first.len(), 3);

        let second = write_blobs(streams(&defs, &[]), &[vec![1]], &primary())
            .expect("write_blobs")
            .blob_paths;
        assert_eq!(second.len(), 1);
        assert!(!std::path::Path::new(&first[1]).exists(), "stale blob 1");
        assert!(!std::path::Path::new(&first[2]).exists(), "stale blob 2");
    }

    #[test]
    fn write_blobs_without_payloads_still_writes_the_primary_blob() {
        let _guard = test_output::LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _state = StateDir::new();

        let defs = vec![component_def(5, None)];
        let paths = write_blobs(streams(&defs, &[]), &[], &primary())
            .expect("write_blobs")
            .blob_paths;
        assert_eq!(paths.len(), 1, "a payload-less world still ships blob 0");
        let (meta, payload_start) = read_cnb(&paths[0]).expect("blob 0 parses");
        assert_eq!(meta.defs.len(), 1);
        assert_eq!(fs::read(&paths[0]).unwrap().len(), payload_start);
    }

    // A blob path that cannot be written (here a directory sits where the file
    // belongs) fails the build rather than shipping a partial data set.
    #[test]
    fn write_blobs_surfaces_a_write_failure() {
        let _guard = test_output::LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let state = StateDir::new();
        fs::create_dir_all(state.data_dir().join("0")).expect("occupy blob 0");

        let result = write_blobs(streams(&[component_def(1, None)], &[]), &[], &primary());
        assert!(result.is_err(), "an unwritable blob path must fail");
    }

    // A named primary blob owns its own directory: blob 0 is the file asked
    // for, overflow blobs are its siblings by index, and the directory is
    // created if it does not exist yet.
    #[test]
    fn write_blobs_honors_a_named_primary_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let primary = dir.path().join("out").join("world");
        let defs = vec![component_def(3, None)];

        let paths = write_blobs(streams(&defs, &[]), &[vec![1], vec![2]], &primary)
            .expect("write_blobs")
            .blob_paths;

        assert_eq!(paths[0], primary.to_string_lossy());
        assert_eq!(paths[1], primary.with_file_name("1").to_string_lossy());
        let (meta, _) = read_cnb(&paths[0]).expect("the named blob parses");
        assert_eq!(meta.defs.len(), 1);
    }

    #[test]
    fn write_lock_records_blob_checksums_payload_sizes_and_provenance() {
        let _guard = test_output::LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _state = StateDir::new();
        let _lock_file = LockFile;

        let defs = vec![
            component_def(3, Some(locator(0, 0, 3))),
            component_def(4, None),
        ];
        let paths = write_blobs(streams(&defs, &[]), &[vec![1, 2, 3]], &primary())
            .expect("write_blobs")
            .blob_paths;
        let named: Vec<(&str, &BlobAssetDef)> = vec![("floor", &defs[0]), ("wall", &defs[1])];
        let resources = vec![LockedResource {
            name: "clip".to_string(),
            id: Some(2),
            kind: "AudioClip".to_string(),
            handle: 2,
            args_hash: "ff".to_string(),
            payload_blob: None,
            ..Default::default()
        }];
        let injected = vec![crate::build_only::InjectedAsset {
            name: "debug_hud".to_string(),
            asset_type: "DebugHud".to_string(),
            args: serde_json::json!({"enabled": true}),
            injected_by: "engine",
        }];
        let shadowed = vec![crate::build_only::ShadowedAsset {
            name: "bistro_mat_wood".to_string(),
            asset_type: "Material".to_string(),
            generated_by: "bistro".to_string(),
            args: serde_json::json!({}),
        }];

        write_lock(&named, &resources, &injected, &shadowed, &paths).expect("write_lock");
        let blob_bytes = fs::read(&paths[0]).expect("blob 0 readable");
        let written = fs::read_to_string(LOCK_PATH).expect("lock written to the working directory");

        let lock: BlobLock = serde_json::from_str(&written).expect("lock is valid json");
        assert_eq!(lock.engine_version, env!("CARGO_PKG_VERSION"));
        assert!(OffsetDateTime::parse(&lock.built_at, &Rfc3339).is_ok());

        assert_eq!(lock.blobs.len(), 1);
        assert_eq!(lock.blobs[0].path, paths[0]);
        assert_eq!(lock.blobs[0].checksum, checksum(&blob_bytes));
        assert_eq!(
            lock.blobs[0].payload_bytes, 3,
            "payload bytes exclude the header and the metadata section"
        );

        assert_eq!(lock.assets.len(), 2);
        assert_eq!(lock.assets[0].name, "floor");
        assert_eq!(lock.assets[0].kind, "Component");
        assert_eq!(lock.assets[0].discriminant, 3);
        assert_eq!(lock.assets[0].args_hash, checksum(&defs[0].args_bytes));
        assert_eq!(lock.assets[0].payload_blob, Some(0));
        assert_eq!(lock.assets[1].name, "wall");
        assert_eq!(lock.assets[1].payload_blob, None);

        assert_eq!(lock.resources[0].name, "clip");
        assert_eq!(lock.resources[0].handle, 2);
        assert_eq!(lock.injected[0].name, "debug_hud");
        assert_eq!(lock.injected[0].args["enabled"], true);
        assert_eq!(lock.injected[0].injected_by, "engine");
        assert_eq!(lock.shadowed[0].generated_by, "bistro");
    }

    // A lock is still written when a listed blob is unreadable: the entry
    // records the empty checksum and no payload bytes rather than failing.
    #[test]
    fn write_lock_tolerates_a_missing_blob_file() {
        let _guard = test_output::LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _lock_file = LockFile;

        let paths = vec!["/no/such/data/0".to_string()];
        write_lock(&[], &[], &[], &[], &paths).expect("write_lock");
        let written = fs::read_to_string(LOCK_PATH).expect("lock written");

        let lock: BlobLock = serde_json::from_str(&written).expect("lock is valid json");
        assert_eq!(lock.blobs[0].payload_bytes, 0);
        assert_eq!(lock.blobs[0].checksum, checksum(b""));
        assert!(lock.assets.is_empty());
    }

    #[test]
    fn packer_appends_within_the_limit() {
        let mut p = PayloadPacker::new(100);
        let a = p.push(&[1, 2, 3]);
        let b = p.push(&[4, 5]);

        assert_eq!((a.blob_index, a.offset, a.len), (0, 0, 3));
        assert_eq!((b.blob_index, b.offset, b.len), (0, 3, 2));
        assert_eq!(p.finish(), vec![vec![1, 2, 3, 4, 5]]);
    }

    #[test]
    fn packer_rolls_to_a_new_blob_at_the_limit() {
        let mut p = PayloadPacker::new(4);
        let a = p.push(&[1, 2, 3]);
        // 3 + 2 exceeds the 4-byte cap, so this payload starts blob 1.
        let b = p.push(&[4, 5]);

        assert_eq!(a.blob_index, 0);
        assert_eq!((b.blob_index, b.offset, b.len), (1, 0, 2));
        assert_eq!(p.finish(), vec![vec![1, 2, 3], vec![4, 5]]);
    }

    #[test]
    fn packer_keeps_an_oversized_payload_in_an_empty_blob() {
        // A single payload larger than the cap cannot be split, so it stays
        // in the current (empty) blob rather than rolling forever.
        let mut p = PayloadPacker::new(4);
        let a = p.push(&[7; 10]);
        assert_eq!((a.blob_index, a.offset, a.len), (0, 0, 10));

        // The next payload rolls to a fresh blob.
        let b = p.push(&[1]);
        assert_eq!((b.blob_index, b.offset), (1, 0));
    }

    #[test]
    fn packer_zero_length_payload_gets_a_valid_locator() {
        let mut p = PayloadPacker::new(8);
        let a = p.push(&[]);
        let b = p.push(&[1]);
        assert_eq!((a.blob_index, a.offset, a.len), (0, 0, 0));
        assert_eq!((b.blob_index, b.offset, b.len), (0, 0, 1));
    }

    #[test]
    fn packer_group_boundary_starts_a_fresh_blob() {
        let mut p = PayloadPacker::new(1024);
        let a = p.push(&[1, 2]);
        p.start_group();
        let b = p.push(&[3]);
        let c = p.push(&[4]);
        assert_eq!(a.blob_index, 0);
        assert_eq!((b.blob_index, b.offset), (1, 0));
        assert_eq!((c.blob_index, c.offset), (1, 1));
        assert_eq!(p.finish(), vec![vec![1, 2], vec![3, 4]]);
    }

    #[test]
    fn packer_empty_group_produces_no_blob() {
        let mut p = PayloadPacker::new(1024);
        let a = p.push(&[1]);
        p.start_group();
        p.start_group();
        let b = p.push(&[2]);
        assert_eq!(a.blob_index, 0);
        assert_eq!(b.blob_index, 1);
        assert_eq!(p.finish().len(), 2);
    }

    #[test]
    fn packer_group_never_lands_in_blob_zero() {
        // Even with no earlier content, a group leaves the eagerly-read blob 0
        // empty and starts at blob 1.
        let mut p = PayloadPacker::new(1024);
        p.start_group();
        let a = p.push(&[1]);
        assert_eq!((a.blob_index, a.offset), (1, 0));
        assert_eq!(p.finish(), vec![vec![], vec![1]]);
    }

    #[test]
    fn packer_size_rollover_still_applies_within_a_group() {
        let mut p = PayloadPacker::new(2);
        p.push(&[1]);
        p.start_group();
        let a = p.push(&[2, 3]);
        let b = p.push(&[4]);
        assert_eq!(a.blob_index, 1);
        assert_eq!(b.blob_index, 2, "group content exceeding the cap rolls on");
    }

    // The byte-format round-trip tests live in the concinnity-blob crate with
    // the encoder; here the tests cover cook's packing policy and the lock.

    #[test]
    fn checksum_matches_known_sha256_vectors() {
        assert_eq!(
            checksum(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            checksum(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn now_iso8601_is_rfc3339_parseable() {
        let stamp = now_iso8601();
        assert!(OffsetDateTime::parse(&stamp, &Rfc3339).is_ok());
    }

    #[test]
    fn blob_lock_serializes_injected_type_field_as_type() {
        // The lock file is read by humans and tools; the serde rename on
        // LockedInjection keeps the JSON key `type`, matching world.jsonl.
        let lock = BlobLock {
            engine_version: "0.0.0".to_string(),
            built_at: "2026-01-01T00:00:00Z".to_string(),
            blobs: vec![BlobEntry {
                path: "data/0".to_string(),
                checksum: "00".to_string(),
                payload_bytes: 4,
            }],
            assets: vec![],
            resources: vec![
                LockedResource {
                    name: "clip".to_string(),
                    id: Some(0),
                    kind: "AudioClip".to_string(),
                    handle: 0,
                    args_hash: "00".to_string(),
                    payload_blob: Some(0),
                    ..Default::default()
                },
                LockedResource {
                    name: "wall_tex".to_string(),
                    id: Some(1),
                    kind: "Texture".to_string(),
                    handle: 0,
                    args_hash: "00".to_string(),
                    payload_blob: Some(0),
                    texture_source: Some(LockedTextureSource {
                        source: "wall.png".to_string(),
                        image_index: 2,
                    }),
                    ..Default::default()
                },
            ],
            injected: vec![LockedInjection {
                name: "debug_hud".to_string(),
                asset_type: "DebugHud".to_string(),
                args: serde_json::json!({}),
                injected_by: "engine".to_string(),
            }],
            shadowed: vec![LockedShadow {
                name: "bistro_mat_wood".to_string(),
                asset_type: "Material".to_string(),
                generated_by: "bistro".to_string(),
            }],
        };
        let json = serde_json::to_value(&lock).unwrap();
        assert_eq!(json["injected"][0]["type"], "DebugHud");
        assert!(json["injected"][0].get("asset_type").is_none());
        // LockedShadow carries the same rename, and names what it overrides.
        assert_eq!(json["shadowed"][0]["type"], "Material");
        assert_eq!(json["shadowed"][0]["generated_by"], "bistro");

        // Source info serializes only where present, and round-trips.
        assert!(json["resources"][0].get("texture_source").is_none());
        assert!(json["resources"][0].get("mesh_source").is_none());
        assert_eq!(json["resources"][1]["texture_source"]["source"], "wall.png");

        let back: BlobLock = serde_json::from_value(json).unwrap();
        assert_eq!(back.injected[0].asset_type, "DebugHud");
        assert_eq!(back.blobs[0].payload_bytes, 4);
        assert_eq!(back.resources[0].kind, "AudioClip");
        assert!(back.resources[0].texture_source.is_none());
        let tex = back.resources[1].texture_source.as_ref().unwrap();
        assert_eq!(tex.source, "wall.png");
        assert_eq!(tex.image_index, 2);
        assert_eq!(back.shadowed[0].name, "bistro_mat_wood");
    }

    #[test]
    fn blob_lock_reads_a_lock_without_its_optional_fields() {
        // Locks written before resource and shadow provenance landed have no
        // `resources` / `shadowed` key; reading one must not fail.
        let json = serde_json::json!({
            "engine_version": "0.0.0",
            "built_at": "2026-01-01T00:00:00Z",
            "blobs": [],
            "assets": [],
            "injected": [],
        });
        let back: BlobLock = serde_json::from_value(json).unwrap();
        assert!(back.resources.is_empty());
        assert!(back.shadowed.is_empty());

        // A resource recorded before source info landed parses with none.
        let json = serde_json::json!({
            "engine_version": "0.0.0",
            "built_at": "2026-01-01T00:00:00Z",
            "blobs": [],
            "assets": [],
            "resources": [{
                "name": "tex", "kind": "Texture", "handle": 0,
                "args_hash": "", "payload_blob": null,
            }],
            "injected": [],
        });
        let back: BlobLock = serde_json::from_value(json).unwrap();
        assert!(back.resources[0].texture_source.is_none());
        assert!(back.resources[0].mesh_source.is_none());
    }
}
