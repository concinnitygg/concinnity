//! Blob file reading and lazy payload residency live in concinnity-store;
//! re-export them under the historical crate::blob::* paths. `pub` so the editor
//! crate's in-memory build path can construct `BlobData`.
pub use concinnity_store::blob::*;

use crate::ecs::ComponentAsset;
use crate::ecs::asset_id::AssetId;
use crate::result::CnResult;

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
// section is read into memory by `load_raw`; overflow blobs are read from disk
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
    pub(crate) scene_groups: Vec<concinnity_core::ecs::SceneGroup>,
    pub(crate) mesh_bounds: Vec<concinnity_core::ecs::MeshBoundsRecord>,
    pub(crate) physics_budget: Option<concinnity_core::ecs::PhysicsBudgetRecord>,
    pub(crate) manifest: WorldManifest,
    pub(crate) blob: BlobData,
}

pub(crate) fn load() -> Result<LoadedBlob, CnResult> {
    resolve(concinnity_store::blob::load_raw()?)
}

// `load` against a primary blob file named directly, rather than the
// state root's `data/` layout. Overflow blobs are its siblings by index.
pub(crate) fn load_at(primary: &std::path::Path) -> Result<LoadedBlob, CnResult> {
    resolve(concinnity_store::blob::load_raw_at(primary)?)
}

fn resolve((meta, blob_data): (BlobMeta, BlobData)) -> Result<LoadedBlob, CnResult> {
    let components = meta
        .defs
        .iter()
        .map(|def| {
            // Every record is baked: the bytes are the serialized runtime
            // component (cook already ran the asset -> component translation).
            let mut component = ComponentAsset::from_baked(def)?;
            if let Some(locator) = &def.payload {
                component.inject_locator(locator.clone());
            }
            Ok((def.name, component))
        })
        .collect::<Result<Vec<_>, CnResult>>()?;

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
