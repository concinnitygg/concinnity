// src/blob.rs
//
// The .cnb blob binary format (read/write, lock file, lazy payload load) lives
// in concinnity-core; re-export it under the historical crate::blob::* paths.
// `pub` so the editor crate's in-memory build path can construct `BlobData`.
pub use concinnity_core::blob::*;

use crate::ecs::ComponentAsset;
use crate::result::CnResult;

// Load the primary blob, resolve every stored def to a `ComponentAsset`, and
// return the resource stream and `BlobData` alongside them. The blob carries a
// component stream (systems are internal and constructed at runtime) and a
// resource stream (compiled resources addressed by a per-kind handle, which the
// caller loads into per-kind tables).
//
// Each component that has a compiled payload carries its `PayloadLocator`
// injected into it (see `ComponentAsset::inject_locator`). Only blob 0's payload
// section is read into memory by `load_raw`; overflow blobs are read from disk
// lazily on first access.
pub(crate) fn load() -> Result<(Vec<ComponentAsset>, Vec<ResourceRecord>, BlobData), CnResult> {
    let (defs, resources, blob_data) = concinnity_core::blob::load_raw()?;

    let components = defs
        .iter()
        .map(|def| {
            // Every record is baked: the bytes are the serialized runtime
            // component (cook already ran the asset -> component translation).
            let mut component = ComponentAsset::from_baked(def)?;
            if let Some(locator) = &def.payload {
                component.inject_locator(locator.clone());
            }
            Ok(component)
        })
        .collect::<Result<Vec<_>, CnResult>>()?;

    Ok((components, resources, blob_data))
}
