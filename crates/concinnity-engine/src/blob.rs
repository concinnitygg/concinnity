// src/blob.rs
//
// The .cnb blob binary format (read/write, lock file, lazy payload load) lives
// in concinnity-core; re-export it under the historical crate::blob::* paths.
// `pub` so the editor crate's in-memory build path can construct `BlobData`.
pub use concinnity_core::blob::*;

use crate::ecs::{ComponentAsset, RecordKind};
use crate::result::CnResult;

// Load the primary blob, resolve every stored def to a `ComponentAsset` (the
// blob carries only components, systems are internal and constructed at
// runtime), and return the `BlobData` alongside them.
//
// Each component that has a compiled payload carries its `PayloadLocator`
// injected into it (see `ComponentAsset::inject_locator`). Only blob 0's payload
// section is read into memory by `load_raw`; overflow blobs are read from disk
// lazily on first access.
pub(crate) fn load() -> Result<(Vec<ComponentAsset>, BlobData), CnResult> {
    let (defs, blob_data) = concinnity_core::blob::load_raw()?;

    let components = defs
        .iter()
        .map(|def| {
            // `Authored` records carry the authored `Args` (reconstructed via
            // `from_args`); `Baked` records carry the already-translated runtime
            // component. A blob may hold both while the migration is in flight.
            let mut component = match def.record {
                RecordKind::Authored => ComponentAsset::from_def(def)?,
                RecordKind::Baked => ComponentAsset::from_baked(def)?,
            };
            if let Some(locator) = &def.payload {
                component.inject_locator(locator.clone());
            }
            Ok(component)
        })
        .collect::<Result<Vec<_>, CnResult>>()?;

    Ok((components, blob_data))
}
