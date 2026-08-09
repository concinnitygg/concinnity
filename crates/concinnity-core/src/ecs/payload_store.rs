// The payload-access seam a `PipelineContext` exposes to systems.
//
// Systems reach compiled binary payloads through this trait, not a concrete
// store, so the ECS mechanism carries no dependency on blob file I/O or the
// `.concinnity/` layout. The runtime implementor is `concinnity_blob`-backed
// `BlobData` in `crate::blob`; tests supply their own.

use crate::ecs::PayloadLocator;
use crate::result::CnResult;

// A source of compiled payload bytes addressed by `PayloadLocator`.
pub trait PayloadStore {
    // Read the bytes a locator points at. `&mut self` because a disk-backed
    // store may load an overflow section lazily on first access. Errors when
    // the payload was released, the locator is out of range, or a lazy load
    // fails.
    fn read(&mut self, locator: &PayloadLocator) -> Result<&[u8], CnResult>;

    // Release an entire blob's in-memory payload once every system that needs
    // it has finished (e.g. after GPU upload). A store with nothing resident
    // for `blob_index` treats this as a no-op.
    fn release(&mut self, blob_index: u32);

    // Whether the payloads are backed by files still on disk, so a released
    // payload can be re-read on demand rather than kept RAM-resident.
    fn disk_backed(&self) -> bool;
}
