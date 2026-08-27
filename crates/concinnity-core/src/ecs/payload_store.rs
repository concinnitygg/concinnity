// The payload-access seam a `PipelineContext` exposes to systems.
//
// Systems reach compiled binary payloads through this trait, not a concrete
// store, so the ECS mechanism carries no dependency on blob file I/O or the
// state root's layout. The runtime implementor is the `BlobData` that
// `concinnity_host::store` builds over `crate::blob`; tests supply their own.

use crate::ecs::PayloadLocator;
use crate::result::CnResult;

/// A source of compiled payload bytes addressed by `PayloadLocator`.
pub trait PayloadStore {
    /// Read the bytes a locator points at. `&mut self` because a disk-backed
    /// store may load an overflow section lazily on first access. Errors when
    /// the payload was released, the locator is out of range, or a lazy load
    /// fails.
    fn read(&mut self, locator: &PayloadLocator) -> Result<&[u8], CnResult>;

    /// Release an entire blob's in-memory payload once every system that needs
    /// it has finished (e.g. after GPU upload). A store with nothing resident
    /// for `blob_index` treats this as a no-op.
    fn release(&mut self, blob_index: u32);

    /// Whether the payloads are backed by files still on disk, so a released
    /// payload can be re-read on demand rather than kept RAM-resident.
    fn disk_backed(&self) -> bool;

    /// Release every resident section at once, returning the bytes freed.
    /// `World::start` calls this after init: systems read compiled payloads
    /// only while initing and cache what they keep, so nothing consults the
    /// store again. A store with nothing resident frees nothing.
    fn release_all_resident(&mut self) -> usize {
        0
    }
}

/// A payload store holding nothing: every read errors and every release is a
/// no-op. What a [`World`](crate::ecs::World) built without a blob runs on --
/// unit tests, and worlds assembled entirely from runtime-only components.
pub struct NoPayloads;

impl PayloadStore for NoPayloads {
    fn read(&mut self, locator: &PayloadLocator) -> Result<&[u8], CnResult> {
        tracing::error!(
            "NoPayloads: world has no compiled payloads, cannot read blob {}",
            locator.blob_index
        );
        Err(CnResult::FileIo)
    }

    fn release(&mut self, _blob_index: u32) {}

    fn disk_backed(&self) -> bool {
        false
    }
}
