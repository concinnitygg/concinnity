//! Decals and particle emitters, added and removed while the world runs.
//!
//! Both are slot tables the backend owns: the caller hands over a record and
//! gets back the index it will remove by. Neither is part of the draw set, so
//! neither reaches the per-frame contract.

use crate::render::error::{RenderError, RenderResult};

/// The projected-decal and particle-emitter slot tables.
///
/// All defaulted to [`RenderError::Unsupported`], so a backend without the
/// pass reports the failure to whoever asked rather than silently accepting a
/// record it will never draw.
pub trait SceneEffects {
    /// Append a projected-decal record at runtime, returning a stable slot
    /// index the caller hands to [`Self::remove_decal`] later. Lets a
    /// gameplay system stamp bullet holes, footprints, or other ad-hoc
    /// decals after the world has built.
    fn add_decal(&mut self, record: crate::render::decal::DecalRecord) -> RenderResult<usize> {
        let _ = record;
        Err(RenderError::Unsupported { op: "add_decal" })
    }

    /// Tombstone a runtime decal slot. The id returned by
    /// [`Self::add_decal`] becomes invalid; the next add may reuse it.
    fn remove_decal(&mut self, decal_id: usize) -> RenderResult<()> {
        let _ = decal_id;
        Err(RenderError::Unsupported { op: "remove_decal" })
    }

    /// Append a particle-emitter record at runtime, returning a stable slot
    /// index. The backend allocates the per-emitter GPU pool + atomic
    /// spawn counter (matching the init-time path) so the compute kernel
    /// can begin ticking on the next frame.
    fn add_emitter(
        &mut self,
        record: crate::render::particles::ParticleEmitterRecord,
    ) -> RenderResult<usize> {
        let _ = record;
        Err(RenderError::Unsupported { op: "add_emitter" })
    }

    /// Tombstone a runtime emitter slot and release its GPU pool +
    /// counter buffers (the GPU keeps them alive via its own refcount
    /// until any in-flight command buffer that referenced them completes).
    fn remove_emitter(&mut self, emitter_id: usize) -> RenderResult<()> {
        let _ = emitter_id;
        Err(RenderError::Unsupported {
            op: "remove_emitter",
        })
    }
}
