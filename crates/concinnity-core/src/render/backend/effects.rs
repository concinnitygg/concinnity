//! Decals and particle emitters, added and removed while the world runs.
//!
//! Both are slot tables the backend owns: the caller hands over a record and
//! gets back the index it will remove by. Neither is part of the draw set, so
//! neither reaches the per-frame contract.

use alloc::string::String;
use alloc::string::ToString;

/// The projected-decal and particle-emitter slot tables.
///
/// All defaulted to an `Err`, so a backend without the pass reports the
/// failure to whoever asked rather than silently accepting a record it will
/// never draw.
pub trait SceneEffects {
    /// Append a projected-decal record at runtime, returning a stable slot
    /// index the caller hands to [`Self::remove_decal`] later. Lets a
    /// gameplay system stamp bullet holes, footprints, or other ad-hoc
    /// decals after the world has built. Backends that have not implemented
    /// the runtime path return `Err`; the caller logs and drops the request.
    fn add_decal(&mut self, record: crate::render::decal::DecalRecord) -> Result<usize, String> {
        let _ = record;
        Err("add_decal: not implemented on this backend".to_string())
    }

    /// Tombstone a runtime decal slot. The id returned by
    /// [`Self::add_decal`] becomes invalid; the next add may reuse it.
    /// Default no-op-with-Err: backends without a runtime path leave the
    /// remove logged + skipped at the caller.
    fn remove_decal(&mut self, decal_id: usize) -> Result<(), String> {
        let _ = decal_id;
        Err("remove_decal: not implemented on this backend".to_string())
    }

    /// Append a particle-emitter record at runtime, returning a stable slot
    /// index. The backend allocates the per-emitter GPU pool + atomic
    /// spawn counter (matching the init-time path) so the compute kernel
    /// can begin ticking on the next frame. Default no-op-with-Err.
    fn add_emitter(
        &mut self,
        record: crate::render::particles::ParticleEmitterRecord,
    ) -> Result<usize, String> {
        let _ = record;
        Err("add_emitter: not implemented on this backend".to_string())
    }

    /// Tombstone a runtime emitter slot and release its GPU pool +
    /// counter buffers (the GPU keeps them alive via its own refcount
    /// until any in-flight command buffer that referenced them completes).
    /// Default no-op-with-Err.
    fn remove_emitter(&mut self, emitter_id: usize) -> Result<(), String> {
        let _ = emitter_id;
        Err("remove_emitter: not implemented on this backend".to_string())
    }
}
