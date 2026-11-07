//! Decals and particle emitters, added and removed while the world runs.
//!
//! Both are slot tables: the caller hands over a record and gets back the
//! index it will remove by. The decals' table is the scene's, and a backend
//! answers only for the pass that draws it; the emitters' is the backend's
//! own.

use crate::render::backend::error::{RenderError, RenderResult};

/// The GPU side of the projected-decal pass: the pipeline a runtime decal
/// draws with and each slot's albedo binding. The slot table itself is the
/// scene's [`DecalSet`](crate::render::decal::DecalSet).
///
/// Defaulted for a backend without the pass: it has no pipeline to make
/// ready, so a runtime decal is refused with [`RenderError::Unsupported`]
/// rather than silently accepted and never drawn.
pub trait DecalPass {
    /// Make the decal pass ready to draw a runtime decal, building what it
    /// draws with on first use where the backend builds it lazily.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without the pass, or
    /// an error if its pipeline cannot be built.
    fn ensure_decal_pass(&mut self) -> RenderResult<()> {
        Err(RenderError::Unsupported { op: "add_decal" })
    }

    /// Point decal slot `id`'s albedo at `texture_slot` of the texture pool,
    /// for a backend that binds each slot's albedo itself.
    fn bind_decal_texture(&mut self, id: usize, texture_slot: usize) {
        let _ = (id, texture_slot);
    }

    /// Forget decal slot `id`'s albedo binding, once the slot is removed.
    fn release_decal_texture(&mut self, id: usize) {
        let _ = id;
    }
}

/// The particle-emitter slot table.
///
/// Defaulted to [`RenderError::Unsupported`], so a backend without the pass
/// reports the failure to whoever asked rather than silently accepting a
/// record it will never draw.
pub trait RuntimeEmitters {
    /// Append a particle-emitter record at runtime, returning a stable slot
    /// index. The backend allocates the per-emitter GPU pool + atomic
    /// spawn counter (matching the init-time path) so the compute kernel
    /// can begin ticking on the next frame.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without runtime
    /// emitters, or an error if the emitter's GPU pool cannot be allocated.
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
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without runtime
    /// emitters, or an error if `emitter_id` names no live emitter.
    fn remove_emitter(&mut self, emitter_id: usize) -> RenderResult<()> {
        let _ = emitter_id;
        Err(RenderError::Unsupported {
            op: "remove_emitter",
        })
    }
}
