//! Decals and particle emitters, added and removed while the world runs.
//!
//! Both are slot tables the scene keeps: the caller hands over a record and
//! gets back the index it will remove by, and a backend answers only for the
//! passes that draw them.

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

/// The GPU side of the particle passes: the pipelines every emitter runs
/// through and each emitter's own pool. The emitter table itself is the
/// scene's [`ParticleSystem`](crate::render::particles::ParticleSystem).
///
/// Defaulted for a backend without the passes: it has no pipelines to make
/// ready, so a runtime emitter is refused with [`RenderError::Unsupported`]
/// rather than silently accepted and never drawn.
pub trait EmitterPass {
    /// Make the particle passes ready to run an emitter, building their
    /// pipelines on first use, so a world that never had an emitter pays
    /// nothing until the first add.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without the passes,
    /// or an error if their pipelines cannot be built.
    fn ensure_emitter_pass(&mut self) -> RenderResult<()> {
        Err(RenderError::Unsupported { op: "add_emitter" })
    }

    /// Allocate emitter `id`'s GPU pool and spawn counter for `record`, and
    /// bind its albedo, so the simulation can tick it from the next frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool cannot be allocated.
    fn create_emitter(
        &mut self,
        id: usize,
        record: &crate::render::particles::ParticleEmitterRecord,
    ) -> RenderResult<()> {
        let _ = (id, record);
        Ok(())
    }

    /// Release emitter `id`'s GPU pool once it is removed, keeping it alive
    /// for whatever frame in flight still reads it.
    fn destroy_emitter(&mut self, id: usize) {
        let _ = id;
    }
}
