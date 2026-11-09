//! The light tables a backend's GPU reads, rewritten when the world's local
//! lights move.
//!
//! The light block itself is the scene's
//! [`LightState`](crate::render::light::state::LightState), which every
//! backend uploads from; only the per-light storage the passes index is a
//! backend's own.

use crate::render::backend::error::RenderResult;
use crate::render::light::lights::LightData;

/// The per-light tables the passes read: the local, area and spot-shadow
/// records, sized once at init.
///
/// Its one caller is the
/// [`RenderBackend::move_local_lights`](super::RenderBackend::move_local_lights)
/// default, which checks the counts and drains the GPU before it asks for the
/// rewrite. Defaulted for a backend that draws no local lights: it has no spot
/// slices and no tables to rewrite.
pub trait LightTables {
    /// The spot-shadow slices the backend allocated at init, one per shadowed
    /// spot light.
    fn spot_slice_count(&self) -> usize {
        0
    }

    /// Rewrite the local-light, area-light and spot-shadow tables from `data`
    /// in place, and re-derive the spot slices' frusta. `data` holds the same
    /// lights, in the same order, as the tables were built from, and the caller
    /// has drained the GPU, since every frame in flight reads these tables.
    ///
    /// # Errors
    ///
    /// Returns an error if a table cannot be written.
    fn rewrite_light_tables(&mut self, data: &LightData) -> RenderResult<()> {
        let _ = data;
        Ok(())
    }
}
