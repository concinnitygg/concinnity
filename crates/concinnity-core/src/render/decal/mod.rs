//! Backend-agnostic decal support. Owns the per-decal model / inverse-model
//! matrix math the projected-decal pass needs at runtime, the `DecalRecord` the
//! backends consume, and the `DecalSet` slot table they drive the pass from.
//! Decals are stamped onto the scene depth buffer by drawing a unit-box volume
//! per decal: the fragment shader reconstructs the world-space point of each
//! rasterized pixel from depth and tests whether it lies inside the box.

mod record;
mod set;

/// Live decals a backend that reserves a descriptor block per decal at init
/// can hold; adds past it return an error.
pub const MAX_DECALS: usize = 256;

pub use record::{DecalRecord, build_decal_records, decal_model_matrix, invert_decal_model};
pub use set::{AtCapacity, DecalSet, RemoveError, VisibleDecal};
