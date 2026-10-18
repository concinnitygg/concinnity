//! Audio policy with no mixer behind it: clip decode bookkeeping, the capped
//! one-shot voice pool, occlusion smoothing, impact gain and attenuation range
//! sanitizing. Each is generic over the mixer's own handle and payload types,
//! so the engine's mixer stores its sounds in them.

pub mod clips;
pub mod distance;
pub mod impact;
pub mod occlusion;
pub mod voices;

/// Opaque handle to a spatial emitter inside the engine's mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmitterId(pub usize);
