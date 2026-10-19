//! Audio: the system that decides what plays (`system`), the mixer state that
//! decides how (`mixer`), and the policy pieces under them: clip decode
//! bookkeeping, the capped one-shot voice pool, occlusion smoothing, impact gain
//! and attenuation range sanitizing. The host's mixer is reached through
//! [`out::AudioOut`].

pub mod clips;
pub mod distance;
pub mod impact;
pub mod mixer;
pub mod occlusion;
pub mod out;
pub mod system;
pub mod voices;

/// A spatial emitter's slot in the mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmitterId(pub usize);
