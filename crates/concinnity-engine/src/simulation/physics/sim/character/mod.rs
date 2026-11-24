// The character controller: a capsule driven by a desired translation and
// resolved by queries, changing nothing; the caller applies the answer. Sensors
// never block a move.

mod capsule;
mod config;
mod resolve;
mod slide;

pub use capsule::CharacterCapsule;
pub(crate) use config::CharacterConfig;

pub(crate) use resolve::resolve;
