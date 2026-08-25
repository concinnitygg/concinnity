// concinnity-physics/src/sim/character/mod.rs
//
// The character controller: a capsule driven by a desired translation rather
// than by forces.
//
// It sits beside the step rather than in it, on the query side of the
// simulation, because that is what it is -- a move is resolved by asking the
// scene where the capsule can get to, and nothing in the scene is changed by
// the asking. The caller applies the answer itself, which is what lets one
// character be resolved, judged, and only then moved.
//
// The parts are split by what they need: the capsule and the tuning are plain
// data, the deflection arithmetic is two vectors in and one out, and only the
// resolve loop needs a world to sweep against.
//
// Sensors are excluded where every other query excludes them, in the filter
// the sweeps here already run through, so a region that records overlap never
// blocks a move.

mod capsule;
mod config;
mod resolve;
mod slide;

pub use capsule::CharacterCapsule;
pub(crate) use config::CharacterConfig;

pub(crate) use resolve::resolve;
