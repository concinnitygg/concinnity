// concinnity-physics/src/character.rs
//
// One character-capsule move and its result. A character is moved by asking
// the simulation to resolve a desired translation against the scene rather
// than by simulating it, so the exchange is a request in and a resolved move
// out.

use crate::BodyHandle;
use crate::LayerMask;

/// One character-capsule move request.
#[derive(Debug, Clone, Copy)]
pub struct CharacterMoveInput {
    /// World-space capsule centre before the move.
    pub center: [f32; 3],
    /// Desired translation for this tick.
    pub desired: [f32; 3],
    /// Seconds this move covers.
    pub dt: f32,
    /// The moving capsule's own body, left out of the collision query.
    pub exclude: BodyHandle,
    /// Which layers the collision query considers.
    pub mask: LayerMask,
}

/// Result of moving a character capsule for one tick.
#[derive(Debug, Clone, Copy)]
pub struct CharacterMove {
    /// The translation actually applied after collision resolution.
    pub translation: [f32; 3],
    /// True when the capsule is resting on a surface after the move.
    pub grounded: bool,
}
