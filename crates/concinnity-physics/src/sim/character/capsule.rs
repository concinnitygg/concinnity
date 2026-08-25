// concinnity-physics/src/sim/character/capsule.rs
//
// The capsule a character move is resolved against, held by the caller across
// the fixed ticks rather than rebuilt per move. Here that cache buys nothing --
// the shape is two floats and building one allocates nothing -- but the shape
// is the one thing a simulation genuinely owns, so it is the caller that keeps
// it and a simulation whose shape does cost something needs no other change.

use crate::ColliderShape;

/// A character's collision capsule: a cylinder of `2 * half_height` capped by
/// hemispheres of `radius`, standing on the y axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterCapsule {
    half_height: f32,
    radius: f32,
}

impl CharacterCapsule {
    /// Build the capsule for a cylinder of `2 * half_height` capped by
    /// hemispheres of `radius`.
    pub fn new(half_height: f32, radius: f32) -> Self {
        CharacterCapsule {
            half_height,
            radius,
        }
    }

    /// Adopt new dimensions, for a character whose capsule was re-authored.
    pub fn resize(&mut self, half_height: f32, radius: f32) {
        self.half_height = half_height;
        self.radius = radius;
    }

    /// Half the cylindrical section's height, excluding the caps.
    pub fn half_height(&self) -> f32 {
        self.half_height
    }

    /// Cap and cylinder radius.
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// The capsule as a collider shape, which is what a sweep is asked in
    /// terms of.
    pub(crate) fn shape(&self) -> ColliderShape {
        ColliderShape::Capsule {
            half_height: self.half_height,
            radius: self.radius,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_capsule_carries_the_dimensions_it_was_built_from() {
        let capsule = CharacterCapsule::new(0.6, 0.3);
        assert_eq!(capsule.half_height(), 0.6);
        assert_eq!(capsule.radius(), 0.3);
        assert_eq!(
            capsule.shape(),
            ColliderShape::Capsule {
                half_height: 0.6,
                radius: 0.3,
            }
        );
    }

    #[test]
    fn resizing_replaces_both_dimensions() {
        let mut capsule = CharacterCapsule::new(0.6, 0.3);
        capsule.resize(0.6, 0.3);
        assert_eq!(capsule, CharacterCapsule::new(0.6, 0.3));
        capsule.resize(0.9, 0.4);
        assert_eq!(capsule.half_height(), 0.9);
        assert_eq!(capsule.radius(), 0.4);
    }
}
