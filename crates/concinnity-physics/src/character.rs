// concinnity-physics/src/character.rs
//
// The capsule a character move is resolved against. Rapier's controller
// borrows a shape per call, and building one allocates a reference-counted
// handle, so each character keeps its capsule across the fixed ticks and
// rebuilds it only when its dimensions change.

use core::fmt;

use rapier3d::prelude::SharedShape;

// A character's collision capsule, cached next to the dimensions it was built
// from.
#[derive(Clone)]
pub struct CharacterShape {
    // Cylinder half-height, excluding the hemisphere caps.
    half_height: f32,
    radius: f32,
    shape: SharedShape,
}

impl CharacterShape {
    // Build the capsule for a cylinder of `2 * half_height` capped by
    // hemispheres of `radius`.
    pub fn capsule(half_height: f32, radius: f32) -> Self {
        Self {
            half_height,
            radius,
            shape: SharedShape::capsule_y(half_height, radius),
        }
    }

    // Adopt new dimensions, rebuilding the capsule only when they differ from
    // the cached one's.
    pub fn resize(&mut self, half_height: f32, radius: f32) {
        if self.half_height == half_height && self.radius == radius {
            return;
        }
        *self = Self::capsule(half_height, radius);
    }

    pub(crate) fn shape(&self) -> &SharedShape {
        &self.shape
    }
}

impl fmt::Debug for CharacterShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharacterShape")
            .field("half_height", &self.half_height)
            .field("radius", &self.radius)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn capsule_carries_its_dimensions() {
        let shape = CharacterShape::capsule(0.6, 0.3);
        let capsule = shape.shape().as_capsule().expect("a capsule");
        assert!((capsule.half_height() - 0.6).abs() < 1e-6);
        assert!((capsule.radius - 0.3).abs() < 1e-6);
    }

    #[test]
    fn resize_keeps_the_shape_until_a_dimension_changes() {
        let mut shape = CharacterShape::capsule(0.6, 0.3);
        let original = shape.shape().0.clone();

        shape.resize(0.6, 0.3);
        assert!(
            Arc::ptr_eq(&original, &shape.shape().0),
            "unchanged dimensions must reuse the cached capsule"
        );

        shape.resize(0.9, 0.3);
        assert!(
            !Arc::ptr_eq(&original, &shape.shape().0),
            "a new half-height rebuilds the capsule"
        );
        let rebuilt = shape.shape().0.clone();
        shape.resize(0.9, 0.4);
        assert!(
            !Arc::ptr_eq(&rebuilt, &shape.shape().0),
            "a new radius rebuilds the capsule"
        );
        let capsule = shape.shape().as_capsule().expect("a capsule");
        assert!((capsule.half_height() - 0.9).abs() < 1e-6);
        assert!((capsule.radius - 0.4).abs() < 1e-6);
    }
}
