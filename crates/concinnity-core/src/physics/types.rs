// The collision shapes and body parameters the simulation is asked to build.
// Plain data in the engine's `[f32; 3]` representation: no simulation math type
// appears here, so a caller never has to name one.

/// A collision shape, in the body's local space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColliderShape {
    /// Box with the given half-extents along x, y, z.
    Cuboid {
        /// Half-extents along each axis.
        half_extents: [f32; 3],
    },
    /// Sphere of the given radius.
    Ball {
        /// Sphere radius.
        radius: f32,
    },
    /// Y-axis capsule: a cylinder of `2 * half_height` capped by hemispheres.
    Capsule {
        /// Half the cylindrical section's height.
        half_height: f32,
        /// Cap and cylinder radius.
        radius: f32,
    },
}

/// Physical parameters for a dynamic (freely simulated) body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicParams {
    /// Mass in kilograms. `0.0` derives mass from the shape's volume.
    pub mass: f32,
    /// Coulomb friction coefficient.
    pub friction: f32,
    /// Bounciness in `[0, 1]`.
    pub restitution: f32,
    /// Multiplier on the world gravity for this body.
    pub gravity_scale: f32,
    /// Linear velocity damping (air drag).
    pub linear_damping: f32,
}
