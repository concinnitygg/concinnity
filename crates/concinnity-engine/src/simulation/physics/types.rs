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
    /// Angular velocity damping, slowing spin on its own.
    pub angular_damping: f32,
    /// Rolling resistance: the share of the load a rolling contact resists
    /// with, acting at the body's rolling radius (a ball's or a capsule's
    /// radius; a box does not roll). `0.0` rolls freely.
    pub rolling_resistance: f32,
}

/// A dynamic body's state between steps, for whatever acts on it from outside
/// the solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyState {
    /// Center of mass, in world units.
    pub position: [f32; 3],
    /// Orientation as an `[x, y, z, w]` quaternion.
    pub rotation: [f32; 4],
    /// Linear velocity, in world units per second.
    pub linear_velocity: [f32; 3],
    /// Angular velocity, in radians per second.
    pub angular_velocity: [f32; 3],
    /// Mass, in kilograms.
    pub mass: f32,
    /// Moments of inertia about the body's own axes, in kilogram square
    /// meters.
    pub inertia: [f32; 3],
    /// Whether the body is asleep, held still until something wakes it.
    pub sleeping: bool,
}

/// A 1 kg body with moderate friction, no bounce, full gravity, and no drag.
impl Default for DynamicParams {
    fn default() -> Self {
        Self {
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            rolling_resistance: 0.0,
        }
    }
}
