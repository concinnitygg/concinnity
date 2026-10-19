//! The scalar, vector and rotation primitives the rest of the vocabulary is
//! built from: the f32 transcendentals `core` leaves to a math library, the
//! 3-component vector ops every layout and transform reaches for (and their
//! double-precision twins), and the quaternion / Euler convention rotations
//! are authored and stepped in.
mod angle;
pub mod dvec3;
pub mod noise;
pub mod pick;
mod rotation;
mod scalar;
pub mod vec3;

pub use angle::{flat_forward, flat_right, forward, heading_of, shortest_turn};
pub use rotation::{
    Quat, euler_yxz_deg_from_quat, quat_from_axis_angle, quat_from_euler_yxz_deg, quat_mul,
    quat_normalize, quat_rotate,
};

pub use scalar::{
    acos, asin, atan2, ceil, cos, exp, exp2, floor, fract, half_life_decay, hypot, ln, log2, log10,
    mul_add, powf, powf_f64, powi, rem_euclid, round, round_f64, sin, sin_cos, sqrt, tan, trunc,
};
