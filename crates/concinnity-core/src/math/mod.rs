//! The scalar, vector and rotation primitives the rest of the vocabulary is
//! built from: the f32 transcendentals `core` leaves to a math library, the
//! 3-component vector ops every layout and transform reaches for, and the
//! quaternion / Euler convention rotations are authored and stepped in.
mod rotation;
mod scalar;
pub mod vec3;

pub use rotation::{Quat, euler_yxz_deg_from_quat, quat_from_euler_yxz_deg, quat_normalize};

pub use scalar::{
    acos, asin, atan2, ceil, cos, exp, exp2, floor, fract, hypot, ln, log2, mul_add, powf, powi,
    rem_euclid, round, sin, sin_cos, sqrt, tan, trunc,
};
