// src/math/mod.rs
//
// The scalar and vector primitives the rest of the vocabulary is built from:
// the f32 transcendentals `core` leaves to a math library, and the
// 3-component vector ops every layout and transform reaches for.
mod scalar;
pub mod vec3;

pub use scalar::{
    acos, asin, atan2, cos, exp, exp2, floor, fract, powf, rem_euclid, sin, sin_cos, sqrt, tan,
};
