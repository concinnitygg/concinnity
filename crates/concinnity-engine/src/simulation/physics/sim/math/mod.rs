// The simulation's own math, plain f32 arithmetic and `libm` rather than std's
// platform transcendentals, so results are bit-identical across platforms.

mod mat3;
mod quat;
mod vec3;

pub(crate) use mat3::Mat3;
pub(crate) use quat::Quat;
pub(crate) use vec3::{Vec3, vec3};
