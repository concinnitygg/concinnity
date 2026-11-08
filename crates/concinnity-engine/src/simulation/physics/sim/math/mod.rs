// The simulation's own small math layer. It is here rather than borrowed from
// a general math crate because the simulation asks one thing of its arithmetic
// that a general crate does not promise: identical bits on every platform.
// That rules out the std float methods, whose transcendentals come from the
// platform's libm and differ in the last unit in the last place between glibc,
// musl, MSVC, and Apple; every function below is plain f32 arithmetic or a
// `libm` call.

mod mat3;
mod quat;
mod vec3;

pub(crate) use mat3::Mat3;
pub(crate) use quat::Quat;
pub(crate) use vec3::{Vec3, vec3};
