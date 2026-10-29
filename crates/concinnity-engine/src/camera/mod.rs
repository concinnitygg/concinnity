//! The camera: the 3D camera an entity views the world through, the probe that
//! reports what it is looking at, and the scripted track that can drive it.
//! The input-driven controllers that move it live in [`crate::controller`].

pub(crate) mod camera3d;
pub(crate) mod probe;
pub(crate) mod track;
