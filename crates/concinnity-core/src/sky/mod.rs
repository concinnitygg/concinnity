//! The rotating celestial sphere: the orientation a world's
//! [`SkyRotation`](crate::components::SkyRotation) is at this tick, and the
//! system that advances it.
//!
//! One rotation drives every consumer, so the sky, the image-based lighting it
//! provides, the directional lights, and the props hung on it never disagree:
//! the environment cubes are sampled through it, each directional light's
//! authored direction is turned by it, and the component's own entity carries
//! it as a transform for the hierarchy to compose.

mod orientation;
mod system;

pub use orientation::SkyOrientation;
pub use system::SkyRotationSystem;
