//! Reflection probes: capture math, automatic placement, the probe book every
//! backend keeps, the staggered bake sequence, and the radiance storage cap.

pub mod bake;
pub mod book;
pub mod capture;
pub mod radiance;
pub(crate) mod reflection_probe;
pub mod seed;
