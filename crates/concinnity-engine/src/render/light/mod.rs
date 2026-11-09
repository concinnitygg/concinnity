//! Light preparation: the drained light components packed into the GPU light
//! block, area lights and their LTC tables, the cluster grid's depth range,
//! and the spheres that keep part of the sky's light off their surroundings.

pub(crate) mod area;
#[cfg(test)]
mod bench;
pub(crate) mod cluster_range;
pub(crate) mod directional;
pub mod lights;
pub mod ltc;
pub(crate) mod point;
pub(crate) mod rect_area;
pub mod sky_occluders;
pub(crate) mod spot;
pub mod state;
