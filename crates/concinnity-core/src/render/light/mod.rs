//! Light preparation: the drained light components packed into the GPU light
//! block, area lights and their LTC tables, and the cluster grid's depth range.

pub(crate) mod area;
pub(crate) mod cluster_range;
pub(crate) mod directional;
pub mod lights;
pub mod ltc;
pub(crate) mod point;
pub(crate) mod rect_area;
pub(crate) mod spot;
pub mod state;
