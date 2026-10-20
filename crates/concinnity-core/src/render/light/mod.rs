//! Light preparation: the drained light components packed into the GPU light
//! block, area lights and their LTC tables, and the cluster grid's depth range.

pub(crate) mod area;
pub(crate) mod cluster_range;
pub mod lights;
pub mod ltc;
pub mod state;
