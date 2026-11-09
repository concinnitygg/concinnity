//! Shadow-map preparation: cascade fitting and its re-render schedule, the spot
//! shadow array, and the depth-bias state every shadow pass binds.

pub mod bias;
pub mod csm;
pub(crate) mod schedule;
pub mod spot;
