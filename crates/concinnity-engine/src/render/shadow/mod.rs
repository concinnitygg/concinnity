//! Shadow-map preparation: cascade fitting and its re-render schedule, the far
//! shadow past the cascades, the spot shadow array, and the depth-bias state
//! every shadow pass binds.

pub mod bias;
pub mod csm;
pub mod far;
pub(crate) mod receivers;
pub(crate) mod schedule;
pub mod spot;
