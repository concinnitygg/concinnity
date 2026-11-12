//! Shadow-map preparation: cascade fitting and its re-render schedule, the far
//! shadow past the cascades, the spot shadow array, and the depth-bias state
//! every shadow pass binds.

pub mod bias;
pub mod csm;
pub mod far;
pub mod far_anchor;
pub(crate) mod motion;
pub(crate) mod receivers;
pub(crate) mod schedule;
pub mod spot;
