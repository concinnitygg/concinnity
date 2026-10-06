//! Every debug verb, grouped by what it reaches: each group declares its verbs'
//! table entries beside the handlers that answer them.

pub(super) mod anim;
pub(super) mod camera;
pub(super) mod render;
pub(super) mod scene;
pub(super) mod settings;
pub(super) mod snapshot;
#[cfg(test)]
pub(super) mod testing;
