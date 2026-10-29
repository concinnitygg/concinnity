//! The animation state machine: what an `AnimationGraph` asset compiles into,
//! the blendspace members a state can play, and the cursor that walks it.
//!
//! A [`CompiledGraph`] is read-only once built, except for the clip-duration
//! refresh a hot-reload applies. A [`GraphCursor`] is the live position in one:
//! the client's animation system owns one per graph target, advances it each
//! frame, and samples the blended pose through [`sample_graph_pose_into`].

mod blend;
mod cursor;
mod graph;
mod root;
mod sample;
#[cfg(test)]
mod tests;

pub use blend::{Blend1D, Blend2D, ClipPlay, StatePlay, blend1d_weights, blend2d_weights};
pub use cursor::{GraphCursor, StateFade, normalized_time};
pub use graph::{
    CmpOp, CompiledCondition, CompiledGraph, CompiledState, CompiledTransition, ParamSpec,
};
pub use root::cursor_root_delta;
pub use sample::sample_graph_pose_into;
