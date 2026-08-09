// src/gfx/anim_graph/mod.rs
//
// The compiled animation state machine: what an `AnimGraph` asset compiles
// into, and the blendspace members a state can play. Read-only once built,
// except for the clip-duration refresh a hot-reload applies.
//
// The cursor that walks a graph, the pose sampling, and the root-motion deltas
// are per-frame compute and live above this crate.

mod blend;

pub use blend::{Blend1D, Blend2D, ClipPlay, StatePlay, blend1d_weights, blend2d_weights};

use alloc::string::String;
use alloc::vec::Vec;

/// Comparison operator for a transition condition, evaluated as
/// `parameter <op> value`. `eq` / `ne` compare exactly and are mainly useful
/// for flag-like parameters holding whole numbers such as 0 and 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CmpOp {
    /// Less than.
    #[default]
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
    /// Equal (exact).
    Eq,
    /// Not equal (exact).
    Ne,
}

impl CmpOp {
    // Evaluate `lhs <op> rhs`.
    pub fn eval(self, lhs: f32, rhs: f32) -> bool {
        match self {
            CmpOp::Lt => lhs < rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Gt => lhs > rhs,
            CmpOp::Ge => lhs >= rhs,
            CmpOp::Eq => lhs == rhs,
            CmpOp::Ne => lhs != rhs,
        }
    }
}

// One compiled transition condition: parameter names are resolved to indices
// into the graph's parameter vector at compile time, so evaluation is a
// direct slice read (the runtime blob interner is empty; nothing resolves
// names at runtime).
#[derive(Debug, Clone)]
pub struct CompiledCondition {
    pub param: usize,
    pub op: CmpOp,
    pub value: f32,
}

// One compiled outgoing transition. Conditions AND together; an empty list is
// always true (useful with `exit_time` alone).
#[derive(Debug, Clone)]
pub struct CompiledTransition {
    pub to: usize,
    // Crossfade length between the outgoing and incoming state poses. Zero
    // snaps.
    pub duration_secs: f32,
    // When set, the transition is gated until the state's normalized time
    // reaches this value (see `normalized_time`).
    pub exit_time: Option<f32>,
    pub conditions: Vec<CompiledCondition>,
}

// One compiled state: what it plays (a clip or a blendspace, at `rate`) and
// its outgoing transitions in declaration order (first match wins).
#[derive(Debug, Clone)]
pub struct CompiledState {
    pub name: String,
    // Clock scale; 1.0 plays at authored speed.
    pub rate: f32,
    // Whether this state's clock wraps. Single-clip states default to the
    // clip's own flag, blendspaces default to wrapping; `loop_override`
    // wins over both.
    pub looping: bool,
    pub play: StatePlay,
    pub transitions: Vec<CompiledTransition>,
}

/// A graph parameter: a named float, seeded to `default`.
#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub name: String,
    pub default: f32,
}

// A compiled animation state machine. Built once from the `AnimGraph` asset;
// read-only afterwards except for clip-duration refresh on hot-reload.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    pub params: Vec<ParamSpec>,
    pub states: Vec<CompiledState>,
    pub initial: usize,
}

impl CompiledGraph {
    // Index of a parameter by name, for surfaces (debug commands) that still
    // speak names.
    pub fn param_index(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    // The parameter vector seeded to each parameter's declared default.
    pub fn default_params(&self) -> Vec<f32> {
        self.params.iter().map(|p| p.default).collect()
    }

    // Update every member playing `clip` to a new duration, so wrap and
    // exit-time math keep tracking a hot-reloaded clip.
    pub fn refresh_clip_duration(&mut self, clip: usize, duration_secs: f32) {
        for state in &mut self.states {
            for member in state.play.members_mut() {
                if member.clip == clip {
                    member.duration_secs = duration_secs;
                }
            }
        }
    }
}
