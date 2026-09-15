// The compiled state machine: its states, transitions, and parameters, plus the
// clip-duration refresh a hot-reload applies.

use super::StatePlay;
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
    /// Evaluate `lhs <op> rhs`.
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

/// One compiled transition condition: parameter names are resolved to indices
/// into the graph's parameter vector at compile time, so evaluation is a
/// direct slice read (the runtime blob interner is empty; nothing resolves
/// names at runtime).
#[derive(Debug, Clone)]
pub struct CompiledCondition {
    /// Index into the graph's parameter vector.
    pub param: usize,
    /// Comparison applied between the parameter and `value`.
    pub op: CmpOp,
    /// The value the parameter is compared against.
    pub value: f32,
}

/// One compiled outgoing transition. Conditions AND together; an empty list is
/// always true (useful with `exit_time` alone).
#[derive(Debug, Clone)]
pub struct CompiledTransition {
    /// Index of the state this transition enters.
    pub to: usize,
    /// Crossfade length between the outgoing and incoming state poses. Zero
    /// snaps.
    pub duration_secs: f32,
    /// When set, the transition is gated until the state's normalized time
    /// reaches this value (see `normalized_time`).
    pub exit_time: Option<f32>,
    /// Conditions that must all hold; an empty list is always true.
    pub conditions: Vec<CompiledCondition>,
}

/// One compiled state: what it plays (a clip or a blendspace, at `rate`) and
/// its outgoing transitions in declaration order (first match wins).
#[derive(Debug, Clone)]
pub struct CompiledState {
    /// The state's authored name.
    pub name: String,
    /// Clock scale; 1.0 plays at authored speed.
    pub rate: f32,
    /// Whether this state's clock wraps. Single-clip states default to the
    /// clip's own flag, blendspaces default to wrapping; `loop_override`
    /// wins over both.
    pub looping: bool,
    /// What the state plays.
    pub play: StatePlay,
    /// Outgoing transitions in declaration order; first match wins.
    pub transitions: Vec<CompiledTransition>,
}

/// A graph parameter: a named float, seeded to `default`.
#[derive(Debug, Clone)]
pub struct ParamSpec {
    /// The parameter's authored name.
    pub name: String,
    /// Value the parameter is seeded with.
    pub default: f32,
}

/// A compiled animation state machine. Built once from the `AnimationGraph` asset;
/// read-only afterwards except for clip-duration refresh on hot-reload.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    /// The graph's parameters, in index order.
    pub params: Vec<ParamSpec>,
    /// The graph's states, in index order.
    pub states: Vec<CompiledState>,
    /// Index of the state the graph starts in.
    pub initial: usize,
}

impl CompiledGraph {
    /// Index of a parameter by name, for surfaces (debug commands) that still
    /// speak names.
    pub fn param_index(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    /// The parameter vector seeded to each parameter's declared default.
    pub fn default_params(&self) -> Vec<f32> {
        self.params.iter().map(|p| p.default).collect()
    }

    /// Update every member playing `clip` to a new duration, so wrap and
    /// exit-time math keep tracking a hot-reloaded clip.
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
