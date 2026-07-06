// src/gfx/anim_graph/mod.rs
//
// Animation state-machine runtime: the compiled graph representation plus the
// cursor that walks it. Pure math/state code with no asset or ECS
// dependencies -- the `AnimGraph` asset compiles into a `CompiledGraph` (see
// `assets/anim_graph.rs`), and the client's AnimationSystem owns one
// `GraphCursor` per graph target, advancing it each frame and sampling the
// blended pose through `sample_graph_pose`. Blendspace member math lives in
// `blend`, pose sampling in `sample`.

mod blend;
mod sample;
#[cfg(test)]
mod tests;

pub use blend::{Blend1D, Blend2D, ClipPlay, StatePlay, blend1d_weights, blend2d_weights};
pub use sample::sample_graph_pose;

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

// An in-flight crossfade from the previous state. The outgoing state's clock
// keeps advancing during the fade so its pose stays live rather than frozen.
#[derive(Debug, Clone)]
pub struct StateFade {
    pub from_state: usize,
    pub from_clock: f32,
    pub elapsed_secs: f32,
    pub duration_secs: f32,
}

impl StateFade {
    // Fade progress in [0, 1]: 0 = all outgoing pose, 1 = all incoming.
    pub fn progress(&self) -> f32 {
        (self.elapsed_secs / self.duration_secs.max(1e-6)).clamp(0.0, 1.0)
    }
}

// The live position in a graph: current state, its clock, and any in-flight
// crossfade. One per graph target, owned by the client's AnimationSystem.
//
// Clock units depend on the state's play: seconds (scaled by `rate`) for
// single clips and non-sync blendspaces, normalized phase (one full pass of
// the blend = 1.0) for phase-synced blendspaces, where member clips of
// different lengths must share one wrap point.
#[derive(Debug, Clone)]
pub struct GraphCursor {
    pub state: usize,
    pub clock: f32,
    pub fade: Option<StateFade>,
}

impl GraphCursor {
    // A cursor parked at the graph's initial state.
    pub fn start(graph: &CompiledGraph) -> Self {
        Self {
            state: graph.initial,
            clock: 0.0,
            fade: None,
        }
    }

    // Advance clocks by `dt_secs` and take at most one transition. Transition
    // checks run against the current state's outgoing list in declaration
    // order; the first whose exit-time gate and conditions all pass wins.
    // Taking a transition while a fade is in flight replaces the fade: the
    // new fade blends from the interrupted fade's *incoming* state only, so a
    // rapid double transition can pop the older outgoing pose.
    pub fn advance(&mut self, graph: &CompiledGraph, params: &[f32], dt_secs: f32) {
        let dt = dt_secs.max(0.0);
        let state = &graph.states[self.state];
        advance_clock(state, params, dt, &mut self.clock);
        if let Some(fade) = self.fade.as_mut() {
            fade.elapsed_secs += dt;
            advance_clock(
                &graph.states[fade.from_state],
                params,
                dt,
                &mut fade.from_clock,
            );
            if fade.elapsed_secs >= fade.duration_secs {
                self.fade = None;
            }
        }

        let normalized = normalized_time(state, self.clock, params);
        for tr in &state.transitions {
            if let Some(gate) = tr.exit_time
                && normalized < gate
            {
                continue;
            }
            let hold = tr.conditions.iter().any(|c| {
                let lhs = params.get(c.param).copied().unwrap_or(0.0);
                !c.op.eval(lhs, c.value)
            });
            if hold {
                continue;
            }
            self.fade = (tr.duration_secs > 0.0).then_some(StateFade {
                from_state: self.state,
                from_clock: self.clock,
                elapsed_secs: 0.0,
                duration_secs: tr.duration_secs,
            });
            self.state = tr.to;
            self.clock = 0.0;
            break;
        }
    }
}

// Advance one state's clock: seconds for clips and non-sync blends,
// normalized phase for synced blends (dividing by the blend's current
// effective duration keeps one wall-clock second worth of playback per
// second regardless of which members dominate).
fn advance_clock(state: &CompiledState, params: &[f32], dt: f32, clock: &mut f32) {
    if state.play.sync() {
        let weights = state.play.weights(params);
        let eff = state.play.effective_duration(&weights).max(1e-6);
        *clock += dt * state.rate / eff;
    } else {
        *clock += dt * state.rate;
    }
}

// A state's normalized time in [0, 1]: the fraction of one full pass covered
// by the clock. Looping states report the fraction within the current pass,
// so an `exit_time` gate re-opens every loop; non-looping states saturate at
// 1. Blendspace passes are measured against the weight-averaged member
// duration at the current parameters.
pub fn normalized_time(state: &CompiledState, clock: f32, params: &[f32]) -> f32 {
    let phase = if state.play.sync() {
        clock
    } else {
        let weights = state.play.weights(params);
        let eff = state.play.effective_duration(&weights);
        if eff <= 1e-6 {
            return 1.0;
        }
        clock / eff
    };
    if state.looping {
        phase.fract()
    } else {
        phase.clamp(0.0, 1.0)
    }
}
