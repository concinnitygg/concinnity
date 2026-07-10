// src/assets/anim_graph.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};
use crate::gfx::anim_graph::{
    Blend1D, Blend2D, ClipPlay, CmpOp, CompiledCondition, CompiledGraph, CompiledState,
    CompiledTransition, ParamSpec, StatePlay,
};

/// A named float parameter driving a graph's transitions. Gameplay systems
/// (or the `anim-param` debug command) write parameter values at runtime;
/// transitions compare against them. Flag-like parameters use 0 and 1.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GraphParam {
    /// Parameter name, referenced by transition conditions.
    pub name: String,
    /// Initial value at world start.
    pub default: f32,
}

/// One member of a 1D blendspace: a clip pinned at a parameter `value`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GraphBlendPoint {
    /// Parameter value at which this clip plays alone.
    pub value: f32,
    /// The [Animation](#animation) clip at this point. Must target the same
    /// [SkinnedMesh](#skinnedmesh) as the graph.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub clip: Option<AssetId>,
}

/// A blendspace: several clips mixed continuously by parameter value instead
/// of one clip per state. `kind` selects the shape.
///
/// With `sync` true the members share one normalized phase clock -- a walk
/// and a run cycle stay foot-aligned while the blend moves between them, so
/// speed changes do not slide the feet. Leave it false for members that are
/// not cyclic gaits.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum GraphBlend {
    /// Clips along one parameter. The parameter picks the two neighbouring
    /// `points` (by ascending `value`) and blends them; outside the range
    /// the nearest end clip plays alone.
    Blend1d {
        /// Name of the declared graph parameter driving the blend.
        parameter: String,
        /// Members in ascending `value` order.
        points: Vec<GraphBlendPoint>,
        /// Phase-sync the members (see above).
        #[serde(default)]
        sync: bool,
    },
    /// Clips on a regular grid over two parameters, blended bilinearly
    /// between the four grid neighbours of the parameter point (clamped at
    /// the grid edges).
    Blend2d {
        /// Name of the parameter along the grid's x axis.
        parameter_x: String,
        /// Name of the parameter along the grid's y axis.
        parameter_y: String,
        /// Ascending x-axis sample positions, one per grid column.
        x_values: Vec<f32>,
        /// Ascending y-axis sample positions, one per grid row.
        y_values: Vec<f32>,
        /// One row of [Animation](#animation) clip names per `y_values`
        /// entry, each row holding one clip per `x_values` entry.
        rows: Vec<Vec<AssetId>>,
        /// Phase-sync the members (see above).
        #[serde(default)]
        sync: bool,
    },
}

/// One state of the graph: while active it plays either a single
/// [Animation](#animation) `clip` or a `blend` (a blendspace mixing several
/// clips by parameter value). Exactly one of the two must be set.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GraphState {
    /// State name, referenced by `initial` and by transitions.
    pub name: String,
    /// The [Animation](#animation) clip this state plays. Must target the
    /// same [SkinnedMesh](#skinnedmesh) as the graph. Leave unset when the
    /// state plays a `blend` instead.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub clip: Option<AssetId>,
    /// A blendspace to play instead of a single `clip`.
    pub blend: Option<GraphBlend>,
    /// Playback speed scale; 1.0 plays at authored speed.
    pub rate: f32,
    /// Overrides the loop mode while this state plays: a single `clip`
    /// defaults to its own `looping` flag, a `blend` defaults to looping.
    pub loop_override: Option<bool>,
}

impl Default for GraphState {
    fn default() -> Self {
        Self {
            name: String::new(),
            clip: None,
            blend: None,
            rate: 1.0,
            loop_override: None,
        }
    }
}

/// One two-bone IK chain, pinning the chain's end joint (typically a foot)
/// to the ground the physics scene finds beneath it.
///
/// `joints` names the chain root, middle, and end in the target skeleton --
/// e.g. a hip, knee, and foot. The middle joint must be the direct child of
/// the root and the end the direct child of the middle. Every frame the
/// runtime probes straight down from the animated end joint; when a surface
/// is within range, the chain bends so the end lands `foot_height` above it.
/// Pinning pauses automatically while the character is airborne.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GraphIkChain {
    /// Names of the chain's root, middle, and end joints, in order. Exactly
    /// three are required, matching the target skeleton's joint names.
    pub joints: Vec<String>,
    /// Bend direction in mesh space: the middle joint bows toward this
    /// vector (a knee points forward, an elbow backward).
    pub pole: [f32; 3],
    /// Name of a declared graph parameter scaling the solve in `[0, 1]`;
    /// empty pins at full strength. Lets gameplay fade IK in and out.
    pub weight_parameter: String,
    /// Height the end joint rests above the probed surface, in mesh units
    /// (the sole-to-ankle offset for a foot).
    pub foot_height: f32,
}

impl Default for GraphIkChain {
    fn default() -> Self {
        Self {
            joints: Vec::new(),
            pole: [0.0, 0.0, 1.0],
            weight_parameter: String::new(),
            foot_height: 0.0,
        }
    }
}

/// One transition condition, `parameter <op> value`. All of a transition's
/// conditions must pass for it to fire.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GraphCondition {
    /// Name of a declared graph parameter.
    pub parameter: String,
    /// Comparison operator: `lt`, `le`, `gt`, `ge`, `eq`, or `ne`.
    pub op: CmpOp,
    /// Right-hand side of the comparison.
    pub value: f32,
}

/// One directed transition between two states.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GraphTransition {
    /// Source state name.
    pub from: String,
    /// Destination state name.
    pub to: String,
    /// Crossfade length in seconds between the outgoing and incoming poses.
    /// Zero snaps to the new state's pose immediately.
    pub duration_secs: f32,
    /// When set (0 to 1), the transition waits until the source state has
    /// played this fraction of its clip. On a looping state the gate re-opens
    /// every loop; on a non-looping state it stays open once reached. Useful
    /// for letting a clip finish before leaving, e.g. `0.9` on a jump.
    pub exit_time: Option<f32>,
    /// Conditions that must all pass (in addition to any `exit_time` gate).
    /// An empty list always passes.
    pub conditions: Vec<GraphCondition>,
}

/// An animation state machine for one [SkinnedMesh](#skinnedmesh).
///
/// While a plain set of [Animation](#animation) clips blends every clip all
/// the time, a graph plays exactly one *state* at a time and moves between
/// states along declared transitions, crossfading poses over each
/// transition's `duration_secs`. Transitions fire when their conditions --
/// comparisons against the graph's named float `parameters` -- pass. Gameplay
/// systems write parameter values each frame (the `anim-param` debug command
/// does the same from a `cn debug` session).
///
/// A graph owns its target: every [Animation](#animation) targeting the
/// graph's mesh must be referenced by exactly one state, and at most one
/// graph may target a given mesh (both are build errors otherwise). Clip
/// `weight` and `fade_in_secs` have no effect under a graph.
///
/// Transitions are checked in declaration order and the first match wins.
/// A state with no outgoing transitions (or none passing) keeps playing;
/// looping states wrap, non-looping states hold their final pose.
///
/// ```jsonl
/// // Two single-clip states:
/// {"name":"hero_graph","type":"AnimGraph","args":{
///   "target":"hero",
///   "parameters":[{"name":"speed","default":0.0}],
///   "initial":"idle",
///   "states":[
///     {"name":"idle","clip":"hero_idle"},
///     {"name":"run","clip":"hero_run","rate":1.1}
///   ],
///   "transitions":[
///     {"from":"idle","to":"run","duration_secs":0.2,
///      "conditions":[{"parameter":"speed","op":"gt","value":0.5}]},
///     {"from":"run","to":"idle","duration_secs":0.3,
///      "conditions":[{"parameter":"speed","op":"le","value":0.5}]}
///   ]
/// }}
/// // One locomotion blendspace state mixing idle/walk/run by speed:
/// {"name":"hero_graph","type":"AnimGraph","args":{
///   "target":"hero",
///   "parameters":[{"name":"speed","default":0.0}],
///   "states":[
///     {"name":"locomotion","blend":{"kind":"blend1d","parameter":"speed","sync":true,
///      "points":[
///        {"value":0.0,"clip":"hero_idle"},
///        {"value":1.6,"clip":"hero_walk"},
///        {"value":5.0,"clip":"hero_run"}
///      ]}}
///   ]
/// }}
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AnimGraph {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [SkinnedMesh](#skinnedmesh) asset this graph animates.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub target: Option<AssetId>,
    /// Named float parameters transitions compare against.
    pub parameters: Vec<GraphParam>,
    /// Name of the state the graph starts in. Defaults to the first state.
    pub initial: String,
    /// The graph's states. At least one is required.
    pub states: Vec<GraphState>,
    /// Directed transitions between states.
    pub transitions: Vec<GraphTransition>,
    /// Two-bone IK chains applied on top of every state's pose; see
    /// [GraphIkChain](#graphikchain).
    pub ik_chains: Vec<GraphIkChain>,
}

impl AnimGraph {
    /// Compile the authored graph into the runtime representation, resolving
    /// state and parameter names to indices and clip references through
    /// `resolve_clip`, which maps an [Animation](#animation) asset id to its
    /// index, duration, and looping flag in the target's clip list. Structural
    /// problems (unknown names, missing clips, non-positive rates) are
    /// reported as errors; the build validates the same rules earlier, so a
    /// runtime failure here means the world blob and the clip list disagree.
    pub fn compile(
        &self,
        resolve_clip: impl Fn(AssetId) -> Option<(usize, f32, bool)>,
    ) -> Result<CompiledGraph, String> {
        let ctx = |detail: String| format!("AnimGraph {}: {detail}", self.asset_id);
        if self.states.is_empty() {
            return Err(ctx("graph has no states".into()));
        }

        let params: Vec<ParamSpec> = self
            .parameters
            .iter()
            .map(|p| ParamSpec {
                name: p.name.clone(),
                default: p.default,
            })
            .collect();
        let param_index = |name: &str| params.iter().position(|p| p.name == name);
        let state_index = |name: &str| self.states.iter().position(|s| s.name == name);

        let mut states: Vec<CompiledState> = Vec::with_capacity(self.states.len());
        for s in &self.states {
            if s.rate <= 0.0 {
                return Err(ctx(format!("state '{}': rate must be positive", s.name)));
            }
            // The member resolver: an Animation reference -> a ClipPlay,
            // shared by the single-clip and blendspace arms.
            let play_for = |clip_id: AssetId| -> Result<(ClipPlay, bool), String> {
                let Some((clip, duration_secs, clip_looping)) = resolve_clip(clip_id) else {
                    return Err(ctx(format!(
                        "state '{}': clip {clip_id} is not a clip on the graph's target",
                        s.name
                    )));
                };
                Ok((
                    ClipPlay {
                        clip,
                        duration_secs,
                    },
                    clip_looping,
                ))
            };
            let (play, default_looping) = match (&s.clip, &s.blend) {
                (Some(_), Some(_)) => {
                    return Err(ctx(format!(
                        "state '{}' sets both `clip` and `blend`; pick one",
                        s.name
                    )));
                }
                (None, None) => {
                    return Err(ctx(format!("state '{}' has no `clip` or `blend`", s.name)));
                }
                (Some(clip_id), None) => {
                    let (clip_play, clip_looping) = play_for(*clip_id)?;
                    (StatePlay::Clip(clip_play), clip_looping)
                }
                // Blendspaces default to looping (their members are cyclic
                // gaits far more often than one-shots).
                (None, Some(blend)) => (compile_blend(s, blend, &param_index, &play_for)?, true),
            };
            states.push(CompiledState {
                name: s.name.clone(),
                rate: s.rate,
                looping: s.loop_override.unwrap_or(default_looping),
                play,
                transitions: Vec::new(),
            });
        }

        for t in &self.transitions {
            let Some(from) = state_index(&t.from) else {
                return Err(ctx(format!("transition from unknown state '{}'", t.from)));
            };
            let Some(to) = state_index(&t.to) else {
                return Err(ctx(format!("transition to unknown state '{}'", t.to)));
            };
            let mut conditions = Vec::with_capacity(t.conditions.len());
            for c in &t.conditions {
                let Some(param) = param_index(&c.parameter) else {
                    return Err(ctx(format!(
                        "transition '{}' -> '{}' references undeclared parameter '{}'",
                        t.from, t.to, c.parameter
                    )));
                };
                conditions.push(CompiledCondition {
                    param,
                    op: c.op,
                    value: c.value,
                });
            }
            states[from].transitions.push(CompiledTransition {
                to,
                duration_secs: t.duration_secs.max(0.0),
                exit_time: t.exit_time,
                conditions,
            });
        }

        let initial = if self.initial.is_empty() {
            0
        } else {
            state_index(&self.initial)
                .ok_or_else(|| ctx(format!("initial state '{}' not found", self.initial)))?
        };

        Ok(CompiledGraph {
            params,
            states,
            initial,
        })
    }
}

// Compile one blendspace node: parameter and clip names resolve to indices,
// axis positions must ascend strictly, and 2D grids must be complete.
fn compile_blend(
    state: &GraphState,
    blend: &GraphBlend,
    param_index: &impl Fn(&str) -> Option<usize>,
    play_for: &impl Fn(AssetId) -> Result<(ClipPlay, bool), String>,
) -> Result<StatePlay, String> {
    let err = |detail: String| format!("state '{}': {detail}", state.name);
    let param = |name: &str, axis: &str| {
        param_index(name)
            .ok_or_else(|| err(format!("blend {axis} '{name}' is not a declared parameter")))
    };
    let strictly_ascending = |v: &[f32]| v.windows(2).all(|w| w[0] < w[1]);
    let member = |clip: Option<AssetId>| -> Result<ClipPlay, String> {
        let id = clip.ok_or_else(|| err("blend member has no `clip`".into()))?;
        Ok(play_for(id)?.0)
    };

    match blend {
        GraphBlend::Blend1d {
            parameter,
            points,
            sync,
        } => {
            if points.is_empty() {
                return Err(err("blend has no `points`".into()));
            }
            let thresholds: Vec<f32> = points.iter().map(|p| p.value).collect();
            if !strictly_ascending(&thresholds) {
                return Err(err("blend point `value`s must be strictly ascending".into()));
            }
            let plays = points
                .iter()
                .map(|p| member(p.clip))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(StatePlay::Blend1D(Blend1D {
                param: param(parameter, "parameter")?,
                thresholds,
                plays,
                sync: *sync,
            }))
        }
        GraphBlend::Blend2d {
            parameter_x,
            parameter_y,
            x_values,
            y_values,
            rows,
            sync,
        } => {
            if x_values.is_empty() || y_values.is_empty() {
                return Err(err("blend `x_values` / `y_values` must not be empty".into()));
            }
            if !strictly_ascending(x_values) || !strictly_ascending(y_values) {
                return Err(err(
                    "blend `x_values` and `y_values` must be strictly ascending".into(),
                ));
            }
            if rows.len() != y_values.len() || rows.iter().any(|r| r.len() != x_values.len()) {
                return Err(err(format!(
                    "blend `rows` must be {} row(s) of {} clip(s) to match the grid",
                    y_values.len(),
                    x_values.len()
                )));
            }
            let plays = rows
                .iter()
                .flatten()
                .map(|&clip| member(Some(clip)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(StatePlay::Blend2D(Blend2D {
                param_x: param(parameter_x, "parameter_x")?,
                param_y: param(parameter_y, "parameter_y")?,
                x_values: x_values.clone(),
                y_values: y_values.clone(),
                plays,
                sync: *sync,
            }))
        }
    }
}

impl Component for AnimGraph {
    const NAME: &'static str = "AnimGraph";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_json() -> serde_json::Value {
        serde_json::json!({
            "target": "hero",
            "parameters": [{"name": "speed", "default": 0.5}],
            "initial": "idle",
            "states": [
                {"name": "idle", "clip": "hero_idle"},
                {"name": "run", "clip": "hero_run", "rate": 1.5, "loop_override": false}
            ],
            "transitions": [
                {"from": "idle", "to": "run", "duration_secs": 0.2, "exit_time": 0.5,
                 "conditions": [{"parameter": "speed", "op": "gt", "value": 1.0}]}
            ]
        })
    }

    // Maps every clip id to slot 0 of a 1-second looping clip.
    fn any_clip(_: AssetId) -> Option<(usize, f32, bool)> {
        Some((0, 1.0, true))
    }

    #[test]
    fn deserialises_full_graph() {
        crate::ecs::asset_id::reset_interner();
        let g: AnimGraph = serde_json::from_value(graph_json()).unwrap();
        assert!(g.target.is_some());
        assert_eq!(g.parameters.len(), 1);
        assert_eq!(g.states.len(), 2);
        assert_eq!(g.states[1].rate, 1.5);
        assert_eq!(g.states[1].loop_override, Some(false));
        assert_eq!(g.transitions.len(), 1);
        assert_eq!(g.transitions[0].exit_time, Some(0.5));
        assert_eq!(g.transitions[0].conditions[0].op, CmpOp::Gt);
    }

    #[test]
    fn deserialises_with_defaults() {
        let g: AnimGraph = serde_json::from_str("{}").unwrap();
        assert!(g.target.is_none());
        assert!(g.states.is_empty());
        assert!(g.initial.is_empty());
    }

    #[test]
    fn compiles_names_to_indices() {
        crate::ecs::asset_id::reset_interner();
        let g: AnimGraph = serde_json::from_value(graph_json()).unwrap();
        let compiled = g.compile(any_clip).unwrap();
        assert_eq!(compiled.initial, 0);
        assert_eq!(compiled.states[0].transitions.len(), 1);
        let tr = &compiled.states[0].transitions[0];
        assert_eq!(tr.to, 1);
        assert_eq!(tr.conditions[0].param, 0);
        // loop_override false beats the clip's own looping flag.
        assert!(!compiled.states[1].looping);
        assert!(compiled.states[0].looping);
    }

    #[test]
    fn compile_empty_initial_defaults_to_first_state() {
        crate::ecs::asset_id::reset_interner();
        let mut v = graph_json();
        v["initial"] = serde_json::json!("");
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert_eq!(g.compile(any_clip).unwrap().initial, 0);
    }

    #[test]
    fn compile_rejects_unknown_names() {
        crate::ecs::asset_id::reset_interner();
        let mut v = graph_json();
        v["transitions"][0]["to"] = serde_json::json!("ghost");
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert!(g.compile(any_clip).unwrap_err().contains("ghost"));

        let mut v = graph_json();
        v["transitions"][0]["conditions"][0]["parameter"] = serde_json::json!("nope");
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert!(g.compile(any_clip).unwrap_err().contains("nope"));

        let mut v = graph_json();
        v["initial"] = serde_json::json!("ghost");
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert!(g.compile(any_clip).unwrap_err().contains("ghost"));
    }

    #[test]
    fn compile_rejects_unresolvable_clip_and_bad_rate() {
        crate::ecs::asset_id::reset_interner();
        let g: AnimGraph = serde_json::from_value(graph_json()).unwrap();
        assert!(g.compile(|_| None).unwrap_err().contains("clip"));

        let mut v = graph_json();
        v["states"][0]["rate"] = serde_json::json!(0.0);
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert!(g.compile(any_clip).unwrap_err().contains("rate"));
    }

    #[test]
    fn compile_rejects_empty_graph() {
        let g = AnimGraph::default();
        assert!(g.compile(any_clip).unwrap_err().contains("no states"));
    }

    fn blend1d_graph_json() -> serde_json::Value {
        serde_json::json!({
            "target": "hero",
            "parameters": [{"name": "speed", "default": 0.0}],
            "states": [
                {"name": "locomotion", "blend": {"kind": "blend1d", "parameter": "speed",
                 "sync": true,
                 "points": [
                     {"value": 0.0, "clip": "idle"},
                     {"value": 1.6, "clip": "walk"},
                     {"value": 5.0, "clip": "run"}
                 ]}}
            ]
        })
    }

    fn blend2d_graph_json() -> serde_json::Value {
        serde_json::json!({
            "target": "hero",
            "parameters": [{"name": "speed"}, {"name": "strafe"}],
            "states": [
                {"name": "locomotion", "blend": {"kind": "blend2d",
                 "parameter_x": "speed", "parameter_y": "strafe",
                 "x_values": [0.0, 5.0], "y_values": [-1.0, 1.0],
                 "rows": [["run_l", "run_l"], ["run_r", "run_r"]]}}
            ]
        })
    }

    #[test]
    fn compiles_blend1d_state() {
        crate::ecs::asset_id::reset_interner();
        let g: AnimGraph = serde_json::from_value(blend1d_graph_json()).unwrap();
        let compiled = g.compile(any_clip).unwrap();
        let StatePlay::Blend1D(b) = &compiled.states[0].play else {
            panic!("expected a 1D blendspace");
        };
        assert_eq!(b.param, 0);
        assert_eq!(b.thresholds, vec![0.0, 1.6, 5.0]);
        assert_eq!(b.plays.len(), 3);
        assert!(b.sync);
        assert!(compiled.states[0].looping, "blendspaces default to looping");
    }

    #[test]
    fn compiles_blend2d_state() {
        crate::ecs::asset_id::reset_interner();
        let g: AnimGraph = serde_json::from_value(blend2d_graph_json()).unwrap();
        let compiled = g.compile(any_clip).unwrap();
        let StatePlay::Blend2D(b) = &compiled.states[0].play else {
            panic!("expected a 2D blendspace");
        };
        assert_eq!((b.param_x, b.param_y), (0, 1));
        assert_eq!(b.plays.len(), 4);
        assert!(!b.sync);
    }

    #[test]
    fn compile_rejects_clip_and_blend_together_or_neither() {
        crate::ecs::asset_id::reset_interner();
        let mut v = blend1d_graph_json();
        v["states"][0]["clip"] = serde_json::json!("idle");
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert!(g.compile(any_clip).unwrap_err().contains("pick one"));

        let v = serde_json::json!({"target":"hero","states":[{"name":"empty"}]});
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert!(
            g.compile(any_clip)
                .unwrap_err()
                .contains("no `clip` or `blend`")
        );
    }

    #[test]
    fn compile_rejects_unsorted_blend_points() {
        crate::ecs::asset_id::reset_interner();
        let mut v = blend1d_graph_json();
        v["states"][0]["blend"]["points"][2]["value"] = serde_json::json!(1.0);
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert!(g.compile(any_clip).unwrap_err().contains("ascending"));
    }

    #[test]
    fn compile_rejects_undeclared_blend_parameter() {
        crate::ecs::asset_id::reset_interner();
        let mut v = blend1d_graph_json();
        v["states"][0]["blend"]["parameter"] = serde_json::json!("nope");
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert!(g.compile(any_clip).unwrap_err().contains("nope"));
    }

    #[test]
    fn compile_rejects_mismatched_grid_rows() {
        crate::ecs::asset_id::reset_interner();
        let mut v = blend2d_graph_json();
        v["states"][0]["blend"]["rows"] = serde_json::json!([["a", "b"]]);
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert!(g.compile(any_clip).unwrap_err().contains("rows"));
    }
}
