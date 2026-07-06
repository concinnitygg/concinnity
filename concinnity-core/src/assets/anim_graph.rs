// src/assets/anim_graph.rs

use crate::check::cross_reference::{CrossRef, CrossReferenced, RefKind};
use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};
use crate::gfx::anim_graph::{
    CmpOp, CompiledCondition, CompiledGraph, CompiledState, CompiledTransition, ParamSpec,
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

/// One state of the graph: a single [Animation](#animation) clip played on a
/// loop (or once) while the state is active.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GraphState {
    /// State name, referenced by `initial` and by transitions.
    pub name: String,
    /// The [Animation](#animation) clip this state plays. Must target the
    /// same [SkinnedMesh](#skinnedmesh) as the graph.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub clip: Option<AssetId>,
    /// Playback speed scale; 1.0 plays the clip at its authored speed.
    pub rate: f32,
    /// Overrides the clip's own `looping` flag while this state plays. Leave
    /// unset to keep the clip's flag.
    pub loop_override: Option<bool>,
}

impl Default for GraphState {
    fn default() -> Self {
        Self {
            name: String::new(),
            clip: None,
            rate: 1.0,
            loop_override: None,
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
            let Some(clip_id) = s.clip else {
                return Err(ctx(format!("state '{}' has no clip", s.name)));
            };
            let Some((clip, duration_secs, clip_looping)) = resolve_clip(clip_id) else {
                return Err(ctx(format!(
                    "state '{}': clip {clip_id} is not a clip on the graph's target",
                    s.name
                )));
            };
            if s.rate <= 0.0 {
                return Err(ctx(format!("state '{}': rate must be positive", s.name)));
            }
            states.push(CompiledState {
                name: s.name.clone(),
                clip,
                rate: s.rate,
                looping: s.loop_override.unwrap_or(clip_looping),
                duration_secs,
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

impl CrossReferenced for AnimGraph {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let mut refs = Vec::new();
        match args.get("target").and_then(|v| v.as_str()).unwrap_or("") {
            "" => refs.push(CrossRef::Issue(format!(
                "AnimGraph '{name}': `target` field is required (the SkinnedMesh to animate)"
            ))),
            target => refs.push(CrossRef::Resolve {
                kind: RefKind::SkinnedMesh,
                target: target.to_string(),
                error: format!("AnimGraph '{name}': target SkinnedMesh '{target}' not found"),
            }),
        }
        let states = args
            .get("states")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        for (i, state) in states.iter().enumerate() {
            let state_name = state.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let label = if state_name.is_empty() {
                format!("state #{i}")
            } else {
                format!("state '{state_name}'")
            };
            match state.get("clip").and_then(|v| v.as_str()).unwrap_or("") {
                "" => refs.push(CrossRef::Issue(format!(
                    "AnimGraph '{name}': {label} has no `clip` (an Animation asset name)"
                ))),
                clip => refs.push(CrossRef::Resolve {
                    kind: RefKind::Animation,
                    target: clip.to_string(),
                    error: format!("AnimGraph '{name}': {label} clip '{clip}' not found"),
                }),
            }
        }
        refs
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
        let mut v = graph_json();
        v["initial"] = serde_json::json!("");
        let g: AnimGraph = serde_json::from_value(v).unwrap();
        assert_eq!(g.compile(any_clip).unwrap().initial, 0);
    }

    #[test]
    fn compile_rejects_unknown_names() {
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

    #[test]
    fn cross_refs_cover_target_and_clips() {
        let refs = AnimGraph::cross_refs("g", &graph_json());
        // One target resolve + two clip resolves.
        assert_eq!(refs.len(), 3);
        assert!(refs.iter().all(|r| matches!(r, CrossRef::Resolve { .. })));
    }

    #[test]
    fn cross_refs_flag_missing_target_and_clip() {
        let refs = AnimGraph::cross_refs("g", &serde_json::json!({"states":[{"name":"idle"}]}));
        let issues: Vec<_> = refs
            .iter()
            .filter_map(|r| match r {
                CrossRef::Issue(msg) => Some(msg.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(issues.len(), 2);
        assert!(issues[0].contains("target"));
        assert!(issues[1].contains("clip"));
    }
}
