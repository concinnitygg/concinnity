// src/gfx/animation/ik.rs
//
// Foot-pinning IK for graph targets. At install time each authored
// `AnimationIkChain` resolves its joint names against the target's skeleton and
// its weight parameter against the graph's declaration order; per frame the
// chain's ground probe (answered by PhysicsSystem, one frame behind) turns
// into a mesh-space target for the analytic two-bone solve, applied to the
// sampled locals just before the skinning matrices.

use std::collections::{BTreeMap, HashMap};

use crate::components::{
    AnimationIkChain, AnimationParams, CharacterRig, GroundProbe, GroundProbes,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, SkinnedMeshHandle};
use crate::gfx::ik::TwoBoneChain;
use crate::gfx::pose_scratch::PoseScratch;
use crate::gfx::skeleton::Skeleton;
use crate::gfx::transform::{Mat4, mat4_affine_inverse};

// Probe ray extents around the animated foot: the ray starts `PROBE_UP`
// above it and reaches `PROBE_DOWN` below.
const PROBE_UP: f32 = 0.6;
const PROBE_DOWN: f32 = 0.6;
// Largest vertical correction a chain will apply; ground further from the
// animated foot than this leaves the pose alone.
const SNAP_THRESHOLD: f32 = 0.45;

// One resolved chain: solver indices plus its runtime knobs.
#[derive(Debug, Clone)]
pub(super) struct IkChainRuntime {
    pub chain: TwoBoneChain,
    // Index into the graph's parameter block, or `None` for full strength.
    pub(crate) weight_param: Option<usize>,
    pub(crate) foot_height: f32,
}

// Resolve the authored chains against the target's skeleton and parameter
// declarations. Unresolvable chains are dropped with a warning, so one bad
// name never takes down the rest.
pub(super) fn resolve_chains(
    graph_id: AssetId,
    authored: &[AnimationIkChain],
    parameters: &[crate::components::AnimationParam],
    skeleton: &Skeleton,
) -> Vec<IkChainRuntime> {
    let mut chains = Vec::new();
    for (i, c) in authored.iter().enumerate() {
        let fail = |detail: String| {
            tracing::warn!("AnimationGraph {graph_id}: ik_chains[{i}] {detail}; chain disabled");
        };
        if c.joints.len() != 3 {
            fail(format!("names {} joints, expected 3", c.joints.len()));
            continue;
        }
        let index = |name: &str| {
            let found = skeleton.joint_index(name);
            if found.is_none() {
                fail(format!("joint '{name}' not found in the target skeleton"));
            }
            found
        };
        let (Some(root), Some(mid), Some(end)) = (
            index(&c.joints[0]),
            index(&c.joints[1]),
            index(&c.joints[2]),
        ) else {
            continue;
        };
        // The local-pose write-back assumes direct parentage.
        if skeleton.joints()[mid].parent != Some(root) || skeleton.joints()[end].parent != Some(mid)
        {
            fail(format!(
                "'{}' -> '{}' -> '{}' must be a direct parent chain",
                c.joints[0], c.joints[1], c.joints[2]
            ));
            continue;
        }
        let weight_param = if c.weight_parameter.is_empty() {
            None
        } else {
            let found = parameters.iter().position(|p| p.name == c.weight_parameter);
            if found.is_none() {
                fail(format!(
                    "weight parameter '{}' is not declared",
                    c.weight_parameter
                ));
                continue;
            }
            found
        };
        chains.push(IkChainRuntime {
            chain: TwoBoneChain {
                root,
                mid,
                end,
                pole: c.pole,
            },
            weight_param,
            foot_height: c.foot_height,
        });
    }
    chains
}

// This frame's solve inputs for one target: the rig's model transform pair
// and, per chain, the world-space pin height and weight (or `None` to leave
// the chain animated).
#[derive(Default)]
pub(super) struct IkFrame {
    pub model: Mat4,
    pub inv_model: Mat4,
    // One entry per resolved chain: (pin world Y, weight).
    pub pins: Vec<Option<(f32, f32)>>,
}

// Serial pre-pass: fold each target's probe answers, rig state, and weight
// parameters into per-chain pins for the parallel sample loop. `out` persists
// across frames (a target absent from it gets no solve), so this refresh
// allocates nothing in steady state.
pub(super) fn frame_inputs(
    targets: &BTreeMap<SkinnedMeshHandle, super::TargetState>,
    ctx: &PipelineContext,
    out: &mut HashMap<SkinnedMeshHandle, IkFrame>,
) {
    out.retain(|target, _| targets.contains_key(target));
    for (&target, state) in targets {
        let refreshed = refresh_target(target, state, ctx, out);
        if !refreshed {
            out.remove(&target);
        }
    }
}

// Refresh one target's `IkFrame` in place; `false` means the target has no
// solve this frame and its entry must not survive.
fn refresh_target(
    target: SkinnedMeshHandle,
    state: &super::TargetState,
    ctx: &PipelineContext,
    out: &mut HashMap<SkinnedMeshHandle, IkFrame>,
) -> bool {
    let super::TargetMode::Graph(g) = &state.mode else {
        return false;
    };
    if g.chains.is_empty() {
        return false;
    }
    // Pinning needs the rig: its transform maps mesh space to world, and
    // an airborne capsule suspends the solve so jumps stay authored.
    let Some((model, grounded)) = ctx
        .query::<CharacterRig>()
        .find(|r| r.target == target)
        .map(|r| (r.model(), r.grounded))
    else {
        return false;
    };
    let params = ctx.query::<AnimationParams>().find(|p| p.target == target);
    let probes = ctx.query::<GroundProbes>().find(|p| p.target == target);
    let frame = out.entry(target).or_default();
    frame.model = model;
    frame.inv_model = mat4_affine_inverse(model);
    frame.pins.clear();
    frame
        .pins
        .extend(g.chains.iter().enumerate().map(|(i, chain)| {
            if !grounded {
                return None;
            }
            let (point, _normal) = probes
                .and_then(|p| p.probes.get(i))
                .and_then(|probe| probe.hit)?;
            let weight = match chain.weight_param {
                Some(p) => params
                    .and_then(|a| a.values.get(p))
                    .copied()
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0),
                None => 1.0,
            };
            (weight > 0.0).then_some((point[1] + chain.foot_height, weight))
        }));
    true
}

// Apply every pinned chain to the sampled locals in `scratch.locals` (runs
// inside the parallel sample loop; `scratch.aux` is the world-matrix buffer
// the solves compose through). The pin keeps the animated foot's world X/Z
// and clamps only its height, so the gait's horizontal travel is untouched.
pub(super) fn apply_chains(
    skeleton: &Skeleton,
    scratch: &mut PoseScratch,
    chains: &[IkChainRuntime],
    frame: &IkFrame,
) {
    for (chain, pin) in chains.iter().zip(&frame.pins) {
        let Some((pin_y, weight)) = *pin else {
            continue;
        };
        skeleton.world_matrices_into(&scratch.locals, &mut scratch.aux);
        let Some(w) = scratch.aux.get(chain.chain.end) else {
            continue;
        };
        let foot_mesh = [w[3][0], w[3][1], w[3][2]];
        let foot_world = transform_point(&frame.model, foot_mesh);
        if (pin_y - foot_world[1]).abs() > SNAP_THRESHOLD {
            continue;
        }
        let target_world = [foot_world[0], pin_y, foot_world[2]];
        let target_mesh = transform_point(&frame.inv_model, target_world);
        crate::gfx::ik::apply_two_bone_ik(
            skeleton,
            &mut scratch.locals,
            &chain.chain,
            target_mesh,
            weight,
            &mut scratch.aux,
        );
    }
}

// Serial post-pass: refresh each target's probe rays from the posed foot
// positions, for PhysicsSystem to answer next frame. The skinning matrix
// applied to a joint's bind position is its current mesh-space position, so
// no extra hierarchy walk is needed. `feet_scratch` is caller-owned so the
// per-target foot list reuses one buffer across targets and frames.
pub(super) fn refresh_rays(
    targets: &BTreeMap<SkinnedMeshHandle, super::TargetState>,
    ctx: &mut PipelineContext,
    feet_scratch: &mut Vec<[f32; 3]>,
) {
    for (&target, state) in targets {
        let super::TargetMode::Graph(g) = &state.mode else {
            continue;
        };
        if g.chains.is_empty() {
            continue;
        }
        let Some(model) = ctx
            .query::<CharacterRig>()
            .find(|r| r.target == target)
            .map(|r| r.model())
        else {
            continue;
        };
        {
            let Some(pose) = ctx
                .query::<crate::components::SkeletonPose>()
                .find(|p| p.mesh_id == target)
            else {
                continue;
            };
            feet_scratch.clear();
            feet_scratch.extend(g.chains.iter().map(|c| {
                let end = c.chain.end;
                let mesh = pose
                    .joint_matrices
                    .get(end)
                    .map(|m| transform_point(m, pose.skeleton.bind_position(end)))
                    .unwrap_or([0.0; 3]);
                transform_point(&model, mesh)
            }));
        }
        if let Some(probes) = ctx.query_mut::<GroundProbes>().find(|p| p.target == target) {
            // Refill in place so the component's buffer capacity survives.
            probes.probes.clear();
            probes
                .probes
                .extend(feet_scratch.iter().map(|foot| GroundProbe {
                    origin: [foot[0], foot[1] + PROBE_UP, foot[2]],
                    max_dist: PROBE_UP + PROBE_DOWN,
                    hit: None,
                }));
        }
    }
}

fn transform_point(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::AnimationParam;
    use crate::gfx::skeleton::{Joint, JointPose, Skeleton};

    // A valid hip -> knee -> foot chain (each the direct child of the last),
    // plus an extra unrelated root joint the broken-parentage case names.
    fn leg_skeleton() -> Skeleton {
        let joint = |name: &str, parent: Option<usize>| Joint {
            name: name.to_string(),
            parent,
            bind: JointPose::default(),
        };
        Skeleton::new(vec![
            joint("hip", None),
            joint("knee", Some(0)),
            joint("foot", Some(1)),
            joint("stray", None),
        ])
    }

    fn chain(joints: &[&str], weight_parameter: &str) -> AnimationIkChain {
        AnimationIkChain {
            joints: joints.iter().map(|s| s.to_string()).collect(),
            pole: [0.0, 0.0, 1.0],
            weight_parameter: weight_parameter.to_string(),
            foot_height: 0.05,
        }
    }

    #[test]
    fn resolves_a_valid_full_strength_chain() {
        let skel = leg_skeleton();
        let out = resolve_chains(
            AssetId(1),
            &[chain(&["hip", "knee", "foot"], "")],
            &[],
            &skel,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(
            (out[0].chain.root, out[0].chain.mid, out[0].chain.end),
            (0, 1, 2)
        );
        assert_eq!(
            out[0].weight_param, None,
            "empty weight name pins full strength"
        );
        assert!((out[0].foot_height - 0.05).abs() < 1e-6);
    }

    #[test]
    fn resolves_the_weight_parameter_to_its_declaration_index() {
        let skel = leg_skeleton();
        let params = [
            AnimationParam {
                name: "unused".to_string(),
                default: 0.0,
            },
            AnimationParam {
                name: "ik".to_string(),
                default: 1.0,
            },
        ];
        let out = resolve_chains(
            AssetId(1),
            &[chain(&["hip", "knee", "foot"], "ik")],
            &params,
            &skel,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].weight_param,
            Some(1),
            "resolved to declaration order"
        );
    }

    #[test]
    fn rejects_a_chain_with_the_wrong_joint_count() {
        let skel = leg_skeleton();
        let out = resolve_chains(AssetId(1), &[chain(&["hip", "knee"], "")], &[], &skel);
        assert!(out.is_empty(), "a two-joint chain is dropped");
    }

    #[test]
    fn rejects_a_chain_naming_a_missing_joint() {
        let skel = leg_skeleton();
        let out = resolve_chains(
            AssetId(1),
            &[chain(&["hip", "knee", "toe"], "")],
            &[],
            &skel,
        );
        assert!(out.is_empty(), "an unknown joint disables the chain");
    }

    #[test]
    fn rejects_a_chain_that_is_not_a_direct_parent_line() {
        let skel = leg_skeleton();
        // hip -> stray -> foot: stray is a root, not hip's child, so parentage
        // is broken.
        let out = resolve_chains(
            AssetId(1),
            &[chain(&["hip", "stray", "foot"], "")],
            &[],
            &skel,
        );
        assert!(out.is_empty(), "broken parentage disables the chain");
    }

    #[test]
    fn rejects_a_chain_with_an_undeclared_weight_parameter() {
        let skel = leg_skeleton();
        let out = resolve_chains(
            AssetId(1),
            &[chain(&["hip", "knee", "foot"], "ghost")],
            &[],
            &skel,
        );
        assert!(
            out.is_empty(),
            "an undeclared weight parameter disables the chain"
        );
    }

    #[test]
    fn keeps_the_good_chains_when_one_is_bad() {
        let skel = leg_skeleton();
        let out = resolve_chains(
            AssetId(1),
            &[
                chain(&["hip", "knee"], ""),         // bad: wrong count
                chain(&["hip", "knee", "foot"], ""), // good
            ],
            &[],
            &skel,
        );
        assert_eq!(out.len(), 1, "one bad chain does not take down the rest");
        assert_eq!(out[0].chain.end, 2);
    }
}
