// src/gfx/animation/ik.rs
//
// Foot-pinning IK for graph targets. At install time each authored
// `GraphIkChain` resolves its joint names against the target's skeleton and
// its weight parameter against the graph's declaration order; per frame the
// chain's ground probe (answered by PhysicsSystem, one frame behind) turns
// into a mesh-space target for the analytic two-bone solve, applied to the
// sampled locals just before the skinning matrices.

use std::collections::HashMap;

use crate::assets::{AnimParams, CharacterRig, GraphIkChain, GroundProbe, GroundProbes};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::ik::TwoBoneChain;
use crate::gfx::skinning::{Mat4, Skeleton, mat4_affine_inverse};

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
    pub weight_param: Option<usize>,
    pub foot_height: f32,
}

// Resolve the authored chains against the target's skeleton and parameter
// declarations. Unresolvable chains are dropped with a warning, so one bad
// name never takes down the rest.
pub(super) fn resolve_chains(
    graph_id: AssetId,
    authored: &[GraphIkChain],
    parameters: &[crate::assets::GraphParam],
    skeleton: &Skeleton,
) -> Vec<IkChainRuntime> {
    let mut chains = Vec::new();
    for (i, c) in authored.iter().enumerate() {
        let fail = |detail: String| {
            tracing::warn!("AnimGraph {graph_id}: ik_chains[{i}] {detail}; chain disabled");
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
pub(super) struct IkFrame {
    pub model: Mat4,
    pub inv_model: Mat4,
    // One entry per resolved chain: (pin world Y, weight).
    pub pins: Vec<Option<(f32, f32)>>,
}

// Serial pre-pass: fold each target's probe answers, rig state, and weight
// parameters into per-chain pins for the parallel sample loop.
pub(super) fn frame_inputs(
    targets: &HashMap<AssetId, super::TargetState>,
    ctx: &mut PipelineContext,
) -> HashMap<AssetId, IkFrame> {
    let mut out = HashMap::new();
    for (&target, state) in targets {
        let super::TargetMode::Graph(g) = &state.mode else {
            continue;
        };
        if g.chains.is_empty() {
            continue;
        }
        // Pinning needs the rig: its transform maps mesh space to world, and
        // an airborne capsule suspends the solve so jumps stay authored.
        let Some((model, grounded)) = ctx
            .query::<CharacterRig>()
            .find(|r| r.target == target)
            .map(|r| (r.model(), r.grounded))
        else {
            continue;
        };
        let params = ctx
            .query::<AnimParams>()
            .find(|p| p.target == target)
            .map(|p| p.values.clone())
            .unwrap_or_default();
        let hits: Vec<Option<([f32; 3], [f32; 3])>> = ctx
            .query::<GroundProbes>()
            .find(|p| p.target == target)
            .map(|p| p.probes.iter().map(|probe| probe.hit).collect())
            .unwrap_or_default();
        let pins = g
            .chains
            .iter()
            .enumerate()
            .map(|(i, chain)| {
                if !grounded {
                    return None;
                }
                let (point, _normal) = hits.get(i).copied().flatten()?;
                let weight = match chain.weight_param {
                    Some(p) => params.get(p).copied().unwrap_or(0.0).clamp(0.0, 1.0),
                    None => 1.0,
                };
                (weight > 0.0).then_some((point[1] + chain.foot_height, weight))
            })
            .collect();
        out.insert(
            target,
            IkFrame {
                model,
                inv_model: mat4_affine_inverse(model),
                pins,
            },
        );
    }
    out
}

// Apply every pinned chain to one target's sampled locals (runs inside the
// parallel sample loop). The pin keeps the animated foot's world X/Z and
// clamps only its height, so the gait's horizontal travel is untouched.
pub(super) fn apply_chains(
    skeleton: &Skeleton,
    locals: &mut Vec<Mat4>,
    chains: &[IkChainRuntime],
    frame: &IkFrame,
) {
    for (chain, pin) in chains.iter().zip(&frame.pins) {
        let Some((pin_y, weight)) = *pin else {
            continue;
        };
        let worlds = skeleton.world_matrices(locals);
        let Some(w) = worlds.get(chain.chain.end) else {
            continue;
        };
        let foot_mesh = [w[3][0], w[3][1], w[3][2]];
        let foot_world = transform_point(&frame.model, foot_mesh);
        if (pin_y - foot_world[1]).abs() > SNAP_THRESHOLD {
            continue;
        }
        let target_world = [foot_world[0], pin_y, foot_world[2]];
        let target_mesh = transform_point(&frame.inv_model, target_world);
        crate::gfx::ik::apply_two_bone_ik(skeleton, locals, &chain.chain, target_mesh, weight);
    }
}

// Serial post-pass: refresh each target's probe rays from the posed foot
// positions, for PhysicsSystem to answer next frame. The skinning matrix
// applied to a joint's bind position is its current mesh-space position, so
// no extra hierarchy walk is needed.
pub(super) fn refresh_rays(
    targets: &HashMap<AssetId, super::TargetState>,
    ctx: &mut PipelineContext,
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
        let feet: Vec<[f32; 3]> = {
            let Some(pose) = ctx
                .query::<crate::assets::SkeletonPose>()
                .find(|p| p.mesh_id == target)
            else {
                continue;
            };
            g.chains
                .iter()
                .map(|c| {
                    let end = c.chain.end;
                    let mesh = pose
                        .joint_matrices
                        .get(end)
                        .map(|m| transform_point(m, pose.skeleton.bind_position(end)))
                        .unwrap_or([0.0; 3]);
                    transform_point(&model, mesh)
                })
                .collect()
        };
        if let Some(probes) = ctx.query_mut::<GroundProbes>().find(|p| p.target == target) {
            probes.probes = feet
                .into_iter()
                .map(|foot| GroundProbe {
                    origin: [foot[0], foot[1] + PROBE_UP, foot[2]],
                    max_dist: PROBE_UP + PROBE_DOWN,
                    hit: None,
                })
                .collect();
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
