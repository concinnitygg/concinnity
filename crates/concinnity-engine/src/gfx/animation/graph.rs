// src/gfx/animation/graph.rs
//
// The state-machine drive for a clip bucket: an `AnimGraph` compiled at init
// owns the target, playing one state at a time and crossfading over
// transitions. Transition conditions read the target's `AnimParams`
// component, which gameplay systems (or the `anim-param` debug command)
// write; the graph math itself lives in `concinnity_core::gfx::anim_graph`.

use std::collections::HashMap;

use crate::assets::{AnimGraph, AnimParams};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::anim_graph::{CompiledGraph, GraphCursor};

use super::{TargetMode, TargetState};

// The graph drive for one target bucket.
pub(super) struct GraphTarget {
    pub graph: CompiledGraph,
    pub cursor: GraphCursor,
    // Parameter values as of this step, copied from the target's
    // `AnimParams` component before the cursor advances. Also the snapshot
    // the `anim-state` debug command reports.
    pub params: Vec<f32>,
    // Writes queued by the `anim-param` debug command, applied to the
    // `AnimParams` component at the top of the next step (the component
    // stays the single authoritative store).
    pub pending: Vec<(usize, f32)>,
    // Foot-pinning IK chains, resolved against the target skeleton at
    // install (see `super::ik`). Empty when the graph authored none.
    pub chains: Vec<super::ik::IkChainRuntime>,
}

// Compile every declared `AnimGraph` onto its target bucket and publish one
// seeded `AnimParams` per graph. `clip_slots` maps an Animation asset id to
// its (target, clip index) slot from the clip drain. Returns the number of
// graphs installed. The build validates graph shape and ownership, so a
// failure here (blob/clip-list disagreement) downgrades the bucket to its
// flat drive with a warning rather than dropping the world.
pub(super) fn install_graphs(
    targets: &mut HashMap<AssetId, TargetState>,
    ctx: &mut PipelineContext,
    clip_slots: &HashMap<AssetId, (AssetId, usize)>,
    skinned_map: &crate::gfx::skinned_mesh_map::SkinnedMeshHandleMap,
) -> usize {
    let mut installed = 0usize;
    for g in ctx.drain::<AnimGraph>() {
        // Resolve the authored SkinnedMesh handle to the mesh's asset id, which
        // keys the target bucket (shared with the clip drain) and the runtime
        // `AnimParams` / `SkeletonPose` / `GroundProbes` this publishes below.
        let Some(target) = g.target.map(|h| skinned_map.get(h)) else {
            tracing::warn!(
                "AnimationSystem: AnimGraph {} has no target, ignored",
                g.asset_id
            );
            continue;
        };
        let Some(bucket) = targets.get_mut(&target) else {
            tracing::warn!(
                "AnimationSystem: AnimGraph {} targets {} which has no clips, ignored",
                g.asset_id,
                target
            );
            continue;
        };
        let compiled = g.compile(|anim_id| {
            let &(slot_target, index) = clip_slots.get(&anim_id)?;
            if slot_target != target {
                return None;
            }
            let clip = &bucket.clips[index].clip;
            Some((index, clip.duration, clip.looping))
        });
        match compiled {
            Ok(graph) => {
                let params = graph.default_params();
                ctx.push(AnimParams::new(target, params.clone()));
                // IK chains resolve joint names against the target's
                // skeleton (published by GraphicsSystem init, which ran
                // before this one) and get a ground-probe exchange for
                // PhysicsSystem to answer.
                let chains = if g.ik_chains.is_empty() {
                    Vec::new()
                } else if let Some(skeleton) = ctx
                    .query::<crate::assets::SkeletonPose>()
                    .find(|p| p.mesh_id == target)
                    .map(|p| p.skeleton.clone())
                {
                    super::ik::resolve_chains(g.asset_id, &g.ik_chains, &g.parameters, &skeleton)
                } else {
                    tracing::warn!(
                        "AnimationSystem: AnimGraph {} has ik_chains but target {} has no \
                         skeleton pose; IK disabled",
                        g.asset_id,
                        target
                    );
                    Vec::new()
                };
                if !chains.is_empty() {
                    ctx.push(crate::assets::GroundProbes {
                        target,
                        probes: Vec::new(),
                    });
                }
                bucket.mode = TargetMode::Graph(GraphTarget {
                    cursor: GraphCursor::start(&graph),
                    graph,
                    params,
                    pending: Vec::new(),
                    chains,
                });
                installed += 1;
            }
            Err(e) => tracing::warn!("AnimationSystem: {e}; falling back to weighted blend"),
        }
    }
    installed
}

// Per-step drive for one graph target: flush queued debug writes into the
// `AnimParams` component, snapshot its values, and advance the cursor.
pub(super) fn step_target(
    g: &mut GraphTarget,
    target: AssetId,
    ctx: &mut PipelineContext,
    dt_secs: f32,
) {
    if let Some(params) = ctx.query_mut::<AnimParams>().find(|p| p.target == target) {
        for (index, value) in g.pending.drain(..) {
            params.set(index, value);
        }
        g.params.clone_from(&params.values);
    } else {
        // The component was published at init; losing it means something
        // despawned it. Keep the last snapshot and still apply debug writes
        // so the graph stays drivable.
        for (index, value) in g.pending.drain(..) {
            if let Some(slot) = g.params.get_mut(index) {
                *slot = value;
            }
        }
    }
    g.cursor.advance(&g.graph, &g.params, dt_secs);
}
