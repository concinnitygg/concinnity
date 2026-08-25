// src/gfx/animation/graph.rs
//
// The state-machine drive for a clip bucket: an `AnimationGraph` compiled at init
// owns the target, playing one state at a time and crossfading over
// transitions. Transition conditions read the target's `AnimationParams`
// component, which gameplay systems (or the `anim-param` debug command)
// write; the graph math itself lives in `concinnity_cpu::gfx::anim_graph`.

use std::collections::{BTreeMap, HashMap};

use crate::components::{AnimationGraph, AnimationParams};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, SkinnedMeshHandle};
use crate::gfx::anim_graph::{CompiledGraph, GraphCursor};

use super::{TargetMode, TargetState};

// The graph drive for one target bucket.
pub(super) struct GraphTarget {
    pub graph: CompiledGraph,
    pub cursor: GraphCursor,
    // Parameter values as of this step, copied from the target's
    // `AnimationParams` component before the cursor advances. Also the snapshot
    // the `anim-state` debug command reports.
    pub params: Vec<f32>,
    // Writes queued by the `anim-param` debug command, applied to the
    // `AnimationParams` component at the top of the next step (the component
    // stays the single authoritative store).
    pub pending: Vec<(usize, f32)>,
    // Foot-pinning IK chains, resolved against the target skeleton at
    // install (see `super::ik`). Empty when the graph authored none.
    pub chains: Vec<super::ik::IkChainRuntime>,
}

// Compile every declared `AnimationGraph` onto its target bucket and publish one
// seeded `AnimationParams` per graph. `clip_slots` maps an Animation asset id to
// its (target, clip index) slot from the clip drain. Returns the number of
// graphs installed. The build validates graph shape and ownership, so a
// failure here (blob/clip-list disagreement) downgrades the bucket to its
// flat drive with a warning rather than dropping the world.
pub(super) fn install_graphs(
    targets: &mut BTreeMap<SkinnedMeshHandle, TargetState>,
    ctx: &mut PipelineContext,
    clip_slots: &HashMap<AssetId, (SkinnedMeshHandle, usize)>,
) -> usize {
    let mut installed = 0usize;
    for g in ctx.drain::<AnimationGraph>() {
        // The authored SkinnedMesh handle keys the target bucket (shared with
        // the clip drain) and the runtime `AnimationParams` / `SkeletonPose` /
        // `GroundProbes` this publishes below.
        let Some(target) = g.target else {
            tracing::warn!(
                "AnimationSystem: AnimationGraph {} has no target, ignored",
                g.asset_id
            );
            continue;
        };
        let Some(bucket) = targets.get_mut(&target) else {
            tracing::warn!(
                "AnimationSystem: AnimationGraph {} targets handle {} which has no clips, ignored",
                g.asset_id,
                target.index()
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
                ctx.push(AnimationParams::new(target, params.clone()));
                // IK chains resolve joint names against the target's
                // skeleton (published by GraphicsSystem init, which ran
                // before this one) and get a ground-probe exchange for
                // PhysicsSystem to answer.
                let chains = if g.ik_chains.is_empty() {
                    Vec::new()
                } else if let Some(skeleton) = ctx
                    .query::<crate::components::SkeletonPose>()
                    .find(|p| p.mesh_id == target)
                    .map(|p| p.skeleton.clone())
                {
                    super::ik::resolve_chains(g.asset_id, &g.ik_chains, &g.parameters, &skeleton)
                } else {
                    tracing::warn!(
                        "AnimationSystem: AnimationGraph {} has ik_chains but target handle {} has \
                         no skeleton pose; IK disabled",
                        g.asset_id,
                        target.index()
                    );
                    Vec::new()
                };
                if !chains.is_empty() {
                    ctx.push(crate::components::GroundProbes {
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
// `AnimationParams` component, snapshot its values, and advance the cursor.
pub(super) fn step_target(
    g: &mut GraphTarget,
    target: SkinnedMeshHandle,
    ctx: &mut PipelineContext,
    dt_secs: f32,
) {
    if let Some(params) = ctx
        .query_mut::<AnimationParams>()
        .find(|p| p.target == target)
    {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::BlobData;
    use crate::ecs::asset_id::intern;
    use crate::ecs::{ComponentSlot, ComponentStorage, Resources};
    use crate::gfx::profile::FrameProfile;

    use super::super::TargetState;
    use super::super::flat::{ClipEntry, FlatState};

    // A bare clip; installing a graph reads only its duration and loop flag.
    fn clip_entry() -> ClipEntry {
        ClipEntry {
            clip: crate::gfx::skinning::AnimationClip {
                morph_keys: Vec::new(),
                duration: 1.0,
                looping: true,
                tracks: Vec::new(),
                root: None,
            },
            declared_weight: 1.0,
            fade_in_secs: 0.0,
        }
    }

    // A flat bucket for `target`, holding the one clip a graph can claim.
    fn flat_bucket(target: SkinnedMeshHandle) -> BTreeMap<SkinnedMeshHandle, TargetState> {
        BTreeMap::from([(
            target,
            TargetState {
                clips: vec![clip_entry()],
                mode: TargetMode::Flat(FlatState {
                    current_weights: vec![1.0],
                    transition: None,
                }),
            },
        )])
    }

    // The handle an authored `"target": "<name>"` reference deserializes to:
    // the resolver's interner fallback puts the interned id in the handle.
    fn handle(name: &str) -> SkinnedMeshHandle {
        SkinnedMeshHandle(intern(name).0)
    }

    // Owns the storage a PipelineContext borrows from. Graph install reads no
    // payloads, so the blob stays empty.
    struct TestWorld {
        components: ComponentStorage,
        blob: BlobData,
        profile: FrameProfile,
        resources: Resources,
        scratch: crate::ecs::Arena,
    }

    impl TestWorld {
        fn new() -> Self {
            Self {
                components: ComponentStorage::default(),
                blob: BlobData::new(vec![Some(Vec::new())]),
                profile: FrameProfile::default(),
                resources: Resources::new(),
                scratch: crate::ecs::Arena::with_capacity(64 * 1024),
            }
        }

        fn push<C: ComponentSlot>(&mut self, c: C) {
            self.components.push_typed(c);
        }

        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
                frame: crate::ecs::FrameContext::new(&self.scratch),
            }
        }

        fn count<C: ComponentSlot>(&mut self) -> usize {
            self.ctx().query::<C>().count()
        }
    }

    fn is_graph(targets: &BTreeMap<SkinnedMeshHandle, TargetState>, t: SkinnedMeshHandle) -> bool {
        matches!(targets[&t].mode, TargetMode::Graph(_))
    }

    // A one-state graph on `target` playing `clip`, plus the slot map resolving
    // that clip onto the bucket's only entry.
    fn graph_json(target: &str, clip: &str) -> serde_json::Value {
        serde_json::json!({
            "target": target,
            "parameters": [{"name": "speed", "default": 1.5}],
            "states": [{"name": "idle", "clip": clip}],
        })
    }

    fn parse(v: serde_json::Value) -> AnimationGraph {
        crate::ecs::asset_id::ensure_name_resolver();
        serde_json::from_value(v).unwrap()
    }

    // A graph with no target names nothing to own, so it is skipped and the
    // bucket keeps whatever drive it had.
    #[test]
    fn install_graphs_ignores_a_graph_with_no_target() {
        let mut w = TestWorld::new();
        w.push(AnimationGraph::default());
        let mut targets = BTreeMap::new();
        assert_eq!(
            install_graphs(&mut targets, &mut w.ctx(), &HashMap::new()),
            0
        );
    }

    // A graph aimed at a mesh with no clips has nothing to drive: it is skipped
    // rather than installing an empty bucket.
    #[test]
    fn install_graphs_ignores_a_graph_whose_target_has_no_clips() {
        let target = handle("gi_no_clips");
        let mut w = TestWorld::new();
        w.push(parse(graph_json("gi_no_clips", "gi_missing_clip")));
        let mut targets = BTreeMap::new();
        assert_eq!(
            install_graphs(&mut targets, &mut w.ctx(), &HashMap::new()),
            0
        );
        assert!(!targets.contains_key(&target));
        assert_eq!(w.count::<AnimationParams>(), 0, "no params published");
    }

    // A graph the blob and clip list disagree on (here: an unresolvable clip
    // reference) downgrades its bucket to the weighted blend rather than
    // dropping the world.
    #[test]
    fn install_graphs_falls_back_to_the_blend_when_compile_fails() {
        let target = handle("gi_bad_compile");
        let mut w = TestWorld::new();
        w.push(parse(graph_json("gi_bad_compile", "gi_unresolvable")));
        let mut targets = flat_bucket(target);
        // An empty slot map resolves no clip, so the compile fails.
        assert_eq!(
            install_graphs(&mut targets, &mut w.ctx(), &HashMap::new()),
            0
        );
        assert!(
            !is_graph(&targets, target),
            "the bucket keeps its flat drive"
        );
        assert_eq!(w.count::<AnimationParams>(), 0);
    }

    // A graph claiming a clip that belongs to a different target cannot resolve
    // it either: ownership is enforced through the slot map.
    #[test]
    fn install_graphs_rejects_a_clip_owned_by_another_target() {
        let target = handle("gi_owner");
        let clip = intern("gi_owned_clip");
        let mut w = TestWorld::new();
        w.push(parse(graph_json("gi_owner", "gi_owned_clip")));
        let mut targets = flat_bucket(target);
        // The clip exists, but in someone else's bucket.
        let slots = HashMap::from([(clip, (handle("gi_stranger"), 0))]);
        assert_eq!(install_graphs(&mut targets, &mut w.ctx(), &slots), 0);
        assert!(!is_graph(&targets, target));
    }

    // The success path: the graph takes the bucket and publishes one AnimationParams
    // seeded with the declared defaults. No IK chains means no ground probes.
    #[test]
    fn install_graphs_takes_the_bucket_and_seeds_params() {
        let target = handle("gi_ok");
        let clip = intern("gi_ok_clip");
        let mut w = TestWorld::new();
        w.push(parse(graph_json("gi_ok", "gi_ok_clip")));
        let mut targets = flat_bucket(target);
        let slots = HashMap::from([(clip, (target, 0usize))]);
        assert_eq!(install_graphs(&mut targets, &mut w.ctx(), &slots), 1);
        assert!(is_graph(&targets, target));

        let ctx = w.ctx();
        let params: Vec<&AnimationParams> = ctx.query::<AnimationParams>().collect();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].target, target);
        assert_eq!(
            params[0].values,
            vec![1.5],
            "seeded from the declared default"
        );
        assert_eq!(
            w.count::<crate::components::GroundProbes>(),
            0,
            "no chains, no probe exchange"
        );
    }

    // IK chains resolve joint names against the target's skeleton. Without a
    // pose to resolve against the graph still installs, but with IK off and no
    // probe exchange -- the rest of the machine keeps running.
    #[test]
    fn install_graphs_disables_ik_when_the_target_has_no_skeleton() {
        let target = handle("gi_no_skel");
        let clip = intern("gi_no_skel_clip");
        let mut v = graph_json("gi_no_skel", "gi_no_skel_clip");
        v["ik_chains"] = serde_json::json!([{"joints": ["hip", "knee", "foot"],
                                             "pole": [0.0, 0.0, 1.0]}]);
        let mut w = TestWorld::new();
        w.push(parse(v));
        let mut targets = flat_bucket(target);
        let slots = HashMap::from([(clip, (target, 0usize))]);
        assert_eq!(install_graphs(&mut targets, &mut w.ctx(), &slots), 1);

        let TargetMode::Graph(g) = &targets[&target].mode else {
            panic!("graph installed");
        };
        assert!(g.chains.is_empty(), "IK disabled without a skeleton");
        assert_eq!(w.count::<crate::components::GroundProbes>(), 0);
    }

    // A graph bucket parked at its initial state, for driving `step_target`.
    fn graph_target(name: &str) -> GraphTarget {
        let clip = intern(&format!("{name}_clip"));
        let g = parse(graph_json(name, &format!("{name}_clip")));
        let compiled = g
            .compile(|id| (id == clip).then_some((0, 1.0, true)))
            .unwrap();
        GraphTarget {
            cursor: GraphCursor::start(&compiled),
            params: compiled.default_params(),
            graph: compiled,
            pending: Vec::new(),
            chains: Vec::new(),
        }
    }

    // The component is the authoritative parameter store: a queued debug write
    // lands in it, and the bucket's snapshot then reads back from it.
    #[test]
    fn step_target_flushes_queued_writes_into_the_component() {
        let target = handle("gs_component");
        let mut w = TestWorld::new();
        w.push(AnimationParams::new(target, vec![0.0]));
        let mut g = graph_target("gs_component");
        g.pending.push((0, 3.0));

        step_target(&mut g, target, &mut w.ctx(), 0.1);
        assert!(g.pending.is_empty(), "the queue drains");
        assert_eq!(g.params, vec![3.0], "snapshot taken from the component");
        let ctx = w.ctx();
        let params: Vec<&AnimationParams> = ctx.query::<AnimationParams>().collect();
        assert_eq!(params[0].values, vec![3.0], "the write landed in the store");
    }

    // Losing the component (something despawned it) must not wedge the graph:
    // debug writes still apply to the last snapshot so it stays drivable.
    #[test]
    fn step_target_keeps_applying_writes_without_the_component() {
        let target = handle("gs_orphan");
        let mut w = TestWorld::new();
        let mut g = graph_target("gs_orphan");
        g.pending.push((0, 4.0));
        // An out-of-range index is dropped rather than panicking.
        g.pending.push((9, 1.0));

        step_target(&mut g, target, &mut w.ctx(), 0.1);
        assert!(g.pending.is_empty());
        assert_eq!(g.params, vec![4.0], "the snapshot took the write");
    }

    // A component belonging to another target is not this bucket's store.
    #[test]
    fn step_target_ignores_another_targets_params() {
        let target = handle("gs_mine");
        let mut w = TestWorld::new();
        w.push(AnimationParams::new(handle("gs_theirs"), vec![9.0]));
        let mut g = graph_target("gs_mine");
        g.pending.push((0, 2.0));

        step_target(&mut g, target, &mut w.ctx(), 0.1);
        assert_eq!(g.params, vec![2.0], "the stranger's values are not read");
        let ctx = w.ctx();
        let params: Vec<&AnimationParams> = ctx.query::<AnimationParams>().collect();
        assert_eq!(params[0].values, vec![9.0], "and are not written either");
    }
}
