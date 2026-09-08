// src/render_graph/validate.rs
//
// Barrier-coverage check over a `CompiledGraph`. The compile pass derives each
// pass's barrier lists by walking each resource's timeline; this module replays
// those barriers in execution order -- `barriers_before` ahead of the pass's own
// accesses, `barriers_after` once they have been checked -- and checks the
// resulting state against what each pass's read / write declarations require.
// The two directions are structurally independent -- a per-resource timeline
// versus a per-pass replay -- so a deriver bug (a dropped transition, a
// mis-ordered run, a read-run stage union that misses a consumer) shows up as a
// gap here.
//
// The replay respects the schedule's partial order, not just the serial index
// order: a state has to hold along every path to the pass that relies on it, so
// the transition that established it must be ordered before that pass and no
// pass free to run concurrently with it may touch the resource.
//
// [`sync_point_gaps`] is the other half: it checks the cross-queue signal / wait
// pairs actually order every dependency that crosses queues, rather than
// trusting the derivation that produced them.
//
// The checks are pure and GPU-free, so they run both as a headless sweep over
// the `FrameGraphInputs` space and as a per-frame `debug_assertions` assertion
// inside each backend executor, where they cover the graphs a test sweep never
// builds.

use super::compile::CompiledGraph;
use super::passes::PassId;
use super::reach::Reachability;
use super::schedule::{PassQueue, realised_reachability};
use super::types::{BarrierOp, ReadStages, ResourceState};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

// What kind of coverage a pass is missing for one resource.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GapKind {
    // The pass reads the resource, but the replayed barrier state is not `Read`:
    // no transition made the producing write visible to this consumer.
    UncoveredRead,
    // The pass writes the resource, but the replayed barrier state is not `Write`:
    // no transition opened it for writing.
    UncoveredWrite,
    // The resource is in `Read`, but the barrier that opened the read run does not
    // name this consumer's shader stage, so the producing write was never made
    // visible to it.
    MissingReadStage,
    // The barrier that put the resource in the state this pass needs runs on a
    // pass the schedule does not order before this one, so the state does not
    // hold along every path here.
    UnorderedTransition,
    // Another pass the schedule leaves free to run concurrently with this one
    // touches the resource, and at least one of the two writes it, so whatever
    // state this pass replays is not stable while it runs.
    ConcurrentAccess,
}

// One pass / resource pair whose declared access is not covered by a barrier.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BarrierGap {
    pub pass: PassId,
    pub resource_label: &'static str,
    pub kind: GapKind,
}

impl core::fmt::Display for BarrierGap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let what = match self.kind {
            GapKind::UncoveredRead => "reads",
            GapKind::UncoveredWrite => "writes",
            GapKind::MissingReadStage => "reads (stage not in the run union)",
            GapKind::UnorderedTransition => "relies on an unordered transition to",
            GapKind::ConcurrentAccess => "races a concurrent pass on",
        };
        write!(f, "pass {:?} {} {}", self.pass, what, self.resource_label)
    }
}

// Every declared access in `graph` that no barrier covers, in execution order.
// Empty for a correctly compiled graph.
#[cfg(test)]
pub(crate) fn barrier_coverage_gaps(graph: &CompiledGraph) -> Vec<BarrierGap> {
    gaps_over(graph, &|_| true)
}

/// The same check restricted to the resources `driven[resource_index]` marks, i.e.
/// the ones a backend executor resolves to a native target and emits transitions
/// for. A backend calls this on the graphs a real session builds, so it covers the
/// input combinations a headless sweep does not reach, and it asserts specifically
/// that everything the backend claims to drive is fully covered. Resources outside
/// the driven set keep whatever synchronisation their encoder owns and are skipped.
pub fn barrier_coverage_gaps_for_driven(graph: &CompiledGraph, driven: &[bool]) -> Vec<BarrierGap> {
    gaps_over(graph, &|i| driven.get(i).copied().unwrap_or(false))
}

/// The access each resource is left in once the graph's last barrier for it has
/// run, indexed by resource id: the state plus the stage union that barrier
/// carried, since a `Read`'s native state can depend on its consuming stages.
/// `(Undefined, empty)` for a resource no barrier touches.
///
/// A backend pairs this with its own state translation to check the cross-frame
/// contract: a resource whose first-use transition names a resting state must
/// actually end the frame in it, or the next frame's producer barrier declares a
/// source state the resource is not in.
pub fn final_states(graph: &CompiledGraph) -> Vec<(ResourceState, ReadStages)> {
    let mut state = vec![(ResourceState::Undefined, ReadStages::empty()); graph.resources.len()];
    for pass in &graph.passes {
        for op in pass
            .barriers_before
            .iter()
            .chain(pass.barriers_after.iter())
        {
            state[op.resource_index()] = (op.to_state(), op.read_stages());
        }
    }
    state
}

/// What is wrong with one of the schedule's cross-queue sync points.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SyncGapKind {
    /// A dependency crossing queues that neither the waits nor the per-queue
    /// order puts in order, so the consumer may start before the producer has
    /// finished.
    Unsynchronised,
    /// A wait whose producer does not signal the waiting pass's queue: the two
    /// halves of the handoff disagree.
    MissingSignal,
    /// A wait the graph does not ask for: it names a producer on the waiter's
    /// own queue, a producer that is not on the queue the wait claims, or one
    /// the waiter does not depend on.
    SpuriousWait,
    /// A wait naming a producer the compiled order records later, which no
    /// single serial recording of the passes could satisfy.
    WaitBeforeSignal,
    /// A pass signalling a queue on which nothing waits for it.
    UnusedSignal,
}

/// One malformed cross-queue sync point in a compiled graph.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SyncGap {
    /// The producing side, or the offending pass itself for
    /// [`SyncGapKind::UnusedSignal`].
    pub producer: PassId,
    /// The consuming side, `None` for a gap with only one.
    pub consumer: Option<PassId>,
    /// The queue the gap is about: the producer's queue for a wait-side gap,
    /// the signalled queue for [`SyncGapKind::UnusedSignal`].
    pub queue: PassQueue,
    /// What is wrong.
    pub kind: SyncGapKind,
}

impl core::fmt::Display for SyncGap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let what = match self.kind {
            SyncGapKind::Unsynchronised => "is not ordered by any sync point",
            SyncGapKind::MissingSignal => "waits without a matching signal",
            SyncGapKind::SpuriousWait => "waits on a producer it does not depend on",
            SyncGapKind::WaitBeforeSignal => "waits on a producer recorded later",
            SyncGapKind::UnusedSignal => "signals a queue nothing waits on",
        };
        match self.consumer {
            Some(consumer) => write!(
                f,
                "{:?} ({}) -> {:?} {}",
                self.producer,
                self.queue.name(),
                consumer,
                what
            ),
            None => write!(f, "{:?} ({}) {}", self.producer, self.queue.name(), what),
        }
    }
}

/// Every cross-queue sync point in `graph` that is malformed, and every
/// cross-queue dependency the sync points fail to order.
///
/// The coverage half is transitive on purpose: a consumer keeps one wait per
/// producing queue because a queue runs its own passes in order, so an
/// individual edge is legitimately covered by a later producer's wait. What has
/// to hold is that the relation the waits and the per-queue order generate
/// contains every cross-queue dependency, which is what this checks.
///
/// That relation is rebuilt here from the passes' recorded queues and waits
/// rather than read back from the compile pass, so a dropped pair reports
/// instead of being justified by the derivation that dropped it.
pub fn sync_point_gaps(graph: &CompiledGraph) -> Vec<SyncGap> {
    let n = graph.passes.len();
    let realised = realised_reachability(&graph.passes);
    let mut gaps = Vec::new();

    for (consumer, pass) in graph.passes.iter().enumerate() {
        for wait in &pass.waits_before {
            let producer = wait.producer();
            let queue = wait.producer_queue();
            let mut gap = |kind| {
                gaps.push(SyncGap {
                    producer: graph.passes[producer].id,
                    consumer: Some(pass.id),
                    queue,
                    kind,
                });
            };
            if queue == pass.queue
                || graph.passes[producer].queue != queue
                || !graph.depends_on(producer, consumer)
            {
                gap(SyncGapKind::SpuriousWait);
            }
            if producer >= consumer {
                gap(SyncGapKind::WaitBeforeSignal);
            }
            if !graph.passes[producer].signals_after.contains(&pass.queue) {
                gap(SyncGapKind::MissingSignal);
            }
        }
    }

    for (producer, pass) in graph.passes.iter().enumerate() {
        for &queue in &pass.signals_after {
            let awaited = graph.passes.iter().any(|consumer| {
                consumer.queue == queue
                    && consumer
                        .waits_before
                        .iter()
                        .any(|wait| wait.producer() == producer)
            });
            if !awaited {
                gaps.push(SyncGap {
                    producer: pass.id,
                    consumer: None,
                    queue,
                    kind: SyncGapKind::UnusedSignal,
                });
            }
        }
    }

    for producer in 0..n {
        for consumer in 0..n {
            if graph.passes[producer].queue == graph.passes[consumer].queue
                || !graph.depends_on(producer, consumer)
                || realised.reaches(producer, consumer)
            {
                continue;
            }
            gaps.push(SyncGap {
                producer: graph.passes[producer].id,
                consumer: Some(graph.passes[consumer].id),
                queue: graph.passes[producer].queue,
                kind: SyncGapKind::Unsynchronised,
            });
        }
    }

    gaps
}

/// Panic unless recording `graph.passes` in index order honours the schedule.
///
/// This is the contract every backend executor currently relies on: it records
/// one serial stream, so the compiled order has to be a topological order for
/// each queue at once and every wait has to name a producer already recorded.
/// An executor that later submits the queues separately stops needing this;
/// until then it is what lets a two-queue schedule be flattened back to one
/// without changing the recorded command order. `backend` names the caller in
/// the message.
pub fn assert_serial_order_honours_schedule(graph: &CompiledGraph, backend: &str) {
    let problems = serial_order_problems(graph);
    assert!(
        problems.is_empty(),
        "render graph ({backend}): the serial pass order does not honour the schedule: {}",
        problems.join(", ")
    );
}

// Everything that would stop a single serial recording of `graph.passes` from
// honouring the schedule, as human-readable lines.
fn serial_order_problems(graph: &CompiledGraph) -> Vec<String> {
    let mut problems = Vec::new();
    for (i, pass) in graph.passes.iter().enumerate() {
        for wait in &pass.waits_before {
            if wait.producer() >= i {
                problems.push(format!(
                    "{:?} waits on {:?}, which the compiled order records no earlier",
                    pass.id,
                    graph.passes[wait.producer()].id
                ));
            }
        }
        for j in (i + 1)..graph.passes.len() {
            if graph.depends_on(j, i) {
                problems.push(format!(
                    "{:?} depends on {:?}, which the compiled order records later",
                    pass.id, graph.passes[j].id
                ));
            }
        }
    }
    for gap in sync_point_gaps(graph) {
        problems.push(format!("{gap}"));
    }
    problems
}

fn gaps_over(graph: &CompiledGraph, driven: &dyn Fn(usize) -> bool) -> Vec<BarrierGap> {
    let n_resources = graph.resources.len();
    // Rebuilt from the passes' own queues and waits, for the same reason
    // `sync_point_gaps` rebuilds it: the replay has to be able to disagree with
    // the schedule the compile pass planned.
    let realised = realised_reachability(&graph.passes);
    let mut state = vec![ResourceState::Undefined; n_resources];
    // Stage union carried by the barrier that opened each resource's current read
    // run; meaningless unless that resource is in `Read`.
    let mut run_stages = vec![ReadStages::empty(); n_resources];
    // Pass index of the barrier that put each resource in `state`, so the replay
    // can ask whether that transition is ordered before the pass relying on it.
    // `None` while the resource is still `Undefined`.
    let mut setter: Vec<Option<usize>> = vec![None; n_resources];
    let mut gaps = Vec::new();

    // Applying one pass's barrier list: the same effect whichever end of the
    // pass it runs at, so the replay differs only in when it calls this.
    let apply = |ops: &[BarrierOp],
                 pass_idx: usize,
                 state: &mut [ResourceState],
                 run_stages: &mut [ReadStages],
                 setter: &mut [Option<usize>]| {
        for op in ops {
            let i = op.resource_index();
            state[i] = op.to_state();
            setter[i] = Some(pass_idx);
            if op.to_state() == ResourceState::Read {
                run_stages[i] = op.read_stages();
            }
        }
    };

    for (pass_idx, pass) in graph.passes.iter().enumerate() {
        apply(
            &pass.barriers_before,
            pass_idx,
            &mut state,
            &mut run_stages,
            &mut setter,
        );

        let mut report = |i: usize, kind: GapKind| {
            gaps.push(BarrierGap {
                pass: pass.id,
                resource_label: graph.resources[i].label,
                kind,
            });
        };
        // A pass that writes a resource leaves it in `Write` whether or not it also
        // reads it, so writes are checked first and shadow the read check.
        for w in &pass.writes {
            let i = w.resource_index();
            if !driven(i) {
                continue;
            }
            if state[i] != ResourceState::Write {
                report(i, GapKind::UncoveredWrite);
            }
        }
        for r in &pass.reads {
            let i = r.resource_index();
            if !driven(i) || pass.writes.iter().any(|w| w.resource_index() == i) {
                continue;
            }
            if state[i] != ResourceState::Read {
                report(i, GapKind::UncoveredRead);
            } else if !run_stages[i].contains(r.stage()) {
                report(i, GapKind::MissingReadStage);
            }
        }
        // Partial-order half of the replay. The serial walk above credits each
        // access to the last transition by index; that is only the transition
        // this pass actually sees if the schedule orders it before this pass on
        // every path, and only stable if nothing concurrent touches the
        // resource. One report per pass / resource, writes and reads together,
        // since both rest on the same state.
        for v in pass.accesses() {
            let i = v.resource_index();
            if !driven(i) {
                continue;
            }
            if let Some(b) = setter[i]
                && b != pass_idx
                && !realised.reaches(b, pass_idx)
            {
                report(i, GapKind::UnorderedTransition);
            }
            if concurrent_conflict(graph, &realised, pass_idx, i) {
                report(i, GapKind::ConcurrentAccess);
            }
        }
        // A producer-side transition runs once this pass's own work has
        // completed, so it lands after the pass's accesses have been checked
        // against the state it was itself given. Every consumer of it is a
        // descendant of this pass on the realised relation, which is what the
        // `UnorderedTransition` check above tests for them.
        apply(
            &pass.barriers_after,
            pass_idx,
            &mut state,
            &mut run_stages,
            &mut setter,
        );
    }

    gaps
}

// Whether some pass the schedule leaves free to run alongside `pass_idx` touches
// resource `i` with at least one of the two writing it. Concurrent reads are
// harmless; a concurrent write means the state `pass_idx` relies on is not
// stable for its duration.
fn concurrent_conflict(
    graph: &CompiledGraph,
    realised: &Reachability,
    pass_idx: usize,
    i: usize,
) -> bool {
    let writes = |p: usize| {
        graph.passes[p]
            .writes
            .iter()
            .any(|w| w.resource_index() == i)
    };
    let mine = writes(pass_idx);
    graph.resources[i]
        .touches
        .iter()
        .any(|&other| realised.concurrent(pass_idx, other) && (mine || writes(other)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::render_graph::builder::GraphBuilder;
    use crate::render::render_graph::frame::{FrameGraphInputs, build_frame_graph};
    use crate::render::render_graph::schedule::CrossQueueWait;
    use crate::render::render_graph::types::{
        BufferDesc, BufferUsage, PassKind, PixelFormat, TextureDesc, TextureSize, TextureUsage,
    };
    use alloc::string::ToString;
    use alloc::vec;

    // Every gated flag on `FrameGraphInputs`, so the sweep below can name the
    // combination that failed rather than reporting an opaque struct. Extend when a
    // gated pass is added; the sweep is only as wide as this table.
    use super::super::frame::GATED_FLAGS as FLAGS;

    // Compile the graph for `combo`, or panic naming the combination.
    fn compile_combo(combo: &[usize]) -> (CompiledGraph, Vec<&'static str>) {
        let mut inputs = FrameGraphInputs::all_off();
        for &f in combo {
            FLAGS[f].1(&mut inputs);
        }
        let names: Vec<&'static str> = combo.iter().map(|&f| FLAGS[f].0).collect();
        let graph = build_frame_graph(&inputs)
            .unwrap_or_else(|e| panic!("graph failed to compile for {names:?}: {e}"));
        (graph, names)
    }

    // Join a list of reportable items into one message body.
    fn joined<T: core::fmt::Display>(items: &[T]) -> alloc::string::String {
        items
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }

    // Compile the graph for `combo` and assert it has no barrier gaps.
    fn assert_covered(combo: &[usize]) {
        let (graph, names) = compile_combo(combo);
        let gaps = barrier_coverage_gaps(&graph);
        assert!(
            gaps.is_empty(),
            "barrier gaps for {names:?}: {}",
            joined(&gaps)
        );
    }

    // Compile the graph for `combo` and assert its two-queue schedule is sound:
    // every cross-queue dependency is ordered by a signal / wait pair, the
    // compiled order is a topological order for both queues at once, and no two
    // passes on different queues share a resource they could race on.
    fn assert_scheduled(combo: &[usize]) {
        let (graph, names) = compile_combo(combo);
        let gaps = sync_point_gaps(&graph);
        assert!(
            gaps.is_empty(),
            "sync-point gaps for {names:?}: {}",
            joined(&gaps)
        );
        assert_serial_order_honours_schedule(&graph, "sweep");
        let shared = unsynchronised_sharing(&graph);
        assert!(
            shared.is_empty(),
            "unsynchronised cross-queue sharing for {names:?}: {}",
            joined(&shared)
        );
    }

    // One resource two passes on different queues could race on: they both touch
    // it, at least one of them writes it, and the schedule leaves them free to
    // run at the same time.
    #[derive(Debug)]
    struct SharedRace {
        a: PassId,
        b: PassId,
        resource_label: &'static str,
    }

    impl core::fmt::Display for SharedRace {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(
                f,
                "{:?} and {:?} share {} across queues",
                self.a, self.b, self.resource_label
            )
        }
    }

    // Every such race in `graph`. Concurrent reads are left alone: only a write
    // on one of the two sides makes the overlap a hazard.
    fn unsynchronised_sharing(graph: &CompiledGraph) -> Vec<SharedRace> {
        let mut races = Vec::new();
        for (i, res) in graph.resources.iter().enumerate() {
            let writes = |p: usize| {
                graph.passes[p]
                    .writes
                    .iter()
                    .any(|w| w.resource_index() == i)
            };
            for (k, &a) in res.touches.iter().enumerate() {
                for &b in &res.touches[k + 1..] {
                    if graph.passes[a].queue == graph.passes[b].queue
                        || !(writes(a) || writes(b))
                        || !graph.passes_may_overlap(a, b)
                    {
                        continue;
                    }
                    races.push(SharedRace {
                        a: graph.passes[a].id,
                        b: graph.passes[b].id,
                        resource_label: res.label,
                    });
                }
            }
        }
        races
    }

    #[test]
    fn every_single_and_paired_flag_graph_is_fully_covered() {
        // Exhaustive over the whole flag space is 2^24 graphs; singles + pairs is
        // ~300 and catches the interaction bugs that matter (a pass inserted between
        // a producer and its consumer, a substituted pass -- unified G-buffer for
        // the split pre-passes, RT for SSR -- rerouting a read). The all-off and
        // all-on ends are covered separately below.
        assert_covered(&[]);
        for a in 0..FLAGS.len() {
            assert_covered(&[a]);
            for b in (a + 1)..FLAGS.len() {
                assert_covered(&[a, b]);
            }
        }
    }

    #[test]
    fn the_driven_subset_check_ignores_resources_outside_it() {
        // A backend drives only the resources its registry resolves; the rest keep
        // whatever synchronisation their encoder owns. Stripping a barrier for an
        // undriven resource must stay silent, and the same strip on a driven one
        // must report -- otherwise the subset check would either alarm on every
        // partially-migrated frame or never alarm at all.
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex());
        let b = g.create_texture("b", tex());
        let (a1, b1) = {
            let mut p = g.add_pass(PassId::Main, PassKind::Render);
            (p.write_texture(a), p.write_texture(b))
        };
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(a1)
            .read_texture(b1)
            .presents();
        let mut g = g.compile().expect("compiles");
        let composite = g.passes.len() - 1;
        g.passes[composite].barriers_before.clear();

        let only_a = {
            let mut d = vec![false; g.resources.len()];
            d[a.resource.index()] = true;
            d
        };
        let gaps = barrier_coverage_gaps_for_driven(&g, &only_a);
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        assert_eq!(gaps[0].resource_label, "a");

        let none = vec![false; g.resources.len()];
        assert_eq!(barrier_coverage_gaps_for_driven(&g, &none), vec![]);
    }

    #[test]
    fn every_frame_graph_resource_classifies_as_its_backend_expects() {
        // Both explicit backends now take a resource's barrier class from the
        // graph instead of restating it, so a desc edit here silently changes what
        // transitions they emit. These are the classes the executors' translators
        // are written against; changing one means changing the translator too.
        use crate::render::render_graph::types::GraphResourceClass as C;

        let mut inputs = FrameGraphInputs::all_off();
        for (name, set) in FLAGS {
            if *name != "world_hidden" {
                set(&mut inputs);
            }
        }
        // Multisampled, so the separate `hdr_color` attachment is in the graph;
        // without MSAA the single colour target is the spine and only
        // `hdr_resolve` is declared.
        inputs.hdr_sample_count = 4;
        let graph = build_frame_graph(&inputs).expect("compiles");

        let expected = [
            ("draw_args", C::IndirectBuffer),
            ("draw_args2", C::IndirectBuffer),
            ("cull_status", C::UnorderedBuffer),
            ("cluster_light_list", C::StorageBuffer),
            ("ao_output", C::ColorTarget),
            ("shadow_map", C::DepthTarget),
            ("spot_shadow_map", C::DepthTarget),
            ("fog_froxel_volume", C::StorageImage),
            ("hdr_depth", C::DepthTarget),
            ("hdr_color", C::ColorTarget),
            ("hiz_pyramid", C::StorageImage),
            // The unified pre-pass's four attachments are four resources, and
            // the depth one is why: it is a different class from its three
            // colour siblings, so one handle could not have carried it.
            ("gbuffer_normal_depth", C::ColorTarget),
            ("gbuffer_roughness", C::ColorTarget),
            ("gbuffer_velocity", C::ColorTarget),
            ("gbuffer_depth", C::DepthTarget),
        ];
        for (label, want) in expected {
            let res = graph
                .resources
                .iter()
                .find(|r| r.label == label)
                .unwrap_or_else(|| panic!("{label} missing from the fully-loaded graph"));
            assert_eq!(res.class(), Some(want), "{label}");
        }

        // Nothing in the graph may be unclassifiable: a resource with no class
        // gets no registry entry and so silently loses its barriers.
        for res in &graph.resources {
            assert!(res.class().is_some(), "{} has no class", res.label);
        }
    }

    #[test]
    fn every_single_and_paired_flag_schedule_is_sound() {
        // The same singles + pairs space as the barrier sweep, over the schedule
        // instead: the pass set changes which compute passes have independent
        // graphics work, so a pair that moves one onto the async queue is where
        // an unsynchronised handoff would first appear.
        assert_scheduled(&[]);
        for a in 0..FLAGS.len() {
            assert_scheduled(&[a]);
            for b in (a + 1)..FLAGS.len() {
                assert_scheduled(&[a, b]);
            }
        }
    }

    #[test]
    fn the_fully_loaded_graph_schedule_is_sound() {
        let all: Vec<usize> = (0..FLAGS.len())
            .filter(|&f| FLAGS[f].0 != "world_hidden")
            .collect();
        assert_scheduled(&all);
    }

    #[test]
    fn the_fully_loaded_graph_is_covered() {
        // Every gated pass at once except `world_hidden`, which masks them all off
        // (its collapsed graph is covered as a single above).
        let all: Vec<usize> = (0..FLAGS.len())
            .filter(|&f| FLAGS[f].0 != "world_hidden")
            .collect();
        assert_covered(&all);
    }

    fn tex() -> TextureDesc {
        TextureDesc::texture_2d(
            TextureSize::Drawable,
            TextureSize::Drawable,
            PixelFormat::Rgba16Float,
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        )
    }

    #[test]
    fn a_well_formed_graph_has_no_gaps() {
        let mut g = GraphBuilder::new();
        let t = g.create_texture("t", tex());
        let t1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();
        let g = g.compile().expect("compiles");
        assert_eq!(barrier_coverage_gaps(&g), vec![]);
    }

    #[test]
    fn a_mixed_stage_read_run_is_covered_for_both_consumers() {
        // The case the read-stage union exists for: one write consumed by a compute
        // pass and a render pass. A single producer barrier must name both stages,
        // or the second consumer races the write.
        let mut g = GraphBuilder::new();
        let t = g.create_texture("t", tex());
        let t1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        g.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_texture(t1);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();
        let g = g.compile().expect("compiles");
        assert_eq!(barrier_coverage_gaps(&g), vec![]);
    }

    #[test]
    fn a_dropped_barrier_is_reported() {
        // Negative control: strip the consumer's barrier and the replay must flag
        // the read it no longer covers. Without this the test above could pass on a
        // validator that never reports anything.
        let mut g = GraphBuilder::new();
        let t = g.create_texture("t", tex());
        let t1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();
        let mut g = g.compile().expect("compiles");
        let composite = g.passes.len() - 1;
        g.passes[composite].barriers_before.clear();

        let gaps = barrier_coverage_gaps(&g);
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        assert_eq!(gaps[0].kind, GapKind::UncoveredRead);
        assert_eq!(gaps[0].pass, PassId::Composite);
        assert_eq!(gaps[0].resource_label, "t");
    }

    #[test]
    fn a_read_run_missing_a_consumer_stage_is_reported() {
        // Negative control for the stage union: narrow the producer barrier to the
        // fragment stage only and the compute consumer must be flagged, even though
        // the resource is correctly in `Read`.
        let mut g = GraphBuilder::new();
        let t = g.create_texture("t", tex());
        let t1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        g.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_texture(t1);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();
        let mut g = g.compile().expect("compiles");
        for pass in &mut g.passes {
            for op in pass
                .barriers_before
                .iter_mut()
                .chain(pass.barriers_after.iter_mut())
            {
                if op.to_state() == ResourceState::Read {
                    op.read_stages = ReadStages::FRAGMENT;
                }
            }
        }

        let gaps = barrier_coverage_gaps(&g);
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        assert_eq!(gaps[0].kind, GapKind::MissingReadStage);
        assert_eq!(gaps[0].pass, PassId::AutoExposure);
    }

    // The producer's own barrier stripped: the write it was opened for is no
    // longer covered, which is the other half of the replay's report and the
    // one a dropped first-use transition produces.
    #[test]
    fn a_dropped_producer_barrier_is_reported_as_an_uncovered_write() {
        let mut g = GraphBuilder::new();
        let t = g.create_texture("t", tex());
        let t1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();
        let mut g = g.compile().expect("compiles");
        g.passes[0].barriers_before.clear();

        let gaps = barrier_coverage_gaps(&g);
        assert!(
            gaps.iter().any(|gap| gap.kind == GapKind::UncoveredWrite
                && gap.pass == PassId::Main
                && gap.resource_label == "t"),
            "{gaps:?}"
        );
    }

    fn buf() -> BufferDesc {
        BufferDesc {
            size_bytes: None,
            usage: BufferUsage::STORAGE,
        }
    }

    // A compute producer feeding a render consumer, with an independent render
    // pass beside it so the producer earns the async queue. The base graph every
    // sync-point control below mutates.
    fn cross_queue_graph() -> CompiledGraph {
        let mut g = GraphBuilder::new();
        let args = g.create_buffer("draw_args", buf());
        let shadow = g.create_texture("shadow_map", tex());
        let scene = g.create_texture("scene", tex());

        let args1 = g
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(args);
        g.add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(shadow);
        let scene1 = {
            let mut p = g.add_pass(PassId::Main, PassKind::Render);
            p.read_buffer(args1);
            p.write_texture(scene)
        };
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();
        let g = g.compile().expect("compiles");
        // The control tests are only meaningful on a graph that really splits.
        assert_eq!(
            g.pass(PassId::Cull).expect("present").queue,
            PassQueue::AsyncCompute
        );
        assert!(sync_point_gaps(&g).is_empty(), "{:?}", sync_point_gaps(&g));
        g
    }

    #[test]
    fn a_dropped_wait_is_reported() {
        // Negative control: strip the consumer's wait and both halves of the
        // handoff must report -- the dependency is no longer ordered, and the
        // producer is left signalling a queue nothing waits on.
        let mut g = cross_queue_graph();
        let main = g.pass_index(PassId::Main).expect("present");
        g.passes[main].waits_before.clear();

        let gaps = sync_point_gaps(&g);
        assert!(
            gaps.iter()
                .any(|gap| gap.kind == SyncGapKind::Unsynchronised
                    && gap.producer == PassId::Cull
                    && gap.consumer == Some(PassId::Main)),
            "{gaps:?}"
        );
        assert!(
            gaps.iter()
                .any(|gap| gap.kind == SyncGapKind::UnusedSignal && gap.producer == PassId::Cull),
            "{gaps:?}"
        );
    }

    #[test]
    fn a_dropped_signal_is_reported() {
        // The other half: the wait stands but the producer no longer signals the
        // waiter's queue, so the pair no longer matches.
        let mut g = cross_queue_graph();
        let cull = g.pass_index(PassId::Cull).expect("present");
        g.passes[cull].signals_after.clear();

        let gaps = sync_point_gaps(&g);
        assert!(
            gaps.iter()
                .any(|gap| gap.kind == SyncGapKind::MissingSignal
                    && gap.consumer == Some(PassId::Main)),
            "{gaps:?}"
        );
    }

    #[test]
    fn a_wait_on_a_later_producer_is_reported() {
        // A wait no serial recording could satisfy: the producer it names is
        // recorded after the waiter.
        let mut g = cross_queue_graph();
        let cull = g.pass_index(PassId::Cull).expect("present");
        let composite = g.pass_index(PassId::Composite).expect("present");
        g.passes[cull].waits_before.push(CrossQueueWait {
            producer: composite,
            producer_queue: PassQueue::Graphics,
        });

        let gaps = sync_point_gaps(&g);
        assert!(
            gaps.iter()
                .any(|gap| gap.kind == SyncGapKind::WaitBeforeSignal
                    && gap.consumer == Some(PassId::Cull)),
            "{gaps:?}"
        );
        // It is also a stall the graph never asked for.
        assert!(
            gaps.iter().any(|gap| gap.kind == SyncGapKind::SpuriousWait),
            "{gaps:?}"
        );
    }

    #[test]
    fn a_dropped_wait_is_reported_by_the_coverage_replay() {
        // Negative control for the concurrency half of the replay. Main reads
        // what Cull wrote on the other queue; with the wait gone nothing orders
        // the two, so the producer is free to overwrite the buffer while Main
        // reads it.
        let mut g = cross_queue_graph();
        let main = g.pass_index(PassId::Main).expect("present");
        g.passes[main].waits_before.clear();

        let gaps = barrier_coverage_gaps(&g);
        assert!(
            gaps.iter().any(|gap| gap.kind == GapKind::ConcurrentAccess
                && gap.pass == PassId::Main
                && gap.resource_label == "draw_args"),
            "{gaps:?}"
        );
        // The graph it came from reports nothing, so the control is measuring
        // the dropped wait rather than a validator that always alarms.
        assert_eq!(barrier_coverage_gaps(&cross_queue_graph()), vec![]);
    }

    #[test]
    fn a_continuation_reader_moved_across_queues_is_reported() {
        // Negative control for the path half of the replay. `draw_args` has one
        // barrier for its whole read run, recorded on the run's first pass; a
        // continuation reader relies on it without carrying one. Move that
        // reader to the other queue and the transition it relies on is no
        // longer ordered before it.
        let mut g = GraphBuilder::new();
        let args = g.create_buffer("draw_args", buf());
        let shadow = g.create_texture("shadow_map", tex());
        let scene = g.create_texture("scene", tex());

        let args1 = g
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(args);
        g.add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(shadow);
        let scene1 = {
            let mut p = g.add_pass(PassId::Main, PassKind::Render);
            p.read_buffer(args1);
            p.write_texture(scene)
        };
        // A second reader of the same version, so it coalesces into Main's run.
        g.add_pass(PassId::Cull2, PassKind::Compute)
            .read_buffer(args1);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();
        let mut g = g.compile().expect("compiles");

        // The compile pass records the run's transition on its producer for
        // exactly this reason, which is what frees the continuation reader to
        // leave the graphics queue, and the graph is clean as compiled.
        let cull = g.pass_index(PassId::Cull).expect("present");
        let main = g.pass_index(PassId::Main).expect("present");
        let cull2 = g.pass_index(PassId::Cull2).expect("present");
        assert_eq!(g.passes[cull2].queue, PassQueue::AsyncCompute);
        assert!(g.passes[cull2].barriers_before.is_empty());
        assert_eq!(g.passes[cull].barriers_after.len(), 1);
        assert_eq!(barrier_coverage_gaps(&g), vec![]);

        // Put the transition back on the run's first reader, which is where a
        // deriver blind to the queue split would leave it. Nothing then orders
        // it before the reader on the other queue.
        let op = g.passes[cull].barriers_after.remove(0);
        g.passes[main].barriers_before.push(op);
        let gaps = barrier_coverage_gaps(&g);
        assert!(
            gaps.iter()
                .any(|gap| gap.kind == GapKind::UnorderedTransition
                    && gap.pass == PassId::Cull2
                    && gap.resource_label == "draw_args"),
            "{gaps:?}"
        );
    }

    // A gap prints what the pass did and to what, so a sweep failure names the
    // access rather than dumping a struct.
    #[test]
    fn every_gap_kind_says_what_the_pass_did() {
        let gap = |kind| {
            BarrierGap {
                pass: PassId::Composite,
                resource_label: "t",
                kind,
            }
            .to_string()
        };
        assert_eq!(gap(GapKind::UncoveredRead), "pass Composite reads t");
        assert_eq!(gap(GapKind::UncoveredWrite), "pass Composite writes t");
        assert_eq!(
            gap(GapKind::MissingReadStage),
            "pass Composite reads (stage not in the run union) t"
        );
        assert_eq!(
            gap(GapKind::UnorderedTransition),
            "pass Composite relies on an unordered transition to t"
        );
        assert_eq!(
            gap(GapKind::ConcurrentAccess),
            "pass Composite races a concurrent pass on t"
        );
    }

    // A sync gap prints both sides and the queue, so a sweep failure names the
    // handoff rather than dumping a struct.
    #[test]
    fn every_sync_gap_kind_says_what_is_wrong() {
        let gap = |kind, consumer| {
            SyncGap {
                producer: PassId::Cull,
                consumer,
                queue: PassQueue::AsyncCompute,
                kind,
            }
            .to_string()
        };
        assert_eq!(
            gap(SyncGapKind::Unsynchronised, Some(PassId::Main)),
            "Cull (async_compute) -> Main is not ordered by any sync point"
        );
        assert_eq!(
            gap(SyncGapKind::MissingSignal, Some(PassId::Main)),
            "Cull (async_compute) -> Main waits without a matching signal"
        );
        assert_eq!(
            gap(SyncGapKind::SpuriousWait, Some(PassId::Main)),
            "Cull (async_compute) -> Main waits on a producer it does not depend on"
        );
        assert_eq!(
            gap(SyncGapKind::WaitBeforeSignal, Some(PassId::Main)),
            "Cull (async_compute) -> Main waits on a producer recorded later"
        );
        assert_eq!(
            gap(SyncGapKind::UnusedSignal, None),
            "Cull (async_compute) signals a queue nothing waits on"
        );
    }

    // The resting state is the last barrier's, and a resource no barrier
    // touches never leaves `Undefined`: a backend pairs this with its own
    // translation to check that the next frame's producer barrier declares a
    // source state the resource is actually in.
    #[test]
    fn the_final_state_is_the_last_barrier_each_resource_took() {
        let mut g = GraphBuilder::new();
        let t = g.create_texture("t", tex());
        let untouched = g.create_texture("untouched", tex());
        let t1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();
        let g = g.compile().expect("compiles");

        let states = final_states(&g);
        assert_eq!(states.len(), g.resources.len());
        let (state, stages) = states[t.resource.index()];
        assert_eq!(
            state,
            ResourceState::Read,
            "the last barrier opened it for the composite's read"
        );
        assert!(
            !stages.is_empty(),
            "a read barrier names its consumer stage"
        );
        assert_eq!(
            states[untouched.resource.index()],
            (ResourceState::Undefined, ReadStages::empty())
        );
    }
}
