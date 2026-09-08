// src/render_graph/schedule.rs
//
// Two-queue schedule derived from the compiled pass order. The compile pass
// hands this module the topologically sorted passes plus the dependency edges
// in compiled-index space; this module:
//
//   1. Assigns each pass a [`PassQueue`]. A `Compute` pass moves to the async
//      queue when the graph shows work it could actually overlap with (at least
//      one render pass that is neither its ancestor nor its descendant) and
//      when every barrier it takes part in is already ordered by a data
//      dependency. Everything else stays on the graphics queue, so its position
//      in the serial order is unchanged.
//   2. Derives the cross-queue signal / wait pairs from the same edges, one
//      wait per (consumer, producing queue) naming the latest producer on that
//      queue: waiting on it covers every earlier producer there, because a
//      queue runs its own passes in order.
//   3. Builds the two happens-before relations the rest of the graph reasons
//      with: `dependency`, the closure of the dependency DAG (what correctness
//      requires), and `realised`, the closure of each queue's serial order plus
//      the wait edges (what the schedule delivers). `super::validate` checks the
//      first is contained in the second; the aliasing planner packs against the
//      second, so a missing sync point makes it more conservative rather than
//      quietly unsound.
//
// The assignment is a pure function of the compiled graph, so a graph cached by
// `FrameGraphInputs` stays valid. Wiring a native compute queue is per-backend
// work this module does not do: every executor still records the passes in one
// serial order.

use alloc::vec;
use alloc::vec::Vec;

use super::compile::CompiledPass;
use super::reach::Reachability;
use super::types::{PassKind, ResourceState};

/// The hardware queue an executor records a pass onto.
///
/// A backend that has not created a compute queue records every pass onto its
/// one graphics queue and honours the schedule by keeping the serial order; the
/// assignment is then a plan the executor is free to flatten.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PassQueue {
    /// The graphics queue. Carries every render pass, the presenting pass, and
    /// any compute pass with no independent graphics work to overlap.
    Graphics,
    /// The asynchronous compute queue. Carries the compute passes the graph
    /// shows can run alongside graphics work.
    AsyncCompute,
}

impl PassQueue {
    /// Every queue, in a stable order so a derived list is deterministic.
    pub const ALL: [PassQueue; 2] = [PassQueue::Graphics, PassQueue::AsyncCompute];
    /// Number of distinct queues, the width of any `[T; PassQueue::COUNT]`
    /// companion array.
    pub const COUNT: usize = PassQueue::ALL.len();

    /// Dense index into a `[T; PassQueue::COUNT]` companion array.
    pub const fn index(self) -> usize {
        match self {
            PassQueue::Graphics => 0,
            PassQueue::AsyncCompute => 1,
        }
    }

    /// Stable display name, for a diagnostic that names the queue.
    pub const fn name(self) -> &'static str {
        match self {
            PassQueue::Graphics => "graphics",
            PassQueue::AsyncCompute => "async_compute",
        }
    }
}

/// One cross-queue wait a pass performs before its own work runs: the consumer
/// waits for `producer`, which runs on `producer_queue`, to have completed.
///
/// The matching half is the producer's `signals_after`, which lists the queues
/// that wait on it. A wait covers every earlier producer on the same queue as
/// well, since a queue runs its passes in order, so the compile pass keeps only
/// the latest producer per queue.
///
/// A cross-queue edge is more than an execution dependency, and the pass's
/// `barriers_before` for the shared resource has to carry the extra semantics
/// each API needs at the handoff:
///
///   * Vulkan: when the two queues come from different queue families, the
///     resource needs a release barrier on the producer's family and a matching
///     acquire barrier on the consumer's, with equal `srcQueueFamilyIndex` /
///     `dstQueueFamilyIndex` on both halves. A single-family pair (one family
///     with several queues) needs only the semaphore.
///   * DirectX: the compute queue accepts a narrower set of resource states than
///     the direct queue, so the resource must already be in a compute-legal
///     state (a `UNORDERED_ACCESS` / `NON_PIXEL_SHADER_RESOURCE` / `COMMON`
///     rather than a render-target or pixel-shader state) when the handoff
///     happens; a transition into a direct-queue-only state has to be recorded
///     back on the direct queue.
///
/// Neither native half is implemented: no backend creates a compute queue yet.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CrossQueueWait {
    pub(super) producer: usize,
    pub(super) producer_queue: PassQueue,
}

impl CrossQueueWait {
    /// Index of the producing pass in `CompiledGraph::passes`.
    pub fn producer(self) -> usize {
        self.producer
    }
    /// The queue the producing pass runs on, which is never the waiter's own.
    pub fn producer_queue(self) -> PassQueue {
        self.producer_queue
    }
}

// The two happens-before relations a compiled graph carries. Both are over
// compiled pass indices.
#[derive(Debug, Clone)]
pub(crate) struct Schedule {
    // Closure of the dependency DAG: the ordering the reads / writes require.
    dependency: Reachability,
    // Closure of each queue's serial order plus the cross-queue wait edges: the
    // ordering the schedule actually delivers.
    realised: Reachability,
}

impl Schedule {
    // Whether `b` depends on `a`, transitively, through the read / write edges.
    pub(crate) fn depends(&self, a: usize, b: usize) -> bool {
        self.dependency.reaches(a, b)
    }

    // Whether the schedule guarantees `a` completes before `b` starts.
    //
    // The per-queue edges in this relation are submission order, which on its
    // own only orders when work *starts*: two same-queue dispatches with no
    // barrier between them may overlap on every backend. What makes the edge a
    // completion order is that the graph emits a barrier on every state change
    // (`super::compile::derive_barriers`) and each executor emits an aliasing
    // barrier at every slot reuse boundary. Weakening either of those weakens
    // this relation, and the aliasing planner rests on it.
    pub(crate) fn precedes(&self, a: usize, b: usize) -> bool {
        self.realised.reaches(a, b)
    }

    // Whether the schedule leaves two distinct passes free to run at once.
    pub(crate) fn may_overlap(&self, a: usize, b: usize) -> bool {
        self.realised.concurrent(a, b)
    }
}

// Assign queues, derive the cross-queue sync points onto `passes`, and return
// the resulting schedule relations. `dag[i]` lists the successors of compiled
// pass `i`; every edge points forward because the list is topologically sorted.
pub(crate) fn schedule(
    passes: &mut [CompiledPass],
    dag: &[Vec<usize>],
    n_resources: usize,
) -> Schedule {
    let n = passes.len();
    let dependency = Reachability::new(n, dag);
    let reliance = barrier_reliance(passes, n_resources);

    for i in 0..n {
        passes[i].queue = if may_run_async(passes, &dependency, &reliance, i) {
            PassQueue::AsyncCompute
        } else {
            PassQueue::Graphics
        };
    }

    derive_sync_points(passes, dag);

    let realised = Reachability::new(n, &realised_edges(passes));
    Schedule {
        dependency,
        realised,
    }
}

// Whether pass `i` belongs on the async-compute queue: it is a compute pass, it
// has graphics work to overlap with, and every barrier it takes part in is
// already ordered by a data dependency.
//
// The last condition is what keeps the move sound against the barrier
// derivation as it stands. `barriers_before` records one transition per read
// run, on the run's first pass, and a continuation reader relies on it without
// carrying one of its own. On a single queue the serial order covers that; once
// the pass sits on another queue, only a data dependency does. So a pass may
// leave the graphics queue only when every transition it relies on runs on an
// ancestor (or on itself), and every transition it records that another pass
// relies on runs for a descendant. Splitting a read run's barrier per queue
// would relax this; until then the assignment respects the barriers in hand.
fn may_run_async(
    passes: &[CompiledPass],
    dependency: &Reachability,
    reliance: &[(usize, usize)],
    i: usize,
) -> bool {
    if passes[i].kind != PassKind::Compute {
        return false;
    }
    // The partner is looked for among the render passes rather than among the
    // passes already assigned to the graphics queue, which keeps the rule
    // non-circular: a render pass is on the graphics queue by construction.
    let overlaps = passes
        .iter()
        .enumerate()
        .any(|(j, p)| p.kind == PassKind::Render && dependency.concurrent(i, j));
    if !overlaps {
        return false;
    }
    reliance.iter().all(|&(barrier, relying)| {
        if relying == i {
            dependency.reaches(barrier, i)
        } else if barrier == i {
            dependency.reaches(i, relying)
        } else {
            true
        }
    })
}

// Every `(barrier pass, relying pass)` pair in the graph: the pass whose
// `barriers_before` last put a resource in the state some other pass's declared
// access needs. Replayed in serial order, which is the order the transitions are
// recorded in. Pairs where the relying pass carries the transition itself are
// omitted: they need no ordering.
fn barrier_reliance(passes: &[CompiledPass], n_resources: usize) -> Vec<(usize, usize)> {
    let mut setter: Vec<Option<usize>> = vec![None; n_resources];
    let mut reliance: Vec<(usize, usize)> = Vec::new();
    for (i, pass) in passes.iter().enumerate() {
        for op in &pass.barriers_before {
            debug_assert_ne!(op.to_state(), ResourceState::Undefined);
            setter[op.resource_index()] = Some(i);
        }
        for v in pass.writes.iter().chain(pass.reads.iter()) {
            if let Some(barrier) = setter[v.resource_index()]
                && barrier != i
                && !reliance.contains(&(barrier, i))
            {
                reliance.push((barrier, i));
            }
        }
    }
    reliance
}

// Fill each pass's `waits_before` / `signals_after` from the cross-queue
// dependency edges. Deduplicated in both directions: a consumer keeps one wait
// per producing queue (the latest producer there, which covers the rest), and a
// producer lists each waiting queue once.
fn derive_sync_points(passes: &mut [CompiledPass], dag: &[Vec<usize>]) {
    let n = passes.len();
    for pass in passes.iter_mut() {
        pass.waits_before.clear();
        pass.signals_after.clear();
    }

    let mut latest: Vec<[Option<usize>; PassQueue::COUNT]> = vec![[None; PassQueue::COUNT]; n];
    for (producer, successors) in dag.iter().enumerate() {
        let producer_queue = passes[producer].queue;
        for &consumer in successors {
            if passes[consumer].queue == producer_queue {
                continue;
            }
            let slot = &mut latest[consumer][producer_queue.index()];
            if slot.is_none_or(|current| producer > current) {
                *slot = Some(producer);
            }
        }
    }

    for consumer in 0..n {
        for queue in PassQueue::ALL {
            if let Some(producer) = latest[consumer][queue.index()] {
                passes[consumer].waits_before.push(CrossQueueWait {
                    producer,
                    producer_queue: queue,
                });
            }
        }
    }

    // Second walk so each producer learns which queues wait on it. Collected
    // first because it reads one pass's waits while writing another's signals.
    let signals: Vec<(usize, PassQueue)> = passes
        .iter()
        .flat_map(|pass| {
            pass.waits_before
                .iter()
                .map(move |wait| (wait.producer, pass.queue))
        })
        .collect();
    for (producer, queue) in signals {
        if !passes[producer].signals_after.contains(&queue) {
            passes[producer].signals_after.push(queue);
        }
    }
}

// The relation the recorded schedule delivers, rebuilt from the passes' own
// queue assignments and waits. `super::validate` calls it rather than reading
// the relation the compile pass cached, so a missing signal / wait pair shows up
// there instead of being justified by the same derivation that dropped it.
pub(super) fn realised_reachability(passes: &[CompiledPass]) -> Reachability {
    Reachability::new(passes.len(), &realised_edges(passes))
}

// The edges the schedule actually delivers: each queue's consecutive passes in
// serial order, plus one edge per cross-queue wait. Both point forward, since a
// wait's producer always precedes its consumer in the compiled order; a wait
// that does not is dropped here and reported by `super::validate` instead.
fn realised_edges(passes: &[CompiledPass]) -> Vec<Vec<usize>> {
    let n = passes.len();
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    for queue in PassQueue::ALL {
        let mut previous: Option<usize> = None;
        for (i, pass) in passes.iter().enumerate() {
            if pass.queue != queue {
                continue;
            }
            if let Some(p) = previous {
                edges[p].push(i);
            }
            previous = Some(i);
        }
    }
    for (consumer, pass) in passes.iter().enumerate() {
        for wait in &pass.waits_before {
            if wait.producer < consumer {
                edges[wait.producer].push(consumer);
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::render_graph::builder::GraphBuilder;
    use crate::render::render_graph::passes::PassId;
    use crate::render::render_graph::types::{
        BufferDesc, BufferUsage, PixelFormat, TextureDesc, TextureSize, TextureUsage,
    };

    fn tex() -> TextureDesc {
        TextureDesc::texture_2d(
            TextureSize::Drawable,
            TextureSize::Drawable,
            PixelFormat::Rgba16Float,
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        )
    }

    fn buf() -> BufferDesc {
        BufferDesc {
            size_bytes: None,
            usage: BufferUsage::STORAGE,
        }
    }

    #[test]
    fn queue_names_and_indices_are_dense_and_distinct() {
        for (i, q) in PassQueue::ALL.iter().enumerate() {
            assert_eq!(q.index(), i, "{q:?}");
            assert!(!q.name().is_empty());
        }
        assert_eq!(PassQueue::COUNT, 2);
        assert_ne!(PassQueue::Graphics.name(), PassQueue::AsyncCompute.name());
    }

    #[test]
    fn an_independent_compute_pass_moves_to_the_async_queue() {
        // Cull writes draw_args, read by Main. Shadow writes its own target and
        // neither depends on Cull nor is depended on by it, so Cull has a render
        // pass to overlap with and belongs on the async queue.
        let mut b = GraphBuilder::new();
        let args = b.create_buffer("draw_args", buf());
        let shadow = b.create_texture("shadow_map", tex());
        let scene = b.create_texture("scene", tex());

        let args1 = b
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(args);
        b.add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(shadow);
        let scene1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_buffer(args1);
            p.write_texture(scene)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();

        let g = b.compile().expect("compiles");
        let cull = g.pass(PassId::Cull).expect("Cull present");
        assert_eq!(cull.queue, PassQueue::AsyncCompute);
        for id in [PassId::Shadow, PassId::Main, PassId::Composite] {
            assert_eq!(
                g.pass(id).expect("present").queue,
                PassQueue::Graphics,
                "{id:?}"
            );
        }
    }

    #[test]
    fn a_serialised_compute_pass_stays_on_graphics() {
        // Every render pass is either an ancestor or a descendant of the compute
        // pass, so moving it buys no overlap and the serial order must not change
        // for it.
        let mut b = GraphBuilder::new();
        let scene = b.create_texture("scene", tex());
        let lum = b.create_texture("lum", tex());

        let scene1 = b
            .add_pass(PassId::Main, PassKind::Render)
            .write_texture(scene);
        let lum1 = {
            let mut p = b.add_pass(PassId::AutoExposure, PassKind::Compute);
            p.read_texture(scene1);
            p.write_texture(lum)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(lum1)
            .presents();

        let g = b.compile().expect("compiles");
        for pass in &g.passes {
            assert_eq!(pass.queue, PassQueue::Graphics, "{:?}", pass.id);
            assert!(pass.waits_before.is_empty());
            assert!(pass.signals_after.is_empty());
        }
    }

    #[test]
    fn a_cross_queue_edge_gets_one_signal_and_one_wait() {
        let mut b = GraphBuilder::new();
        let args = b.create_buffer("draw_args", buf());
        let shadow = b.create_texture("shadow_map", tex());
        let scene = b.create_texture("scene", tex());

        let args1 = b
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(args);
        b.add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(shadow);
        let scene1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_buffer(args1);
            p.write_texture(scene)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();

        let g = b.compile().expect("compiles");
        let cull_idx = g.pass_index(PassId::Cull).expect("Cull present");
        let main = g.pass(PassId::Main).expect("Main present");
        assert_eq!(main.waits_before.len(), 1);
        assert_eq!(main.waits_before[0].producer(), cull_idx);
        assert_eq!(
            main.waits_before[0].producer_queue(),
            PassQueue::AsyncCompute
        );
        assert_eq!(
            g.passes[cull_idx].signals_after,
            alloc::vec![PassQueue::Graphics]
        );
        // The consumer's own queue never appears as a producer queue.
        for pass in &g.passes {
            for wait in &pass.waits_before {
                assert_ne!(wait.producer_queue(), pass.queue, "{:?}", pass.id);
            }
        }
    }

    #[test]
    fn a_consumer_waits_once_for_the_latest_producer_on_the_other_queue() {
        // Two independent compute producers both feed Main. Both land on the async
        // queue, which runs them in order, so Main needs only the later wait.
        let mut b = GraphBuilder::new();
        let args = b.create_buffer("draw_args", buf());
        let lights = b.create_buffer("cluster_light_list", buf());
        let shadow = b.create_texture("shadow_map", tex());
        let scene = b.create_texture("scene", tex());

        let args1 = b
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(args);
        let lights1 = b
            .add_pass(PassId::LightCull, PassKind::Compute)
            .write_buffer(lights);
        b.add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(shadow);
        let scene1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_buffer(args1);
            p.read_buffer(lights1);
            p.write_texture(scene)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();

        let g = b.compile().expect("compiles");
        let cull = g.pass_index(PassId::Cull).expect("present");
        let light_cull = g.pass_index(PassId::LightCull).expect("present");
        let main = g.pass(PassId::Main).expect("present");
        assert_eq!(main.waits_before.len(), 1, "{:?}", main.waits_before);
        assert_eq!(main.waits_before[0].producer(), cull.max(light_cull));
        // The earlier producer signals nothing: the queue order carries it.
        assert!(g.passes[cull.min(light_cull)].signals_after.is_empty());
    }

    #[test]
    fn the_realised_order_serialises_each_queue() {
        // Two async-compute passes with no edge between them are still ordered,
        // because one queue runs its own passes in order. That is what keeps the
        // aliasing planner from treating them as concurrent.
        let mut b = GraphBuilder::new();
        let args = b.create_buffer("draw_args", buf());
        let lights = b.create_buffer("cluster_light_list", buf());
        let shadow = b.create_texture("shadow_map", tex());
        let scene = b.create_texture("scene", tex());

        let args1 = b
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(args);
        let lights1 = b
            .add_pass(PassId::LightCull, PassKind::Compute)
            .write_buffer(lights);
        b.add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(shadow);
        let scene1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_buffer(args1);
            p.read_buffer(lights1);
            p.write_texture(scene)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();

        let g = b.compile().expect("compiles");
        let cull = g.pass_index(PassId::Cull).expect("present");
        let light_cull = g.pass_index(PassId::LightCull).expect("present");
        assert_eq!(
            g.passes[cull].queue,
            PassQueue::AsyncCompute,
            "both producers should be async"
        );
        assert_eq!(g.passes[light_cull].queue, PassQueue::AsyncCompute);
        assert!(!g.passes_may_overlap(cull, light_cull));
        assert!(g.pass_precedes(cull.min(light_cull), cull.max(light_cull)));
        // Without an edge between them the dependency DAG leaves them unordered,
        // so the queue's serial order is what this is measuring.
        assert!(!g.depends_on(cull, light_cull));
        assert!(!g.depends_on(light_cull, cull));
    }

    #[test]
    fn an_async_pass_and_an_independent_render_pass_may_overlap() {
        let mut b = GraphBuilder::new();
        let args = b.create_buffer("draw_args", buf());
        let shadow = b.create_texture("shadow_map", tex());
        let scene = b.create_texture("scene", tex());

        let args1 = b
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(args);
        b.add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(shadow);
        let scene1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_buffer(args1);
            p.write_texture(scene)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();

        let g = b.compile().expect("compiles");
        let cull = g.pass_index(PassId::Cull).expect("present");
        let shadow_idx = g.pass_index(PassId::Shadow).expect("present");
        assert!(g.passes_may_overlap(cull, shadow_idx));
        // The overlap is symmetric and a pass never overlaps itself.
        assert!(g.passes_may_overlap(shadow_idx, cull));
        assert!(!g.passes_may_overlap(cull, cull));
    }

    #[test]
    fn the_fully_loaded_graph_overlaps_real_work_across_the_two_queues() {
        // Non-vacuity. A schedule that put nothing on the async queue, or that
        // put passes there with no graphics work beside them, would satisfy every
        // structural check in this module and buy nothing, so this names the
        // overlap the real graph actually has.
        use crate::render::render_graph::frame::{
            FrameGraphInputs, GATED_FLAGS, build_frame_graph,
        };
        use alloc::format;
        use alloc::string::String;

        let mut inputs = FrameGraphInputs::all_off();
        for (name, set) in GATED_FLAGS {
            // `world_hidden` masks every gated pass off again.
            if *name != "world_hidden" {
                set(&mut inputs);
            }
        }
        let g = build_frame_graph(&inputs).expect("frame graph compiles");

        let async_passes: Vec<PassId> = g
            .passes
            .iter()
            .filter(|p| p.queue == PassQueue::AsyncCompute)
            .map(|p| p.id)
            .collect();
        assert!(
            async_passes.contains(&PassId::Cull) && async_passes.contains(&PassId::LightCull),
            "expected the two pre-Main compute passes on the async queue, got {async_passes:?}"
        );

        let mut pairs: Vec<String> = Vec::new();
        for a in 0..g.passes.len() {
            for b in (a + 1)..g.passes.len() {
                if g.passes[a].queue != g.passes[b].queue && g.passes_may_overlap(a, b) {
                    pairs.push(format!("{:?} || {:?}", g.passes[a].id, g.passes[b].id));
                }
            }
        }
        assert!(
            !pairs.is_empty(),
            "the two-queue schedule overlaps nothing, so it is vacuous"
        );

        // The concrete overlap: the clustered-light binning runs while the
        // graphics queue is still drawing the shadow maps and the G-buffer
        // pre-pass, and Main waits for it.
        let light_cull = g.pass_index(PassId::LightCull).expect("present");
        for graphics in [PassId::Shadow, PassId::SpotShadow, PassId::GBufferPrepass] {
            let idx = g.pass_index(graphics).expect("present");
            assert!(
                g.passes_may_overlap(light_cull, idx),
                "LightCull should overlap {graphics:?}; overlaps are {pairs:?}"
            );
        }
        let main = g.pass(PassId::Main).expect("present");
        assert!(
            main.waits_before.iter().any(|w| w.producer() == light_cull),
            "Main consumes the light list, so it must wait for LightCull"
        );
    }

    #[test]
    fn every_wait_names_a_producer_earlier_in_the_serial_order() {
        // The serial order the executors record in has to satisfy each wait at the
        // point it reaches it, which needs the producer to have already run.
        let mut b = GraphBuilder::new();
        let args = b.create_buffer("draw_args", buf());
        let shadow = b.create_texture("shadow_map", tex());
        let scene = b.create_texture("scene", tex());

        let args1 = b
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(args);
        b.add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(shadow);
        let scene1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_buffer(args1);
            p.write_texture(scene)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();

        let g = b.compile().expect("compiles");
        for (i, pass) in g.passes.iter().enumerate() {
            for wait in &pass.waits_before {
                assert!(wait.producer() < i, "{:?} waits forward", pass.id);
            }
        }
    }
}
