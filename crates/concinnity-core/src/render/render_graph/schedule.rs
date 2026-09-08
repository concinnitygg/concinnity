// src/render_graph/schedule.rs
//
// Two-queue schedule derived from the compiled pass order. The compile pass
// hands this module the topologically sorted passes plus the dependency edges
// in compiled-index space; this module:
//
//   1. Assigns each pass a [`PassQueue`] and, jointly with it, decides where
//      each read run's transition is recorded. A `Compute` pass moves to the
//      async queue when the graph shows work it could actually overlap with (at
//      least one render pass that is neither its ancestor nor its descendant)
//      and when every transition it takes part in is ordered by a data
//      dependency; a read run whose readers end up split across queues has its
//      transition recorded on the producing side, which is what makes the
//      second condition reachable for a continuation reader (see
//      [`super::barrier_place`]). Everything else stays on the graphics queue,
//      so its position in the serial order is unchanged.
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
// `FrameGraphInputs` stays valid. Creating the native queues is per-backend work
// this module does not do. The Metal executor submits both queues; the Vulkan
// and DirectX ones still flatten the schedule back into one serial order, which
// stays legal because the compiled order is a topological order for both queues
// at once.

use alloc::vec;
use alloc::vec::Vec;

use super::barrier_place;
use super::compile::CompiledPass;
use super::reach::Reachability;
use super::types::PassKind;

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
/// A cross-queue edge is more than an execution dependency, and the barriers for
/// the shared resource have to carry the extra semantics each API needs at the
/// handoff. Both halves belong on the producing side, which is why a run read
/// from two queues transitions in the producer's
/// [`barriers_after`](super::CompiledPass::barriers_after):
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
///     happens; the transition into it has to be recorded on the direct queue
///     before the signal, as does any transition back into a direct-queue-only
///     state.
///
/// A third requirement is not expressible as a wait at all, because it crosses
/// the frame boundary: the in-graph edges order one frame, and a resource that
/// persists across frames and is touched from both queues needs the async queue
/// to wait, at frame start, on the previous frame's graphics completion. The
/// particle pools are the live case -- `ParticlesSim` integrates each emitter's
/// persistent pool in place on the async queue while `ParticlesDraw` reads it on
/// the graphics queue -- so on a backend with a native queue, frame N+1's
/// simulation would otherwise be free to overwrite a pool frame N's draw is
/// still reading. The fog froxel volume is the same shape, and so is every
/// pool-aliased transient a pass on one queue reads and the next frame's other
/// queue rewrites. A backend that creates the queue owes that frame-start wait
/// in both directions; Metal's is in `metal/graph_events.rs`.
///
/// Metal creates the queue and honours both halves. Vulkan and DirectX create
/// no compute queue yet, so neither native barrier half above is implemented
/// there and both flatten the schedule into one serial stream.
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

// Assign queues, place each read run's transition, derive the cross-queue sync
// points onto `passes`, and return the resulting schedule relations. `dag[i]`
// lists the successors of compiled pass `i`; every edge points forward because
// the list is topologically sorted.
//
// The assignment and the placement depend on each other -- a pass may move only
// when every transition it relies on is ordered, and a transition moves onto its
// producer only when the run spans queues -- so the two are resolved as a fixed
// point rather than in one pass. It is the *greatest* fixed point: the iteration
// starts from every compute pass that has graphics work to overlap and takes
// passes back off the async queue until nothing else has to move. Starting from
// the empty async set would be a fixed point too, and a vacuous one: with
// nothing scheduled asynchronously no run spans queues, so no transition moves
// and no pass is ever freed to move.
//
// Each round strictly shrinks the async set, so the loop runs at most `n` times,
// and both halves read only the graph, keeping the result a pure function of it.
pub(crate) fn schedule(
    passes: &mut [CompiledPass],
    dag: &[Vec<usize>],
    n_resources: usize,
) -> Schedule {
    let n = passes.len();
    let dependency = Reachability::new(n, dag);
    let runs = barrier_place::read_runs(passes, n_resources);

    let mut queues: Vec<PassQueue> = (0..n)
        .map(|i| {
            if has_overlap(passes, &dependency, i) {
                PassQueue::AsyncCompute
            } else {
                PassQueue::Graphics
            }
        })
        .collect();
    loop {
        let reliance = barrier_place::reliance(passes, &runs, &queues, n_resources);
        let mut settled = true;
        for (i, queue) in queues.iter_mut().enumerate() {
            if *queue == PassQueue::AsyncCompute
                && !transitions_are_ordered(&dependency, &reliance, i)
            {
                *queue = PassQueue::Graphics;
                settled = false;
            }
        }
        if settled {
            break;
        }
    }

    for (pass, &queue) in passes.iter_mut().zip(queues.iter()) {
        pass.queue = queue;
    }
    barrier_place::apply(passes, &runs, &queues);

    derive_sync_points(passes, dag);

    let realised = Reachability::new(n, &realised_edges(passes));
    Schedule {
        dependency,
        realised,
    }
}

// Whether pass `i` is a compute pass with graphics work the graph leaves it free
// to overlap: at least one render pass that is neither its ancestor nor its
// descendant. The partner is looked for among the render passes rather than
// among the passes already assigned to the graphics queue, which keeps the rule
// non-circular: a render pass is on the graphics queue by construction.
fn has_overlap(passes: &[CompiledPass], dependency: &Reachability, i: usize) -> bool {
    passes[i].kind == PassKind::Compute
        && passes
            .iter()
            .enumerate()
            .any(|(j, p)| p.kind == PassKind::Render && dependency.concurrent(i, j))
}

// Whether every transition pass `i` takes part in is ordered against it by a
// data dependency, which is what a pass needs before it can leave the graphics
// queue.
//
// On a single queue the serial order covers every transition; once the pass sits
// on another queue, only a data dependency does. So a pass may run
// asynchronously when every transition it relies on runs on an ancestor (or on
// itself) and every transition it records that another pass relies on runs for a
// descendant. A read run recorded on its producer (see
// [`super::barrier_place`]) satisfies both directions by construction, since the
// producer is an ancestor of every reader of the version it wrote; a run still
// recorded on its first reader is what holds a continuation reader back.
fn transitions_are_ordered(
    dependency: &Reachability,
    reliance: &[(usize, usize)],
    i: usize,
) -> bool {
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
        // The three pre-Main compute passes, plus the two the producer-side
        // placement freed: FogFroxel taps the shadow map as a continuation
        // reader after Main, and HizFinal reads the final depth after every
        // decoration pass. All are compute passes with real GPU cost, and
        // neither of the latter two could move while its run's transition sat on
        // the run's first reader.
        for id in [
            PassId::Cull,
            PassId::LightCull,
            PassId::ParticlesSim,
            PassId::FogFroxel,
            PassId::HizFinal,
        ] {
            assert!(
                async_passes.contains(&id),
                "expected {id:?} on the async queue, got {async_passes:?}"
            );
        }
        // Every compute pass still on the graphics queue is there because the
        // graph orders it against every render pass, not because of where a
        // transition sits: HizBuild and Cull2 are the two-phase occlusion cull's
        // own chain between the pre-pass and Main2, AutoExposure reads the frame
        // Main just wrote and feeds the tonemap, and Upscale sits between the
        // last scene pass and Bloom.
        for id in [
            PassId::HizBuild,
            PassId::Cull2,
            PassId::AutoExposure,
            PassId::Upscale,
        ] {
            let i = g.pass_index(id).expect("present");
            assert_eq!(g.passes[i].queue, PassQueue::Graphics, "{id:?}");
            // Asked of the dependency DAG, not of the realised relation: the
            // latter serialises the graphics queue, so every graphics pair looks
            // ordered there and the claim would be vacuous.
            assert!(
                !g.passes
                    .iter()
                    .enumerate()
                    .any(|(j, p)| p.kind == PassKind::Render
                        && !g.depends_on(i, j)
                        && !g.depends_on(j, i)),
                "{id:?} has render work to overlap, so a transition is what holds it back"
            );
        }

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

        // The particle simulation is the widest of them: it reads nothing the
        // frame produces, so the whole raster front is beside it and only the
        // draw that consumes its pools waits.
        let sim = g.pass_index(PassId::ParticlesSim).expect("present");
        for graphics in [
            PassId::Shadow,
            PassId::SpotShadow,
            PassId::GBufferPrepass,
            PassId::SsaoBlur,
            PassId::Main,
        ] {
            let idx = g.pass_index(graphics).expect("present");
            assert!(
                g.passes_may_overlap(sim, idx),
                "ParticlesSim should overlap {graphics:?}; overlaps are {pairs:?}"
            );
        }
        let draw = g.pass(PassId::ParticlesDraw).expect("present");
        assert!(
            draw.waits_before.iter().any(|w| w.producer() == sim),
            "ParticlesDraw reads the simulated pools, so it must wait for ParticlesSim"
        );
    }

    #[test]
    fn a_graph_that_schedules_nothing_asynchronously_moves_no_transition() {
        // The byte-identity guarantee: a run whose readers share a queue keeps
        // the first-reader placement the deriver gave it, so a graph with no
        // async pass compiles to exactly the barrier lists it did before the
        // producer-side placement existed. Swept over the reachable graphs
        // rather than asserted on one, since which graphs those are is what the
        // frame builder's flags decide.
        use crate::render::render_graph::frame::{
            FrameGraphInputs, GATED_FLAGS, build_frame_graph,
        };

        let mut checked = 0;
        for (_, set) in GATED_FLAGS {
            let mut inputs = FrameGraphInputs::all_off();
            set(&mut inputs);
            let g = build_frame_graph(&inputs).expect("frame graph compiles");
            if g.passes.iter().any(|p| p.queue == PassQueue::AsyncCompute) {
                continue;
            }
            checked += 1;
            for pass in &g.passes {
                assert!(
                    pass.barriers_after.is_empty(),
                    "{:?} moved a transition with nothing on the async queue",
                    pass.id
                );
            }
        }
        assert!(checked > 0, "no single-flag graph stays on one queue");
    }

    #[test]
    fn a_moved_transition_is_always_a_producer_opening_a_read_run() {
        // What `barriers_after` may carry, over every reachable graph: only the
        // `Write -> Read` that opens a run, only on a pass that wrote the
        // resource, and only when the readers really are split across queues.
        use crate::render::render_graph::frame::{
            FrameGraphInputs, GATED_FLAGS, build_frame_graph,
        };
        use crate::render::render_graph::types::ResourceState;

        let mut inputs = FrameGraphInputs::all_off();
        for (name, set) in GATED_FLAGS {
            if *name != "world_hidden" {
                set(&mut inputs);
            }
        }
        let g = build_frame_graph(&inputs).expect("frame graph compiles");
        let mut moved = 0;
        for (i, pass) in g.passes.iter().enumerate() {
            for op in &pass.barriers_after {
                moved += 1;
                assert_eq!(op.source_state(), ResourceState::Write, "{:?}", pass.id);
                assert_eq!(op.to_state(), ResourceState::Read, "{:?}", pass.id);
                assert!(
                    pass.writes
                        .iter()
                        .any(|w| w.resource_index() == op.resource_index()),
                    "{:?} carries a transition for a resource it does not write",
                    pass.id
                );
                let readers: Vec<usize> = g.resources[op.resource_index()]
                    .touches
                    .iter()
                    .copied()
                    .filter(|&p| p > i && g.depends_on(i, p))
                    .collect();
                assert!(
                    readers
                        .iter()
                        .any(|&r| g.passes[r].queue != g.passes[i].queue),
                    "{:?} moved a transition no other queue reads",
                    pass.id
                );
                for &r in &readers {
                    assert!(
                        g.pass_precedes(i, r),
                        "{:?} must complete before {:?} sees the transition",
                        pass.id,
                        g.passes[r].id
                    );
                }
            }
        }
        assert!(moved > 0, "the loaded graph moves no transition at all");
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
