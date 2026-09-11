// src/metal/graph_events.rs
//
// Event values for a two-queue graph submission, as a pure function of the
// compiled graph. `metal/graph_queues.rs` owns the `MTLEvent` objects and the
// running value counter; `metal/graph_exec.rs` encodes what this module plans.
//
// One event per queue, not one shared event. Apple's `MTLEvent` documentation
// is explicit that "you can signal an event only with a new value that's
// greater than its current value" and that "multiple producing workloads can't
// combine their signals with one event ... Instead, signal when each producing
// workload finishes by updating its own separate event". Two queues signaling
// one event race on that monotonicity, so each queue signals only its own
// event and a consumer waits on the producing queue's event.
//
// The value space is a frame-major slice: one value per compiled pass index,
// then one terminal value per queue. Within a frame a queue signals in
// ascending compiled-index order, which is the order the executor commits that
// queue's command buffers in, so a queue's own signals increase monotonically;
// the terminals are above every pass value, and the next frame's slice starts
// above every value this frame reserved.

use concinnity_core::render::render_graph::{CompiledGraph, PassQueue};

// Values a frame reserves past one per compiled pass: a terminal per queue.
const TERMINAL_SLOTS: u64 = PassQueue::COUNT as u64;

// One frame's slice of the event value space. Values start at `base + 1` so a
// freshly created event, which reads 0, never satisfies a wait by accident.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) struct FrameEvents {
    base: u64,
    n_passes: usize,
}

impl FrameEvents {
    pub(super) fn new(base: u64, n_passes: usize) -> Self {
        Self { base, n_passes }
    }

    // The value a pass signals, and the value its consumers wait for.
    fn pass_value(&self, idx: usize) -> u64 {
        self.base + 1 + idx as u64
    }

    // The value marking all of a queue's work for this frame as complete.
    fn terminal(&self, queue: PassQueue) -> u64 {
        self.base + 1 + self.n_passes as u64 + queue.index() as u64
    }

    // Base of the next frame's slice, above every value this frame reserved.
    fn next_base(&self) -> u64 {
        self.base + 1 + self.n_passes as u64 + TERMINAL_SLOTS
    }
}

// The event operations one pass's command buffer carries. Waits are encoded
// before the pass's own encoders, signals after them: Metal only accepts either
// while the command buffer has no open encoder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct PassSync {
    // `(the queue whose event carries the value, the value)`. Never names the
    // waiting pass's own queue: same-queue ordering is submission order.
    pub(super) waits: Vec<(PassQueue, u64)>,
    // Values signaled on the pass's own queue's event, ascending.
    pub(super) signals: Vec<u64>,
}

// Every event operation a frame's submission encodes.
#[derive(Clone, Debug)]
pub(super) struct FramePlan {
    passes: Vec<PassSync>,
    terminals: [Option<u64>; PassQueue::COUNT],
    next_base: u64,
}

impl FramePlan {
    pub(super) fn pass(&self, idx: usize) -> &PassSync {
        &self.passes[idx]
    }

    // The value that queue's last pass this frame signals, or `None` when the
    // frame put no pass on it. A queue with no pass signals nothing, so the
    // next frame must keep waiting on whatever terminal was last signaled
    // rather than on one this frame never reached.
    pub(super) fn terminal(&self, queue: PassQueue) -> Option<u64> {
        self.terminals[queue.index()]
    }

    pub(super) fn next_base(&self) -> u64 {
        self.next_base
    }
}

// Lay the frame's event operations over `graph`. `previous[q]` is the last
// terminal value queue `q` actually signaled, which is what this frame's other
// queues wait for at frame start.
//
// The frame-start wait is the cross-frame half of the schedule that in-graph
// edges cannot express: a resource that persists across frames and is touched
// from both queues (the particle pools, the fog froxel volume, the Hi-Z
// pyramid, and every pool-aliased transient) would otherwise let frame N+1's
// work on one queue race frame N's on the other. Making each queue's first pass
// wait on every other queue's previous terminal gives up cross-frame overlap
// and keeps the intra-frame overlap, which is the win the second queue buys.
pub(super) fn plan_frame(
    graph: &CompiledGraph,
    events: FrameEvents,
    previous: [Option<u64>; PassQueue::COUNT],
) -> FramePlan {
    let n = graph.passes.len();
    let mut passes = vec![PassSync::default(); n];

    let mut first = [None; PassQueue::COUNT];
    let mut last = [None; PassQueue::COUNT];
    for (i, pass) in graph.passes.iter().enumerate() {
        let q = pass.queue.index();
        first[q].get_or_insert(i);
        last[q] = Some(i);
    }

    for queue in PassQueue::ALL {
        let Some(i) = first[queue.index()] else {
            continue;
        };
        for other in PassQueue::ALL {
            if other == queue {
                continue;
            }
            if let Some(value) = previous[other.index()] {
                passes[i].waits.push((other, value));
            }
        }
    }

    for (i, pass) in graph.passes.iter().enumerate() {
        for wait in &pass.waits_before {
            passes[i]
                .waits
                .push((wait.producer_queue(), events.pass_value(wait.producer())));
        }
        if !pass.signals_after.is_empty() {
            passes[i].signals.push(events.pass_value(i));
        }
    }

    let mut terminals = [None; PassQueue::COUNT];
    for queue in PassQueue::ALL {
        if let Some(i) = last[queue.index()] {
            let value = events.terminal(queue);
            passes[i].signals.push(value);
            terminals[queue.index()] = Some(value);
        }
    }

    FramePlan {
        passes,
        terminals,
        next_base: events.next_base(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::render::render_graph::{FrameGraphInputs, build_frame_graph};
    use std::collections::BTreeSet;

    // A graph with fog, particles, clustered lighting and the two-pass GPU-cull
    // path on, i.e. one that puts passes on both queues.
    fn loaded_graph() -> CompiledGraph {
        let mut i = FrameGraphInputs::all_off();
        i.hdr_sample_count = 1;
        i.shadow_enabled = true;
        i.bindless_cull_enabled = true;
        i.two_pass_occlusion_enabled = true;
        i.hiz_build_enabled = true;
        i.clustered_lighting_enabled = true;
        i.auto_exposure_enabled = true;
        i.particles_enabled = true;
        i.fog_enabled = true;
        i.decals_enabled = true;
        i.taa_enabled = true;
        i.velocity_enabled = true;
        build_frame_graph(&i).expect("the loaded graph compiles")
    }

    // A graph with nothing on the async queue.
    fn graphics_only_graph() -> CompiledGraph {
        build_frame_graph(&FrameGraphInputs::all_off()).expect("the minimum graph compiles")
    }

    fn queue_of(graph: &CompiledGraph, idx: usize) -> PassQueue {
        graph.passes[idx].queue
    }

    // Every value signaled on `queue` this frame, in the order the executor
    // commits that queue's command buffers (ascending compiled index).
    fn signals_on(graph: &CompiledGraph, plan: &FramePlan, queue: PassQueue) -> Vec<u64> {
        (0..graph.passes.len())
            .filter(|&i| queue_of(graph, i) == queue)
            .flat_map(|i| plan.pass(i).signals.iter().copied())
            .collect()
    }

    #[test]
    fn a_loaded_graph_uses_both_queues() {
        let graph = loaded_graph();
        for queue in PassQueue::ALL {
            assert!(
                graph.passes.iter().any(|p| p.queue == queue),
                "the loaded graph put nothing on {}",
                queue.name()
            );
        }
        assert!(
            graph.passes.iter().any(|p| !p.waits_before.is_empty()),
            "the loaded graph has no cross-queue wait to plan"
        );
    }

    #[test]
    fn signals_are_monotonic_per_queue() {
        let graph = loaded_graph();
        let plan = plan_frame(&graph, FrameEvents::new(0, graph.passes.len()), [None; 2]);
        for queue in PassQueue::ALL {
            let signals = signals_on(&graph, &plan, queue);
            assert!(
                signals.windows(2).all(|w| w[0] < w[1]),
                "{} signals out of order: {signals:?}",
                queue.name()
            );
        }
    }

    #[test]
    fn every_wait_names_a_value_the_other_queue_signals() {
        let graph = loaded_graph();
        let previous = [Some(7), Some(9)];
        let plan = plan_frame(&graph, FrameEvents::new(64, graph.passes.len()), previous);
        let scheduled: [BTreeSet<u64>; PassQueue::COUNT] = PassQueue::ALL.map(|q| {
            let mut set: BTreeSet<u64> = signals_on(&graph, &plan, q).into_iter().collect();
            if let Some(v) = previous[q.index()] {
                set.insert(v);
            }
            set
        });
        for i in 0..graph.passes.len() {
            for &(queue, value) in &plan.pass(i).waits {
                assert_ne!(
                    queue,
                    queue_of(&graph, i),
                    "pass {i} waits on its own queue's event"
                );
                assert!(
                    scheduled[queue.index()].contains(&value),
                    "pass {i} waits for {value} on {}, which nothing signals",
                    queue.name()
                );
            }
        }
    }

    #[test]
    fn a_wait_never_names_a_producer_recorded_later_on_its_queue() {
        // A queue runs its own passes in commit order, so a wait can only be
        // satisfied by a value that queue signals no later than the waiter.
        let graph = loaded_graph();
        let plan = plan_frame(&graph, FrameEvents::new(0, graph.passes.len()), [None; 2]);
        for i in 0..graph.passes.len() {
            for &(queue, value) in &plan.pass(i).waits {
                let signaled_before: Vec<u64> = (0..i)
                    .filter(|&j| queue_of(&graph, j) == queue)
                    .flat_map(|j| plan.pass(j).signals.iter().copied())
                    .collect();
                assert!(
                    signaled_before.contains(&value),
                    "pass {i} waits for {value} on {}, signaled only later",
                    queue.name()
                );
            }
        }
    }

    #[test]
    fn the_frame_start_wait_names_the_previous_frame_terminal() {
        let graph = loaded_graph();
        let first = plan_frame(&graph, FrameEvents::new(0, graph.passes.len()), [None; 2]);
        let previous = PassQueue::ALL.map(|q| first.terminal(q));
        let second = plan_frame(
            &graph,
            FrameEvents::new(first.next_base(), graph.passes.len()),
            previous,
        );
        for queue in PassQueue::ALL {
            let head = (0..graph.passes.len())
                .find(|&i| queue_of(&graph, i) == queue)
                .expect("the loaded graph puts a pass on every queue");
            for other in PassQueue::ALL.into_iter().filter(|&o| o != queue) {
                let expected = first
                    .terminal(other)
                    .expect("the loaded graph puts a pass on every queue");
                assert!(
                    second.pass(head).waits.contains(&(other, expected)),
                    "{}'s first pass does not wait on the previous {} terminal",
                    queue.name(),
                    other.name()
                );
            }
        }
    }

    #[test]
    fn consecutive_frames_reserve_disjoint_ascending_values() {
        let graph = loaded_graph();
        let first = plan_frame(&graph, FrameEvents::new(0, graph.passes.len()), [None; 2]);
        let all_first: Vec<u64> = PassQueue::ALL
            .iter()
            .flat_map(|&q| signals_on(&graph, &first, q))
            .collect();
        assert!(
            all_first.iter().all(|&v| v > 0 && v < first.next_base()),
            "a value fell outside the frame's slice: {all_first:?}"
        );
        let second = plan_frame(
            &graph,
            FrameEvents::new(first.next_base(), graph.passes.len()),
            PassQueue::ALL.map(|q| first.terminal(q)),
        );
        let smallest_second = PassQueue::ALL
            .iter()
            .flat_map(|&q| signals_on(&graph, &second, q))
            .min()
            .expect("the second frame signals something");
        let largest_first = all_first.iter().copied().max().expect("nonempty");
        assert!(
            smallest_second > largest_first,
            "frame 2 reuses values from frame 1 ({smallest_second} <= {largest_first})"
        );
    }

    #[test]
    fn terminals_are_the_last_signal_on_their_queue() {
        let graph = loaded_graph();
        let plan = plan_frame(&graph, FrameEvents::new(0, graph.passes.len()), [None; 2]);
        for queue in PassQueue::ALL {
            let signals = signals_on(&graph, &plan, queue);
            assert_eq!(
                signals.last().copied(),
                plan.terminal(queue),
                "{} does not end on its terminal",
                queue.name()
            );
        }
    }

    #[test]
    fn a_single_queue_graph_plans_no_cross_queue_work() {
        let graph = graphics_only_graph();
        let plan = plan_frame(&graph, FrameEvents::new(0, graph.passes.len()), [None; 2]);
        assert!(plan.terminal(PassQueue::AsyncCompute).is_none());
        assert!(plan.terminal(PassQueue::Graphics).is_some());
        for i in 0..graph.passes.len() {
            assert!(plan.pass(i).waits.is_empty(), "pass {i} waits on nothing");
        }
    }

    #[test]
    fn an_unused_queue_leaves_the_next_frames_frame_start_wait_off() {
        // A frame that puts nothing on the async queue signals no async
        // terminal, so the next frame's graphics head must not wait for one.
        let graph = graphics_only_graph();
        let first = plan_frame(&graph, FrameEvents::new(0, graph.passes.len()), [None; 2]);
        let second = plan_frame(
            &graph,
            FrameEvents::new(first.next_base(), graph.passes.len()),
            PassQueue::ALL.map(|q| first.terminal(q)),
        );
        assert!(second.pass(0).waits.is_empty());
    }
}
