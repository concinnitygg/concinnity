// src/metal/graph_queues.rs
//
// The command queues and events the render-graph executor submits a frame over:
// one `MTLCommandQueue` per [`PassQueue`], and one `MTLEvent` per queue that
// only that queue signals (see `metal/graph_events.rs` for why the events are
// not shared).
//
// Capability gate. The whole two-queue path hangs off `MtlContext::graph_queues`
// being `Some`: if the second queue or either event cannot be created,
// [`GraphQueues::new`] returns `None` and the executor records every pass onto
// the one graphics queue in compiled order, exactly as it did before this
// existed. There is no knob for it; every Metal device that reaches this code
// supports a second queue, so a runtime toggle would only ever be dead weight.

use concinnity_core::render::render_graph::PassQueue;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice, MTLEvent};

use super::graph_events::{FrameEvents, FramePlan};

pub(super) struct GraphQueues {
    // The graphics queue is `MtlContext::command_queue`, which the rest of the
    // backend also submits over (probe bakes, RT builds, screenshot blits); only
    // the async-compute queue is owned here.
    async_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    events: [Retained<ProtocolObject<dyn MTLEvent>>; PassQueue::COUNT],
    // Base of the next frame's value slice. Only ever increases, so a value is
    // never reused for the life of the context.
    next_base: u64,
    // Per queue, the last terminal value whose command buffer was committed.
    // The next frame's other queues wait on these at frame start; a terminal
    // whose buffer never reached the queue is never recorded, so a wait can
    // never name a value nothing will signal.
    signaled: [Option<u64>; PassQueue::COUNT],
}

impl GraphQueues {
    pub(super) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        let async_queue = device.newCommandQueue()?;
        let mut slots: [Option<Retained<ProtocolObject<dyn MTLEvent>>>; PassQueue::COUNT] =
            Default::default();
        for queue in PassQueue::ALL {
            slots[queue.index()] = Some(device.newEvent()?);
        }
        let events = slots.map(|slot| slot.expect("one event per queue"));
        Some(Self {
            async_queue,
            events,
            next_base: 0,
            signaled: [None; PassQueue::COUNT],
        })
    }

    // The queue a pass is submitted on. `graphics` is the context's own queue,
    // borrowed in rather than cloned so both come back with one lifetime.
    pub(super) fn queue<'a>(
        &'a self,
        queue: PassQueue,
        graphics: &'a ProtocolObject<dyn MTLCommandQueue>,
    ) -> &'a ProtocolObject<dyn MTLCommandQueue> {
        match queue {
            PassQueue::Graphics => graphics,
            PassQueue::AsyncCompute => &self.async_queue,
        }
    }

    pub(super) fn event(&self, queue: PassQueue) -> &ProtocolObject<dyn MTLEvent> {
        &self.events[queue.index()]
    }

    // This frame's value slice and the terminals the frame starts by waiting on.
    // The slice is only taken by [`Self::end_submission`], so a frame that fails
    // before it commits anything leaves the values unused and the next frame
    // reuses them rather than skipping ahead of signals nothing reached.
    pub(super) fn begin_frame(
        &self,
        n_passes: usize,
    ) -> (FrameEvents, [Option<u64>; PassQueue::COUNT]) {
        let events = FrameEvents::new(self.next_base, n_passes);
        (events, self.signaled)
    }

    // Record the terminals of every queue whose carrying command buffer has been
    // committed, and take the frame's value slice. `pending` names a terminal
    // whose buffer the caller commits later (the composite pass rides the
    // command buffer `draw_frame` owns); it is recorded by
    // [`Self::record_terminal`] once that commit has happened.
    pub(super) fn end_submission(&mut self, plan: &FramePlan, pending: Option<PassQueue>) {
        for queue in PassQueue::ALL {
            if Some(queue) == pending {
                continue;
            }
            if let Some(value) = plan.terminal(queue) {
                self.signaled[queue.index()] = Some(value);
            }
        }
        self.next_base = plan.next_base();
    }

    // Record a terminal whose command buffer has now been committed.
    pub(super) fn record_terminal(&mut self, queue: PassQueue, value: u64) {
        self.signaled[queue.index()] = Some(value);
    }
}
