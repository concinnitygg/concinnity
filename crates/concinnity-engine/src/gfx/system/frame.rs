// GraphicsSystem per-frame step: core's frame extraction into the owned
// RenderSnapshot, then submission of that snapshot to the backend (see
// `submit`), or its hand-off to the render thread. Asset streaming + the
// camera-relative screen rebase run in StreamingSystem, scheduled just before
// this system.

use concinnity_core::ecs::{PipelineContext, StepResult};
use concinnity_core::input::snapshot::InputPacket;
use concinnity_core::profile;
use concinnity_core::render::snapshot::RenderSnapshot;

use super::*;

// Deposit a sampled input packet for the next tick's InputSystem, merging onto
// an unconsumed one so no edge is lost.
fn deposit_input(ctx: &mut PipelineContext, packet: InputPacket) {
    let sampled_us = concinnity_host::clock::monotonic_micros();
    concinnity_core::input::mailbox::InputMailbox::deposit_in(ctx.resources, packet, sampled_us);
}

// The engine-owned skinned instance pool's free count, for the profiler's
// pool chip.
fn skinned_pool_free(ctx: &PipelineContext) -> u32 {
    ctx.resource::<concinnity_core::render::queues::ActiveRenderQueues>()
        .and_then(|slot| slot.0.as_ref())
        .map(|queues| queues.slots.skinned_free() as u32)
        .unwrap_or(0)
}

// Apply one frame's submit outcome to the world: the memory-pressure signal,
// the op failures the simulation side must roll back (streamed-mesh uploads,
// chunk adds; StreamingSystem drains them next tick), and this frame's render
// stats for the profiler overlay. Backends without GPU-timed stats return the
// trait's default (all zeros), which the HUD displays as "--".
fn apply_frame_outcome(
    ctx: &mut PipelineContext,
    frame: u64,
    memory_pressure: bool,
    failures: Vec<concinnity_core::render::ops::OpFailure>,
    stats: Option<profile::RenderStats>,
) {
    if memory_pressure {
        publish_memory_pressure(ctx, frame);
    }
    if !failures.is_empty() {
        match ctx.resource_mut::<crate::ecs::RenderOpFailures>() {
            Some(pending) => pending.0.extend_from_slice(&failures),
            None => {
                ctx.insert_resource(crate::ecs::RenderOpFailures(failures));
            }
        }
    }
    if let Some(mut stats) = stats {
        stats.skinned_pool_free = skinned_pool_free(ctx);
        ctx.profile.render = stats;
    }
}

// Record a device-memory failure in the shared [`GpuMemoryPressure`] resource,
// where the streaming valve can observe it.
fn publish_memory_pressure(ctx: &mut PipelineContext, frame: u64) {
    use concinnity_core::ecs::GpuMemoryPressure;
    match ctx.resource_mut::<GpuMemoryPressure>() {
        Some(pressure) => {
            pressure.events += 1;
            pressure.last_frame = frame;
        }
        None => {
            ctx.insert_resource(GpuMemoryPressure {
                events: 1,
                last_frame: frame,
            });
        }
    }
}

impl GraphicsSystem {
    // Whether the launch's frame cap is reached, logged on the frame that first
    // reaches it. A reached cap, like a failed init, ends the run with Stop
    // rather than Done: `Done` only retires this system, and the process would
    // spin on headlessly with nothing to present.
    fn frame_cap_reached(&self) -> bool {
        let Some(max) = self.max_frames else {
            return false;
        };
        if self.frame_count == max.max(1) {
            tracing::info!("GraphicsSystem: max_frames ({max}) reached");
        }
        self.frame_count >= max
    }

    pub(super) fn run_step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        if self.failed {
            return StepResult::Stop;
        }
        // Pipelined frames: the backend lives with the render half on the
        // main thread; extract and send the snapshot instead of submitting.
        if ctx.resources.contains::<crate::ecs::PipelinedFrames>() {
            return self.run_step_pipelined(ctx);
        }
        // Take the parked backend for this step (see `ActiveRenderBackend`);
        // it is a plain local from here on, so `ctx` stays freely borrowable.
        let Some(mut backend) = crate::ecs::ActiveRenderBackend::take(ctx.resources) else {
            return StepResult::Done;
        };

        // Fill the frame snapshot from world state, then submit it. The
        // snapshot's buffers ride along and keep their capacity.
        let mut snapshot = std::mem::take(&mut self.snapshot);
        self.extract(ctx, &mut snapshot);
        let submitted = std::time::Instant::now();
        let outcome =
            super::submit::submit(&mut self.frame_policy, &mut snapshot, backend.as_mut());
        ctx.profile
            .record_render_handoff(concinnity_host::clock::micros_since(submitted));
        if outcome.skipped {
            self.extractor.reissue_history_reset(&snapshot.frame);
        }
        self.snapshot = snapshot;

        apply_frame_outcome(
            ctx,
            self.frame_count,
            outcome.replay.memory_pressure,
            outcome.replay.failures,
            outcome.render_stats,
        );

        let mut result = outcome.result;
        if result == StepResult::Continue {
            self.frame_count += 1;
            if self.frame_cap_reached() {
                backend.wait_idle();
                result = StepResult::Stop;
            }
        }

        // Sample the window input the draw's event pump just produced and
        // deposit it for the next tick's InputSystem, which runs before
        // anything else reads input.
        if result == StepResult::Continue {
            deposit_input(ctx, InputPacket::sample(backend.as_mut()));
        }

        crate::ecs::ActiveRenderBackend::put(ctx.resources, backend);
        result
    }

    // The pipelined step: extract into a recycled snapshot, hand it to the
    // render half (a rendezvous send, so at most one frame is in flight), and
    // apply the feedback the render half produced for the previous frame,
    // which the completed send guarantees has already arrived. A closed
    // channel in either direction means the render half stopped.
    fn run_step_pipelined(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let Some(pipe) = crate::ecs::PipelinedFrames::take(ctx.resources) else {
            return StepResult::Done;
        };
        let mut snapshot = std::mem::take(&mut self.snapshot);
        self.extract(ctx, &mut snapshot);
        // The rendezvous blocks until the render thread has finished the frame
        // before and takes this one, which is render-side time.
        let sent = std::time::Instant::now();
        if pipe.snapshot_tx.send(snapshot).is_err() {
            return StepResult::Stop;
        }
        ctx.profile
            .record_render_handoff(concinnity_host::clock::micros_since(sent));
        // The rendezvous send completed, so the render half has this snapshot
        // and submits it unconditionally: count the frame here, once per world
        // step, the way the serial path and StreamingSystem's clock do. Counting
        // drained feedback instead would tie the count to a non-blocking
        // `try_recv`, and a frame-capped run would render `max` or `max + 1`
        // frames depending on which side won the race.
        self.frame_count += 1;

        let mut stop = false;
        while let Ok(feedback) = pipe.feedback_rx.try_recv() {
            stop |= feedback.stop;
            // A dropped frame's reset is redrawn on the next frame this side
            // extracts, a frame later than serial execution manages.
            if feedback.skipped {
                self.extractor
                    .reissue_history_reset(&feedback.recycled.frame);
            }
            apply_frame_outcome(
                ctx,
                self.frame_count,
                feedback.replay.memory_pressure,
                feedback.replay.failures,
                Some(feedback.render_stats),
            );
            self.snapshot = feedback.recycled;
        }

        stop = stop || self.frame_cap_reached();

        crate::ecs::PipelinedFrames::put(ctx.resources, pipe);
        if stop {
            StepResult::Stop
        } else {
            StepResult::Continue
        }
    }

    // Fill `snap` with everything this frame's draw consumes: a requested
    // screenshot burst starts, then core extracts the frame.
    pub(crate) fn extract(&mut self, ctx: &mut PipelineContext, snap: &mut RenderSnapshot) {
        super::screenshot::start_requested(ctx, &mut self.extractor);
        self.extractor.extract(ctx, snap, self.frame_count);
    }

    // Gray out and disable every settings row whose feature the device cannot
    // provide. Runs once at init after the backend reports `self.caps`, while
    // the HitRegions / TextLabels / ScrollPanels are still present.
    pub(super) fn apply_capability_gating(&mut self, ctx: &mut PipelineContext) {
        let vendor = ctx
            .resource::<backend::GpuProfile>()
            .map_or(backend::GpuVendor::Other, |g| g.vendor);
        concinnity_core::settings::system::gate_unavailable_rows(ctx, &self.caps, vendor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{GlobalTransform, RenderHandle};
    use concinnity_core::ecs::World;
    use concinnity_core::gfx::render_types::DrawIndex;
    use concinnity_core::render::feedback;

    fn translated(x: f32) -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, 0.0, 0.0, 1.0],
        ]
    }

    // The pipelined step: the extracted snapshot crosses the channel, the
    // render half's feedback is applied (stats, recycled buffer), and a
    // feedback-flagged stop propagates as StepResult::Stop.
    #[test]
    fn pipelined_step_sends_the_snapshot_and_applies_feedback() {
        let mut world = World::new();
        {
            let mut ctx = world.context();
            let e = ctx.components.spawn();
            ctx.insert(e, GlobalTransform(translated(2.0)));
            ctx.insert(
                e,
                RenderHandle {
                    draws: [DrawIndex(3)].into(),
                },
            );
        }
        let (snapshot_tx, snapshot_rx) = std::sync::mpsc::sync_channel(0);
        let (feedback_tx, feedback_rx) = std::sync::mpsc::channel();
        world.insert_resource(crate::ecs::PipelinedFrames::new(
            crate::ecs::PipelineChannels {
                snapshot_tx,
                feedback_rx,
            },
        ));
        // Stand-in render half: consume each snapshot and reply with a
        // feedback whose second frame flags a stop.
        let consumer = std::thread::spawn(move || {
            for stop in [false, true] {
                let snapshot = snapshot_rx.recv().expect("a snapshot arrives");
                let render_stats = profile::RenderStats {
                    draw_calls: 7,
                    ..Default::default()
                };
                feedback_tx
                    .send(feedback::FrameFeedback {
                        render_stats,
                        replay: Default::default(),
                        recycled: snapshot,
                        stop,
                        skipped: false,
                    })
                    .expect("feedback is consumed");
            }
        });

        let mut gs = GraphicsSystem::new(None);
        // Frame 0: the send completes, but its feedback may or may not have
        // landed yet; either way the step continues.
        assert_eq!(gs.run_step(&mut world.context()), StepResult::Continue);
        // Frame 1: the rendezvous completing proves feedback 0 arrived; the
        // second feedback carries the stop, applied this frame or next.
        let mut result = gs.run_step(&mut world.context());
        if result == StepResult::Continue {
            result = gs.run_step(&mut world.context());
        }
        assert_eq!(result, StepResult::Stop, "the feedback stop propagates");
        consumer.join().expect("the stand-in render half exits");

        let ctx = world.context();
        assert_eq!(
            ctx.profile.render.draw_calls, 7,
            "the feedback's render stats reached the profile"
        );
    }

    // The pipelined step's wait for the render thread to take its snapshot is
    // recorded as render-side time, not as the step's own work.
    #[test]
    fn pipelined_step_records_its_wait_on_the_render_thread_as_the_handoff() {
        let mut world = World::new();
        let (snapshot_tx, snapshot_rx) = std::sync::mpsc::sync_channel(0);
        let (_feedback_tx, feedback_rx) = std::sync::mpsc::channel();
        world.insert_resource(crate::ecs::PipelinedFrames::new(
            crate::ecs::PipelineChannels {
                snapshot_tx,
                feedback_rx,
            },
        ));
        // A render thread still busy with the frame before for 4 ms.
        let consumer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(4));
            snapshot_rx.recv().expect("a snapshot arrives");
        });
        let mut gs = GraphicsSystem::new(None);
        assert_eq!(gs.run_step(&mut world.context()), StepResult::Continue);
        consumer.join().expect("the stand-in render half exits");

        let ctx = world.context();
        ctx.profile.record_system("GraphicsSystem", 0);
        ctx.profile.begin_frame();
        let handoffs = ctx.profile.render_handoffs();
        assert_eq!(handoffs.len(), 1);
        assert!(handoffs[0].1 >= 3_000, "{} us", handoffs[0].1);
    }

    #[test]
    fn the_frame_cap_is_reached_at_its_count() {
        let mut gs = GraphicsSystem::new(None);
        gs.frame_count = 5;
        assert!(!gs.frame_cap_reached(), "no cap");
        gs.max_frames = Some(6);
        assert!(!gs.frame_cap_reached());
        gs.frame_count = 6;
        assert!(gs.frame_cap_reached());
    }

    // With the render half gone (channel closed), the pipelined step stops
    // instead of blocking or panicking.
    #[test]
    fn pipelined_step_stops_when_the_render_half_is_gone() {
        let mut world = World::new();
        let (snapshot_tx, snapshot_rx) = std::sync::mpsc::sync_channel(0);
        let (_feedback_tx, feedback_rx) = std::sync::mpsc::channel();
        world.insert_resource(crate::ecs::PipelinedFrames::new(
            crate::ecs::PipelineChannels {
                snapshot_tx,
                feedback_rx,
            },
        ));
        drop(snapshot_rx);
        let mut gs = GraphicsSystem::new(None);
        assert_eq!(gs.run_step(&mut world.context()), StepResult::Stop);
    }
}
