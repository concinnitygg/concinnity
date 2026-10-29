// GraphicsSystem per-frame step: core's frame extraction into the owned
// RenderSnapshot, then submission of that snapshot to the backend (see
// `submit`), or its hand-off to the render thread. Asset streaming and the
// camera-relative view rebase run in core's StreamingSystem just before it.

use crate::app::pipeline::PipelinedFrames;
use concinnity_engine::ecs::ActiveRenderBackend;
use concinnity_engine::ecs::{Clock, PipelineContext, StepResult};
use concinnity_engine::input::mailbox::InputMailbox;
use concinnity_engine::input::snapshot::InputPacket;
use concinnity_engine::render::frame::outcome::{self, Submitted};
use concinnity_engine::render::frame::snapshot::RenderSnapshot;
use concinnity_host::clock::micros_since;
use std::time::Instant;

use super::{GraphicsSystem, backend, screenshot, submit};

// Deposit a sampled input packet for the next tick's InputSystem, merging onto
// an unconsumed one so no edge is lost.
fn deposit_input(ctx: &mut PipelineContext<'_>, packet: InputPacket) {
    let sampled_us = ctx.resource::<Clock>().map_or(0, Clock::read);
    InputMailbox::deposit_in(ctx.resources, packet, sampled_us);
}

impl GraphicsSystem {
    pub(super) fn run_step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        if self.failed {
            return StepResult::Stop;
        }
        // Pipelined frames: the backend lives with the render half on the
        // main thread; extract and send the snapshot instead of submitting.
        if ctx.resources.contains::<PipelinedFrames>() {
            return self.run_step_pipelined(ctx);
        }
        // Take the parked backend for this step (see `ActiveRenderBackend`);
        // it is a plain local from here on, so `ctx` stays freely borrowable.
        let Some(mut backend) = ActiveRenderBackend::take(ctx.resources) else {
            return StepResult::Done;
        };

        // Fill the frame snapshot from world state, then submit it. The
        // snapshot's buffers ride along and keep their capacity.
        let mut snapshot = std::mem::take(&mut self.snapshot);
        self.extract(ctx, &mut snapshot);
        let submitted = Instant::now();
        let outcome = submit::submit(&mut self.frame_policy, &mut snapshot, backend.as_mut());
        ctx.profile
            .record_render_handoff(snapshot.extract_us.saturating_add(micros_since(submitted)));
        outcome::settle(
            ctx,
            &mut self.extractor,
            self.frames.frames(),
            Submitted {
                skipped: outcome.skipped,
                frame: &snapshot.frame,
                replay: outcome.replay,
                stats: outcome.render_stats,
            },
        );
        self.snapshot = snapshot;

        let (result, capped) = self.frames.end_step(outcome.result);
        if capped {
            backend.wait_idle();
        }

        // Sample the window input the draw's event pump just produced and
        // deposit it for the next tick's InputSystem, which runs before
        // anything else reads input.
        if result == StepResult::Continue {
            deposit_input(ctx, InputPacket::sample(backend.as_mut()));
        }

        ActiveRenderBackend::put(ctx.resources, backend);
        result
    }

    // The pipelined step: extract into a recycled snapshot, hand it to the
    // render half (a rendezvous send, so at most one frame is in flight), and
    // apply the feedback the render half produced for the previous frame,
    // which the completed send guarantees has already arrived. A closed
    // channel in either direction means the render half stopped.
    fn run_step_pipelined(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        let Some(pipe) = PipelinedFrames::take(ctx.resources) else {
            return StepResult::Done;
        };
        let mut snapshot = std::mem::take(&mut self.snapshot);
        self.extract(ctx, &mut snapshot);
        // The rendezvous blocks until the render thread has finished the frame
        // before and takes this one, which is render-side time.
        let extract_us = snapshot.extract_us;
        let sent = Instant::now();
        if pipe.snapshot_tx.send(snapshot).is_err() {
            return StepResult::Stop;
        }
        ctx.profile
            .record_render_handoff(extract_us.saturating_add(micros_since(sent)));
        // The rendezvous send completed, so the render half has this snapshot
        // and submits it unconditionally: count the frame here, once per world
        // step, the way the serial path and StreamingSystem's clock do. Counting
        // drained feedback instead would tie the count to a non-blocking
        // `try_recv`, and a frame-capped run would render `max` or `max + 1`
        // frames depending on which side won the race.
        let mut stop = self.frames.count();

        while let Ok(feedback) = pipe.feedback_rx.try_recv() {
            stop |= feedback.stop;
            // A dropped frame's reset is redrawn on the next frame this side
            // extracts, a frame later than serial execution manages.
            outcome::settle(
                ctx,
                &mut self.extractor,
                self.frames.frames(),
                Submitted {
                    skipped: feedback.skipped,
                    frame: &feedback.recycled.frame,
                    replay: feedback.replay,
                    stats: Some(feedback.render_stats),
                },
            );
            self.snapshot = feedback.recycled;
        }

        PipelinedFrames::put(ctx.resources, pipe);
        if stop {
            StepResult::Stop
        } else {
            StepResult::Continue
        }
    }

    // Fill `snap` with everything this frame's draw consumes: a requested
    // screenshot burst starts, then core extracts the frame. The extraction
    // is render-side work, so it is timed onto the snapshot.
    pub(crate) fn extract(&mut self, ctx: &mut PipelineContext<'_>, snap: &mut RenderSnapshot) {
        let started = Instant::now();
        screenshot::start_requested(ctx, &mut self.extractor);
        self.extractor.extract(ctx, snap, self.frames.frames());
        snap.extract_us = micros_since(started);
    }

    // Gray out and disable every settings row whose feature the device cannot
    // provide. Runs once at init after the backend reports `self.caps`, while
    // the HitRegions / TextLabels / ScrollPanels are still present.
    pub(super) fn apply_capability_gating(&self, ctx: &mut PipelineContext<'_>) {
        let vendor = ctx
            .resource::<backend::GpuProfile>()
            .map_or(backend::GpuVendor::Other, |g| g.vendor);
        concinnity_engine::settings::system::gate_unavailable_rows(ctx, &self.caps, vendor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::{GlobalTransform, RenderHandle};
    use concinnity_engine::ecs::World;
    use concinnity_engine::gfx::render_types::DrawIndex;
    use concinnity_engine::render::frame::feedback;

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
        world.insert_resource(PipelinedFrames::new(
            crate::app::pipeline::PipelineChannels {
                snapshot_tx,
                feedback_rx,
            },
        ));
        // Stand-in render half: consume each snapshot and reply with a
        // feedback whose second frame flags a stop.
        let consumer = std::thread::spawn(move || {
            for stop in [false, true] {
                let snapshot = snapshot_rx.recv().expect("a snapshot arrives");
                let render_stats = concinnity_engine::profile::RenderStats {
                    draw_calls: 7,
                    ..Default::default()
                };
                feedback_tx
                    .send(feedback::FrameFeedback {
                        render_stats,
                        replay: concinnity_engine::render::frame::ops::ReplayOutcome::default(),
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
        world.insert_resource(PipelinedFrames::new(
            crate::app::pipeline::PipelineChannels {
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

    // With the render half gone (channel closed), the pipelined step stops
    // instead of blocking or panicking.
    #[test]
    fn pipelined_step_stops_when_the_render_half_is_gone() {
        let mut world = World::new();
        let (snapshot_tx, snapshot_rx) = std::sync::mpsc::sync_channel(0);
        let (_feedback_tx, feedback_rx) = std::sync::mpsc::channel();
        world.insert_resource(PipelinedFrames::new(
            crate::app::pipeline::PipelineChannels {
                snapshot_tx,
                feedback_rx,
            },
        ));
        drop(snapshot_rx);
        let mut gs = GraphicsSystem::new(None);
        assert_eq!(gs.run_step(&mut world.context()), StepResult::Stop);
    }
}
