//! Pipelined frame driver: simulation and rendering on separate threads. The
//! main thread keeps everything the OS and GPU pin to it -- the event pumps
//! (which run inside draw_frame / window_closed / pump_events), the backend,
//! frame submission, and swapchain recreation -- while a "sim" thread owns the
//! Runtime (pacer, fixed-tick clock, World::step) and produces RenderSnapshots.
//!
//! Each step waits for the render half's go-ahead, which carries the input it
//! reads (`frame_gate`). While frames wait on the GPU, the render half reserves
//! the frame's GPU slot before sampling that input, so GPU and vsync
//! backpressure land in the sim's wait for the go-ahead -- where the fixed-tick
//! accumulator counts it as ordinary frame time -- and the step simulates right
//! before its frame is recorded rather than a GPU wait ahead of it. While they
//! do not, the next step goes as soon as a snapshot arrives and overlaps that
//! frame's submission.
//!
//! The snapshot channel is a rendezvous (bound 0): exactly one frame is in
//! flight, and every snapshot is consumed exactly once in order (the contract
//! the model-push dedupe and the slot-allocation ops rely on). The feedback
//! channel returns render stats, op-replay results, and the consumed snapshot
//! for buffer reuse.

use concinnity_core::ecs::StepResult;
use concinnity_core::input::snapshot::InputPacket;
use concinnity_core::render::backend::RenderBackend;
use concinnity_core::render::feedback::FrameFeedback;
use concinnity_core::render::snapshot::RenderSnapshot;
use concinnity_host::thread::set_current_thread_role;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

use crate::app::budget::main_thread_role;
use crate::app::frame_gate::FrameGo;
use crate::app::runtime::Runtime;
use crate::ecs::PipelinedFrames;
use crate::gfx::system::submit::submit;
use concinnity_core::render::frame_submit::FramePolicy;
use concinnity_core::render::frame_submit::GoTiming;
use concinnity_core::shutdown::ShutdownToken;
use concinnity_host::clock::micros_since;

// Drive a started Runtime with pipelined frames until it stops. Evicts the
// backend from the world onto this (main) thread, publishes the channel pair,
// moves the Runtime to the sim thread, and runs the render half here. A world
// that never built a backend (headless) falls back to the serial loop. A
// panic on the sim thread is re-raised here after the render half drains, so
// the process fails the same way a serial panic does. `screenshot` captures
// the last presented frame on the way out (skipped after a device loss).
pub(crate) fn run_pipelined(mut runtime: Runtime, screenshot: Option<&str>) {
    let Some(mut backend) = crate::ecs::take_render_backend(runtime.world_mut()) else {
        crate::app::runloop::run_loop(&mut runtime, false, |_| {});
        return;
    };
    let role = main_thread_role(runtime.world());
    set_current_thread_role(role);
    let shutdown = runtime.shutdown_token();
    let (snapshot_tx, snapshot_rx) = std::sync::mpsc::sync_channel::<RenderSnapshot>(0);
    let (feedback_tx, feedback_rx) = std::sync::mpsc::channel::<FrameFeedback>();
    let (go, gate) = crate::app::frame_gate::frame_gate();
    runtime.gate_steps_on(gate);
    runtime
        .world_mut()
        .insert_resource(PipelinedFrames::new(crate::ecs::PipelineChannels {
            snapshot_tx,
            feedback_rx,
        }));

    let sim_shutdown = shutdown.clone();
    let sim = std::thread::Builder::new()
        .name("sim".to_string())
        .spawn(move || {
            set_current_thread_role(role);
            loop {
                if sim_shutdown.is_canceled() {
                    return;
                }
                match runtime.world_step() {
                    StepResult::Continue => {}
                    StepResult::Stop | StepResult::Done => return,
                }
            }
            // The Runtime (with the channel ends inside its world) drops here,
            // which unblocks the render half's receive with Disconnected.
        })
        .expect("failed to spawn the sim thread");

    let rendered = render_half(backend.as_mut(), go, snapshot_rx, &feedback_tx, &shutdown);
    // What the run actually drew, which `--frames N` only requested: a pixel A/B
    // of a temporal world is a comparison of the frame it stopped on, so the
    // capture harness reads this back and refuses a run that stopped elsewhere.
    tracing::info!("pipeline: {} frame(s) submitted", rendered.submitted);
    // Whatever ended the render half, make the stop mutual: the sim exits at
    // its next token check, and its blocked send (if any) already errored
    // when the receiver dropped at the end of render_half.
    shutdown.cancel();
    drop(feedback_tx);
    if let Err(payload) = sim.join() {
        // The crash hook already wrote the report on the sim thread; exit
        // through the same unwind a serial panic takes.
        std::panic::resume_unwind(payload);
    }
    // The submit stop paths that can wait already did; a sim-initiated stop
    // leaves in-flight frames to drain here. Never after a device loss: that
    // queue can no longer signal.
    if !rendered.device_lost {
        backend.wait_idle();
        backend.finish_captures();
        if let Some(path) = screenshot {
            match backend.screenshot(path) {
                Ok(saved) => tracing::info!("screenshot saved: {}", saved),
                Err(e) => tracing::warn!("screenshot failed: {}", e),
            }
        }
    }
}

// Drive frames until a stop or the sim side closes a channel. While frames
// wait on the GPU, each one reserves its GPU slot first, pumps and samples the
// window input, and only then lets the simulation step,
// so the tick reads input taken just before its frame is recorded; the time
// spent reserving is the frame's wait on the GPU too. While they do not wait,
// the next step goes as soon as a snapshot arrives and overlaps its submit
// (`GoTiming`). Each frame then replays its ops, submits, and sends the
// feedback. Returns whether the stop was a device loss, and how many frames
// reached the backend.
fn render_half(
    backend: &mut dyn RenderBackend,
    go: FrameGo,
    snapshot_rx: Receiver<RenderSnapshot>,
    feedback_tx: &Sender<FrameFeedback>,
    shutdown: &ShutdownToken,
) -> RenderHalfOutcome {
    let mut policy = FramePolicy::default();
    let mut timing = GoTiming::default();
    let mut submitted = 0u64;
    // The step's go-ahead already went, ahead of its snapshot.
    let mut went = false;
    loop {
        if shutdown.is_canceled() {
            return RenderHalfOutcome::stopped(submitted);
        }
        let mut prepare_us = 0;
        if !went {
            let prepared = Instant::now();
            backend.prepare_frame();
            prepare_us = micros_since(prepared);
            if !let_step_go(&go, backend) {
                return RenderHalfOutcome::stopped(submitted);
            }
        }
        let Some(mut snapshot) = wait_for_snapshot(&snapshot_rx, shutdown) else {
            return RenderHalfOutcome::stopped(submitted);
        };
        // A loop that is not waiting on the GPU has no wait to move: the next
        // step goes now and overlaps this frame's submit. A simulation already
        // gone ends the loop at the next snapshot wait.
        went = !timing.just_in_time() && let_step_go(&go, backend);

        let outcome = submit(&mut policy, &mut snapshot, backend);
        submitted += 1;
        let stop = outcome.result != StepResult::Continue;
        let mut render_stats = outcome.render_stats.unwrap_or_default();
        render_stats.gpu_wait_us = render_stats.gpu_wait_us.saturating_add(prepare_us);
        timing.record(render_stats.gpu_wait_us);
        let feedback = FrameFeedback {
            render_stats,
            replay: outcome.replay,
            recycled: snapshot,
            stop,
            skipped: outcome.skipped,
        };
        // A send failure means the sim already stopped; nothing left to tell.
        let _ = feedback_tx.send(feedback);
        if stop {
            return RenderHalfOutcome {
                device_lost: outcome.device_lost.is_some(),
                submitted,
            };
        }
    }
}

// Pump the window's pending events, sample the input they produced, and let
// the simulation step on it. False once the simulation is gone.
fn let_step_go(go: &FrameGo, backend: &mut dyn RenderBackend) -> bool {
    backend.pump_events();
    let input = InputPacket::sample(backend);
    go.send(input, concinnity_host::clock::monotonic_micros())
}

// What the render half did, read by the caller after the loop ends.
struct RenderHalfOutcome {
    // The stop was a device loss: nothing may wait on that queue on the way out.
    device_lost: bool,
    // Frames handed to the backend, which the frame cap makes exact.
    submitted: u64,
}

impl RenderHalfOutcome {
    // Ended by a shutdown or a closed channel rather than by a failed frame.
    fn stopped(submitted: u64) -> Self {
        Self {
            device_lost: false,
            submitted,
        }
    }
}

// Wait for the next snapshot, or `None` once the sim side closed the channel
// or a shutdown was asked for. On macOS the wait is sliced so the Cocoa run
// loop keeps draining while the simulation runs long (the window stays
// responsive); elsewhere the event pumps live inside the draw itself, so a
// plain blocking receive is right.
#[cfg(target_os = "macos")]
fn wait_for_snapshot(
    rx: &Receiver<RenderSnapshot>,
    shutdown: &ShutdownToken,
) -> Option<RenderSnapshot> {
    use std::sync::mpsc::RecvTimeoutError;
    loop {
        crate::app::runloop::drain_cocoa_events();
        match rx.recv_timeout(std::time::Duration::from_millis(2)) {
            Ok(snapshot) => return Some(snapshot),
            Err(RecvTimeoutError::Timeout) if !shutdown.is_canceled() => {}
            Err(_) => return None,
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn wait_for_snapshot(
    rx: &Receiver<RenderSnapshot>,
    _shutdown: &ShutdownToken,
) -> Option<RenderSnapshot> {
    rx.recv().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::render::backend::recording::{Call, recording_backend};

    // The input a go-ahead carries is sampled after the window's events are
    // pumped, so it never misses events that arrived since the last draw.
    #[test]
    fn the_go_ahead_pumps_events_before_sampling_input() {
        let (state, mut backend) = recording_backend();
        let (go, gate) = crate::app::frame_gate::frame_gate();
        assert!(let_step_go(&go, &mut backend));
        assert!(gate.wait().is_some());
        let calls = &state.lock().calls;
        let pumped = calls.iter().position(|c| matches!(c, Call::PumpEvents));
        let sampled = calls.iter().position(|c| matches!(c, Call::TakeInput));
        assert!(
            pumped.is_some() && pumped < sampled,
            "pump before sample: {calls:?}"
        );
    }
}
