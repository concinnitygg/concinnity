// src/app/pipeline.rs
//
// Pipelined frame driver: simulation of frame N+1 overlaps rendering of frame
// N. The main thread keeps everything the OS and GPU pin to it -- the event
// pumps (which run inside draw_frame / window_closed), the backend, frame
// submission, and swapchain recreation -- while a "sim" thread owns the App
// (pacer, fixed-tick clock, World::step) and produces RenderSnapshots.
//
// The snapshot channel is a rendezvous (bound 0): exactly one frame is in
// flight, every snapshot is consumed exactly once in order (the contract the
// model-push dedupe and the slot-allocation ops rely on), and vsync
// backpressure propagates into the sim's blocked send -- where the fixed-tick
// accumulator counts it as ordinary frame time. The feedback channel returns
// the sampled input, render stats, op-replay results, and the consumed
// snapshot for buffer reuse.

use crate::app::state::App;
use crate::ecs::{PipelinedFrames, StepResult};
use crate::gfx::backend::RenderBackend;
use crate::gfx::feedback::FrameFeedback;
use crate::gfx::graphics_system::frame_policy::FramePolicy;
use crate::gfx::graphics_system::submit::submit;
use crate::gfx::input::InputPacket;
use crate::gfx::snapshot::RenderSnapshot;
use crate::shutdown::ShutdownToken;
use std::sync::mpsc::{Receiver, Sender};

// Drive a started App with pipelined frames until it stops. Evicts the
// backend from the world onto this (main) thread, publishes the channel pair,
// moves the App to the sim thread, and runs the render half here. A world
// that never built a backend (headless) falls back to the serial loop. A
// panic on the sim thread is re-raised here after the render half drains, so
// the process fails the same way a serial panic does. `screenshot` captures
// the last presented frame on the way out (skipped after a device loss).
pub(crate) fn run_pipelined(mut app: App, screenshot: Option<&str>) {
    let Some(mut backend) = app.world_mut().take_render_backend() else {
        crate::app::runloop::run_loop(&mut app, false, |_| {});
        return;
    };
    let shutdown = app.shutdown_token();
    let (snapshot_tx, snapshot_rx) = std::sync::mpsc::sync_channel::<RenderSnapshot>(0);
    let (feedback_tx, feedback_rx) = std::sync::mpsc::channel::<FrameFeedback>();
    app.world_mut()
        .insert_resource(PipelinedFrames(Some(crate::ecs::PipelineChannels {
            snapshot_tx,
            feedback_rx,
        })));

    let sim_shutdown = shutdown.clone();
    let sim = std::thread::Builder::new()
        .name("sim".to_string())
        .spawn(move || {
            loop {
                if sim_shutdown.is_cancelled() {
                    return;
                }
                match app.world_step() {
                    StepResult::Continue => {}
                    StepResult::Stop | StepResult::Done => return,
                }
            }
            // The App (with the channel ends inside its world) drops here,
            // which unblocks the render half's receive with Disconnected.
        })
        .expect("failed to spawn the sim thread");

    let rendered = render_half(backend.as_mut(), snapshot_rx, &feedback_tx, &shutdown);
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
        if let Some(path) = screenshot {
            match backend.screenshot(path) {
                Ok(saved) => tracing::info!("screenshot saved: {}", saved),
                Err(e) => tracing::warn!("screenshot failed: {}", e),
            }
        }
    }
}

// Consume snapshots until a stop or the sim side closes the channel: replay
// the ops and submit each frame, sample input right after the draw (whose
// event pump produced it), and send the feedback. Returns whether the stop
// was a device loss, and how many frames reached the backend.
fn render_half(
    backend: &mut dyn RenderBackend,
    snapshot_rx: Receiver<RenderSnapshot>,
    feedback_tx: &Sender<FrameFeedback>,
    shutdown: &ShutdownToken,
) -> RenderHalfOutcome {
    let mut policy = FramePolicy::default();
    let mut submitted = 0u64;
    loop {
        if shutdown.is_cancelled() {
            return RenderHalfOutcome::stopped(submitted);
        }
        let mut snapshot = match wait_for_snapshot(&snapshot_rx) {
            Ok(snapshot) => snapshot,
            #[cfg(target_os = "macos")]
            Err(SnapshotWaitEnd::Empty) => continue,
            Err(SnapshotWaitEnd::Closed) => return RenderHalfOutcome::stopped(submitted),
        };

        let mut outcome = submit(&mut policy, &mut snapshot, backend);
        submitted += 1;
        outcome.replay.memory_pressure |= outcome.memory_pressure;
        let stop = outcome.result != StepResult::Continue;
        let feedback = FrameFeedback {
            input: InputPacket::sample(backend),
            render_stats: outcome.render_stats.unwrap_or_default(),
            replay: outcome.replay,
            recycled: snapshot,
            stop,
        };
        // A send failure means the sim already stopped; nothing left to tell.
        let _ = feedback_tx.send(feedback);
        if stop {
            return RenderHalfOutcome {
                device_lost: outcome.device_lost,
                submitted,
            };
        }
    }
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

// Why a wait returned without a snapshot. The snapshot itself travels as the
// `Ok` value so the per-frame path never boxes it.
enum SnapshotWaitEnd {
    // Only the macOS wait is sliced (to keep the Cocoa run loop draining);
    // elsewhere the receive blocks until a snapshot or disconnect.
    #[cfg(target_os = "macos")]
    Empty,
    Closed,
}

// Wait for the next snapshot. On macOS the wait is sliced so the Cocoa run
// loop keeps draining while the simulation runs long (the window stays
// responsive); elsewhere the event pumps live inside the draw itself, so a
// plain blocking receive is right.
#[cfg(target_os = "macos")]
fn wait_for_snapshot(rx: &Receiver<RenderSnapshot>) -> Result<RenderSnapshot, SnapshotWaitEnd> {
    use std::sync::mpsc::RecvTimeoutError;
    crate::app::runloop::drain_cocoa_events();
    match rx.recv_timeout(std::time::Duration::from_millis(2)) {
        Ok(snapshot) => Ok(snapshot),
        Err(RecvTimeoutError::Timeout) => Err(SnapshotWaitEnd::Empty),
        Err(RecvTimeoutError::Disconnected) => Err(SnapshotWaitEnd::Closed),
    }
}

#[cfg(not(target_os = "macos"))]
fn wait_for_snapshot(rx: &Receiver<RenderSnapshot>) -> Result<RenderSnapshot, SnapshotWaitEnd> {
    match rx.recv() {
        Ok(snapshot) => Ok(snapshot),
        Err(_) => Err(SnapshotWaitEnd::Closed),
    }
}
