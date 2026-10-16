//! The pipelined loop's per-frame go-ahead. While frames wait on the GPU, the
//! render half reserves the next frame's GPU slot, samples the window input,
//! and only then lets the simulation step, so the tick reads input taken just
//! before the frame it ends with is recorded rather than input that waited out
//! the GPU. While they do not, there is no wait to move, and the go-ahead
//! comes as soon as a snapshot arrives so the next step overlaps its submit.

use concinnity_core::input::snapshot::InputPacket;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};

// A frame's input and when it was sampled on the host clock.
type SampledInput = (InputPacket, u64);

/// The render half's end: one go-ahead per frame, carrying that frame's input
/// and when it was sampled.
pub(crate) struct FrameGo(SyncSender<SampledInput>);

/// The simulation's end: waits for the go-ahead before each step.
#[derive(Debug)]
pub(crate) struct FrameGate(Receiver<SampledInput>);

/// A connected pair. One go-ahead is ever outstanding, since the render half
/// waits for the step's snapshot before it reserves the next frame.
pub(crate) fn frame_gate() -> (FrameGo, FrameGate) {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    (FrameGo(tx), FrameGate(rx))
}

impl FrameGo {
    /// Let the simulation step with `input`, sampled at `sampled_us`. False
    /// once the simulation is gone.
    pub(crate) fn send(&self, input: InputPacket, sampled_us: u64) -> bool {
        !matches!(
            self.0.try_send((input, sampled_us)),
            Err(TrySendError::Disconnected(_))
        )
    }
}

impl FrameGate {
    /// Block until the render half lets the next step go, returning that
    /// frame's input and when it was sampled. `None` once the render half is
    /// gone.
    pub(crate) fn wait(&self) -> Option<SampledInput> {
        self.0.recv().ok()
    }
}

// A frame that waited this long on the GPU was GPU-bound.
const GPU_BOUND_WAIT_US: u32 = 1_000;
// Frames in a row on the other side of the bound before the timing switches,
// so a frame that happens to straddle it does not flip the loop's shape.
const SWITCH_AFTER_FRAMES: u32 = 4;

/// When the render half lets the next step go: just in time, once the next
/// frame's GPU slot is reserved, while frames wait on the GPU; or as soon as
/// the current snapshot arrives while they do not, so the step overlaps that
/// frame's submit instead of queueing behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GoTiming {
    just_in_time: bool,
    // Consecutive frames that disagreed with the current timing.
    against: u32,
}

impl Default for GoTiming {
    fn default() -> Self {
        Self {
            just_in_time: true,
            against: 0,
        }
    }
}

impl GoTiming {
    /// Whether the next step waits for its frame's GPU slot.
    pub(crate) fn just_in_time(self) -> bool {
        self.just_in_time
    }

    /// Account for a frame that spent `gpu_wait_us` blocked on the GPU.
    pub(crate) fn record(&mut self, gpu_wait_us: u32) {
        let gpu_bound = gpu_wait_us >= GPU_BOUND_WAIT_US;
        if gpu_bound == self.just_in_time {
            self.against = 0;
            return;
        }
        self.against += 1;
        if self.against >= SWITCH_AFTER_FRAMES {
            self.just_in_time = gpu_bound;
            self.against = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn after(waits: &[u32]) -> GoTiming {
        let mut timing = GoTiming::default();
        for &wait in waits {
            timing.record(wait);
        }
        timing
    }

    #[test]
    fn a_gpu_bound_loop_stays_just_in_time() {
        assert!(after(&[20_000; 10]).just_in_time());
    }

    #[test]
    fn a_loop_that_never_waits_overlaps_after_a_few_frames() {
        let waits = [0; SWITCH_AFTER_FRAMES as usize];
        assert!(after(&waits[..waits.len() - 1]).just_in_time());
        assert!(!after(&waits).just_in_time());
    }

    #[test]
    fn a_loop_that_starts_waiting_goes_back_to_just_in_time() {
        let mut waits = vec![0; SWITCH_AFTER_FRAMES as usize];
        waits.extend([GPU_BOUND_WAIT_US; SWITCH_AFTER_FRAMES as usize]);
        assert!(after(&waits).just_in_time());
    }

    // One frame across the bound does not flip the timing.
    #[test]
    fn a_lone_frame_across_the_bound_changes_nothing() {
        let mut waits = vec![0; SWITCH_AFTER_FRAMES as usize];
        waits.push(GPU_BOUND_WAIT_US);
        waits.extend([0; SWITCH_AFTER_FRAMES as usize - 1]);
        assert!(!after(&waits).just_in_time());
    }

    #[test]
    fn the_go_ahead_carries_the_frames_input() {
        let (go, gate) = frame_gate();
        let mut input = InputPacket::default();
        input.raw.jump = true;
        assert!(go.send(input, 7));
        assert!(gate.wait().is_some_and(|(p, at)| p.raw.jump && at == 7));
    }

    #[test]
    fn a_gone_render_half_ends_the_wait() {
        let (go, gate) = frame_gate();
        drop(go);
        assert!(gate.wait().is_none());
    }

    #[test]
    fn a_gone_simulation_refuses_the_go_ahead() {
        let (go, gate) = frame_gate();
        drop(gate);
        assert!(!go.send(InputPacket::default(), 0));
    }

    // The gate releases the simulation from another thread, the way the
    // render half does.
    #[test]
    fn the_wait_ends_when_the_other_thread_goes() {
        let (go, gate) = frame_gate();
        let render = std::thread::spawn(move || go.send(InputPacket::default(), 0));
        assert!(gate.wait().is_some());
        assert!(render.join().unwrap());
    }
}
