//! The pipelined loop's per-frame go-ahead. While frames wait on the GPU, the
//! render half reserves the next frame's GPU slot, samples the window input,
//! and only then lets the simulation step, so the tick reads input taken just
//! before the frame it ends with is recorded rather than input that waited out
//! the GPU. While they do not, there is no wait to move, and the go-ahead
//! comes as soon as a snapshot arrives so the next step overlaps its submit.

use concinnity_engine::input::snapshot::InputPacket;
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

#[cfg(test)]
mod tests {
    use super::*;

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
