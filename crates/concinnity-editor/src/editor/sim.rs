// src/editor/sim.rs
//
// The editor's simulation transport: the Play / Pause / Step / Stop state
// machine over the live preview world. Pure state -- the hook's drive
// (`hook/sim_control.rs`) maps it onto the engine's freeze gate
// (`MenuOverride`) each frame and onto the preview rebuild on Stop.
//
// Stopped is the editing baseline: the world sits frozen at its authored
// state. Play unfreezes it (and hands the cursor to the world, like the old
// capture toggle). Pause freezes mid-state for inspection with the cursor
// free. Stop rebuilds the preview world from the in-memory entries, which is
// the same restore-to-authored path every committed edit already takes --
// so a committed edit while playing or paused also drops the transport to
// Stopped (`on_edit`).

// Where the transport stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SimState {
    #[default]
    Stopped,
    Playing,
    Paused,
}

// The transport plus its one-frame step pulse. `step_queued` is consumed by
// the hook's per-frame publish: the frame that takes it runs the world for
// exactly one step, then the freeze resumes.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SimControl {
    pub(crate) state: SimState,
    step_queued: bool,
}

impl SimControl {
    // Whether the world is running (and owns the cursor / gameplay input).
    pub(crate) fn playing(&self) -> bool {
        self.state == SimState::Playing
    }

    // The Play / Pause control: run from Stopped or Paused, freeze mid-state
    // from Playing.
    pub(crate) fn toggle_play(&mut self) {
        self.state = match self.state {
            SimState::Playing => SimState::Paused,
            SimState::Stopped | SimState::Paused => SimState::Playing,
        };
        self.step_queued = false;
    }

    // Freeze mid-state (Escape while playing). A no-op elsewhere.
    pub(crate) fn pause(&mut self) {
        if self.state == SimState::Playing {
            self.state = SimState::Paused;
        }
    }

    // Advance exactly one frame. From Stopped or Paused the world lands
    // paused one step further on; while Playing this is just a pause (the
    // frame already runs).
    pub(crate) fn step(&mut self) {
        match self.state {
            SimState::Playing => self.state = SimState::Paused,
            SimState::Stopped | SimState::Paused => {
                self.state = SimState::Paused;
                self.step_queued = true;
            }
        }
    }

    // Drop to Stopped. Returns whether the authored state must be restored
    // (a rebuild): true from Playing / Paused, false when already there.
    #[must_use]
    pub(crate) fn stop(&mut self) -> bool {
        let restore = self.state != SimState::Stopped;
        self.state = SimState::Stopped;
        self.step_queued = false;
        restore
    }

    // A committed edit rebuilds the preview world, which discards the run's
    // state -- so the transport honestly drops to Stopped. The rebuild is
    // already flagged by the edit path; nothing more to restore.
    pub(crate) fn on_edit(&mut self) {
        self.state = SimState::Stopped;
        self.step_queued = false;
    }

    // Whether the world simulates this frame, consuming a queued step. The
    // hook publishes `MenuOverride(Some(!run))` from this once per frame.
    pub(crate) fn take_run_frame(&mut self) -> bool {
        self.playing() || std::mem::take(&mut self.step_queued)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_pause_cycle() {
        let mut sim = SimControl::default();
        assert_eq!(sim.state, SimState::Stopped);
        sim.toggle_play();
        assert!(sim.playing());
        sim.toggle_play();
        assert_eq!(sim.state, SimState::Paused);
        sim.toggle_play();
        assert!(sim.playing(), "Play resumes from Paused");
    }

    #[test]
    fn pause_only_freezes_a_running_world() {
        let mut sim = SimControl::default();
        sim.pause();
        assert_eq!(
            sim.state,
            SimState::Stopped,
            "Pause from Stopped is a no-op"
        );
        sim.toggle_play();
        sim.pause();
        assert_eq!(sim.state, SimState::Paused);
        sim.pause();
        assert_eq!(sim.state, SimState::Paused);
    }

    #[test]
    fn step_advances_one_frame_then_freezes() {
        let mut sim = SimControl::default();
        sim.step();
        assert_eq!(sim.state, SimState::Paused);
        assert!(sim.take_run_frame(), "the queued step runs one frame");
        assert!(!sim.take_run_frame(), "then the freeze resumes");
        sim.step();
        sim.step();
        assert!(sim.take_run_frame());
        assert!(!sim.take_run_frame(), "steps do not accumulate past one");
    }

    #[test]
    fn step_while_playing_pauses() {
        let mut sim = SimControl::default();
        sim.toggle_play();
        sim.step();
        assert_eq!(sim.state, SimState::Paused);
        assert!(
            !sim.take_run_frame(),
            "the playing frame already ran; no extra pulse"
        );
    }

    #[test]
    fn stop_reports_when_a_restore_is_due() {
        let mut sim = SimControl::default();
        assert!(!sim.stop(), "already at the authored state");
        sim.toggle_play();
        assert!(sim.stop());
        assert_eq!(sim.state, SimState::Stopped);
        sim.step();
        assert!(sim.stop(), "a paused run also restores");
        assert!(!sim.take_run_frame(), "Stop drops a queued step");
    }

    #[test]
    fn an_edit_drops_the_transport_without_a_restore_request() {
        let mut sim = SimControl::default();
        sim.toggle_play();
        sim.on_edit();
        assert_eq!(sim.state, SimState::Stopped);
        assert!(!sim.take_run_frame());
    }

    #[test]
    fn playing_runs_every_frame() {
        let mut sim = SimControl::default();
        sim.toggle_play();
        assert!(sim.take_run_frame());
        assert!(sim.take_run_frame());
    }
}
