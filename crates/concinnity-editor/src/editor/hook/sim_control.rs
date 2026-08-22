// src/editor/hook/sim_control.rs
//
// EditorHook: the simulation transport's per-frame drive. The pure state
// machine lives in `editor/sim.rs`; this maps it onto the engine's freeze
// gate (`MenuOverride`), onto the preview rebuild that restores the authored
// state on Stop, and onto the transport's keyboard shortcuts. The tick runs
// before the world step, so a publish here gates that same frame's systems.

use super::*;
use crate::assets::InputKey;

impl EditorHook {
    // Transport shortcuts: Ctrl+P plays / pauses, Ctrl+Shift+P stops,
    // Ctrl+Period steps. Unlike the history shortcuts these stay live while
    // the world runs (pausing it is their whole point; `captured_key`
    // surfaces through play mode), standing down only while a text field
    // owns the keyboard or a gizmo drag is mid-flight.
    pub(super) fn sim_keys(&mut self, input: &FrameInput) {
        if !input.ctrl || self.text_focus_active() || self.gizmo_drag.is_some() {
            return;
        }
        match input.captured_key {
            Some(InputKey::P) if input.shift => self.sim_stop(),
            Some(InputKey::P) => self.sim_toggle_play(),
            Some(InputKey::Period) => self.sim.step(),
            _ => {}
        }
    }

    // The Play / Pause control (top-bar chip, Preview row, Ctrl+P). Entering
    // play hands the cursor to the world; the fly camera stands down (the two
    // are exclusive cursor owners).
    pub(super) fn sim_toggle_play(&mut self) {
        self.sim.toggle_play();
        if self.sim.playing() {
            self.fly = false;
            self.fly_clock = None;
        }
    }

    // Stop: drop the transport and restore the authored state by rebuilding
    // the preview world from the in-memory entries -- the same path every
    // committed edit takes, so simulated transforms, spawns, behavior state,
    // and physics are all discarded for what edit mode showed before Play.
    pub(super) fn sim_stop(&mut self) {
        if self.sim.stop() {
            self.rebuild_preview = true;
        }
    }

    // The per-frame freeze publish: the world simulates only while Playing or
    // on the one frame a queued Step runs.
    pub(super) fn drive_sim(&mut self, world: &mut World) {
        let run = self.sim.take_run_frame();
        world.insert_resource(MenuOverride(Some(!run)));
    }
}
