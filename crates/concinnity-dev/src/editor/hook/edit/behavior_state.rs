//! EditorHook: the Behavior panel's session state. The actions that read the
//! world or the entry list live in `behavior.rs`; this holds what the panel
//! remembers between frames and the resets that touch nothing else.

use crate::editor::behavior::clip::Clip;
use crate::editor::behavior::panel::{Status, ViewMode};
use crate::editor::behavior::path::Path;
use crate::editor::behavior::pulse::NodePulse;

// Shown state, which of the world's Behavior entries is open (an ordinal into
// them, so an unrelated add / delete cannot retarget it), the selected outline
// row, the outline and palette scrolls, whether the palette is up and which of
// its options the keyboard is on, whether the value field holds keyboard focus,
// and the world checker's verdict on the open behavior. The name field carries
// its own focus, and `remove_armed` is the removal chip waiting on the press
// that carries it out.
#[derive(Debug, Default)]
pub(in crate::editor::hook) struct BehaviorState {
    pub(in crate::editor::hook) open: bool,
    pub(in crate::editor::hook) index: usize,
    pub(in crate::editor::hook) row: Option<usize>,
    pub(in crate::editor::hook) scroll: usize,
    pub(in crate::editor::hook) picking: bool,
    pub(in crate::editor::hook) pick_scroll: usize,
    pub(in crate::editor::hook) pick: usize,
    // The palette's filter text, mirrored off its field once a frame so the
    // presses and draws that follow all narrow by the same query.
    pub(in crate::editor::hook) filter: String,
    pub(in crate::editor::hook) focus: bool,
    pub(in crate::editor::hook) name_focus: bool,
    pub(in crate::editor::hook) remove_armed: bool,
    pub(in crate::editor::hook) status: Option<Status>,
    pub(in crate::editor::hook) mode: ViewMode,
    // The chart's scroll offset, and the anchor an in-flight canvas pan holds.
    pub(in crate::editor::hook) pan: [f32; 2],
    pub(in crate::editor::hook) pan_drag: Option<[f32; 2]>,
    // The list member held for a paste, with the kind of list it came out of so
    // it can only land in one of the same kind. Session state, not an edit, and
    // deliberately not cleared by opening another behavior: carrying a node
    // between two of them is most of the point.
    pub(in crate::editor::hook) clip: Option<Clip>,
    // The overview's selected card. The map's cards stand for whole behaviors
    // and the things they reach rather than for places inside one, so the
    // outline row the other two views share cannot address them.
    pub(in crate::editor::hook) overview_card: Option<usize>,
    // Live-debug state fed by the runtime's execution trace while a play
    // session runs with the Behavior or Variables panel open
    // (`hook/drive/trace.rs`). Pulses cover the OPEN behavior only (paths are
    // per-body); breakpoints are held by behavior NAME + node path so they
    // survive preview rebuilds and body edits shifting node ids.
    pub(in crate::editor::hook) pulses: Vec<NodePulse>,
    pub(in crate::editor::hook) breakpoints: Vec<(String, Path)>,
}

impl BehaviorState {
    // Release every input the panel can hold the keyboard with.
    pub(in crate::editor::hook) fn blur_inputs(&mut self) {
        self.focus = false;
        self.name_focus = false;
        self.remove_armed = false;
        self.picking = false;
    }

    // Forget what indexed the world being left. The clip and the breakpoints
    // survive a switch: both are keyed by content, not by the old entry list.
    pub(in crate::editor::hook) fn reset_for_world(&mut self) {
        self.index = 0;
        self.row = None;
        self.scroll = 0;
        self.status = None;
    }

    // Step to the next view. The pan does not survive, because each chart is
    // its own shape.
    pub(in crate::editor::hook) fn toggle_view(&mut self) {
        self.mode = self.mode.other();
        self.picking = false;
        self.pan_drag = None;
        self.pan = [0.0, 0.0];
    }

    // Toggle the breakpoint on `path` inside the behavior called `name`.
    pub(in crate::editor::hook) fn toggle_breakpoint(&mut self, name: String, path: &Path) {
        match self
            .breakpoints
            .iter()
            .position(|(n, p)| n == &name && p == path)
        {
            Some(i) => {
                self.breakpoints.remove(i);
            }
            None => self.breakpoints.push((name, path.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::behavior::outline::List;

    fn dirty() -> BehaviorState {
        BehaviorState {
            open: true,
            index: 3,
            row: Some(2),
            scroll: 4,
            picking: true,
            pick_scroll: 5,
            pick: 6,
            filter: "wait".into(),
            focus: true,
            name_focus: true,
            remove_armed: true,
            status: Some(Status::message("bad")),
            mode: ViewMode::Chart,
            pan: [1.0, 2.0],
            pan_drag: Some([3.0, 4.0]),
            clip: Some(Clip {
                list: List::Nodes,
                value: serde_json::json!({ "kind": "wait" }),
            }),
            overview_card: Some(1),
            pulses: Vec::new(),
            breakpoints: vec![("b".into(), Vec::new())],
        }
    }

    #[test]
    fn blur_inputs_releases_only_the_input_flags() {
        let mut s = dirty();
        s.blur_inputs();
        assert!(!s.focus && !s.name_focus && !s.remove_armed && !s.picking);
        assert!(s.open);
        assert_eq!((s.index, s.row, s.scroll), (3, Some(2), 4));
        assert_eq!((s.pick, s.pick_scroll, s.filter.as_str()), (6, 5, "wait"));
        assert!(s.status.is_some());
        assert_eq!(s.mode, ViewMode::Chart);
        assert_eq!((s.pan, s.pan_drag), ([1.0, 2.0], Some([3.0, 4.0])));
    }

    #[test]
    fn reset_for_world_keeps_the_clip_and_breakpoints() {
        let mut s = dirty();
        s.reset_for_world();
        assert_eq!((s.index, s.row, s.scroll, s.status), (0, None, 0, None));
        assert!(s.clip.is_some());
        assert_eq!(s.breakpoints.len(), 1);
        assert!(s.open && s.focus && s.name_focus && s.remove_armed && s.picking);
        assert_eq!((s.pick, s.pick_scroll, s.filter.as_str()), (6, 5, "wait"));
        assert_eq!((s.mode, s.overview_card), (ViewMode::Chart, Some(1)));
        assert_eq!((s.pan, s.pan_drag), ([1.0, 2.0], Some([3.0, 4.0])));
    }
}
