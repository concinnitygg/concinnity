//! EditorHook: the Map panel's session state. The panel writes no entry, so
//! this is its shown state and where the canvas is looked at.

#[derive(Debug, Default)]
pub(in crate::editor::hook) struct MapState {
    pub(in crate::editor::hook) open: bool,
    // The canvas's offset, and the anchor an in-flight pan holds.
    pub(in crate::editor::hook) pan: [f32; 2],
    pub(in crate::editor::hook) pan_drag: Option<[f32; 2]>,
}

impl MapState {
    // Another world is another shape, so where the last one was looked at says
    // nothing about this one.
    pub(in crate::editor::hook) fn reset_for_world(&mut self) {
        self.pan = [0.0, 0.0];
        self.pan_drag = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_for_world_keeps_shown_state_and_drops_the_pan() {
        let mut s = MapState {
            open: true,
            pan: [120.0, 40.0],
            pan_drag: Some([1.0, 2.0]),
        };
        s.reset_for_world();
        assert!(s.open, "the panel the user opened stays open");
        assert_eq!(s.pan, [0.0, 0.0]);
        assert_eq!(s.pan_drag, None);
    }
}
