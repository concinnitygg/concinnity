//! `EditorHook`: the Map panel's session state. The panel writes no entry, so
//! this is where the canvas is looked at.

use crate::editor::asset_handle::AssetHandle;
use crate::editor::behavior::graph::Chart;

#[derive(Debug, Default)]
pub(in crate::editor::hook) struct MapState {
    // The canvas's offset, and the card under the cursor.
    pub(in crate::editor::hook) pan: [f32; 2],
    pub(in crate::editor::hook) hovered: Option<usize>,
    // Whether the canvas has been put on where the world it is drawing starts.
    // Cleared whenever the map becomes a fresh look, so the panel opens on the
    // world's entry point rather than on wherever it was last left.
    pub(in crate::editor::hook) rooted: bool,
    // The place the canvas was last brought to, so a selection made elsewhere
    // moves it once rather than every frame it stays selected.
    pub(in crate::editor::hook) shown: Option<AssetHandle>,
    // The chart last drawn, and the entry revision it was built from.
    pub(in crate::editor::hook) chart: Option<(u64, Chart)>,
    // Scratch: where the wire words went, while the chart is written.
    pub(in crate::editor::hook) taken: Vec<[f32; 4]>,
}

impl MapState {
    // Another world is another shape, so where the last one was looked at says
    // nothing about this one.
    pub(in crate::editor::hook) fn reset_for_world(&mut self) {
        self.pan = [0.0, 0.0];
        self.hovered = None;
        self.reroot();
    }

    // Put the canvas back on where the world starts, the next frame the panel
    // drives.
    pub(in crate::editor::hook) fn reroot(&mut self) {
        self.rooted = false;
        self.shown = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::entry_list::EntryList;

    fn a_handle() -> AssetHandle {
        AssetHandle::Entry(
            EntryList::new(vec![serde_json::json!({"type": "Scene"})])
                .key_at(0)
                .unwrap(),
        )
    }

    fn looked_at() -> MapState {
        MapState {
            pan: [120.0, 40.0],
            hovered: Some(1),
            rooted: true,
            shown: Some(a_handle()),
            chart: None,
            taken: Vec::new(),
        }
    }

    #[test]
    fn reset_for_world_drops_the_pan() {
        let mut s = looked_at();
        s.reset_for_world();
        assert_eq!(s.pan, [0.0, 0.0]);
        assert_eq!(s.hovered, None);
    }

    // The canvas is rooted per world, so the next frame puts it on where this
    // world starts rather than on the place the last one was left showing.
    #[test]
    fn reset_for_world_leaves_the_canvas_to_be_rooted_again() {
        let mut s = looked_at();
        s.reset_for_world();
        assert!(!s.rooted);
        assert_eq!(s.shown, None);
    }

    #[test]
    fn rerooting_leaves_the_pan_alone_until_the_canvas_is_driven() {
        let mut s = looked_at();
        s.reroot();
        assert!(!s.rooted);
        assert_eq!(s.shown, None);
        assert_eq!(s.pan, [120.0, 40.0], "nothing here knows the map's shape");
    }
}
