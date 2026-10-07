//! EditorHook: the Character Shape panel's session state: the row window's
//! scroll, the row count sampled once a frame (the rows come from the live
//! world, which the panel sizing cannot reach), the last rejected commit, the
//! seed counter behind Randomize, and a slider drag in flight
//! (`hook/drag/shape.rs`).

use crate::editor::hook::drag::shape::ShapeDrag;

#[derive(Default)]
pub(in crate::editor::hook) struct ShapeState {
    pub(in crate::editor::hook) scroll: usize,
    pub(in crate::editor::hook) rows: usize,
    pub(in crate::editor::hook) status: Option<String>,
    pub(in crate::editor::hook) seed: u64,
    pub(in crate::editor::hook) drag: Option<ShapeDrag>,
}
