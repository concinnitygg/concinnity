//! `EditorHook`: the Character Shape panel's session state: the last rejected
//! commit, the seed counter behind Randomize, a slider drag in flight
//! (`hook/drag/shape.rs`), what the panel shows, and the rows built into the
//! world.

use crate::editor::hook::drag::shape::ShapeDrag;
use crate::editor::hook::edit::character_shape::ShapeCache;
use crate::editor::panels::character_shape_panel::ShapeIds;

#[derive(Default)]
pub(in crate::editor::hook) struct ShapeState {
    pub(in crate::editor::hook) status: Option<String>,
    pub(in crate::editor::hook) seed: u64,
    pub(in crate::editor::hook) drag: Option<ShapeDrag>,
    pub(in crate::editor::hook) cache: Option<ShapeCache>,
    pub(in crate::editor::hook) shown: Option<ShapeIds>,
}
