//! `EditorHook`: the Content panel's session state (the visual-asset thumbnail
//! grid): the type button's cycle position (0 = All, i = `VISUAL_TYPES[i-1]`),
//! the search text and whether its field holds keyboard focus, the cells and
//! what they were built from, and an in-flight drag-out placement
//! (`hook/drag/content.rs`), if any.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_engine::components::UiItem;

use std::sync::Arc;

use crate::editor::hook::drag::content::ContentDrag;
use crate::editor::selection::Selection;
use crate::editor::thumbs::ThumbSet;

#[derive(Default)]
pub(in crate::editor::hook) struct ContentState {
    pub(in crate::editor::hook) type_chip: usize,
    pub(in crate::editor::hook) search_focus: bool,
    pub(in crate::editor::hook) filter: String,
    // The shown assets, their cells, and what they were built from.
    pub(in crate::editor::hook) items: Vec<(String, RegisteredType)>,
    pub(in crate::editor::hook) cells: Vec<UiItem>,
    pub(in crate::editor::hook) cells_rev: u64,
    pub(in crate::editor::hook) key: Option<CellsKey>,
    pub(in crate::editor::hook) drag: Option<ContentDrag>,
}

// What the cells are built from.
pub(in crate::editor::hook) struct CellsKey {
    pub(in crate::editor::hook) tree: u64,
    pub(in crate::editor::hook) type_chip: usize,
    pub(in crate::editor::hook) filter: String,
    // The thumbnail set, held so its address cannot be reused while compared.
    pub(in crate::editor::hook) thumbs: Arc<ThumbSet>,
    pub(in crate::editor::hook) entries: u64,
    pub(in crate::editor::hook) selection: Selection,
}
