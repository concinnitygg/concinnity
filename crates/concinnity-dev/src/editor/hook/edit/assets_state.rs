//! `EditorHook`: the Assets panel's session state, beside its actions in
//! `asset_tree.rs`.

use std::collections::BTreeSet;

use concinnity_engine::components::UiItem;

use crate::editor::asset_handle::AssetHandle;
use crate::editor::panels::asset_tree::{TreeGroup, TreeRow};
use crate::editor::panels::assets_panel::PickerOption;
use crate::editor::selection::Selection;

// The Assets panel: the expanded world's tree by origin (recomputed when
// `stale` and shown), unfolded groups, the open row menu, a status line, and
// the "+" type picker.
#[derive(Debug)]
pub(in crate::editor::hook) struct AssetsState {
    pub(in crate::editor::hook) groups: Vec<TreeGroup>,
    // Counts the times `groups` was rebuilt.
    pub(in crate::editor::hook) tree_rev: u64,
    pub(in crate::editor::hook) unfolded: Vec<usize>,
    pub(in crate::editor::hook) stale: bool,
    pub(in crate::editor::hook) status: Option<String>,
    pub(in crate::editor::hook) search_focus: bool,
    // The search field's text, as the panel last showed it.
    pub(in crate::editor::hook) filter: String,
    pub(in crate::editor::hook) row_menu: Option<AssetHandle>,
    // The row the open menu's asset sits at, found when the menu opens and
    // whenever the rows are rebuilt.
    pub(in crate::editor::hook) menu_at: Option<usize>,
    pub(in crate::editor::hook) picker_open: bool,
    // The tree's rows and their list items, and what they were built from.
    pub(in crate::editor::hook) rows: Vec<TreeRow>,
    pub(in crate::editor::hook) rows_key: Option<RowsKey>,
    pub(in crate::editor::hook) items: Vec<UiItem>,
    pub(in crate::editor::hook) items_rev: u64,
    pub(in crate::editor::hook) decor: Option<DecorKey>,
    // The picker's options, and the filter they were narrowed by.
    pub(in crate::editor::hook) picker: Vec<PickerOption>,
    pub(in crate::editor::hook) picker_filter: Option<String>,
    pub(in crate::editor::hook) picker_rev: u64,
    // A row to scroll into view: asked for, and being shown this frame.
    pub(in crate::editor::hook) reveal: Option<usize>,
    pub(in crate::editor::hook) reveal_now: Option<usize>,
}

// What the tree's rows are flattened from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::editor::hook) struct RowsKey {
    pub(in crate::editor::hook) tree: u64,
    pub(in crate::editor::hook) unfolded: Vec<usize>,
    pub(in crate::editor::hook) filter: Option<String>,
}

// What an asset row's session state is read from.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::editor::hook) struct DecorKey {
    pub(in crate::editor::hook) entries: u64,
    pub(in crate::editor::hook) selection: Selection,
    pub(in crate::editor::hook) hidden: BTreeSet<AssetHandle>,
    pub(in crate::editor::hook) locked: BTreeSet<AssetHandle>,
}

impl Default for AssetsState {
    fn default() -> Self {
        Self {
            groups: Vec::new(),
            tree_rev: 0,
            unfolded: Vec::new(),
            stale: true,
            status: None,
            search_focus: false,
            filter: String::new(),
            row_menu: None,
            menu_at: None,
            picker_open: false,
            rows: Vec::new(),
            rows_key: None,
            items: Vec::new(),
            items_rev: 0,
            decor: None,
            picker: Vec::new(),
            picker_filter: None,
            picker_rev: 0,
            reveal: None,
            reveal_now: None,
        }
    }
}
