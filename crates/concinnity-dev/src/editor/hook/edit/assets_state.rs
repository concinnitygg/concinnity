//! EditorHook: the Assets panel's session state, beside its actions in
//! `asset_tree.rs`.

use crate::editor::asset_handle::AssetHandle;
use crate::editor::panels::asset_tree::TreeGroup;

// The panel's body: every asset of the expanded world as one tree grouped by
// origin. `groups` is the cooked model (it costs a world expansion, so it is
// recomputed only when `stale` and the panel is up), `unfolded` holds the
// groups the user unfolded, `row_menu` the asset whose Delete menu is open,
// and `status` carries a cook failure to the status line. The header "+" type
// picker has its open state and scroll here too: while it is open the search
// field narrows its options instead of the tree.
#[derive(Debug)]
pub(in crate::editor::hook) struct AssetsState {
    pub(in crate::editor::hook) groups: Vec<TreeGroup>,
    pub(in crate::editor::hook) unfolded: Vec<usize>,
    pub(in crate::editor::hook) scroll: usize,
    pub(in crate::editor::hook) stale: bool,
    pub(in crate::editor::hook) status: Option<String>,
    pub(in crate::editor::hook) search_focus: bool,
    pub(in crate::editor::hook) row_menu: Option<AssetHandle>,
    pub(in crate::editor::hook) picker_open: bool,
    pub(in crate::editor::hook) picker_scroll: usize,
}

impl Default for AssetsState {
    fn default() -> Self {
        Self {
            groups: Vec::new(),
            unfolded: Vec::new(),
            scroll: 0,
            stale: true,
            status: None,
            search_focus: false,
            row_menu: None,
            picker_open: false,
            picker_scroll: 0,
        }
    }
}
