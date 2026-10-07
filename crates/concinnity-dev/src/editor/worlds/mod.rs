//! The Worlds panel: the project's worlds, most recently edited first, a `+`
//! that starts an untitled one, and a per-row triple-dot menu. It has two
//! presentations of the same list (`Mode`): the start screen a session with no
//! world named opens on, a sidebar docked down the window's left edge over the
//! world it previews, and the in-session switcher, where a row click opens that
//! world behind the unsaved-changes guard.
//!
//! Layout half only: `geometry.rs` owns the rects and the hit test, `draw.rs`
//! the per-frame layout, and `hook/edit/worlds.rs` / `hook/worlds_start.rs` the
//! actions. `cinematic.rs` is the start screen's attract camera over the world
//! the sidebar previews, and `loading.rs` the cover that stands over it while
//! that world is compiled.

pub(crate) mod cinematic;
mod draw;
pub(crate) mod files;
mod geometry;
pub(crate) mod loading;

pub(crate) use draw::place;
pub(crate) use geometry::{Layout, Mode, hit_test};

use super::hud_ids::{hud_ids, panel_base};
use super::panels::registry::PanelKey;

// Row slots the panel has elements for. The docked sidebar fills a tall window
// with them; the switcher shows fewer. A longer listing scrolls.
pub(crate) const POOL: usize = 24;

// Chrome, then the rows, then the overlays that float over them: the scrollbar,
// the triple-dot button (one set, positioned on whichever row offers it this
// frame) and the row menu it opens. `NEW_BG` is the `+` that starts an
// untitled world.
hud_ids! {
    base: panel_base(PanelKey::Worlds);
    sprites: [
        pub(crate) PANEL_BG,
        pub(crate) CLOSE_BG,
        pub(crate) NEW_BG,
        pub(crate) row_bg[POOL],
        pub(crate) LIST_TRACK,
        pub(crate) LIST_THUMB,
        pub(crate) DOT_BG,
        pub(crate) DOT1,
        pub(crate) DOT2,
        pub(crate) DOT3,
        pub(crate) MENU_BG,
        pub(crate) MENU_OPEN_BG,
        pub(crate) MENU_DELETE_BG,
    ];
    labels: [
        pub(crate) TITLE_LABEL,
        pub(crate) CLOSE_LABEL,
        pub(crate) NEW_LABEL,
        pub(crate) STATUS_LABEL,
        pub(crate) LIST_HEADER,
        pub(crate) row_label[POOL],
        pub(crate) MENU_OPEN_LABEL,
        pub(crate) MENU_DELETE_LABEL,
    ];
}

// One listed world, as the panel draws it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorldRow {
    pub name: String,
    pub path: String,
    // Whether this is the world the session currently has open.
    pub open: bool,
}

// What switching to another world does once the unsaved-changes guard clears.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WorldTarget {
    // Open the world file at this path.
    Open(String),
    // Start on an empty world that is not on disk yet. Nothing is written and
    // nothing is named until the first SAVE asks what to call it.
    Untitled,
}

// A Worlds-panel decision the confirmation dialog is holding: the dialog's
// button hands one of these back when it is pressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WorldsConfirm {
    // Delete the world file at this path.
    Delete(String),
    // Save the open world, then go to the target.
    Save(WorldTarget),
    // Go to the target, dropping the open world's unsaved edits.
    Discard(WorldTarget),
}

// The per-frame view the hook assembles.
pub(crate) struct WorldsView<'a> {
    pub rows: &'a [WorldRow],
    pub scroll: usize,
    // The presentation drawn (and hit-tested) this frame, resolved against the
    // window it is laid out in.
    pub layout: Layout,
    // The start screen's selected row: it carries the Open chip and reads as
    // picked. Always `None` in the switcher, which has no selection model.
    pub selected: Option<usize>,
    // The row whose world is compiled into the background behind the start
    // screen. Clicking it again opens it, which is what makes the second click
    // a commit rather than another preview.
    pub previewing: Option<usize>,
    // The row whose triple-dot menu is open. While one is, the panel is modal
    // over itself: the menu picks, and any other press dismisses it.
    pub menu: Option<usize>,
    // Why the last preview failed, if it did.
    pub status: Option<&'a str>,
    pub mouse: [f32; 2],
}

impl WorldsView<'_> {
    // The visible slot listed world `i` sits at, or `None` while it is scrolled
    // out of the window.
    pub(crate) fn slot_of(&self, i: usize) -> Option<usize> {
        let slot = i.checked_sub(self.scroll)?;
        (slot < self.layout.rows() && i < self.rows.len()).then_some(slot)
    }
}

// A resolved Worlds-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorldsAction {
    // Start an untitled world and edit it.
    New,
    // Preview listed world `i` behind the start screen (an index into the
    // view's rows). Never resolved by the switcher.
    Select(usize),
    // Open listed world `i`.
    Open(usize),
    // Show listed world `i`'s row menu.
    OpenMenu(usize),
    // Delete listed world `i`, behind the confirmation dialog.
    Delete(usize),
    // Dismiss the open row menu without picking from it.
    CloseMenu,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}
