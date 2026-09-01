// src/editor/worlds/mod.rs
//
// The Worlds panel: the project's worlds, most recently edited first, a `+`
// that starts an untitled one, and a per-row triple-dot menu. It has two
// presentations of the same list (`Mode`): the start screen a session with no
// world named opens on, a sidebar docked down the window's left edge over the
// world it previews, and the in-session switcher, where a row click opens that
// world behind the unsaved-changes guard.
//
// Layout half only: `geometry.rs` owns the rects and the hit test, `draw.rs`
// the per-frame layout, and `hook/worlds_edit.rs` / `hook/worlds_start.rs` the
// actions. `cinematic.rs` is the start screen's attract camera over the world
// the sidebar previews, and `loading.rs` the cover that stands over it while
// that world is compiled.

pub(crate) mod cinematic;
mod draw;
mod geometry;
pub(crate) mod loading;

pub(crate) use draw::apply;
pub(crate) use geometry::{Layout, Mode, hit_test};

use super::registry::{self, PanelKey};
use super::widget;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Worlds);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 1);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 3);
// The `+` that starts an untitled world.
pub(crate) const NEW_BG: AssetId = AssetId(BASE + 4);
pub(crate) const NEW_LABEL: AssetId = AssetId(BASE + 5);
pub(crate) const STATUS_LABEL: AssetId = AssetId(BASE + 6);
pub(crate) const LIST_HEADER: AssetId = AssetId(BASE + 7);
pub(crate) const LIST_TRACK: AssetId = AssetId(BASE + 8);
pub(crate) const LIST_THUMB: AssetId = AssetId(BASE + 9);
// The triple-dot button. One set of elements, positioned on whichever row is
// offering it this frame, as in the Assets panel.
pub(crate) const DOT_BG: AssetId = AssetId(BASE + 0xA);
pub(crate) const DOT1: AssetId = AssetId(BASE + 0xB);
pub(crate) const DOT2: AssetId = AssetId(BASE + 0xC);
pub(crate) const DOT3: AssetId = AssetId(BASE + 0xD);
// The floating row menu the triple-dot opens.
pub(crate) const MENU_BG: AssetId = AssetId(BASE + 0xE);
pub(crate) const MENU_OPEN_BG: AssetId = AssetId(BASE + 0xF);
pub(crate) const MENU_OPEN_LABEL: AssetId = AssetId(BASE + 0x10);
pub(crate) const MENU_DELETE_BG: AssetId = AssetId(BASE + 0x11);
pub(crate) const MENU_DELETE_LABEL: AssetId = AssetId(BASE + 0x12);

pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x20 + i as u32)
}
pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}

// Row slots the panel has elements for. The docked sidebar fills a tall window
// with them; the switcher shows fewer. A longer listing scrolls.
pub(crate) const POOL: usize = 24;

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

// Hide every panel element.
pub(crate) fn hide_all(world: &mut crate::ecs::World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &[]);
}

// Every panel sprite id, in draw (insertion) order: chrome, then the rows, then
// the overlays that float over them (scrollbar, triple-dot, row menu).
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, NEW_BG];
    ids.extend((0..POOL).map(row_bg));
    ids.extend([
        LIST_TRACK,
        LIST_THUMB,
        DOT_BG,
        DOT1,
        DOT2,
        DOT3,
        MENU_BG,
        MENU_OPEN_BG,
        MENU_DELETE_BG,
    ]);
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        NEW_LABEL,
        STATUS_LABEL,
        LIST_HEADER,
    ];
    ids.extend((0..POOL).map(row_label));
    ids.extend([MENU_OPEN_LABEL, MENU_DELETE_LABEL]);
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_lists_cover_every_slot_without_repeats() {
        let mut all: Vec<AssetId> = all_sprite_ids()
            .into_iter()
            .chain(all_label_ids())
            .collect();
        let n = all.len();
        all.sort_by_key(|id| id.0);
        all.dedup();
        assert_eq!(all.len(), n, "no duplicate reserved ids");
    }
}
