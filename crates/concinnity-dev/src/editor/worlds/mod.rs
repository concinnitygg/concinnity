//! The Worlds panel: the project's worlds, most recently edited first, a `+`
//! that starts an untitled one, and a per-row context menu. It has two
//! presentations of the same list (`Mode`): the start screen a session with no
//! world named opens on, a sidebar docked down the window's left edge over the
//! world it previews, and the in-session switcher, where a row click opens that
//! world behind the unsaved-changes guard.
//!
//! `geometry.rs` holds what the two presentations differ by, `panel.rs` the
//! widgets, and `hook/edit/worlds.rs` / `hook/worlds_start.rs` the actions.
//! `cinematic.rs` is the start screen's attract camera over the world the
//! sidebar previews, and `loading.rs` the cover that stands over it while that
//! world is compiled.

pub(crate) mod cinematic;
pub(crate) mod files;
mod geometry;
pub(crate) mod loading;
pub(crate) mod panel;

pub(crate) use geometry::{Layout, Mode};

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

// What the user asked of the Worlds panel.
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
}
