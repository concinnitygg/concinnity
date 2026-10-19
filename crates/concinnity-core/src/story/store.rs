//! What a story saves and where a host keeps it.

use alloc::collections::BTreeMap;
use alloc::string::String;

/// A story's persisted position and variables, auto-written page by page
/// (resumed by the story's `continue`) and written to numbered slots by the
/// slot overlay. Position is kept by node slug, which stays stable across story
/// edits where an index would not.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StorySave {
    /// The slug of the node play is at.
    pub slug: String,
    /// The page within that node.
    pub page: u32,
    /// The story's variables (an unset one reads as 0).
    #[serde(default)]
    pub vars: BTreeMap<String, i32>,
}

/// Where a save lives: the auto-save, or a manual slot (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SaveSlot {
    /// The position written page by page, resumed by `continue`.
    Auto,
    /// A slot the player saved to from the slot overlay.
    Manual(usize),
}

/// A host's store for story saves. One per game: a world plays a single story,
/// so saves carry no per-story key.
pub trait StoryStore: core::fmt::Debug + Send {
    /// The save in `slot`, or `None` when nothing was stored or it could not
    /// be read. Sees every earlier write and removal.
    fn read(&self, slot: SaveSlot) -> Option<StorySave>;
    /// Store `save` in `slot`, replacing what was there. Only the implementor
    /// knows what the write was to, so reporting a failure is its job.
    fn write(&self, slot: SaveSlot, save: &StorySave);
    /// Drop whatever `slot` holds.
    fn remove(&self, slot: SaveSlot);
    /// Whether `slot` holds a save. Sees every earlier write and removal.
    fn exists(&self, slot: SaveSlot) -> bool;
}
