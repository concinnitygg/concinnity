//! The project state tree: where the engine's state is anchored, and the names
//! of the directories hanging off it.
//!
//! Everything the engine writes for a project lives under one tree: the
//! compiled blobs (`data/`), the regenerable cache (`cache/`: `0` for the
//! running application, `1` for a build, the baked asset thumbnails included),
//! fetched source assets (`assets/`), named worlds (`worlds/`), the runtime save
//! files (`saves/`), and the mutable settings file (`settings`).
//!
//! [`StateTree`] is a value, not a process-wide install. Whatever runs the
//! process builds one and passes it down: the dev CLI hides it inside the
//! project, a shipped application puts it beside its executable, an embedder
//! points it wherever its own layout implies. Library code is handed the tree,
//! or the single path it needs out of it, and resolves nothing for itself --
//! which is what lets one process drive two trees, and a sandboxed host drive
//! one it chose.
//!
//! A tree has three roots because real installs pull them apart: the content a
//! build produces, the state a run writes, and the caches either regenerates.
//! [`StateTree`] documents when each splits away.
//!
//! Resolution touches no files: these functions compute paths. Reading the tree
//! is `super::source` (finding a source asset) and `super::blob` (the compiled
//! blob).

mod tree;

pub use tree::StateTree;
