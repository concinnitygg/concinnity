//! Test scaffolding shared by the workspace.
//!
//! A dev-dependency only. It depends on `concinnity-core` and nothing else in
//! the workspace, so any crate can take it without a cycle and without pulling
//! an engine tier into its graph.
//!
//! What it offers:
//!
//!   - [`TempTree`]: every path a test writes, under a directory that deletes
//!     itself. A test that builds its own path under the system temporary
//!     directory leaves that tree behind on every run.
//!   - [`fixtures`]: synthetic asset bytes, so a test needs no checked-in
//!     binary and reads nothing outside its own crate.
//!   - [`source`]: reading the workspace's own sources, for the guard tests
//!     that forbid a shape which at runtime would hang rather than fail.
//!   - [`exclusive`] / [`shared`]: the one reader/writer lock over this
//!     binary's process-global state. Readers stay parallel; writers run alone.
//!     [`GlobalState`] is that exclusive guard plus the cwd and window-policy
//!     moves, both put back on drop.
//!   - [`shared_cache_dir`] / [`shared_state_dir`]: the two roots a suite keeps
//!     outside any one test -- a content-addressed cache that exists to avoid
//!     recompiling, and the process-wide state that is emptied once per run.

mod access;
mod global;
mod shared_dirs;
mod temp;

pub mod fixtures;
pub mod source;

pub use access::{ExclusiveAccess, SharedAccess, exclusive, shared};
pub use global::GlobalState;
pub use shared_dirs::{shared_cache_dir, shared_state_dir};
pub use temp::{TempTree, utf8, write_into};

/// Forbid window creation for the rest of the process.
///
/// A world run on the windowed loop stands up a window and blocks on an event
/// loop the harness cannot end, so a test that reaches one hangs rather than
/// failing. Arming this turns that hang into a panic naming the backend.
///
/// [`GlobalState::without_windows`] does the same for the life of a guard, and
/// lifts it afterwards. Call this directly from a test binary that should never
/// open a window at all.
pub fn forbid_windows() {
    concinnity_core::window_policy::forbid_windows();
}
