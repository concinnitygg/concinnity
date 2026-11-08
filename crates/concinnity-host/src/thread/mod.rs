//! The thread-scoped services the rest of the engine builds on, kept out of the
//! vocabulary and compute layer below them (`concinnity-engine`) so that layer
//! needs no operating system.
//!
//! Six of them, sharing nothing but a need for real threads:
//!
//!   - [`jobs`]: the process-wide worker pool a single system fans its own
//!     data-parallel work across (pose sampling, particle update, the
//!     environment-map convolutions).
//!   - [`role`]: how the operating system schedules a thread, set by whoever
//!     spawns it.
//!   - [`worker`]: named background threads that answer requests off the
//!     frame thread (streaming loads, decodes, ground patches).
//!   - [`asset_id`]: the build-time name -> dense id interner, whose table is
//!     per-thread so two concurrent builds cannot see each other's ids.
//!   - [`pending`]: a count of handed-off work a caller can wait to drain.
//!   - [`ThreadBound`]: a value only its own thread may use, held in state
//!     that moves between threads.
//!
//! It knows where nothing lives: it names no path and opens no file. The
//! identity types the interner hands back are `concinnity-engine`'s, and the
//! resolver seam it installs is `concinnity_engine::ecs::resolver`'s.

pub mod asset_id;
mod bound;
pub mod jobs;
mod name_interner;
pub mod pending;
pub mod role;
pub mod worker;

pub use bound::ThreadBound;
pub use role::{ThreadRole, set_current_thread_role};
