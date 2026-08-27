//! The thread-scoped services the rest of the engine builds on, kept out of the
//! vocabulary and compute layer below them (`concinnity-core`) so that layer
//! needs no operating system.
//!
//! Two of them, sharing nothing but a need for real threads:
//!
//!   - [`jobs`]: the process-wide worker pool a single system fans its own
//!     data-parallel work across (pose sampling, particle update, the
//!     environment-map convolutions).
//!   - [`asset_id`]: the build-time name -> dense id interner, whose table is
//!     per-thread so two concurrent builds cannot see each other's ids.
//!
//! It knows where nothing lives: it names no path and opens no file. The
//! identity types the interner hands back are `concinnity-core`'s, and the
//! resolver seam it installs is `concinnity-asset`'s.

pub mod asset_id;
pub mod jobs;
mod name_interner;
