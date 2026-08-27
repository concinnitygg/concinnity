//! concinnity-host: what the engine needs from the machine it runs on.
//!
//! Two charters, sharing nothing but that dependency:
//!
//!   - [`store`]: the project's state tree on disk, and the reads over it.
//!   - [`thread`]: the worker pool and the per-thread interner.
//!
//! They stay separate all the way down. Nothing in `store` reaches into
//! `thread`, and nothing in `thread` names a path.

pub mod store;
pub mod thread;
