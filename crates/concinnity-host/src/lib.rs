//! concinnity-host: what the engine needs from the machine it runs on.
//!
//! Three charters, sharing nothing but that dependency:
//!
//!   - [`store`]: the project's state tree on disk, and the reads over it.
//!   - [`thread`]: the worker pool and the per-thread interner.
//!   - [`scratch`]: the temporary paths an external tool is handed.
//!
//! They stay separate all the way down. Nothing in `store` reaches into
//! `thread`, and nothing in `thread` names a path. `scratch` names paths no
//! project owns, which is why it is not part of `store`.

pub mod scratch;
pub mod store;
pub mod thread;
