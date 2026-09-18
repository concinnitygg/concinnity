//! Include: the authored schema and its resolution.
//!
//! The one build-only type that does not go through `expand_world`. An
//! `Include` stands for the entries of another world file, and which entries a
//! world holds decides everything after it (the `<Type>#<n>` labels, `$id`
//! uniqueness, every expansion pass), so it resolves in a pre-pass straight
//! after the world text is parsed, before labeling, validation and expansion.

pub(crate) mod resolve;
pub(crate) mod schema;

pub use resolve::{SourcedEntry, is_include, resolve_includes, with_includes};
