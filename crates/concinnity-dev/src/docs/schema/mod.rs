//! Reads the authored schema out of the engine's own sources.
//!
//! The asset reference is prose: rustdoc bodies, serde keys, `Default`
//! literals. None of that survives compilation, so it has to come from the
//! source text, which is why `cn docs` runs from a checkout of the engine
//! rather than from what a compiled dependency can hand it.
//!
//! What a name in a field's type refers to is deliberately left undecided here:
//! an ident may be an enum in one source tree and a struct in another, so
//! [`DocFieldType::Name`] carries it through to [`super::reference`], which
//! sees the whole schema at once.

mod attrs;
mod defaults;
mod model;
mod parse;

pub(crate) use model::{DocField, DocFieldType, DocShape, DocType, DocValue};

use std::io;
use std::path::PathBuf;

/// Every named-field struct and string-valued enum declared at the top level of
/// a `.rs` file under `roots`, sorted by name. Paths in `exclude` are skipped.
///
/// Only top-level items participate: an item inside a `mod` block (a `#[cfg]`d
/// test module, say) is not part of the schema.
pub(crate) fn extract(roots: &[PathBuf], exclude: &[PathBuf]) -> io::Result<Vec<DocType>> {
    parse::types(roots, exclude)
}
