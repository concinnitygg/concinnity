//! Character bodies at build time: the bundled humanoid schema, conformance
//! of a `CharacterModel`'s sources to its schema, the morph targets the
//! schema synthesizes from a source, and baking a `CharacterShape` into its
//! target.

pub(crate) mod bake;
/// The bundled `builtin:humanoid` schema and schema lookup by name.
pub mod builtin_schema;
pub(crate) mod frame;
pub(crate) mod import;
pub(crate) mod synth;
pub(crate) mod synthesize;
pub(crate) mod validate;
