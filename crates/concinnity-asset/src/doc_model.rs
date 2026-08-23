//! The shape of an authored schema, as extracted from source.
//!
//! A crate whose sources define authorable types emits its own table of these
//! at build time (`concinnity_asset::ASSET_DOCS`,
//! `concinnity_core::RUNTIME_ASSET_DOCS`), which is what lets the asset
//! reference be assembled from published crates rather than from a directory of
//! sibling sources. Every field is a `&'static str`, so a consumer that never
//! reads a table pays nothing for it.
//!
//! This is the vocabulary only. The extractor that fills it in lives in
//! `concinnity_toolchain::doc_extract`, whose owned counterparts of these types
//! must stay in step with them.

/// One type read out of a crate's schema sources.
pub struct DocType {
    /// The type's Rust identifier.
    pub name: &'static str,
    /// The type's rustdoc, verbatim, one line per `///` line.
    pub doc: &'static str,
    /// What the type contributes to the schema.
    pub shape: DocShape,
}

/// A documented type's contents.
pub enum DocShape {
    /// A struct's serialized fields, in declaration order.
    Fields(&'static [DocField]),
    /// A string-valued enum's serialized values, in declaration order.
    Values(&'static [DocValue]),
}

/// One serialized field of a struct.
pub struct DocField {
    /// The key the field serializes under.
    pub key: &'static str,
    /// The field's rustdoc, collapsed to a single line.
    pub doc: &'static str,
    /// The field's type.
    pub ty: DocFieldType,
    /// True when the field is declared `Option<T>`.
    pub optional: bool,
    /// The literal `impl Default` assigns to the field, when there is one.
    pub default: Option<&'static str>,
}

/// One serialized value of a string-valued enum.
pub struct DocValue {
    /// The string the variant serializes to.
    pub value: &'static str,
    /// The variant's rustdoc, collapsed to a single line.
    pub doc: &'static str,
}

/// A field's JSON-shaped type, classified as far as one crate's sources allow.
/// A type named in the sources stays a [`DocFieldType::Name`]: whether it is an
/// enum, another asset, or a nested object is only decidable against the whole
/// schema, which spans more than one crate.
pub enum DocFieldType {
    /// A `bool`.
    Bool,
    /// An `f32` or `f64`.
    Float,
    /// Any Rust integer.
    Integer,
    /// A string, including the by-name asset references.
    Str,
    /// An open-ended JSON object: a map, or a type with no documented shape.
    Object,
    /// A `[T; n]` array or a `Vec<T>`.
    Array {
        /// The element type.
        elem: &'static DocFieldType,
        /// The fixed length of a Rust array, unset for a `Vec`.
        len: Option<usize>,
    },
    /// A type named in the sources, left for the consumer to resolve.
    Name(&'static str),
}
