// The build-time half of the doc model: owned counterparts of the
// `&'static`-based types in `concinnity_asset::doc_model`, which is what the
// emitter writes these out as. The two shapes have to stay in step, and cannot
// be one type: concinnity-asset's own build script depends on this crate, so a
// dependency the other way would be a cycle.

/// One type read out of a crate's schema sources.
pub struct DocType {
    /// The type's Rust identifier.
    pub name: String,
    /// The type's rustdoc, verbatim, one line per `///` line.
    pub doc: String,
    /// What the type contributes to the schema.
    pub shape: DocShape,
}

/// A documented type's contents.
pub enum DocShape {
    /// A struct's serialized fields, in declaration order.
    Fields(Vec<DocField>),
    /// A string-valued enum's serialized values, in declaration order.
    Values(Vec<DocValue>),
}

/// One serialized field of a struct.
pub struct DocField {
    /// The key the field serializes under.
    pub key: String,
    /// The field's rustdoc, collapsed to a single line.
    pub doc: String,
    /// The field's type.
    pub ty: DocFieldType,
    /// True when the field is declared `Option<T>`.
    pub optional: bool,
    /// The literal `impl Default` assigns to the field, when there is one.
    pub default: Option<String>,
}

/// One serialized value of a string-valued enum.
pub struct DocValue {
    /// The string the variant serializes to.
    pub value: String,
    /// The variant's rustdoc, collapsed to a single line.
    pub doc: String,
}

/// A field's JSON-shaped type, classified as far as one crate's sources allow.
/// A type named in the sources stays a [`DocFieldType::Name`]: whether it is an
/// enum, another asset, or a nested object is only decidable against the whole
/// schema, which spans more than one crate.
#[derive(Debug, PartialEq, Eq)]
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
        elem: Box<DocFieldType>,
        /// The fixed length of a Rust array, unset for a `Vec`.
        len: Option<usize>,
    },
    /// A type named in the sources, left for the consumer to resolve.
    Name(String),
}
