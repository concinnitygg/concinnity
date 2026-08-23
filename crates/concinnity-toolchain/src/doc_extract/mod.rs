//! Reads the authored schema out of a crate's own sources and writes it back as
//! a Rust constant.
//!
//! The asset reference is prose: rustdoc bodies, serde keys, `Default`
//! literals. None of that survives compilation, so it has to come from the
//! source text, and a published `.crate` contains only its own files. Each
//! crate that declares authorable types therefore extracts its own, emitting a
//! table of `concinnity_asset::doc_model` types the reference is assembled from
//! afterwards, whether the build runs from the workspace or from a registry
//! checkout.
//!
//! What a name in a field's type refers to is deliberately left undecided here:
//! an ident may be an enum in one crate and a struct in another, so
//! [`DocFieldType::Name`] carries it through to the consumer that can see the
//! whole schema.

mod attrs;
mod defaults;
mod emit;
mod model;
mod parse;

pub use model::{DocField, DocFieldType, DocShape, DocType, DocValue};

use std::io;
use std::path::PathBuf;

/// How a generated table names itself and the model it is written in terms of.
pub struct TableSpec<'a> {
    /// Name of the emitted constant.
    pub const_name: &'a str,
    /// Rustdoc for the emitted constant, which the workspace's `missing_docs`
    /// lint requires of it.
    pub doc: &'a str,
    /// Path to `concinnity_asset::doc_model` as the module doing the `include!`
    /// can name it: `doc_model` from inside concinnity-asset itself,
    /// `concinnity_asset::doc_model` from a crate that depends on it.
    pub model_path: &'a str,
}

/// Every named-field struct and string-valued enum declared at the top level of
/// a `.rs` file under `roots`, sorted by name. Paths in `exclude` are skipped,
/// which is how a crate keeps the module defining the model out of the table
/// written in terms of it.
///
/// Only top-level items participate: an item inside a `mod` block (a `#[cfg]`d
/// test module, say) is not part of the schema.
pub fn extract(roots: &[PathBuf], exclude: &[PathBuf]) -> io::Result<Vec<DocType>> {
    parse::types(roots, exclude)
}

/// Render `types` as a Rust source file defining one constant, for a build
/// script to write into `OUT_DIR` and the crate to `include!`.
pub fn emit_table(types: &[DocType], spec: &TableSpec) -> String {
    emit::table(types, spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The emitted source has to parse as Rust, or the failure surfaces as a
    // syntax error inside a generated file the author never wrote.
    #[test]
    fn an_extracted_tree_emits_parseable_rust() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("prop.rs"),
            r#"
            /// A prop placed in the world.
            pub struct Prop {
                /// Where it sits.
                pub position: [f32; 3],
                pub tags: Vec<String>,
                pub collider: Option<PropCollider>,
            }
            impl Default for Prop {
                fn default() -> Self {
                    Self { position: [0.0, 0.0, 0.0], tags: Vec::new(), collider: None }
                }
            }
            /// How a prop collides.
            #[serde(rename_all = "snake_case")]
            pub enum ColliderKind {
                /// A box.
                BoxShape,
            }
            "#,
        )
        .expect("write");

        let types = extract(&[dir.path().to_path_buf()], &[]).expect("extract");
        assert_eq!(types.len(), 2);
        let src = emit_table(
            &types,
            &TableSpec {
                const_name: "ASSET_DOCS",
                doc: "Every documented type.",
                model_path: "crate::doc_model",
            },
        );
        syn::parse_file(&src).expect("emitted source parses");
        assert!(src.contains(r#"name: "ColliderKind""#));
        assert!(src.contains(r#"value: "box_shape""#));
        assert!(src.contains(r#"default: Some("[0.0, 0.0, 0.0]")"#));
    }
}
