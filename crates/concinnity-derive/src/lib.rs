//! Derive macros for the Concinnity engine's authored asset schemas.
//!
//! Internal to the engine: `concinnity-core` and `concinnity-cook` derive with
//! it, and nothing re-exports it past them.
//!
//! Both derives also emit the type's static schema (`concinnity_core::ecs::schema`)
//! behind `#[cfg(feature = "schema")]`, which resolves in the deriving crate:
//! a crate that derives declares a `schema` feature enabling core's.

mod asset_fields;
mod docs;
mod schema;
mod serde_attrs;
mod vocabulary;

use proc_macro::TokenStream;

/// Generate `concinnity_core::ecs::AssetFields` for a struct with named
/// fields: its reference fields and closed-vocabulary fields, found by field
/// type, recursing into fields whose type derives `AssetFields` as well.
///
/// Reads serde's field attributes for the authored key: `rename` replaces the
/// field name, `flatten` lifts a nested schema to this level, and `skip` /
/// `skip_deserializing` leave a field out. A container `rename_all` is refused,
/// since no schema uses one. The schema also records serde's `default`
/// attributes, which decide what an omitted key reads as.
#[proc_macro_derive(AssetFields, attributes(serde))]
pub fn derive_asset_fields(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    asset_fields::expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Generate a closed authored vocabulary for an enum of unit variants: `ALL`,
/// `NAMES`, `as_str`, and the `concinnity_core::components::Vocabulary` impl.
///
/// Every variant states its authored name with `#[vocab("Name")]`, beside its
/// doc. Where serde's attributes name the variants too (`rename`,
/// `rename_all`), the two must agree, so the vocabulary is what serde writes.
#[proc_macro_derive(Vocabulary, attributes(vocab, serde))]
pub fn derive_vocabulary(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    vocabulary::expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
