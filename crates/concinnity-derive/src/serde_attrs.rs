//! The slice of serde's attribute grammar that decides a field's authored key.

use syn::{Attribute, LitStr, Token};

/// How serde reads one field from authored input.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct FieldAttrs {
    /// The authored key, when `rename` replaces the field name.
    pub(crate) rename: Option<String>,
    /// The field is never read from input.
    pub(crate) skip: bool,
    /// The field's own fields sit at the parent's level.
    pub(crate) flatten: bool,
}

/// Parse every `#[serde(...)]` attribute on a field.
pub(crate) fn field_attrs(attrs: &[Attribute]) -> syn::Result<FieldAttrs> {
    let mut out = FieldAttrs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") || meta.path.is_ident("skip_deserializing") {
                out.skip = true;
            } else if meta.path.is_ident("flatten") {
                out.flatten = true;
            } else if meta.path.is_ident("rename") {
                if meta.input.peek(Token![=]) {
                    out.rename = Some(meta.value()?.parse::<LitStr>()?.value());
                } else {
                    meta.parse_nested_meta(|inner| {
                        let name = inner.value()?.parse::<LitStr>()?.value();
                        if inner.path.is_ident("deserialize") {
                            out.rename = Some(name);
                        }
                        Ok(())
                    })?;
                }
            } else {
                skip_meta_value(&meta)?;
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// Refuse a container attribute that would change every authored key.
pub(crate) fn check_container_attrs(attrs: &[Attribute]) -> syn::Result<()> {
    for attr in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("rename_all") {
                return Err(meta.error("AssetFields does not support `rename_all`"));
            }
            skip_meta_value(&meta)
        })?;
    }
    Ok(())
}

// Consume an attribute this derive does not read: `key`, `key = value`, or
// `key(...)`.
fn skip_meta_value(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<()> {
    if meta.input.peek(Token![=]) {
        meta.value()?.parse::<syn::Expr>()?;
    } else if meta.input.peek(syn::token::Paren) {
        let content;
        syn::parenthesized!(content in meta.input);
        content.parse::<proc_macro2::TokenStream>()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(tokens: proc_macro2::TokenStream) -> syn::Result<FieldAttrs> {
        let field: syn::Field = syn::parse::Parser::parse2(syn::Field::parse_named, tokens)
            .expect("the test field parses");
        field_attrs(&field.attrs)
    }

    #[test]
    fn a_plain_field_keeps_its_name() {
        let attrs = field(quote::quote! { #[serde(default)] x: f32 }).unwrap();
        assert_eq!(attrs, FieldAttrs::default());
    }

    #[test]
    fn rename_replaces_the_key_in_either_spelling() {
        let attrs = field(quote::quote! { #[serde(default, rename = "do")] body: u8 }).unwrap();
        assert_eq!(attrs.rename.as_deref(), Some("do"));
        let attrs = field(quote::quote! {
            #[serde(rename(serialize = "out", deserialize = "in"))] x: u8
        })
        .unwrap();
        assert_eq!(attrs.rename.as_deref(), Some("in"));
    }

    #[test]
    fn skip_and_flatten_are_recognized_among_other_attributes() {
        let attrs = field(quote::quote! { #[serde(skip)] id: u32 }).unwrap();
        assert!(attrs.skip);
        let attrs = field(quote::quote! {
            #[serde(default = "seven", skip_deserializing)] id: u32
        })
        .unwrap();
        assert!(attrs.skip);
        let attrs = field(quote::quote! {
            #[serde(flatten, deserialize_with = "de", bound(deserialize = "T: X"))] pose: P
        })
        .unwrap();
        assert!(attrs.flatten && !attrs.skip);
    }

    #[test]
    fn a_non_serde_attribute_is_ignored() {
        let attrs = field(quote::quote! { #[doc = "x"] #[expect(dead_code)] x: u8 }).unwrap();
        assert_eq!(attrs, FieldAttrs::default());
    }

    #[test]
    fn rename_all_on_the_container_is_refused() {
        let input: syn::DeriveInput =
            syn::parse_quote! { #[serde(default, rename_all = "camelCase")] struct S {} };
        let err = check_container_attrs(&input.attrs).unwrap_err();
        assert!(err.to_string().contains("rename_all"), "{err}");
        let input: syn::DeriveInput = syn::parse_quote! { #[serde(default)] struct S {} };
        assert!(check_container_attrs(&input.attrs).is_ok());
    }
}
