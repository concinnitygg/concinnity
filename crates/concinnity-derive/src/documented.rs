//! `#[derive(Documented)]`: a type's rustdoc, and each enum variant's, as
//! static text the reference pages print.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields};

use crate::docs::doc_text;
use crate::schema::schema_path;

/// The `Documented` impl for `input`, behind the consuming crate's `schema`
/// feature, or the error naming why it cannot derive.
pub(crate) fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let doc = doc_text(&input.attrs);
    let variant_doc = match &input.data {
        Data::Enum(data) => {
            let arms = data.variants.iter().map(|variant| {
                let ident = &variant.ident;
                let doc = doc_text(&variant.attrs);
                let pattern = match variant.fields {
                    Fields::Unit => quote!(Self::#ident),
                    Fields::Unnamed(_) => quote!(Self::#ident(..)),
                    Fields::Named(_) => quote!(Self::#ident { .. }),
                };
                quote!(#pattern => #doc)
            });
            quote! {
                match self {
                    #(#arms,)*
                }
            }
        }
        Data::Struct(_) => quote!(Self::DOC),
        Data::Union(_) => {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "Documented derives only on structs and enums",
            ));
        }
    };

    let schema = schema_path();
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        #[cfg(feature = "schema")]
        impl #impl_generics #schema::Documented for #ident #ty_generics #where_clause {
            const DOC: &'static str = #doc;

            fn doc(&self) -> &'static str {
                #variant_doc
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expanded(input: &DeriveInput) -> String {
        expand(input).expect("the input derives").to_string()
    }

    #[test]
    fn an_enum_carries_its_doc_and_each_variants() {
        let out = expanded(&syn::parse_quote! {
            /// A tier.
            ///
            /// More.
            enum Tier {
                /// Low.
                Low,
                /// Held.
                Held(u8),
                Named { a: u8 },
            }
        });
        assert!(
            out.contains("const DOC : & 'static str = \"A tier.\\n\\nMore.\""),
            "{out}"
        );
        assert!(out.contains("Self :: Low => \"Low.\""), "{out}");
        assert!(out.contains("Self :: Held (..) => \"Held.\""), "{out}");
        assert!(out.contains("Self :: Named { .. } => \"\""), "{out}");
    }

    #[test]
    fn a_struct_answers_with_its_own_doc_behind_the_consumers_feature() {
        let out = expanded(&syn::parse_quote! {
            /// A ceiling.
            struct Ceiling { a: u8 }
        });
        assert!(out.contains("# [cfg (feature = \"schema\")]"), "{out}");
        assert!(out.contains("schema :: Documented for Ceiling"), "{out}");
        assert!(out.contains("Self :: DOC"), "{out}");
    }

    #[test]
    fn a_union_is_refused() {
        let input: DeriveInput = syn::parse_quote! { union U { a: u8 } };
        assert!(expand(&input).is_err());
    }
}
