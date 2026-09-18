//! `#[derive(Vocabulary)]`: a unit enum's authored names, its `Vocabulary`
//! impl, and its schema.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, Data, DeriveInput, Fields, LitStr};

use crate::docs::doc_text;
use crate::schema::schema_impls;
use crate::serde_attrs::{apply_case, container_attrs, variant_rename};

/// The impls for `input`, or the error naming why it cannot derive.
pub(crate) fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "Vocabulary derives only on enums of unit variants",
        ));
    };
    let rename_all = container_attrs(&input.attrs)?.rename_all;

    let mut variants = Vec::new();
    let mut names = Vec::new();
    let mut docs = Vec::new();
    for variant in &data.variants {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(
                variant,
                "Vocabulary derives only on enums of unit variants",
            ));
        }
        let Some(name) = vocab_name(&variant.attrs)? else {
            return Err(syn::Error::new_spanned(
                variant,
                "every variant names its authored word with `#[vocab(\"...\")]`",
            ));
        };
        if has_serde_naming(input, variant) {
            let serde_name = match variant_rename(&variant.attrs)? {
                Some(name) => name,
                None => apply_case(&variant.ident.to_string(), rename_all.as_deref())
                    .map_err(|e| syn::Error::new_spanned(&input.ident, e))?,
            };
            if name.value() != serde_name {
                return Err(syn::Error::new_spanned(
                    name,
                    format!("serde writes this variant as \"{serde_name}\""),
                ));
            }
        }
        let name = name.value();
        variants.push(&variant.ident);
        names.push(name);
        docs.push(doc_text(&variant.attrs));
    }

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let schema = crate::schema::schema_path();
    let type_name = ident.to_string();
    let type_doc = doc_text(&input.attrs);
    let body = quote! {
        #schema::TypeSchema {
            name: #type_name,
            doc: #type_doc,
            body: #schema::Body::Values(&[
                #( #schema::ValueSchema { name: #names, doc: #docs } ),*
            ]),
            default: ::core::option::Option::None,
        }
    };
    let schema_impls = schema_impls(input, &body, quote!(Enum));

    Ok(quote! {
        impl #impl_generics #ident #ty_generics #where_clause {
            /// Every variant, in the order an editor picker steps through them.
            pub const ALL: &'static [Self] = &[#(Self::#variants),*];

            /// Every variant's authored name, in [`Self::ALL`] order. The
            /// editor's picker list.
            pub const NAMES: &'static [&'static str] = &[#(#names),*];

            /// This variant's canonical authored name: what serde writes for it.
            pub const fn as_str(self) -> &'static str {
                match self {
                    #(Self::#variants => #names),*
                }
            }
        }

        impl #impl_generics ::concinnity_core::components::Vocabulary
            for #ident #ty_generics #where_clause
        {
            const VARIANTS: &'static [&'static str] = Self::NAMES;
        }

        #schema_impls
    })
}

// The `#[vocab("Name")]` on a variant.
fn vocab_name(attrs: &[Attribute]) -> syn::Result<Option<LitStr>> {
    let mut out = None;
    for attr in attrs.iter().filter(|a| a.path().is_ident("vocab")) {
        if out.is_some() {
            return Err(syn::Error::new_spanned(
                attr,
                "a variant takes one `vocab` name",
            ));
        }
        out = Some(attr.parse_args::<LitStr>()?);
    }
    Ok(out)
}

// Whether serde's own attributes spell this variant's name: then a `vocab`
// name must agree with them, since serde's derive writes the variant.
fn has_serde_naming(input: &DeriveInput, variant: &syn::Variant) -> bool {
    let is_serde = |attrs: &[Attribute]| attrs.iter().any(|a| a.path().is_ident("serde"));
    is_serde(&input.attrs) || is_serde(&variant.attrs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expanded(input: DeriveInput) -> String {
        expand(&input).expect("the input derives").to_string()
    }

    #[test]
    fn each_variant_carries_its_name_and_doc() {
        let out = expanded(syn::parse_quote! {
            #[serde(rename_all = "snake_case")]
            enum Quality {
                /// Best.
                #[vocab("quality")]
                Quality,
                #[serde(rename = "fast")]
                #[vocab("fast")]
                UltraPerformance,
                #[vocab("balanced_mode")]
                BalancedMode,
            }
        });
        assert!(
            out.contains(
                "NAMES : & 'static [& 'static str] = & [\"quality\" , \"fast\" , \"balanced_mode\"]"
            ),
            "{out}"
        );
        assert!(
            out.contains("Self :: UltraPerformance => \"fast\""),
            "{out}"
        );
        assert!(
            out.contains("ValueSchema { name : \"quality\" , doc : \"Best.\" }"),
            "{out}"
        );

        // With serde's impls built by hand, the name is whatever `vocab` says.
        let out = expanded(syn::parse_quote! {
            enum Transition { #[vocab("FadeBlack")] FadeBlack, #[vocab("cut")] Cut }
        });
        assert!(out.contains("[\"FadeBlack\" , \"cut\"]"), "{out}");
    }

    #[test]
    fn the_schema_sits_behind_the_consumers_feature() {
        let out = expanded(syn::parse_quote! { enum E { #[vocab("a")] A } });
        assert!(out.contains("# [cfg (feature = \"schema\")]"), "{out}");
        assert!(out.contains("FieldType :: Enum"), "{out}");
        assert!(
            out.contains(":: concinnity_core :: components :: Vocabulary for E"),
            "{out}"
        );
    }

    #[test]
    fn a_vocab_name_serde_would_not_write_is_refused() {
        let input: DeriveInput = syn::parse_quote! {
            #[serde(rename_all = "lowercase")]
            enum E { #[vocab("Loud")] Loud }
        };
        let err = expand(&input).expect_err("serde writes \"loud\"");
        assert!(err.to_string().contains("\"loud\""), "{err}");
    }

    #[test]
    fn data_variants_structs_and_missing_or_doubled_names_are_refused() {
        for input in [
            syn::parse_quote! { enum E { #[vocab("a")] A(u8) } },
            syn::parse_quote! { struct S { a: u8 } },
            syn::parse_quote! { enum E { #[vocab("a")] #[vocab("b")] A } },
            syn::parse_quote! { enum E { #[vocab("a")] A, B } },
        ] {
            assert!(expand(&input).is_err());
        }
    }
}
