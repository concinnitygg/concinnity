//! `#[derive(AssetDefault)]`: a `Default` impl built from each field's
//! `#[asset(default = <expr>)]`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Expr, ExprLit, Fields, Lit};

use crate::asset_attrs::asset_attrs;

/// The `Default` impl for `input`, or the error naming why it cannot derive.
pub(crate) fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "AssetDefault derives only on structs with named fields",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "AssetDefault derives only on structs with named fields",
        ));
    };

    let mut inits = Vec::new();
    for field in &fields.named {
        let ident = &field.ident;
        let value = match asset_attrs(&field.attrs)?.default {
            // A string literal names the text of an owned string field.
            Some(Expr::Lit(ExprLit {
                lit: Lit::Str(text),
                ..
            })) => quote!(::core::convert::From::from(#text)),
            Some(expr) => quote!(#expr),
            None => quote!(::core::default::Default::default()),
        };
        inits.push(quote!(#ident: #value));
    }

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::core::default::Default for #ident #ty_generics #where_clause {
            fn default() -> Self {
                Self {
                    #(#inits),*
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expanded(input: DeriveInput) -> String {
        expand(&input).expect("the input derives").to_string()
    }

    #[test]
    fn a_stated_default_is_used_and_the_rest_take_their_types() {
        let out = expanded(syn::parse_quote! {
            struct Light {
                #[asset(default = [0.0, 4.0, 0.0])]
                position: [f32; 3],
                #[serde(skip)]
                cache: u32,
                #[asset(default = DEFAULT_RANGE)]
                range: f32,
            }
        });
        assert!(out.contains("position : [0.0 , 4.0 , 0.0]"), "{out}");
        assert!(
            out.contains("cache : :: core :: default :: Default :: default ()"),
            "{out}"
        );
        assert!(out.contains("range : DEFAULT_RANGE"), "{out}");
    }

    #[test]
    fn a_string_literal_converts_into_the_field() {
        let out = expanded(syn::parse_quote! {
            struct App {
                #[asset(default = "Concinnity")]
                name: String,
            }
        });
        assert!(
            out.contains("name : :: core :: convert :: From :: from (\"Concinnity\")"),
            "{out}"
        );
    }

    #[test]
    fn only_core_is_named() {
        let out = expanded(syn::parse_quote! { struct S { x: f32 } });
        assert!(!out.contains("std ::"), "{out}");
        assert!(!out.contains("alloc ::"), "{out}");
    }

    #[test]
    fn generics_carry_through_to_the_impl() {
        let out = expanded(syn::parse_quote! { struct S<T: Default> { x: T } });
        assert!(out.contains("impl < T : Default >"), "{out}");
        assert!(out.contains("for S < T >"), "{out}");
    }

    #[test]
    fn enums_tuple_structs_and_unknown_keys_are_refused() {
        for input in [
            syn::parse_quote! { enum E { A } },
            syn::parse_quote! { struct T(u8); },
            syn::parse_quote! { struct U { #[asset(defualt = 1)] x: u8 } },
        ] {
            assert!(expand(&input).is_err());
        }
    }
}
