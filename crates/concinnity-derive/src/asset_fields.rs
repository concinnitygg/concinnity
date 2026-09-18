//! `#[derive(AssetFields)]`: one probe call per authored field.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields};

use crate::serde_attrs::{check_container_attrs, field_attrs};

/// The impl for `input`, or the error naming why it cannot derive.
pub(crate) fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "AssetFields derives only on structs with named fields",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "AssetFields derives only on structs with named fields",
        ));
    };
    check_container_attrs(&input.attrs)?;

    let mut probes = Vec::new();
    for field in &fields.named {
        let attrs = field_attrs(&field.attrs)?;
        if attrs.skip {
            continue;
        }
        let ty = &field.ty;
        let key = if attrs.flatten {
            quote!(::core::option::Option::None)
        } else {
            let name = match attrs.rename {
                Some(rename) => rename,
                None => field
                    .ident
                    .as_ref()
                    .expect("a named field has an ident")
                    .to_string(),
            };
            let name = name.strip_prefix("r#").unwrap_or(&name).to_string();
            quote!(::core::option::Option::Some(#name))
        };
        probes.push(quote! {
            (&&&&::concinnity_core::ecs::asset_fields::probe::Probe::<#ty>::NEW)
                .collect(prefix, #key, out);
        });
    }

    // A schema with no authored fields names neither the probe traits nor its
    // arguments.
    let body = if probes.is_empty() {
        quote! { let _ = (prefix, out); }
    } else {
        quote! {
            #[allow(unused_imports)]
            use ::concinnity_core::ecs::asset_fields::probe::{
                ByNested as _, ByNothing as _, ByReference as _, ByVocabulary as _,
            };
            #(#probes)*
        }
    };

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::concinnity_core::ecs::asset_fields::AssetFields
            for #ident #ty_generics #where_clause
        {
            fn collect_fields(
                prefix: &str,
                out: &mut ::concinnity_core::ecs::asset_fields::FieldTable,
            ) {
                #body
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
    fn every_authored_field_gets_one_probe_under_its_key() {
        let out = expanded(syn::parse_quote! {
            struct S {
                #[serde(skip)]
                asset_id: AssetId,
                #[serde(rename = "do")]
                body: Vec<Node>,
                r#type: Kind,
                #[serde(flatten)]
                pose: Pose,
            }
        });
        assert!(!out.contains("AssetId"), "{out}");
        assert!(out.contains("Probe :: < Vec < Node > >"), "{out}");
        assert!(out.contains("Some (\"do\")"), "{out}");
        assert!(out.contains("Some (\"type\")"), "{out}");
        assert!(
            out.contains(
                "Probe :: < Pose > :: NEW) . collect (prefix , :: core :: option :: Option :: None"
            ),
            "{out}"
        );
        assert_eq!(out.matches(". collect (").count(), 3, "{out}");
    }

    #[test]
    fn only_core_and_the_engine_crate_are_named() {
        let out = expanded(syn::parse_quote! { struct S { x: f32 } });
        assert!(!out.contains("std ::"), "{out}");
        assert!(!out.contains("alloc ::"), "{out}");
        assert!(
            out.contains(":: concinnity_core :: ecs :: asset_fields :: AssetFields"),
            "{out}"
        );
    }

    #[test]
    fn a_fieldless_schema_touches_no_probe() {
        let out = expanded(syn::parse_quote! { struct NoArgs {} });
        assert!(!out.contains("Probe"), "{out}");
    }

    #[test]
    fn generics_carry_through_to_the_impl() {
        let out = expanded(syn::parse_quote! { struct S<T: Clone> { x: T } });
        assert!(out.contains("impl < T : Clone >"), "{out}");
        assert!(out.contains("for S < T >"), "{out}");
    }

    #[test]
    fn enums_tuple_structs_and_rename_all_are_refused() {
        for input in [
            syn::parse_quote! { enum E { A } },
            syn::parse_quote! { struct T(u8); },
            syn::parse_quote! { #[serde(rename_all = "lowercase")] struct R { x: u8 } },
        ] {
            assert!(expand(&input).is_err());
        }
    }
}
