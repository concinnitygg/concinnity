//! The sparse override twin `#[asset(section)]` generates beside a struct:
//! the same fields, each an `Option`, where a set field replaces the section's
//! value and an unset one keeps it.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::DeriveInput;

use crate::asset_attrs::asset_attrs;
use crate::docs::doc_text;
use crate::schema::{AuthoredField, schema_path, type_expr};

/// The twin of `input`, its `AssetFields` impl and its schema.
pub(crate) fn section_twin(
    input: &DeriveInput,
    authored: &[AuthoredField<'_>],
) -> syn::Result<TokenStream> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "a section cannot be generic",
        ));
    }
    let section = &input.ident;
    let twin = format_ident!("{}Overrides", section);
    let vis = &input.vis;
    let fields_path = quote!(::concinnity_core::ecs::asset_fields);
    let schema = schema_path();
    let twin_doc = format!(
        "Sparse overrides of [`{section}`]: each field set replaces the section's value, \
         each field left unset keeps it."
    );

    let mut decls = Vec::new();
    let mut sparse = Vec::new();
    let mut dense = Vec::new();
    let mut applies = Vec::new();
    let mut froms = Vec::new();
    let mut unset = Vec::new();
    let mut clamps = Vec::new();
    let mut entries = Vec::new();
    for AuthoredField { field, attrs, key } in authored {
        if attrs.flatten || attrs.rename.is_some() {
            return Err(syn::Error::new_spanned(
                field,
                "a section field cannot be renamed or flattened",
            ));
        }
        let ident = &field.ident;
        let ty = &field.ty;
        let docs = field.attrs.iter().filter(|a| a.path().is_ident("doc"));
        decls.push(quote! {
            #(#docs)*
            #[serde(default)]
            pub #ident: ::core::option::Option<#ty>,
        });
        sparse.push(quote! {
            match &self.#ident {
                ::core::option::Option::Some(value) => state.serialize_field(#key, value)?,
                ::core::option::Option::None => state.skip_field(#key)?,
            }
        });
        dense.push(quote!(state.serialize_field(#key, &self.#ident)?;));
        applies.push(quote! {
            if let ::core::option::Option::Some(value) = &self.#ident {
                base.#ident = ::core::clone::Clone::clone(value);
            }
        });
        froms.push(quote!(#ident: ::core::option::Option::Some(section.#ident)));
        unset.push(quote!(self.#ident.is_none()));

        let asset = asset_attrs(&field.attrs)?;
        let meta = asset.meta_expr();
        if asset.is_ranged() {
            clamps.push(quote! {
                if let ::core::option::Option::Some(value) = &mut self.#ident {
                    if let ::core::option::Option::Some((from, to)) =
                        #fields_path::clamp_param(value, (#meta).range)
                    {
                        out.push(#fields_path::ClampedParam {
                            section: ::core::option::Option::None,
                            key: #key,
                            from,
                            to,
                        });
                    }
                }
            });
        }
        let doc = doc_text(&field.attrs);
        let inner = type_expr(ty);
        entries.push(quote! {
            #schema::FieldSchema {
                key: #key,
                doc: #doc,
                ty: || #schema::FieldType::Optional(|| #inner),
                default: #schema::FieldDefault::Value(|| {
                    #[allow(unused_imports)]
                    use #schema::probe::{DefaultByNothing as _, DefaultBySerialize as _};
                    let value: #ty = <#section as ::core::default::Default>::default().#ident;
                    (&&#schema::probe::Probe::<#ty>::NEW).boxed(value)
                }),
                param: #meta,
            }
        });
    }

    let clamp_item = if clamps.is_empty() {
        quote!()
    } else {
        quote! {
            fn clamp_params(&mut self, out: &mut #fields_path::ClampedParams) {
                #(#clamps)*
            }
        }
    };
    let name = section.to_string();
    let twin_name = twin.to_string();
    let count = authored.len();
    let doc = doc_text(&input.attrs);
    let unset = if unset.is_empty() {
        quote!(true)
    } else {
        quote!(#(#unset)&&*)
    };
    Ok(quote! {
        #[doc = #twin_doc]
        #[derive(
            Debug,
            Clone,
            Default,
            PartialEq,
            ::serde::Deserialize,
        )]
        #[serde(deny_unknown_fields)]
        #vis struct #twin {
            #(#decls)*
        }

        impl #twin {
            /// Replace each field of `base` this sets.
            pub fn apply_to(&self, base: &mut #section) {
                #(#applies)*
            }

            /// Whether no field is set.
            pub fn is_unset(&self) -> bool {
                #unset
            }
        }

        // A readable format (the authored JSON) leaves unset fields out; a
        // binary one (the cooked blob) writes every field in order.
        impl ::serde::Serialize for #twin {
            fn serialize<S: ::serde::Serializer>(
                &self,
                serializer: S,
            ) -> ::core::result::Result<S::Ok, S::Error> {
                use ::serde::ser::SerializeStruct as _;
                let readable = serializer.is_human_readable();
                let mut state = serializer.serialize_struct(#twin_name, #count)?;
                if readable {
                    #(#sparse)*
                } else {
                    #(#dense)*
                }
                state.end()
            }
        }

        impl ::core::convert::From<#section> for #twin {
            fn from(section: #section) -> Self {
                Self {
                    #(#froms),*
                }
            }
        }

        impl #fields_path::AssetFields for #twin {
            const PARAMS: &'static [(&'static str, #fields_path::ParamMeta)] =
                <#section as #fields_path::AssetFields>::PARAMS;

            fn collect_fields(prefix: &str, out: &mut #fields_path::FieldTable) {
                <#section as #fields_path::AssetFields>::collect_fields(prefix, out);
            }

            #clamp_item
        }

        #[cfg(feature = "schema")]
        const _: () = {
            impl #schema::Schema for #twin {
                const SCHEMA: &'static #schema::TypeSchema = &#schema::TypeSchema {
                    name: #name,
                    doc: #doc,
                    body: #schema::Body::Fields(&[#(#entries),*]),
                    default: ::core::option::Option::Some(|| {
                        #[allow(unused_imports)]
                        use #schema::probe::{DefaultByNothing as _, DefaultBySerialize as _};
                        let value: #twin = ::core::default::Default::default();
                        (&&#schema::probe::Probe::<#twin>::NEW).boxed(value)
                    }),
                };
            }
            impl #schema::Described for #twin {
                const TYPE: #schema::FieldType =
                    #schema::FieldType::Nested(<Self as #schema::Schema>::SCHEMA);
            }
        };
    })
}

#[cfg(test)]
mod tests {
    use crate::asset_fields::expand;

    fn expanded(input: syn::DeriveInput) -> String {
        expand(&input).expect("the input derives").to_string()
    }

    #[test]
    fn a_section_gets_an_optional_twin_with_the_same_keys() {
        let out = expanded(syn::parse_quote! {
            /// Shadows.
            #[asset(section)]
            #[serde(default)]
            pub struct Shadows {
                /// Resolution.
                #[asset(default = 4096, min = 1, apply = restart)]
                pub map_size: u32,
                #[asset(apply = live)]
                pub hybrid: bool,
            }
        });
        assert!(out.contains("pub struct ShadowsOverrides"), "{out}");
        assert!(
            out.contains("pub map_size : :: core :: option :: Option < u32 >"),
            "{out}"
        );
        assert!(out.contains("deny_unknown_fields"), "{out}");
        assert!(out.contains("is_human_readable"), "{out}");
        assert!(out.contains("skip_field (\"map_size\")"), "{out}");
        assert!(
            out.contains("fn apply_to (& self , base : & mut Shadows)"),
            "{out}"
        );
        assert!(out.contains("< Shadows as :: concinnity_core"), "{out}");
        // Only the ranged field clamps, and only when it is set.
        assert_eq!(out.matches("clamp_param (value").count(), 1, "{out}");
        // The twin documents itself under the section's name and defaults.
        assert!(out.contains("name : \"Shadows\""), "{out}");
        assert!(
            out.contains("< Shadows as :: core :: default :: Default > :: default () . map_size"),
            "{out}"
        );
    }

    #[test]
    fn a_renamed_section_field_is_refused() {
        let input = syn::parse_quote! {
            #[asset(section)]
            struct S { #[serde(rename = "y")] x: u8 }
        };
        assert!(expand(&input).is_err());
    }

    #[test]
    fn a_section_field_clamps_under_its_key() {
        let out = expanded(syn::parse_quote! {
            struct GraphicsConfig {
                #[asset(section)]
                #[serde(default)]
                shadows: ShadowsOverrides,
                plain: u32,
            }
        });
        assert!(
            out.contains("clamp_params (& mut self . shadows , & mut nested)"),
            "{out}"
        );
        assert!(out.contains("Some (\"shadows\")"), "{out}");
    }
}
