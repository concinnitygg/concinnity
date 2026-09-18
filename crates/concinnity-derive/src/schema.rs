//! The static schema `#[derive(AssetFields)]` emits for a struct, behind the
//! consuming crate's `schema` feature: its doc, and per authored field the key,
//! doc, JSON-shaped type and what an omitted key deserializes to.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Field, GenericArgument, PathArguments, Type};

use crate::docs::doc_text;
use crate::serde_attrs::{ContainerAttrs, FieldAttrs, SerdeDefault};

/// One authored field, as read by the derive.
pub(crate) struct AuthoredField<'a> {
    pub(crate) field: &'a Field,
    pub(crate) attrs: FieldAttrs,
    /// The authored key; empty for a flattened field.
    pub(crate) key: String,
}

/// The `Schema` and `Described` impls for a struct with these authored fields.
pub(crate) fn struct_schema(
    input: &DeriveInput,
    container: &ContainerAttrs,
    fields: &[AuthoredField<'_>],
) -> TokenStream {
    let schema = schema_path();
    let name = input.ident.to_string();
    let doc = doc_text(&input.attrs);
    let entries = fields.iter().map(|f| field_entry(f, container));
    let default = match &container.default {
        None => quote!(::core::option::Option::None),
        Some(default) => {
            let value = default_value(quote!(Self), default);
            quote!(::core::option::Option::Some(|| #value))
        }
    };
    let body = quote! {
        #schema::TypeSchema {
            name: #name,
            doc: #doc,
            body: #schema::Body::Fields(&[#(#entries),*]),
            default: #default,
        }
    };
    schema_impls(input, &body, quote!(Nested))
}

/// The `Schema` impl holding `body`, and the `Described` impl that describes
/// the type as the `kind` variant of `FieldType`, both behind the consuming
/// crate's `schema` feature.
pub(crate) fn schema_impls(
    input: &DeriveInput,
    body: &TokenStream,
    kind: TokenStream,
) -> TokenStream {
    let schema = schema_path();
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    quote! {
        #[cfg(feature = "schema")]
        const _: () = {
            impl #impl_generics #schema::Schema for #ident #ty_generics #where_clause {
                const SCHEMA: &'static #schema::TypeSchema = &#body;
            }
            impl #impl_generics #schema::Described for #ident #ty_generics #where_clause {
                const TYPE: #schema::FieldType =
                    #schema::FieldType::#kind(<Self as #schema::Schema>::SCHEMA);
            }
        };
    }
}

pub(crate) fn schema_path() -> TokenStream {
    quote!(::concinnity_core::ecs::schema)
}

fn field_entry(f: &AuthoredField<'_>, container: &ContainerAttrs) -> TokenStream {
    let schema = schema_path();
    let key = &f.key;
    let doc = doc_text(&f.field.attrs);
    let ty = &f.field.ty;
    let ty_expr = type_expr(ty);
    let default = match &f.attrs.default {
        Some(default) => {
            let value = default_value(quote!(#ty), default);
            quote!(#schema::FieldDefault::Value(|| #value))
        }
        None if container.default.is_some() => quote!(#schema::FieldDefault::Container),
        None if generic_arg(ty, "Option").is_some() && !f.attrs.custom_deserialize => {
            quote!(#schema::FieldDefault::Null)
        }
        None => quote!(#schema::FieldDefault::Required),
    };
    quote! {
        #schema::FieldSchema {
            key: #key,
            doc: #doc,
            ty: || #ty_expr,
            default: #default,
        }
    }
}

// The default a serde `default` attribute names, boxed for serializing when
// `ty` can be serialized.
fn default_value(ty: TokenStream, default: &SerdeDefault) -> TokenStream {
    let schema = schema_path();
    let make = match default {
        SerdeDefault::Trait => quote!(<#ty as ::core::default::Default>::default()),
        SerdeDefault::Path(path) => quote!(#path()),
    };
    quote! {{
        #[allow(unused_imports)]
        use #schema::probe::{DefaultByNothing as _, DefaultBySerialize as _};
        let value: #ty = #make;
        (&&#schema::probe::Probe::<#ty>::NEW).boxed(value)
    }}
}

// The `FieldType` expression for a field type. The containers serde reads
// structurally (`Option`, `Vec`, `Box`, arrays) are unwrapped by their
// spelling, the way serde itself recognizes `Option`; any other type is
// classified by the traits it implements.
fn type_expr(ty: &Type) -> TokenStream {
    let schema = schema_path();
    match ty {
        Type::Array(array) => {
            let elem = type_expr(&array.elem);
            let len = &array.len;
            quote! {
                #schema::FieldType::Array {
                    elem: || #elem,
                    len: ::core::option::Option::Some(#len),
                }
            }
        }
        Type::Reference(reference) => type_expr(&reference.elem),
        Type::Paren(paren) => type_expr(&paren.elem),
        Type::Group(group) => type_expr(&group.elem),
        _ => {
            if let Some(inner) = generic_arg(ty, "Option") {
                let inner = type_expr(inner);
                quote!(#schema::FieldType::Optional(|| #inner))
            } else if let Some(inner) = generic_arg(ty, "Vec") {
                let elem = type_expr(inner);
                quote! {
                    #schema::FieldType::Array {
                        elem: || #elem,
                        len: ::core::option::Option::None,
                    }
                }
            } else if let Some(inner) = generic_arg(ty, "Box") {
                type_expr(inner)
            } else {
                quote! {{
                    #[allow(unused_imports)]
                    use #schema::probe::{
                        TypeByDescription as _, TypeByNothing as _, TypeByReference as _,
                    };
                    (&&&#schema::probe::Probe::<#ty>::NEW).field_type()
                }}
            }
        }
    }
}

// The single type argument of `ty` when its last path segment is `wrapper`.
fn generic_arg<'t>(ty: &'t Type, wrapper: &str) -> Option<&'t Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let segment = path.path.segments.last()?;
    if segment.ident != wrapper {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    });
    let first = types.next()?;
    types.next().is_none().then_some(first)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ty(tokens: TokenStream) -> String {
        type_expr(&syn::parse2(tokens).expect("a type")).to_string()
    }

    #[test]
    fn containers_unwrap_by_spelling() {
        let out = ty(quote!(Option<Vec<[f32; 3]>>));
        assert!(
            out.starts_with(":: concinnity_core :: ecs :: schema :: FieldType :: Optional"),
            "{out}"
        );
        assert!(
            out.contains("len : :: core :: option :: Option :: None"),
            "{out}"
        );
        assert!(
            out.contains("len : :: core :: option :: Option :: Some (3)"),
            "{out}"
        );
        assert!(out.contains("Probe :: < f32 > :: NEW"), "{out}");
        assert!(!out.contains("Probe :: < Vec"), "{out}");
    }

    #[test]
    fn a_const_array_length_is_kept_as_written() {
        let out = ty(quote!([f32; PARAMS_LEN]));
        assert!(out.contains("Some (PARAMS_LEN)"), "{out}");
    }

    #[test]
    fn a_box_is_its_contents_and_a_map_is_a_leaf() {
        let out = ty(quote!(Box<bool>));
        assert!(out.contains("Probe :: < bool > :: NEW"), "{out}");
        let out = ty(quote!(BTreeMap<String, f32>));
        assert!(
            out.contains("Probe :: < BTreeMap < String , f32 > > :: NEW"),
            "{out}"
        );
    }

    #[test]
    fn a_leaf_tries_reference_then_description_then_nothing() {
        let out = ty(quote!(Ref<Prop>));
        assert!(out.contains("TypeByDescription as _"), "{out}");
        assert!(out.contains("(&& & :: concinnity_core"), "{out}");
    }
}
