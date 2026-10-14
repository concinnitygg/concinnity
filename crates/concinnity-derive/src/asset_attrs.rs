//! The `#[asset(...)]` field attribute: `owned_file` marks a path field as a
//! file the asset owns, `default = <expr>` states the field's default, and
//! `min = <expr>` / `max = <expr>` / `apply = <class>` declare it a parameter:
//! the range an authored number is clamped into, and when a change to it takes
//! effect (`live`, `rebuild`, `variant` or `restart`). `section` on a struct
//! generates its sparse override twin; on a field whose type is such a twin,
//! it clamps that section's parameters along with the parent's.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, Expr, Ident};

/// What a field's `#[asset(...)]` attributes declare.
#[derive(Default)]
pub(crate) struct AssetAttrs {
    /// The field holds a file the asset owns.
    pub(crate) owned_file: bool,
    /// The field's default, when it states one.
    pub(crate) default: Option<Expr>,
    /// The smallest value the field accepts.
    pub(crate) min: Option<Expr>,
    /// The largest value the field accepts.
    pub(crate) max: Option<Expr>,
    /// When a change to the field takes effect.
    pub(crate) apply: Option<Ident>,
    /// On a struct, generate its sparse override twin; on a field, the field
    /// holds such a twin.
    pub(crate) section: bool,
}

const APPLY_CLASSES: [(&str, &str); 4] = [
    ("live", "Live"),
    ("rebuild", "Rebuild"),
    ("variant", "Variant"),
    ("restart", "Restart"),
];

impl AssetAttrs {
    /// Whether the field declares a range or an apply class.
    pub(crate) fn is_param(&self) -> bool {
        self.min.is_some() || self.max.is_some() || self.apply.is_some()
    }

    /// Whether the field declares a range its values are clamped into.
    pub(crate) fn is_ranged(&self) -> bool {
        self.min.is_some() || self.max.is_some()
    }

    /// The field's `ParamMeta` as a const expression.
    pub(crate) fn meta_expr(&self) -> TokenStream {
        let fields = quote!(::concinnity_core::ecs::asset_fields);
        let bound = |expr: &Option<Expr>| match expr {
            Some(expr) => quote!(::core::option::Option::Some((#expr) as f64)),
            None => quote!(::core::option::Option::None),
        };
        let (min, max) = (bound(&self.min), bound(&self.max));
        let apply = match &self.apply {
            Some(class) => {
                let variant = APPLY_CLASSES
                    .iter()
                    .find(|(name, _)| class == name)
                    .map(|(_, variant)| Ident::new(variant, class.span()))
                    .expect("parsing accepts only a known apply class");
                quote!(::core::option::Option::Some(#fields::ApplyClass::#variant))
            }
            None => quote!(::core::option::Option::None),
        };
        quote! {
            #fields::ParamMeta {
                range: #fields::ParamRange { min: #min, max: #max },
                apply: #apply,
            }
        }
    }
}

/// Parse every `#[asset(...)]` attribute on a field. Any other key is an
/// error, so a misspelling cannot silently drop what it meant to declare.
pub(crate) fn asset_attrs(attrs: &[Attribute]) -> syn::Result<AssetAttrs> {
    let mut out = AssetAttrs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("asset")) {
        attr.parse_nested_meta(|meta| {
            let once = |taken: bool, key: &str| match taken {
                true => Err(meta.error(format!("a field states one `{key}`"))),
                false => Ok(()),
            };
            if meta.path.is_ident("owned_file") {
                out.owned_file = true;
            } else if meta.path.is_ident("section") {
                out.section = true;
            } else if meta.path.is_ident("default") {
                once(out.default.is_some(), "default")?;
                out.default = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("min") {
                once(out.min.is_some(), "min")?;
                out.min = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("max") {
                once(out.max.is_some(), "max")?;
                out.max = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("apply") {
                once(out.apply.is_some(), "apply")?;
                let class: Ident = meta.value()?.parse()?;
                if !APPLY_CLASSES.iter().any(|(name, _)| class == name) {
                    return Err(syn::Error::new(
                        class.span(),
                        "expected `live`, `rebuild`, `variant` or `restart`",
                    ));
                }
                out.apply = Some(class);
            } else {
                return Err(meta.error(
                    "expected `owned_file`, `section`, `default = <expr>`, \
                     `min = <expr>`, `max = <expr>` or `apply = <class>`",
                ));
            }
            Ok(())
        })?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs(field: syn::Field) -> syn::Result<AssetAttrs> {
        asset_attrs(&field.attrs)
    }

    #[test]
    fn owned_file_is_read_among_other_attributes() {
        let field: syn::Field = syn::parse_quote! {
            #[serde(default)]
            #[asset(owned_file)]
            source: String
        };
        assert!(attrs(field).unwrap().owned_file);
        let plain: syn::Field = syn::parse_quote! { #[serde(default)] source: String };
        let plain = attrs(plain).unwrap();
        assert!(!plain.owned_file && plain.default.is_none() && !plain.is_param());
    }

    #[test]
    fn a_default_is_any_expression() {
        let field: syn::Field = syn::parse_quote! {
            #[asset(default = [0.0, 4.0, 0.0])]
            position: [f32; 3]
        };
        let default = attrs(field).unwrap().default.expect("a default");
        assert_eq!(quote::quote!(#default).to_string(), "[0.0 , 4.0 , 0.0]");
        let field: syn::Field = syn::parse_quote! {
            #[asset(owned_file, default = "a.glb")]
            source: String
        };
        let both = attrs(field).unwrap();
        assert!(both.owned_file && both.default.is_some());
    }

    #[test]
    fn a_range_and_an_apply_class_are_read_beside_a_default() {
        let field: syn::Field = syn::parse_quote! {
            #[asset(default = 1.0, min = -LIMIT, max = 4, apply = live)]
            strength: f32
        };
        let a = attrs(field).unwrap();
        assert!(a.is_param() && a.is_ranged());
        let min = a.min.as_ref().expect("a min");
        assert_eq!(quote::quote!(#min).to_string(), "- LIMIT");
        assert_eq!(a.apply.as_ref().expect("a class").to_string(), "live");
        let meta = a.meta_expr().to_string();
        assert!(
            meta.contains("min : :: core :: option :: Option :: Some ((- LIMIT) as f64)"),
            "{meta}"
        );
        assert!(
            meta.contains("max : :: core :: option :: Option :: Some ((4) as f64)"),
            "{meta}"
        );
        assert!(meta.contains("ApplyClass :: Live"), "{meta}");

        let field: syn::Field = syn::parse_quote! { #[asset(apply = restart)] hdr: bool };
        let a = attrs(field).unwrap();
        assert!(a.is_param() && !a.is_ranged());
        let meta = a.meta_expr().to_string();
        assert!(
            meta.contains("min : :: core :: option :: Option :: None"),
            "{meta}"
        );
        assert!(meta.contains("ApplyClass :: Restart"), "{meta}");
    }

    #[test]
    fn every_apply_class_parses() {
        for class in ["live", "rebuild", "variant", "restart"] {
            let class = Ident::new(class, proc_macro2::Span::call_site());
            let field: syn::Field = syn::parse_quote! { #[asset(apply = #class)] x: u32 };
            assert!(attrs(field).unwrap().apply.is_some());
        }
    }

    #[test]
    fn an_unknown_key_class_or_a_repeated_key_is_an_error() {
        let refused: [syn::Field; 5] = [
            syn::parse_quote! { #[asset(owned)] source: String },
            syn::parse_quote! { #[asset(apply = later)] n: u8 },
            syn::parse_quote! { #[asset(default = 1)] #[asset(default = 2)] n: u8 },
            syn::parse_quote! { #[asset(min = 0, min = 1)] n: u8 },
            syn::parse_quote! { #[asset(apply = live, apply = restart)] n: u8 },
        ];
        for field in refused {
            assert!(attrs(field).is_err());
        }
    }
}
