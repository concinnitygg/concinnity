//! The `#[asset(...)]` field attribute: `owned_file` marks a path field as a
//! file the asset owns, and `default = <expr>` states the field's default.

use syn::{Attribute, Expr};

/// What a field's `#[asset(...)]` attributes declare.
#[derive(Default)]
pub(crate) struct AssetAttrs {
    /// The field holds a file the asset owns.
    pub(crate) owned_file: bool,
    /// The field's default, when it states one.
    pub(crate) default: Option<Expr>,
}

/// Parse every `#[asset(...)]` attribute on a field. Any other key is an
/// error, so a misspelling cannot silently drop what it meant to declare.
pub(crate) fn asset_attrs(attrs: &[Attribute]) -> syn::Result<AssetAttrs> {
    let mut out = AssetAttrs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("asset")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("owned_file") {
                out.owned_file = true;
                Ok(())
            } else if meta.path.is_ident("default") {
                if out.default.is_some() {
                    return Err(meta.error("a field states one `default`"));
                }
                out.default = Some(meta.value()?.parse()?);
                Ok(())
            } else {
                Err(meta.error("expected `owned_file` or `default = <expr>`"))
            }
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
        assert!(!plain.owned_file && plain.default.is_none());
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
    fn an_unknown_key_or_a_second_default_is_an_error() {
        let field: syn::Field = syn::parse_quote! { #[asset(owned)] source: String };
        assert!(attrs(field).is_err());
        let field: syn::Field = syn::parse_quote! {
            #[asset(default = 1)]
            #[asset(default = 2)]
            n: u8
        };
        assert!(attrs(field).is_err());
    }
}
